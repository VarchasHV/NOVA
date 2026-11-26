"""Perception encoders: one small network per observation type, then fusion into one vector.

All inputs have a single batch dimension (B, ...). Sequences are handled by flattening (T, B) first.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import obs as O
from .config import ModelConfig


def mlp(inp: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.SiLU(), nn.Linear(hidden, out), nn.SiLU())


class RayEncoder(nn.Module):
    """1-D convolution around the 360 degree ray ring.

    Circular padding: ray 63 is next to ray 0, so a wall behind-right looks continuous. The output
    is flattened (not pooled) so that *where* something is stays visible to later layers.
    """

    OUT = 128

    def __init__(self):
        super().__init__()
        layers, inp = [], 1 + O.NUM_HIT_TYPES
        for out, stride in ((32, 1), (48, 2), (64, 2), (64, 2)):
            layers += [nn.Conv1d(inp, out, 5, stride=stride, padding=2, padding_mode="circular"), nn.SiLU()]
            inp = out
        self.conv = nn.Sequential(*layers)
        self.out = nn.Sequential(nn.Linear(64 * (O.NUM_RAYS // 8), self.OUT), nn.SiLU())

    def forward(self, dist: torch.Tensor, hit: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(hit.long(), O.NUM_HIT_TYPES).float()  # (B, 64, 6)
        x = torch.cat([dist.unsqueeze(-1), one_hot], dim=-1).transpose(1, 2)  # (B, 7, 64)
        return self.out(self.conv(x).flatten(1))


class DepthEncoder(nn.Module):
    OUT = 64

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.SiLU(),
        )
        self.out = nn.Sequential(nn.Linear(32 * 3 * 4, self.OUT), nn.SiLU())

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.out(self.conv(depth.unsqueeze(1)).flatten(1))


class EntityEncoder(nn.Module):
    """Encodes a *set* of up to 32 visible things: embed each, let them attend to each other, pool.

    A learned "null" entity is always present, so a map view with nothing in it is still valid input
    (attention over an entirely masked set would produce NaN).
    """

    OUT = 128

    def __init__(self, dim: int = 64, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.type_emb = nn.Embedding(O.NUM_ENTITY_TYPES, 16)
        self.embed = mlp(O.ENTITY_FEATURES + 16, dim, dim)
        self.null = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.out = nn.Sequential(nn.Linear(2 * dim, self.OUT), nn.SiLU())
        # [distance/1024, sin, cos, rel_vel_forward, rel_vel_right]
        self.register_buffer("scale", torch.tensor([1.0, 1.0, 1.0, 1 / 20.0, 1 / 20.0]), persistent=False)

    def forward(self, feat: torch.Tensor, etype: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b = feat.shape[0]
        x = self.embed(torch.cat([feat * self.scale, self.type_emb(etype.long())], dim=-1))
        x = torch.cat([self.null.expand(b, -1, -1), x], dim=1)  # (B, 33, D)
        valid = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=mask.device), mask > 0.5], dim=1)
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        split = lambda t: t.reshape(b, t.shape[1], self.heads, -1).transpose(1, 2)
        att = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=valid[:, None, None, :])
        x = x + self.proj(att.transpose(1, 2).reshape(b, valid.shape[1], -1))
        m = valid.unsqueeze(-1)
        mean = (x * m).sum(1) / m.sum(1)
        mx = x.masked_fill(~m, torch.finfo(x.dtype).min).max(1).values
        return self.out(torch.cat([mean, mx], dim=-1))


def normalize_vitals(v: torch.Tensor) -> torch.Tensor:
    """Fixed, hand-chosen scaling so every input is roughly in [-1, 1]. Returns (B, 24)."""
    health, armor = v[:, 0:1] / 100.0, v[:, 1:2] / 100.0
    weapon = F.one_hot(v[:, 2].long().clamp(0, 7), 8).float()
    sel_ammo = (v[:, 3:4] / 50.0).clamp(-1.0, 4.0)
    ammo = v[:, 4:8] / v.new_tensor([200.0, 50.0, 50.0, 300.0])
    keys = v[:, 8:14]
    vel = v[:, 14:17] / 16.0
    low_health = (v[:, 0:1] < 30).float()
    return torch.cat([health, armor, weapon, sel_ammo, ammo, keys, vel, low_health], dim=-1)


VITALS_FEATURES = 2 + 8 + 1 + 4 + 6 + 3 + 1


class ObservationEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.use_depth = cfg.use_depth
        self.rays = RayEncoder()
        self.entities = EntityEncoder()
        self.vitals = mlp(VITALS_FEATURES + O.NUM_BUTTONS, 128, 64)
        total = RayEncoder.OUT + EntityEncoder.OUT + 64
        if cfg.use_depth:
            self.depth = DepthEncoder()
            total += DepthEncoder.OUT
        self.fuse = nn.Sequential(nn.LayerNorm(total), nn.Linear(total, cfg.d_model), nn.SiLU())

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [
            self.rays(obs["ray_dist"], obs["ray_type"]),
            self.entities(obs["ent_feat"], obs["ent_type"], obs["ent_mask"]),
            self.vitals(torch.cat([normalize_vitals(obs["vitals"]), obs["prev_action"]], dim=-1)),
        ]
        if self.use_depth:
            parts.append(self.depth(obs["depth"]))
        return self.fuse(torch.cat(parts, dim=-1))
