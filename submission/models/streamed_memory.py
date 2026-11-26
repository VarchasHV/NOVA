"""Streamed spatial memory: a fixed budget of slots anchored to self-estimated positions.

Open-world games keep only a limited number of map chunks loaded, streaming new ones in and evicting
unimportant ones. We do the same with the agent's memory:

* stream in   - far from every slot? allocate a new slot at the estimated position.
* update      - otherwise refresh the nearest slot with a learned, gated GRU write.
* stream out  - memory full? overwrite the slot with the lowest *learned relevance*.
* read        - attention over slots, each described relative to the player's current pose
                ("behind-left, 600 units away"), so no global coordinates are needed.

Relevance is trained by the policy gradient because it also biases the read attention: a slot that
is useful to read gets a higher relevance, and is therefore kept.

What is bookkeeping (not a decision): where a slot is written and its relative position.
What is learned: what gets written, how strongly, what is kept, and what is read.

`mode="soft"` is the fallback if position-anchored writes are not allowed: slot choice is then a
learned soft attention over all slots.

Everything is vectorised over slots; there are no Python loops over them.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import ModelConfig
from .obs import RAY_RANGE

GEO_DIM = 6
State = dict[str, torch.Tensor]


def egocentric(slot_pos: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """Slot positions (B,K,2) seen from pose (B,3): [dist, log dist, sin bearing, cos bearing].
    Bearing is 0 straight ahead and positive to the left, like the official entity angles."""
    delta = slot_pos - pose[:, None, :2]
    c, s = torch.cos(pose[:, 2:3]), torch.sin(pose[:, 2:3])
    fwd = delta[..., 0] * c + delta[..., 1] * s
    left = -delta[..., 0] * s + delta[..., 1] * c
    dist = torch.sqrt(fwd**2 + left**2 + 1e-8)
    # Inside one tenth of a ray range the bearing is meaningless, so fade it out.
    safe = dist.clamp(min=0.1)
    return torch.stack([dist.clamp(max=6.0), torch.log1p(dist), left / safe, fwd / safe], dim=-1)


class StreamedMemory(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.memory_mode not in ("anchored", "soft"):
            raise ValueError(f"memory_mode must be 'anchored' or 'soft', not {cfg.memory_mode!r}")
        self.mode = cfg.memory_mode
        self.k, self.d, self.heads = cfg.slots, cfg.slot_dim, cfg.heads
        self.radius = cfg.new_slot_radius / RAY_RANGE
        ctx = cfg.d_model + cfg.core_size
        d = self.d

        self.write_input = nn.Sequential(nn.Linear(ctx, d), nn.SiLU())
        self.cell = nn.GRUCell(d, d)
        self.gate = nn.Linear(2 * d, 1)
        nn.init.constant_(self.gate.bias, 1.0)  # start by writing
        self.relevance = nn.Sequential(nn.Linear(d + 2, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.relevance[-1].weight)
        nn.init.constant_(self.relevance[-1].bias, 1.0)

        self.key_value = nn.Sequential(nn.Linear(d + GEO_DIM, d), nn.SiLU())
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.query = nn.Linear(ctx, d)
        self.readout = nn.Sequential(nn.Linear(d, cfg.readout_dim), nn.SiLU())

        if self.mode == "soft":
            self.address = nn.Sequential(nn.Linear(2 * d + 4, 32), nn.SiLU(), nn.Linear(32, 1))
            self.free_bias = nn.Parameter(torch.tensor(1.0))

    def initial_state(self, batch: int, device) -> State:
        z = lambda *shape: torch.zeros(batch, *shape, device=device)
        return {
            "mem_pos": z(self.k, 2),
            "mem_content": z(self.k, self.d),
            "mem_age": z(self.k),
            "mem_visits": z(self.k),
            "mem_valid": z(self.k),  # 0/1 when anchored, usage in [0, 1] when soft
        }

    def _relevance(self, content, age, visits) -> torch.Tensor:
        stats = torch.stack([torch.log1p(age) / 6.0, torch.log1p(visits) / 6.0], dim=-1)
        return self.relevance(torch.cat([content, stats], dim=-1)).squeeze(-1)

    # ------------------------------------------------------------------ write
    def _write_anchored(self, w_in, pose, s: State):
        pos, content = s["mem_pos"], s["mem_content"]
        valid = s["mem_valid"] > 0.5
        with torch.no_grad():
            dist = torch.linalg.vector_norm(pos - pose[:, None, :2], dim=-1)
            min_d, nearest = dist.masked_fill(~valid, float("inf")).min(dim=1)
            is_new = min_d > self.radius
            score = self._relevance(content, s["mem_age"], s["mem_visits"]).masked_fill(~valid, float("-inf"))
            victim = score.argmin(dim=1)  # a free slot if there is one, else the least relevant
            target = torch.where(is_new, victim, nearest)
            sel = torch.zeros_like(s["mem_valid"]).scatter_(1, target[:, None], 1.0).bool()
            fresh = sel & is_new[:, None]

        old = content.gather(1, target[:, None, None].expand(-1, 1, self.d)).squeeze(1)
        old = torch.where(is_new[:, None], torch.zeros_like(old), old)
        gate = torch.sigmoid(self.gate(torch.cat([w_in, old], dim=-1)))
        new = old + gate * (self.cell(w_in, old) - old)
        content = torch.where(sel[..., None], new[:, None, :], content)
        pos = torch.where(fresh[..., None], pose[:, None, :2], pos)
        valid_f = torch.where(sel, 1.0, s["mem_valid"])
        age = torch.where(sel, 0.0, (s["mem_age"] + 1.0) * valid_f)
        visits = torch.where(fresh, 0.0, s["mem_visits"]) + sel.float()
        return {"mem_pos": pos, "mem_content": content, "mem_age": age, "mem_visits": visits, "mem_valid": valid_f}, target

    def _write_soft(self, w_in, pose, s: State):
        b, k, d = s["mem_content"].shape
        usage = s["mem_valid"]
        w = w_in[:, None, :].expand(-1, k, -1)
        geo = egocentric(s["mem_pos"], pose) * (usage[..., None] > 0)
        logits = self.address(torch.cat([w, s["mem_content"], geo], dim=-1)).squeeze(-1)
        weights = torch.softmax(logits + self.free_bias * (1.0 - usage), dim=1)
        gate = torch.sigmoid(self.gate(torch.cat([w_in, (weights[..., None] * s["mem_content"]).sum(1)], dim=-1)))
        a = weights * gate  # (B, K) how strongly each slot is written
        cand = self.cell(w.reshape(b * k, d), s["mem_content"].reshape(b * k, d)).reshape(b, k, d)
        content = s["mem_content"] + a[..., None] * (cand - s["mem_content"])
        mass = usage * (1.0 - a)
        pos = (mass[..., None] * s["mem_pos"] + a[..., None] * pose[:, None, :2]) / (mass + a + 1e-8)[..., None]
        new_usage = usage + a * (1.0 - usage)
        age = (s["mem_age"] + 1.0) * (1.0 - a)
        visits = s["mem_visits"] + a
        state = {"mem_pos": pos.detach(), "mem_content": content, "mem_age": age, "mem_visits": visits, "mem_valid": new_usage}
        return state, a.argmax(dim=1)

    # ------------------------------------------------------------------ one decision
    def forward(self, feat: torch.Tensor, hidden: torch.Tensor, pose: torch.Tensor, state: State):
        """feat (B,d_model), hidden (B,core), pose (B,3) -> (readout, new state, info)."""
        ctx = torch.cat([feat, hidden], dim=-1)
        w_in = self.write_input(ctx)
        write = self._write_anchored if self.mode == "anchored" else self._write_soft
        state, target = write(w_in, pose, state)

        b, k = state["mem_age"].shape
        geo = torch.cat(
            [
                egocentric(state["mem_pos"], pose),
                (torch.log1p(state["mem_age"]) / 6.0)[..., None],
                (torch.log1p(state["mem_visits"]) / 6.0)[..., None],
            ],
            dim=-1,
        )
        relevance = self._relevance(state["mem_content"], state["mem_age"], state["mem_visits"])
        kv = self.key_value(torch.cat([state["mem_content"], geo], dim=-1))
        dh = self.d // self.heads
        keys = self.key(kv).view(b, k, self.heads, dh).transpose(1, 2)  # (B, H, K, dh)
        values = self.value(kv).view(b, k, self.heads, dh).transpose(1, 2)
        q = self.query(ctx).view(b, self.heads, 1, dh)
        logits = (q @ keys.transpose(-1, -2)).squeeze(2) / math.sqrt(dh)  # (B, H, K)
        logits = logits + nn.functional.logsigmoid(relevance)[:, None, :]
        if self.mode == "soft":
            logits = logits + torch.log(state["mem_valid"] + 1e-6)[:, None, :]
        # The write above guarantees at least one valid slot per row, so softmax is well defined.
        logits = logits.masked_fill((state["mem_valid"] <= 0)[:, None, :], float("-inf"))
        att = torch.softmax(logits, dim=-1)
        read = (att.unsqueeze(2) @ values).squeeze(2).reshape(b, self.d)
        info = {"attention": att.mean(dim=1), "relevance": torch.sigmoid(relevance), "written_slot": target}
        return self.readout(read), state, info
