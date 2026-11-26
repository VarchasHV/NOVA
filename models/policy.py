"""Factorised action heads for the MultiBinary(9) action space.

Opposite buttons share one three-way head (forward/back, strafe, turn), so the policy never wastes
samples pressing both. `idx` is (B, 6) with one choice per head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

HEAD_NAMES = ("move", "strafe", "turn", "attack", "use", "weapon")
HEAD_SIZES = (3, 3, 3, 2, 2, 2)
NUM_LOGITS = sum(HEAD_SIZES)
NUM_HEADS = len(HEAD_SIZES)
# Starting preferences (initialisation only, fully overwritten by training): walk more than
# stand, and rarely cycle weapons.
INITIAL_BIAS = (0.0, 0.8, -1.0, 0.5, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, -1.0, 0.0, -1.5, 0.0, -4.0)


class ActionHead(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU())
        self.logits = nn.Linear(hidden, NUM_LOGITS)
        with torch.no_grad():
            self.logits.weight.mul_(0.01)
            self.logits.bias.copy_(torch.tensor(INITIAL_BIAS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(self.net(x))


def _log_probs(logits: torch.Tensor) -> list[torch.Tensor]:
    return [F.log_softmax(part, dim=-1) for part in logits.split(HEAD_SIZES, dim=-1)]


def sample(logits: torch.Tensor, temperature: float = 1.0, generator=None) -> torch.Tensor:
    out = []
    for part in logits.split(HEAD_SIZES, dim=-1):
        p = F.softmax(part / max(temperature, 1e-3), dim=-1)
        out.append(torch.multinomial(p.reshape(-1, p.shape[-1]), 1, generator=generator).reshape(p.shape[:-1]))
    return torch.stack(out, dim=-1)


def greedy(logits: torch.Tensor) -> torch.Tensor:
    return torch.stack([part.argmax(dim=-1) for part in logits.split(HEAD_SIZES, dim=-1)], dim=-1)


def log_prob(logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Joint log probability = sum over the independent heads."""
    return sum(lp.gather(-1, idx[..., i : i + 1].long()).squeeze(-1) for i, lp in enumerate(_log_probs(logits)))


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-head entropy, shape (..., NUM_HEADS)."""
    return torch.stack([-(lp.exp() * lp).sum(-1) for lp in _log_probs(logits)], dim=-1)


def to_buttons(idx: torch.Tensor) -> torch.Tensor:
    """(..., 6) head choices -> (..., 9) buttons in the official order."""
    move, strafe, turn, attack, use, weapon = idx.unbind(-1)
    return torch.stack(
        [move == 1, move == 2, strafe == 1, strafe == 2, turn == 1, turn == 2, attack == 1, use == 1, weapon == 1],
        dim=-1,
    ).float()
