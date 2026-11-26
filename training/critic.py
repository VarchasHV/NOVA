"""Asymmetric critic: during training the value function may see the whole truth.

It reads the actor's recurrent features plus the privileged vector and predicts two returns
(extrinsic, intrinsic). It is only used to compute advantages, never exported or loaded at inference.
"""

from __future__ import annotations

import torch
from torch import nn

from .privileged import PRIV_DIM


class PrivilegedCritic(nn.Module):
    def __init__(self, core_size: int, detach_actor: bool = False, hidden: int = 256):
        super().__init__()
        self.detach_actor = detach_actor
        self.priv_net = nn.Sequential(nn.Linear(PRIV_DIM, 128), nn.SiLU())
        self.net = nn.Sequential(
            nn.Linear(core_size + 128, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2)
        )

    def forward(self, actor_hidden: torch.Tensor, priv: torch.Tensor) -> torch.Tensor:
        """-> (..., 2) values for the extrinsic and intrinsic reward streams."""
        if self.detach_actor:
            actor_hidden = actor_hidden.detach()
        return self.net(torch.cat([actor_hidden, self.priv_net(priv.clamp(-10, 10))], dim=-1))
