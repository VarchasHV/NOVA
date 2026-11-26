"""Recurrent core: a GRU, optionally with a slow "strategic" GRU on top (two timescales)."""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class RecurrentCore(nn.Module):
    def __init__(self, cfg: ModelConfig, input_dim: int):
        super().__init__()
        self.two = cfg.two_timescale
        self.every = cfg.slow_every
        self.size = cfg.core_size
        if self.two:
            self.slow = nn.GRUCell(input_dim, self.size)
            input_dim += self.size
        self.fast = nn.GRUCell(input_dim, self.size)

    def initial_state(self, batch: int, device) -> dict[str, torch.Tensor]:
        state = {"h": torch.zeros(batch, self.size, device=device)}
        if self.two:
            state["h_slow"] = torch.zeros(batch, self.size, device=device)
            state["t"] = torch.zeros(batch, device=device)
        return state

    def forward(self, x: torch.Tensor, state: dict) -> dict:
        new = {}
        if self.two:
            tick = (torch.remainder(state["t"], self.every) == 0).unsqueeze(-1)
            h_slow = torch.where(tick, self.slow(x, state["h_slow"]), state["h_slow"])
            new["h_slow"], new["t"] = h_slow, state["t"] + 1
            x = torch.cat([x, h_slow], dim=-1)
        new["h"] = self.fast(x, state["h"])
        return new
