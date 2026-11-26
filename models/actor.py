"""The full actor: observation -> encoders -> odometry -> streamed memory -> core -> action heads.

The same code runs one decision at a time (`step`, used to collect experience and in the submitted
agent) and over whole sequences (`unroll`, used by PPO). It only ever sees the official observation
plus its own previous action.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from . import obs as O
from .config import ModelConfig
from .core import RecurrentCore
from .encoders import ObservationEncoder
from .odometry import Odometry, integrate_pose
from .policy import ActionHead
from .streamed_memory import StreamedMemory

State = dict[str, torch.Tensor]


class StepOutput(NamedTuple):
    logits: torch.Tensor
    hidden: torch.Tensor  # core output, also read by the training-only critic
    state: State
    pose: torch.Tensor
    info: dict


class StreamedMindActor(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ObservationEncoder(cfg)
        self.odometry = Odometry()
        self.memory = StreamedMemory(cfg) if cfg.memory else None
        readout = cfg.readout_dim if cfg.memory else 0
        self.core = RecurrentCore(cfg, cfg.d_model + readout)
        self.head = ActionHead(cfg.core_size + cfg.d_model + readout)
        self.aux = nn.Linear(cfg.core_size, cfg.aux_dim)  # training signal only

    # ------------------------------------------------------------------ state
    def initial_state(self, batch: int, device=None) -> State:
        device = device or next(self.parameters()).device
        state = self.core.initial_state(batch, device)
        if self.memory is not None:
            state.update(self.memory.initial_state(batch, device))
        state.update(
            pose=torch.zeros(batch, 3, device=device),
            prev_ray=torch.zeros(batch, O.NUM_RAYS, device=device),
            prev_vel=torch.zeros(batch, 2, device=device),
            prev_cmd=torch.zeros(batch, O.NUM_BUTTONS, device=device),
            started=torch.zeros(batch, device=device),
        )
        return state

    @staticmethod
    def reset_where(state: State, reset: torch.Tensor) -> State:
        """Zero the state of the batch rows where a new episode begins (reset is (B,) of 0/1)."""
        keep = 1.0 - reset.float()
        return {k: v * keep.view(-1, *([1] * (v.dim() - 1))) for k, v in state.items()}

    # ------------------------------------------------------------------ pieces
    def check_obs(self, obs: dict) -> None:
        """Guard: only official-observation fields may enter the actor (no privileged data, ever)."""
        extra = set(obs) - set(O.obs_spec(self.cfg.use_depth))
        if extra:
            raise ValueError(f"the actor must not receive {sorted(extra)}")

    def estimate_pose(self, obs: dict, state: State) -> torch.Tensor:
        """Dead-reckon from the previous pose. Never receives gradients from the policy."""
        with torch.no_grad():
            vel = obs["vitals"][:, 14:16]
            raw = self.odometry(state["prev_ray"], obs["ray_dist"], obs["prev_action"], state["prev_cmd"], vel, state["prev_vel"])
            motion = self.odometry.to_motion(raw) * state["started"].unsqueeze(-1)  # first frame: no motion
            return integrate_pose(state["pose"], motion)

    def _think(self, feat: torch.Tensor, pose: torch.Tensor, state: State):
        new, info = {}, {}
        parts = [feat]
        if self.memory is not None:
            mem = {k: state[k] for k in state if k.startswith("mem_")}
            readout, mem, info = self.memory(feat, state["h"], pose, mem)
            new.update(mem)
            parts.append(readout)
        new.update(self.core(torch.cat(parts, dim=-1), state))
        logits = self.head(torch.cat([new["h"], *parts], dim=-1))
        return logits, new, info

    # ------------------------------------------------------------------ public
    def step(self, obs: dict, state: State, reset: torch.Tensor | None = None) -> StepOutput:
        self.check_obs(obs)
        if reset is not None:
            state = self.reset_where(state, reset)
        pose = self.estimate_pose(obs, state)
        logits, new, info = self._think(self.encoder(obs), pose, state)
        new.update(
            pose=pose,
            prev_ray=obs["ray_dist"].clone(),
            prev_vel=obs["vitals"][:, 14:16].clone(),
            prev_cmd=obs["prev_action"].clone(),
            started=torch.ones_like(state["started"]),
        )
        return StepOutput(logits, new["h"], new, pose, info)

    def unroll(self, obs: dict, state: State, reset: torch.Tensor, pose: torch.Tensor):
        """Replay a sequence for training. obs values are (T,B,...), reset (T,B), pose (T,B,3): the
        pose estimates recorded while collecting. Returns logits (T,B,L) and hidden (T,B,H)."""
        self.check_obs(obs)
        t, b = reset.shape
        feat = self.encoder({k: v.reshape(t * b, *v.shape[2:]) for k, v in obs.items()}).reshape(t, b, -1)
        logits, hidden = [], []
        for i in range(t):
            state = self.reset_where(state, reset[i])
            out, new, _ = self._think(feat[i], pose[i], state)
            state = {**state, **new}
            logits.append(out)
            hidden.append(new["h"])
        return torch.stack(logits), torch.stack(hidden)
