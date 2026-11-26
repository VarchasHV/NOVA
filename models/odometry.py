"""Neural odometry: "how far did I just move and turn?"

The observation has no heading and no position, so we estimate them. Each decision the network looks
at the previous and current ray scans, the player's own velocity and the last buttons pressed, and
predicts the motion in the previous frame: (forward, right, turn). Adding these up gives a pose
relative to where the episode started. It drifts, which is fine because memory slots are room-sized.

Trained only by supervision from the true pose, which exists during training and nowhere else.
"""

from __future__ import annotations

import torch
from torch import nn

from . import obs as O

# Output units: translation in RAY_RANGE (1024 map units), rotation in radians.
TICS = 4  # game tics per decision
TRANSLATION_SCALE = 32.0 / O.RAY_RANGE
TURN_SCALE = 0.25
# inputs besides the two ray scans: prev action(9), the action before it(9), velocity now(2), before(2)
EXTRA = 9 + 9 + 2 + 2


class Odometry(nn.Module):
    def __init__(self):
        super().__init__()
        layers, inp = [], 3  # previous scan, current scan, their difference
        for out, stride in ((24, 1), (32, 2), (32, 2), (16, 2)):
            layers += [nn.Conv1d(inp, out, 5, stride=stride, padding=2, padding_mode="circular"), nn.SiLU()]
            inp = out
        self.conv = nn.Sequential(*layers)
        self.net = nn.Sequential(
            nn.Linear(16 * (O.NUM_RAYS // 8) + EXTRA, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU()
        )
        self.head = nn.Linear(128, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, prev_ray, ray, action, prev_action, vel, prev_vel) -> torch.Tensor:
        """prev_ray/ray (B,64), action = buttons held during this motion (B,9), prev_action = the ones
        before that (turning accelerates while held), vel/prev_vel = (forward, right) units per tic.

        Returns (B,3) normalised motion: divide by the scales' meaning in `to_motion`.
        """
        scans = torch.stack([prev_ray, ray, 10.0 * (ray - prev_ray)], dim=1)
        extra = torch.cat([action, prev_action, vel / 16.0, prev_vel / 16.0], dim=-1)
        residual = self.head(self.net(torch.cat([self.conv(scans).flatten(1), extra], dim=-1)))
        # Starting guess: translation = average velocity x tics. The network learns the correction.
        prior = 0.5 * (vel + prev_vel) * TICS / O.RAY_RANGE / TRANSLATION_SCALE
        return torch.cat([prior + residual[:, :2], residual[:, 2:]], dim=-1)

    @staticmethod
    def to_motion(raw: torch.Tensor) -> torch.Tensor:
        """Normalised output -> (forward, right) in RAY_RANGE units and turn in radians."""
        return raw * raw.new_tensor([TRANSLATION_SCALE, TRANSLATION_SCALE, TURN_SCALE])

    @staticmethod
    def to_target(motion: torch.Tensor) -> torch.Tensor:
        return motion / motion.new_tensor([TRANSLATION_SCALE, TRANSLATION_SCALE, TURN_SCALE])


def integrate_pose(pose: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
    """Dead reckoning. pose (B,3) = x, y, heading in the episode-start frame (counter-clockwise).
    motion (B,3) = forward, right, turn measured in the frame of the *previous* pose."""
    x, y, th = pose.unbind(-1)
    fwd, right, turn = motion.unbind(-1)
    c, s = torch.cos(th), torch.sin(th)
    th = th + turn
    th = torch.remainder(th + torch.pi, 2 * torch.pi) - torch.pi  # keep in [-pi, pi]
    return torch.stack([x + fwd * c + right * s, y + fwd * s - right * c, th], dim=-1)
