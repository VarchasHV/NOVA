"""Auxiliary supervision from privileged signals (training only)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.odometry import Odometry

AUX_NAMES = ("key_seen", "locked_door_seen", "exit_near", "stagnating")


def aux_loss(predicted_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary targets that push the recurrent state to remember what matters."""
    return F.binary_cross_entropy_with_logits(predicted_logits, target)


def odometry_loss(odometry: Odometry, batch: dict) -> tuple[torch.Tensor, dict]:
    """batch holds flat (B, ...) tensors of consecutive-step pairs and the true motion."""
    raw = odometry(batch["prev_ray"], batch["ray"], batch["cmd"], batch["prev_cmd"], batch["prev_cmd2"],
                   batch["vel"], batch["prev_vel"])
    target = odometry.to_target(batch["motion"]).clamp(-4.0, 4.0)
    # A small beta keeps the pull strong for small errors: tiny per-step biases add up to large drift.
    loss = F.smooth_l1_loss(raw, target, beta=0.02)
    with torch.no_grad():
        error = (odometry.to_motion(raw) - batch["motion"]).abs()
        stats = {"odom_translation_error_units": float(error[:, :2].mean() * 1024),
                 "odom_turn_error_deg": float(torch.rad2deg(error[:, 2]).mean())}
    return loss, stats
