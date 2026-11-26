"""Left/right mirroring of whole episodes: free extra data and no turning bias.

The environment worker picks the flag at reset and applies it for the entire episode, so recurrent
state and memory stay consistent. Everything lateral flips together.
"""

from __future__ import annotations

import numpy as np

from models.obs import NUM_RAYS

RAY_ORDER = (-np.arange(NUM_RAYS)) % NUM_RAYS  # ray i (counter-clockwise) <-> ray 64 - i
BUTTON_ORDER = np.array([0, 1, 3, 2, 5, 4, 6, 7, 8])  # swap strafe and turn left/right


def mirror_obs(obs: dict) -> dict:
    """Mirror compact observation arrays (see models.obs). Returns new arrays."""
    out = dict(obs)
    out["ray_dist"] = obs["ray_dist"][RAY_ORDER].copy()
    out["ray_type"] = obs["ray_type"][RAY_ORDER].copy()
    feat = obs["ent_feat"].copy()
    feat[:, 1] *= -1  # sin of the bearing
    feat[:, 4] *= -1  # lateral velocity
    out["ent_feat"] = feat
    vitals = obs["vitals"].copy()
    vitals[15] *= -1  # velocity to the right
    out["vitals"] = vitals
    out["prev_action"] = mirror_buttons(obs["prev_action"])
    if "depth" in obs:
        out["depth"] = obs["depth"][:, ::-1].copy()
    return out


def mirror_buttons(buttons: np.ndarray) -> np.ndarray:
    return np.asarray(buttons)[..., BUTTON_ORDER].copy()


def mirror_motion(motion: np.ndarray) -> np.ndarray:
    """(forward, right, turn) -> mirrored."""
    return motion * np.array([1.0, -1.0, -1.0], motion.dtype)
