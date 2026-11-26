"""Turn the official observation dict into the compact, clean arrays the network reads.

Used by both the training workers and the submitted agent, so the two can never drift apart.
Only fields from the official observation (plus our own previous action) are used.
"""

from __future__ import annotations

import numpy as np

NUM_RAYS = 64
NUM_HIT_TYPES = 6
MAX_ENTITIES = 32
ENTITY_FEATURES = 5
NUM_ENTITY_TYPES = 12
VITALS_DIM = 17
NUM_BUTTONS = 9
DEPTH_H, DEPTH_W = 24, 32
RAY_RANGE = 1024.0  # map units


def obs_spec(use_depth: bool) -> dict[str, tuple[tuple[int, ...], type]]:
    """name -> (shape, dtype) of every compact observation array."""
    spec = {
        "ray_dist": ((NUM_RAYS,), np.float32),
        "ray_type": ((NUM_RAYS,), np.int64),
        "ent_feat": ((MAX_ENTITIES, ENTITY_FEATURES), np.float32),
        "ent_type": ((MAX_ENTITIES,), np.int64),
        "ent_mask": ((MAX_ENTITIES,), np.float32),
        "vitals": ((VITALS_DIM,), np.float32),
        "prev_action": ((NUM_BUTTONS,), np.float32),
    }
    if use_depth:
        spec["depth"] = ((DEPTH_H, DEPTH_W), np.float32)
    return spec


def _clean(x, low: float, high: float) -> np.ndarray:
    a = np.nan_to_num(np.asarray(x, np.float32), nan=0.0, posinf=high, neginf=low)
    return np.clip(a, low, high)


def preprocess(obs: dict, prev_action: np.ndarray, use_depth: bool) -> dict[str, np.ndarray]:
    """Official observation -> compact arrays. Never raises on NaN/Inf or out-of-range ids."""
    rays, ents = obs["rays"], obs["visible_entities"]
    out = {
        "ray_dist": _clean(rays["distance"], 0.0, 1.0),
        "ray_type": np.clip(np.asarray(rays["hit_type"], np.int64), 0, NUM_HIT_TYPES - 1),
        "ent_feat": _clean(ents["features"], -64.0, 64.0),
        "ent_type": np.clip(np.asarray(ents["entity_type"], np.int64), 0, NUM_ENTITY_TYPES - 1),
        "ent_mask": _clean(ents["mask"], 0.0, 1.0),
        "vitals": _clean(obs["vitals"], -10000.0, 10000.0),
        "prev_action": np.asarray(prev_action, np.float32).copy(),
    }
    if use_depth:
        depth = np.asarray(obs["depth"], np.float32)  # (120, 160)
        # Average 5x5 blocks: 120x160 -> 24x32.
        out["depth"] = depth.reshape(DEPTH_H, 5, DEPTH_W, 5).mean(axis=(1, 3)).astype(np.float32) / 255.0
    return out
