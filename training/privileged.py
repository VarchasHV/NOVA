"""Fixed-size ground-truth vector for the asymmetric critic (training only)."""

from __future__ import annotations

import math

import numpy as np

NUM_ENEMIES = 5
PRIV_DIM = 18 + 4 * 4 + NUM_ENEMIES * 4


def _relative(points: np.ndarray, x: float, y: float, heading: float) -> np.ndarray:
    """(N, 4): [distance/1024, sin bearing, cos bearing, distance] sorted nearest first."""
    if len(points) == 0:
        return np.zeros((0, 4))
    dx, dy = points[:, 0] - x, points[:, 1] - y
    c, s = math.cos(heading), math.sin(heading)
    fwd, left = dx * c + dy * s, -dx * s + dy * c
    dist = np.hypot(fwd, left)
    safe = np.maximum(dist, 1.0)
    out = np.stack([np.minimum(dist / 1024.0, 8.0), left / safe, fwd / safe, dist], axis=1)
    return out[np.argsort(dist, kind="stable")]


def privileged_vector(ws: dict, meta, totals: dict, spawn: tuple[float, float], sectors_seen: int, mirror: bool) -> np.ndarray:
    x, y, heading = ws["x"], ws["y"], math.radians(ws["angle"])
    v = np.zeros(PRIV_DIM, np.float32)
    closed = ~ws["door_open"]
    targets = (
        ws["key_positions"],
        meta.door_centers[closed & (meta.door_locks > 0)],
        meta.door_centers[closed & (meta.door_locks == 0)],
        meta.exits,
    )
    m = -1.0 if mirror else 1.0
    v[:18] = [
        (x - spawn[0]) / 2048.0, m * (y - spawn[1]) / 2048.0,
        m * math.sin(heading), math.cos(heading),
        ws["health"] / 100.0, ws["armor"] / 100.0, min(ws["ammo"], 200.0) / 50.0,
        *ws["keys"],
        1.0 - ws["tic"] / max(ws["episode_ticks"], 1),
        ws["kills"] / max(totals["monsters"], 1),
        sectors_seen / max(meta.num_sectors, 1),
        len(ws["monsters"]) / 20.0,
        ws["damage_taken"] / 100.0,
    ]
    i = 18
    for points in targets:
        rel = _relative(points, x, y, heading)
        if len(rel):
            v[i : i + 4] = (1.0, rel[0, 0], m * rel[0, 1], rel[0, 2])
        i += 4
    rel = _relative(ws["monsters"], x, y, heading)[:NUM_ENEMIES]
    for row in rel:
        v[i : i + 4] = (1.0, row[0], m * row[1], row[2])
        i += 4
    return v
