"""Reward shaped after the official scoring categories, from privileged signals (training only).

Two streams:
  extrinsic  - mirrors the scorer: exit, progression, exploration, combat, survival
  intrinsic  - coarse-grid novelty; annealed away during training

Per-episode caps on the repeatable parts keep the invariant that finishing and progressing are worth
more than farming kills or wandering.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .novelty import EpisodicNovelty


@dataclass
class RewardConfig:
    exit: float = 10.0
    exit_health_bonus: float = 1.0
    exit_time_bonus: float = 2.0
    key: float = 2.0
    unlock: float = 2.0
    door: float = 0.3
    sector: float = 0.2
    sector_cap: float = 4.0  # most exploration reward one episode can give
    kill: float = 0.3
    kill_cap: float = 4.0
    damage: float = 0.01  # per HP
    death: float = 3.0
    time: float = 0.001  # per decision
    stagnation: float = 0.02
    stagnation_window: int = 30  # decisions
    stagnation_radius: float = 64.0
    grid: float = 0.05  # intrinsic
    door_reach: float = 192.0  # only credit doors that open near the player


COMPONENTS = ("exit", "key", "unlock", "door", "sector", "kill", "damage", "death", "time", "stagnation", "grid")


class RewardTracker:
    """Create at episode start with the first world state, then call step() after every action."""

    def __init__(self, cfg: RewardConfig, ws: dict, meta, totals: dict):
        self.cfg, self.meta, self.totals = cfg, meta, totals
        self.keys = ws["keys"].copy()
        self.kills = ws["kills"]
        self.damage = ws["damage_taken"]
        self.opened = set(int(i) for i in ws["door_open"].nonzero()[0])  # already open at start
        self.novelty = EpisodicNovelty()
        self.novelty.new_sector(ws["sector"])
        self.novelty.grid_bonus(ws["x"], ws["y"])
        self.positions = deque([(ws["x"], ws["y"])], maxlen=cfg.stagnation_window)
        self.sector_paid = self.kill_paid = 0.0
        self.stagnating = False
        self.counts = {"keys": 0, "unlocks": 0, "doors": 0, "exit": 0, "stagnant_steps": 0}
        self.totals_by_component = dict.fromkeys(COMPONENTS, 0.0)

    def step(self, ws: dict) -> tuple[float, float, dict]:
        """Returns (extrinsic, intrinsic, per-component values)."""
        c, r = self.cfg, dict.fromkeys(COMPONENTS, 0.0)
        remaining = max(0.0, 1.0 - ws["tic"] / max(ws["episode_ticks"], 1))
        r["time"] = -c.time

        gained = int(((ws["keys"] > 0.5) & (self.keys < 0.5)).sum())
        if gained:
            self.keys = ws["keys"].copy()
            self.counts["keys"] += gained
            r["key"] = c.key * gained

        for i in ws["door_open"].nonzero()[0]:
            i = int(i)
            if i in self.opened:
                continue
            cx, cy = self.meta.door_centers[i]
            if math.hypot(cx - ws["x"], cy - ws["y"]) > c.door_reach:
                continue
            self.opened.add(i)
            if self.meta.door_locks[i] > 0:
                self.counts["unlocks"] += 1
                r["unlock"] += c.unlock
            else:
                self.counts["doors"] += 1
                r["door"] += c.door

        if self.novelty.new_sector(ws["sector"]):
            each = min(c.sector, c.sector_cap / max(self.meta.num_sectors, 1) * 2.0)
            r["sector"] = max(0.0, min(each, c.sector_cap - self.sector_paid))
            self.sector_paid += r["sector"]

        kills = ws["kills"] - self.kills
        if kills > 0:
            r["kill"] = max(0.0, min(c.kill * kills, c.kill_cap - self.kill_paid))
            self.kill_paid += r["kill"]
        self.kills = ws["kills"]

        r["damage"] = -c.damage * max(0.0, min(ws["damage_taken"] - self.damage, 200.0))
        self.damage = ws["damage_taken"]

        if ws["dead"]:
            r["death"] = -c.death * (1.0 + 0.5 * remaining)
        if ws["exit_reached"]:
            self.counts["exit"] = 1
            r["exit"] = c.exit + c.exit_health_bonus * min(ws["health"], 100.0) / 100.0 + c.exit_time_bonus * remaining

        self.positions.append((ws["x"], ws["y"]))
        x0, y0 = self.positions[0]
        self.stagnating = (len(self.positions) == self.positions.maxlen
                           and max(math.hypot(x - x0, y - y0) for x, y in self.positions) < c.stagnation_radius)
        if self.stagnating:
            self.counts["stagnant_steps"] += 1
            r["stagnation"] = -c.stagnation

        r["grid"] = c.grid * self.novelty.grid_bonus(ws["x"], ws["y"])
        for k, v in r.items():
            self.totals_by_component[k] += v
        intrinsic = r["grid"]
        return sum(r.values()) - intrinsic, intrinsic, r

    def summary(self, ws: dict, family: str) -> dict:
        """End-of-episode metrics, including the 0..1 success score the curriculum uses."""
        sector_frac = len(self.novelty.sectors) / max(self.meta.num_sectors, 1)
        kill_frac = min(1.0, ws["kills"] / max(self.totals["monsters"], 1))
        exited = bool(ws["exit_reached"])
        need = self.totals["keys"] + self.totals["locks"] + 3
        progression = min(1.0, (self.counts["keys"] + self.counts["unlocks"] + 3 * exited) / need)
        if family == "combat":
            success = 0.75 * kill_frac * (0.5 if ws["dead"] else 1.0) + 0.25 * exited
        elif family in ("movement", "explore"):
            success = 1.0 if exited else 0.7 * sector_frac
        else:
            success = progression
        return {
            "success": float(success), "progression": float(progression), "exit": float(exited),
            "dead": float(bool(ws["dead"])), "kills": float(ws["kills"]), "kill_frac": float(kill_frac),
            "sector_frac": float(sector_frac), "damage": float(ws["damage_taken"]),
            **{k: float(v) for k, v in self.counts.items()},
            **{f"r_{k}": float(v) for k, v in self.totals_by_component.items()},
        }
