"""Episodic exploration bonuses, from privileged position (training only, never seen by the actor)."""

from __future__ import annotations

import math


class EpisodicNovelty:
    """Tracks what has been visited in the current episode.

    * sectors - first visit of an engine sector, the closest thing to the scorer's "unique areas"
    * grid    - 1/sqrt(visits) each time the player enters a coarse cell (densifies the signal)
    """

    def __init__(self, cell: float = 128.0):
        self.cell = cell
        self.sectors: set[int] = set()
        self.counts: dict[tuple[int, int], int] = {}
        self.current: tuple[int, int] | None = None

    def new_sector(self, sector: int) -> bool:
        if sector < 0 or sector in self.sectors:
            return False
        self.sectors.add(sector)
        return True

    def grid_bonus(self, x: float, y: float) -> float:
        key = (int(math.floor(x / self.cell)), int(math.floor(y / self.cell)))
        if key == self.current:  # standing still or spinning earns nothing
            return 0.0
        self.current = key
        self.counts[key] = self.counts.get(key, 0) + 1
        return 1.0 / math.sqrt(self.counts[key])
