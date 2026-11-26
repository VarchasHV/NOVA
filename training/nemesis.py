"""Nemesis scenarios: remember the (family, level, seed) of failed episodes and replay them.

Only the *recipe* is stored. Replays are fresh episodes with the current policy; old transitions are
never fed back into PPO.
"""

from __future__ import annotations

import random


class NemesisDB:
    def __init__(self, capacity: int = 512, fail_below: float = 0.3, solved_above: float = 0.7):
        self.capacity, self.fail_below, self.solved_above = capacity, fail_below, solved_above
        self.entries: dict[tuple[str, int, int], dict] = {}

    @staticmethod
    def category(ep: dict) -> str:
        if ep["dead"]:
            return "died"
        if ep["keys"] and not ep["unlocks"]:
            return "had_key_never_unlocked"
        if ep["stagnant_steps"] > 0.3 * ep["steps"]:
            return "stuck"
        return "ran_out_of_time"

    def report(self, ep: dict, step: int) -> None:
        if ep["family"] == "freedoom" or ep["level"] == 0:
            return
        key = (ep["family"], int(ep["level"]), int(ep["seed"]))
        if ep["success"] >= self.solved_above:
            self.entries.pop(key, None)
        elif ep["success"] < self.fail_below:
            old = self.entries.pop(key, None)  # re-insert: newest failures are kept longest
            self.entries[key] = {
                "category": self.category(ep), "success": ep["success"], "step": step,
                "failures": 1 + (old["failures"] if old else 0),
            }
            while len(self.entries) > self.capacity:
                self.entries.pop(next(iter(self.entries)))

    def sample(self, count: int, rng: random.Random) -> list[tuple[str, int, int]]:
        keys = list(self.entries)
        return rng.sample(keys, min(count, len(keys)))

    def state_dict(self) -> list:
        return [[list(k), v] for k, v in self.entries.items()]

    def load_state_dict(self, state: list) -> None:
        self.entries = {(k[0], int(k[1]), int(k[2])): v for k, v in state}
