"""The curriculum Director: decides which scenarios the workers practise.

Idea (as in Prioritised Level Replay): spend time where learning is happening. A scenario bucket
(family, difficulty) with success around 50% teaches the most; ones always won or always lost get
little. Harder levels unlock when the current one is mastered, and also on a time schedule so a short
training run still reaches the full task. The last stage is vanilla rules at skill 3.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .maps.generator import FAMILIES, NUM_LEVELS
from .nemesis import NemesisDB


@dataclass
class DirectorConfig:
    unlock_success: float = 0.7  # success on the hardest open level that opens the next one
    min_episodes: int = 40
    ema: float = 0.03
    level_every: float = 0.13  # level L opens by progress L * level_every at the latest
    final_start: float = 0.75  # vanilla-only stage
    freedoom_start: float = 0.30
    nemesis_prob: float = 0.2
    nemesis_pack: int = 32
    fixed_level: int | None = None  # smoke tests: only this difficulty
    families: list[str] = field(default_factory=lambda: list(FAMILIES))
    family_start: dict[str, float] = field(default_factory=lambda: {
        "movement": 0.0, "combat": 0.0, "explore": 0.02, "progression": 0.05, "gauntlet": 0.15})
    family_weight: dict[str, float] = field(default_factory=lambda: {
        "movement": 1.0, "combat": 1.5, "explore": 1.5, "progression": 2.5, "gauntlet": 2.5})


class Director:
    def __init__(self, cfg: DirectorConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.nemesis = NemesisDB()
        self.success = {(f, l): 0.0 for f in (*cfg.families, "freedoom") for l in range(NUM_LEVELS)}
        self.count = dict.fromkeys(self.success, 0)
        self.unlocked = dict.fromkeys(cfg.families, 0)

    def report(self, ep: dict, step: int) -> None:
        key = (ep["family"], int(ep["level"]))
        if key not in self.success:
            return
        self.count[key] += 1
        rate = max(self.cfg.ema, 1.0 / self.count[key])
        self.success[key] += rate * (ep["success"] - self.success[key])
        self.nemesis.report(ep, step)
        f, l = key
        if (f in self.unlocked and l == self.unlocked[f] and l + 1 < NUM_LEVELS
                and self.count[key] >= self.cfg.min_episodes and self.success[key] >= self.cfg.unlock_success):
            self.unlocked[f] = l + 1

    def weights(self, progress: float) -> dict[tuple[str, int], float]:
        """Sampling probability of every (family, level) bucket at this point of training."""
        c = self.cfg
        final = progress >= c.final_start
        out: dict[tuple[str, int], float] = {}
        active = [f for f in c.families if progress >= c.family_start.get(f, 0.0)] or list(c.families[:1])
        for family in active:
            if c.fixed_level is not None:
                levels = [c.fixed_level]
            else:
                top = min(NUM_LEVELS - 1, max(self.unlocked[family], int(progress / c.level_every)))
                levels = list(range(top + 1))
                if final:
                    levels = [l for l in levels if l >= 2] or [top]
            raw = []
            for l in levels:
                s, n = self.success[(family, l)], self.count[(family, l)]
                w = 1.0 if n < c.min_episodes else 0.1 + 4.0 * s * (1.0 - s)
                if n >= c.min_episodes and s < 0.15 and l > 0:
                    w *= 0.5
                if final:
                    w *= 1.0 + max(0, l - 2)
                raw.append(w)
            share = c.family_weight.get(family, 1.0)
            if family == "movement" and progress > 0.2:
                share *= 0.25
            for l, w in zip(levels, raw):
                out[(family, l)] = share * w / sum(raw)
        total = sum(out.values())
        return {k: v / total for k, v in out.items()}

    def directive(self, progress: float) -> dict:
        """Plain data sent to the environment workers."""
        return {
            "weights": self.weights(progress),
            "final": progress >= self.cfg.final_start,
            "freedoom": self.cfg.fixed_level is None and progress >= self.cfg.freedoom_start,
            "nemesis": self.nemesis.sample(self.cfg.nemesis_pack, self.rng),
            "nemesis_prob": self.cfg.nemesis_prob,
        }

    def state_dict(self) -> dict:
        pack = lambda d: [[list(k), v] for k, v in d.items()]
        return {"success": pack(self.success), "count": pack(self.count), "unlocked": dict(self.unlocked),
                "nemesis": self.nemesis.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        for name in ("success", "count"):
            target = getattr(self, name)
            for k, v in state[name]:
                target[(k[0], int(k[1]))] = v
        self.unlocked.update(state["unlocked"])
        self.nemesis.load_state_dict(state["nemesis"])
