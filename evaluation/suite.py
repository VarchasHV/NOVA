"""The fixed held-out suite: maps that training never sees.

Generated maps use seeds >= HELDOUT_SEED_START (training draws below it), at skill 3 with vanilla
rules and tick caps in 2100-4200. Tier 4 also includes Freedoom maps excluded from training.
"""

from __future__ import annotations

import random

from training.maps.generator import HELDOUT_SEED_START, write_pack
from training.vec_env import FREEDOOM_HELDOUT

TIERS = {1: "combat", 2: "explore", 3: "progression", 4: "gauntlet"}
LEVELS = (1, 2, 3, 4)  # cycled through, so each tier spans easy to hard


def build_suite(directory: str, maps_per_tier: int = 6, freedoom: bool = True) -> list[dict]:
    """Write the generated maps into `directory`; returns one entry per evaluation episode."""
    from training.env import FREEDOOM_WAD

    scenarios, tiers = [], []
    for tier, family in TIERS.items():
        for i in range(maps_per_tier):
            scenarios.append((family, LEVELS[i % len(LEVELS)], HELDOUT_SEED_START + 1000 * tier + i))
            tiers.append(tier)
    path = f"{directory}/heldout.wad"
    entries = write_pack(path, scenarios)
    for e, tier in zip(entries, tiers):
        e.update(wad=path, tier=tier)
    if freedoom:
        rng = random.Random(7)
        for name in FREEDOOM_HELDOUT:
            entries.append({"wad": FREEDOOM_WAD, "map": name, "tier": 4, "family": "freedoom", "level": 4,
                            "seed": int(name[3:]), "ticks": rng.randint(2100, 4200)})
    return entries
