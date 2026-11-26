"""Our estimate of the official scoring (docs/EVALUATION.md). The real weights are hidden.

Per map, out of roughly 200: exit is the largest item, then progression, exploration and combat.
Tier weights 1:2:3:4 are placeholders for "harder tiers carry more weight".
"""

from __future__ import annotations

TIER_WEIGHT = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}


def score_map(m: dict) -> dict:
    exit_points = 100.0 + 10.0 * m["health_frac"] + 10.0 * m["time_left_frac"] if m["exit"] else 0.0
    progression = min(45.0, 12.0 * m["keys"] + 12.0 * m["unlocks"] + 3.0 * m["doors"])
    exploration = 40.0 * min(1.0, m["sector_frac"])
    combat = 40.0 * min(1.0, m["kill_frac"]) - 0.05 * min(m["damage"], 200.0)
    if m["dead"]:
        combat -= 5.0 + 10.0 * m["time_left_frac"]
    parts = {"exit_points": exit_points, "progression_points": progression,
             "exploration_points": exploration, "combat_points": combat}
    return {**parts, "score": sum(parts.values())}


def aggregate(maps: list[dict]) -> dict:
    """Weighted mean score over maps, plus per-tier means of every metric."""
    tiers: dict[int, list[dict]] = {}
    for m in maps:
        tiers.setdefault(m["tier"], []).append(m)
    table = {}
    for tier, ms in sorted(tiers.items()):
        keys = [k for k, v in ms[0].items() if isinstance(v, (int, float)) and k not in ("tier", "seed", "level")]
        table[tier] = {k: sum(m[k] for m in ms) / len(ms) for k in keys}
    total_weight = sum(TIER_WEIGHT[t] for t in table)
    return {"score": sum(TIER_WEIGHT[t] * table[t]["score"] for t in table) / total_weight, "tiers": table}
