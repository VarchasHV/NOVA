"""Check that a generated level can be finished (training tooling only).

This is a plain graph search over the level grid. It is allowed here because it only tests maps
before training; nothing like it exists anywhere near the agent.
"""

from __future__ import annotations

from .generator import KEY_THINGS, PLAYER
from .udmf import VOID, Level


def reachable(level: Level) -> tuple[set[tuple[int, int]], set[int]]:
    """Cells the player can reach, and lock numbers it can open, picking up keys along the way."""
    key_lock = {t: lock for lock, kinds in KEY_THINGS.items() for t in kinds}
    keys_at = {(int(t.cx), int(t.cy)): key_lock[t.type] for t in level.things if t.type in key_lock}
    start = next((int(t.cx), int(t.cy)) for t in level.things if t.type == PLAYER)
    held: set[int] = set()
    while True:
        seen, stack = {start}, [start]
        while stack:
            x, y = stack.pop()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                r = level.region_at(nx, ny)
                if r == VOID or (nx, ny) in seen:
                    continue
                region = level.regions[r]
                if region.kind == "ledge" or (region.lock and region.lock not in held):
                    continue
                seen.add((nx, ny))
                stack.append((nx, ny))
        found = {lock for cell, lock in keys_at.items() if cell in seen}
        if found <= held:
            return seen, held
        held |= found


def validate_level(level: Level) -> list[str]:
    """Problems with a level; an empty list means it is fine."""
    problems = []
    players = [t for t in level.things if t.type == PLAYER]
    if len(players) != 1:
        return [f"{len(players)} player starts"]
    for t in level.things:
        if level.region_at(int(t.cx), int(t.cy)) == VOID:
            problems.append(f"thing {t.type} inside a wall")
    cells, held = reachable(level)
    if level.switch:
        if level.switch[:2] not in cells:
            problems.append("exit switch unreachable")
    else:
        if not any(level.regions[level.region_at(*c)].kind == "exit" for c in cells):
            problems.append("exit unreachable")
    locks = {r.lock for r in level.regions if r.lock}
    if locks - held:
        problems.append(f"no reachable key for locks {sorted(locks - held)}")
    return problems
