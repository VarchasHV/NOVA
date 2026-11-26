"""Make a generated level less regular: bent corners, steps, pillars and ledges.

Applied on top of a finished level, so rooms, doors, keys, monsters and the exit stay where they
were. Nothing here can cut a route: steps are low enough to walk up, obstacles stand alone with a
free cell all around, and corners move too little to close a gap.
"""

from __future__ import annotations

import numpy as np

from .udmf import VOID, Level, Region

STEP_HEIGHTS = (0, 8, 16, 24)  # any two differ by at most 24, which the player can always climb
LEDGE_HEIGHT = 96  # far more than a step above every floor
MAX_SHIFT = 12  # a one-cell gap (64 units) keeps at least 40; the player is 32 wide


def add_obstacles(level: Level, rng: np.random.Generator, chance: float = 0.7) -> int:
    """Stand pillars (solid) and ledges (see-through) on open floor. Returns how many."""
    h, w = level.grid.shape
    taken = {(int(t.cx), int(t.cy)) for t in level.things}
    if level.switch:
        taken.add(level.switch[:2])
    candidates = [(x, y) for y in range(1, h - 1) for x in range(1, w - 1)]
    rng.shuffle(candidates)
    ledge, placed = None, 0
    limit = max(1, int((level.grid != VOID).sum()) // 24)
    for x, y in candidates:
        if placed >= limit:
            break
        around = level.grid[y - 1 : y + 2, x - 1 : x + 2]
        region = int(level.grid[y, x])
        # The whole 3x3 block must be one open floor: never in a doorway, corridor or beside another obstacle.
        if region == VOID or level.regions[region].kind != "floor" or not np.all(around == region):
            continue
        if any((x + dx, y + dy) in taken for dx in (-1, 0, 1) for dy in (-1, 0, 1)) or rng.random() > chance:
            continue
        if rng.random() < 0.5:
            level.grid[y, x] = VOID
        else:
            if ledge is None:
                level.regions.append(Region("ledge", floor=LEDGE_HEIGHT))
                ledge = len(level.regions) - 1
            level.grid[y, x] = ledge
        placed += 1
    return placed


def add_steps(level: Level, rng: np.random.Generator) -> None:
    for region in level.regions:
        if region.kind != "ledge":
            region.floor = int(rng.choice(STEP_HEIGHTS))


def bend_corners(level: Level, rng: np.random.Generator) -> None:
    h, w = level.grid.shape
    shift = rng.integers(-MAX_SHIFT, MAX_SHIFT + 1, size=(h + 1, w + 1, 2))
    for thing in level.things:  # keep the room around every thing, so nothing starts inside a wall
        x, y = int(thing.cx), int(thing.cy)
        shift[y : y + 2, x : x + 2] = 0
    level.warp = shift


def vary(level: Level, rng: np.random.Generator) -> Level:
    add_obstacles(level, rng)
    add_steps(level, rng)
    bend_corners(level, rng)
    return level
