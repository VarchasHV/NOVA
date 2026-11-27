"""Procedural levels: a room graph on a coarse grid -> corridors -> doors -> key/lock chain -> exit.

Everything is deterministic in (family, level, seed), so any failed episode can be replayed from
three numbers. Seeds below HELDOUT_SEED_START are for training; the rest are reserved for evaluation.

Families (each with difficulty 0..4):
  movement     a few rooms, walk through doors, reach the exit
  combat       one arena with pillars and monsters
  explore      many rooms with loops and dead ends
  progression  1-3 keys and matching locked doors that force backtracking
  gauntlet     everything together
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from doom_env.wad import Lump, write_wad

from .udmf import LOCK_BLUE, LOCK_RED, LOCK_YELLOW, VOID, Level, Region, Thing, to_udmf
from .variety import vary

FAMILIES = ("movement", "combat", "explore", "progression", "gauntlet")
NUM_LEVELS = 5
HELDOUT_SEED_START = 1_000_000_000
SLOT = 9  # cells per macro-grid slot; every room covers cells 3..5 of its slot in both directions

PLAYER = 1
KEY_THINGS = {LOCK_RED: (13, 38), LOCK_BLUE: (5, 40), LOCK_YELLOW: (6, 39)}  # (card, skull)
# monster types in order of difficulty: zombieman, shotgun guy, imp, demon, chaingunner, cacodemon, hell knight
MONSTERS = (3004, 9, 3001, 3002, 65, 3005, 69)
MONSTER_POOL = ((0,), (0, 2), (0, 1, 2), (0, 1, 2, 3), (0, 1, 2, 3, 4, 5, 6))
MONSTER_WEIGHT = (5.0, 3.0, 4.0, 2.0, 1.0, 0.5, 0.3)
HEALTH = (2011, 2012, 2014)  # stimpack, medikit, health bonus
AMMO = (2007, 2048, 2008, 2049)  # clip, bullet box, shells, shell box
WEAPONS = (2001, 2002)  # shotgun, chaingun
ARMOR, BARREL = 2018, 2035

PARAMS: dict[str, dict[str, tuple]] = {
    "movement": dict(rooms=(2, 3, 4, 5, 6), locks=(0, 0, 0, 0, 0), monsters=(0, 0, 0, 0, 1),
                     doors=(0.0, 0.4, 0.7, 0.9, 1.0), loops=(0.0, 0.0, 0.1, 0.2, 0.2)),
    "explore": dict(rooms=(4, 5, 7, 9, 12), locks=(0, 0, 0, 0, 0), monsters=(0, 0, 1, 3, 5),
                    doors=(0.0, 0.3, 0.5, 0.6, 0.7), loops=(0.0, 0.1, 0.2, 0.3, 0.3)),
    "progression": dict(rooms=(3, 4, 6, 8, 10), locks=(1, 1, 1, 2, 3), monsters=(0, 0, 1, 3, 5),
                        doors=(0.0, 0.2, 0.4, 0.6, 0.6), loops=(0.0, 0.0, 0.1, 0.2, 0.2)),
    "gauntlet": dict(rooms=(5, 6, 8, 11, 14), locks=(1, 1, 2, 2, 3), monsters=(3, 6, 9, 15, 22),
                     doors=(0.3, 0.5, 0.6, 0.7, 0.7), loops=(0.0, 0.1, 0.2, 0.2, 0.3)),
}
TICKS = (1050, 1400, 2100, 2800, 3500)


def level_rng(family: str, level: int, seed: int) -> np.random.Generator:
    return np.random.default_rng([FAMILIES.index(family), level, seed])


def generate(family: str, level: int, seed: int, variety: float = 0.0) -> Level:
    """`variety` is the share of maps given uneven shapes, steps and ledges (0 = the plain square maps).
    The same arguments always give the same map."""
    if family not in FAMILIES or not 0 <= level < NUM_LEVELS:
        raise ValueError(f"unknown scenario {family!r} level {level}")
    rng = level_rng(family, level, seed)
    lv = _arena(rng, level) if family == "combat" else _rooms(rng, family, level)
    if seed >= HELDOUT_SEED_START:
        lv.ticks = int(rng.integers(2100, 4201))
    elif level >= 3:
        lv.ticks = int(rng.integers(2100, 4201))
    else:
        lv.ticks = TICKS[level]
    if variety > 0:
        extra = np.random.default_rng([FAMILIES.index(family), level, seed, 1])  # leaves the layout's draws alone
        if extra.random() < variety:
            vary(lv, extra)
    return lv


# --------------------------------------------------------------------------- shared pieces
class _Placer:
    """Hands out distinct free cells for things, keeping them one cell apart where possible."""

    def __init__(self, rng):
        self.rng, self.used = rng, set()

    def pick(self, cells: list[tuple[int, int]]) -> tuple[int, int] | None:
        free = [c for c in cells if c not in self.used]
        if not free:
            return None
        c = free[int(self.rng.integers(len(free)))]
        self.used.add(c)
        return c

    def thing(self, things: list, cells, kind: int, angle: int = 0, ambush: bool = False) -> bool:
        c = self.pick(cells)
        if c is None:
            return False
        things.append(Thing(c[0] + 0.5, c[1] + 0.5, kind, angle, ambush))
        return True


def _inner(x0, y0, x1, y1) -> list[tuple[int, int]]:
    """Room cells, excluding the outermost ring when the room is big enough (keeps doors clear)."""
    mx, my = int(x1 - x0 > 2), int(y1 - y0 > 2)
    return [(x, y) for x in range(x0 + mx, x1 - mx) for y in range(y0 + my, y1 - my)]


def _monster(rng, level: int) -> int:
    pool = MONSTER_POOL[level]
    w = np.array([MONSTER_WEIGHT[i] for i in pool])
    return MONSTERS[int(rng.choice(pool, p=w / w.sum()))]


def _exit(rng, level: int, grid, regions, room, avoid: set, switch_only: bool = False):
    """Add the level exit to room (x0, y0, x1, y1): a wall switch or a walk-over alcove."""
    x0, y0, x1, y1 = room
    if switch_only or (level >= 1 and rng.random() < 0.5):
        walls = []
        for x in range(x0, x1):
            for y in range(y0, y1):
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    inside = 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]
                    if grid[y, x] != VOID and (not inside or grid[ny, nx] == VOID) and (x, y) not in avoid:
                        walls.append((x, y, dx, dy))
        if walls:
            sw = walls[int(rng.integers(len(walls)))]
            avoid.add(sw[:2])
            return sw
    cx = x0 if rng.random() < 0.5 else x1 - 2
    cy = y0 if rng.random() < 0.5 else y1 - 2
    regions.append(Region("exit"))
    grid[cy : cy + 2, cx : cx + 2] = len(regions) - 1
    avoid.update((x, y) for x in range(cx, cx + 2) for y in range(cy, cy + 2))
    return None


# --------------------------------------------------------------------------- combat arena
def _arena(rng, level: int) -> Level:
    side = (8, 10, 12, 15, 18)[level]
    w, h = side + int(rng.integers(0, 3)), side + int(rng.integers(-1, 2))
    grid = np.full((h + 2, w + 2), VOID, np.int64)
    grid[1 : h + 1, 1 : w + 1] = 0
    regions = [Region("floor")]
    # Pillars (cover): single blocks that never touch each other or the outer wall.
    for _ in range((0, 2, 4, 6, 9)[level]):
        x, y = int(rng.integers(3, w - 2)), int(rng.integers(3, h - 2))
        if np.all(grid[y - 1 : y + 3, x - 1 : x + 3] == 0):
            grid[y : y + 2, x : x + 2] = VOID
    # Split the floor into quadrants so exploration has something to count.
    for qx, qy in ((1, 0), (0, 1), (1, 1)):
        regions.append(Region("floor"))
        part = grid[1 + qy * (h // 2) : 1 + (h // 2 if qy == 0 else h), 1 + qx * (w // 2) : 1 + (w // 2 if qx == 0 else w)]
        part[part == 0] = len(regions) - 1
    placer = _Placer(rng)
    cells = [(x, y) for y in range(grid.shape[0]) for x in range(grid.shape[1]) if grid[y, x] != VOID]
    things: list[Thing] = []
    start = [c for c in cells if c[0] <= 3 and c[1] <= 3]
    sx, sy = placer.pick(start or cells)
    things.append(Thing(sx + 0.5, sy + 0.5, PLAYER, int(rng.integers(0, 360))))
    placer.used.update((sx + dx, sy + dy) for dx in range(-2, 3) for dy in range(-2, 3))
    # A switch exit: it takes a deliberate USE, so wandering into it cannot cut the fight short.
    switch = _exit(rng, level, grid, regions, (w - 3, h - 3, w + 1, h + 1), placer.used, switch_only=True)
    cells = [c for c in cells if grid[c[1], c[0]] != VOID and regions[grid[c[1], c[0]]].kind == "floor"]
    far = [c for c in cells if abs(c[0] - sx) + abs(c[1] - sy) >= side // 2] or cells
    for _ in range((2, 3, 5, 8, 12)[level] + int(rng.integers(0, 2 + level))):
        placer.thing(things, far, _monster(rng, level), int(rng.integers(0, 360)), bool(rng.random() < 0.25))
    for _ in range(int(rng.integers(2, 5))):
        placer.thing(things, cells, int(rng.choice(HEALTH)))
    for _ in range(int(rng.integers(2, 6))):
        placer.thing(things, cells, int(rng.choice(AMMO)))
    if rng.random() < 0.6:
        placer.thing(things, cells, int(rng.choice(WEAPONS)))
    if rng.random() < 0.3:
        placer.thing(things, cells, ARMOR)
    if level >= 2:
        for _ in range(int(rng.integers(0, 3))):
            placer.thing(things, far, BARREL)
    return Level(grid, regions, things, switch)


# --------------------------------------------------------------------------- room graph families
def _room_graph(rng, n: int, loops: float):
    """Random tree of adjacent macro-grid slots. Returns (slots, tree edges, spare adjacent pairs)."""
    size = int(np.ceil(np.sqrt(n))) + 1
    start = (int(rng.integers(size)), int(rng.integers(size)))
    slots, index, edges = [start], {start: 0}, []
    while len(slots) < n:
        # Prefer extending the newest rooms: gives longer routes with side branches.
        i = len(slots) - 1 - int(rng.integers(min(len(slots), 2))) if rng.random() < 0.6 else int(rng.integers(len(slots)))
        x, y = slots[i]
        nbrs = [(x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        nbrs = [c for c in nbrs if 0 <= c[0] < size and 0 <= c[1] < size and c not in index]
        if not nbrs:
            continue
        c = nbrs[int(rng.integers(len(nbrs)))]
        index[c] = len(slots)
        slots.append(c)
        edges.append((i, index[c]))
    tree = {frozenset(e) for e in edges}
    spare = []
    for i, (x, y) in enumerate(slots):
        for c in ((x + 1, y), (x, y + 1)):
            if c in index and frozenset((i, index[c])) not in tree and rng.random() < loops:
                spare.append((i, index[c]))
    return size, slots, edges, spare


def _bfs(n: int, edges, source: int):
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    depth, parent = [-1] * n, [-1] * n
    depth[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if depth[v] < 0:
                depth[v], parent[v] = depth[u] + 1, u
                queue.append(v)
    return depth, parent


def _rooms(rng, family: str, level: int) -> Level:
    p = {k: v[level] for k, v in PARAMS[family].items()}
    n = int(p["rooms"]) + int(rng.integers(0, 2))
    size, slots, edges, spare = _room_graph(rng, n, p["loops"])

    depth, parent = _bfs(n, edges, 0)
    exit_room = int(max(range(n), key=lambda i: depth[i] + 0.5 * rng.random()))
    path = [exit_room]
    while parent[path[-1]] >= 0:
        path.append(parent[path[-1]])
    path.reverse()  # start -> exit

    # --- locks sit on the route to the exit; keys sit before their lock -----------------------
    num_locks = min(int(p["locks"]), len(path) - 1)
    cuts = sorted(rng.choice(len(path) - 1, size=num_locks, replace=False).tolist()) if num_locks else []
    colours = rng.permutation([LOCK_RED, LOCK_BLUE, LOCK_YELLOW])[:num_locks].tolist()
    locked = {frozenset((path[c], path[c + 1])): colours[i] for i, c in enumerate(cuts)}
    open_edges = [e for e in edges if frozenset(e) not in locked]
    component = [-1] * n  # how many locks you must pass to get to each room
    for i, root in enumerate([0] + [path[c + 1] for c in cuts]):
        d, _ = _bfs(n, open_edges, root)
        for r in range(n):
            if d[r] >= 0:
                component[r] = i
    key_rooms = []
    for i in range(num_locks):
        options = [r for r in range(n) if component[r] == i and r != 0] or [r for r in range(n) if component[r] <= i]
        side_rooms = [r for r in options if r not in path]
        pool = side_rooms if side_rooms and rng.random() < 0.8 else options
        key_rooms.append(int(max(pool, key=lambda r: depth[r] + 2.0 * rng.random())))
    # Loops may only join rooms behind the same number of locks, otherwise they would bypass one.
    spare = [(a, b) for a, b in spare if component[a] == component[b]]

    # --- geometry -------------------------------------------------------------------------------
    grid = np.full((size * SLOT, size * SLOT), VOID, np.int64)
    regions: list[Region] = []
    rects = []
    for sx, sy in slots:
        x0, y0 = sx * SLOT + int(rng.integers(1, 4)), sy * SLOT + int(rng.integers(1, 4))
        x1, y1 = sx * SLOT + int(rng.integers(6, 9)), sy * SLOT + int(rng.integers(6, 9))
        regions.append(Region("floor"))
        grid[y0:y1, x0:x1] = len(regions) - 1
        rects.append((x0, y0, x1, y1))

    auto = family != "gauntlet" and level <= 1
    for a, b in edges + spare:
        lock = locked.get(frozenset((a, b)), 0)
        has_door = bool(lock) or rng.random() < p["doors"]
        horizontal = slots[a][1] == slots[b][1]
        axis = 0 if horizontal else 1
        if slots[a][axis] > slots[b][axis]:
            a, b = b, a
        lo, hi = rects[a][2 + axis], rects[b][axis]  # corridor spans [lo, hi) along the axis
        band = slots[a][1 - axis] * SLOT + 3 + int(rng.integers(0, 2))  # 2 cells wide
        door_at = int(rng.integers(lo, hi))
        parts: dict[str, int] = {}
        for t in range(lo, hi):
            if has_door and t == door_at:
                name = "door"
            else:
                name = "after" if has_door and t > door_at else "before"
            if name not in parts:
                regions.append(Region("door", lock, auto) if name == "door" else Region("floor"))
                parts[name] = len(regions) - 1
            if horizontal:
                grid[band : band + 2, t] = parts[name]
            else:
                grid[t, band : band + 2] = parts[name]

    # --- things ---------------------------------------------------------------------------------
    placer = _Placer(rng)
    things: list[Thing] = []
    sx, sy = placer.pick(_inner(*rects[0]))
    things.append(Thing(sx + 0.5, sy + 0.5, PLAYER, int(rng.integers(0, 360))))
    switch = _exit(rng, level, grid, regions, rects[exit_room], placer.used)
    cells = [_inner(*r) for r in rects]
    for i, room in enumerate(key_rooms):
        card, skull = KEY_THINGS[colours[i]]
        kind = skull if rng.random() < 0.25 else card
        if not placer.thing(things, cells[room], kind):
            x0, y0, x1, y1 = rects[room]  # tiny room: allow its outer ring too
            if not placer.thing(things, [(x, y) for x in range(x0, x1) for y in range(y0, y1) if grid[y, x] != VOID], kind):
                raise RuntimeError("no room left for a key")

    others = [r for r in range(1, n)] or [0]
    for _ in range(int(p["monsters"]) + int(rng.integers(0, 1 + p["monsters"] // 4))):
        room = int(rng.choice(others))
        placer.thing(things, cells[room], _monster(rng, level), int(rng.integers(0, 360)),
                     bool(family == "gauntlet" and rng.random() < 0.35))
    supply = {"gauntlet": 0.5 - 0.06 * level}.get(family, 0.5)
    for r in range(n):
        if rng.random() < supply:
            placer.thing(things, cells[r], int(rng.choice(HEALTH)))
        if p["monsters"] and rng.random() < supply:
            placer.thing(things, cells[r], int(rng.choice(AMMO)))
    if p["monsters"] >= 5 and rng.random() < 0.7:
        placer.thing(things, cells[int(rng.choice(others))], int(rng.choice(WEAPONS)))
    if family == "gauntlet" and rng.random() < 0.4:
        placer.thing(things, cells[int(rng.choice(others))], ARMOR)
    return Level(grid, regions, things, switch)


# --------------------------------------------------------------------------- WAD packing
def map_lumps_for(level: Level, name: str) -> list[Lump]:
    return [Lump(name, b""), Lump("TEXTMAP", to_udmf(level).encode()), Lump("ENDMAP", b"")]


def write_pack(path: str | Path, scenarios: list[tuple[str, int, int]], variety: float = 0.0) -> list[dict]:
    """Write many maps into one WAD (switching maps inside one WAD costs ~2 ms, switching WADs
    ~140 ms). Returns one entry per map: name, family, level, seed, ticks."""
    lumps, entries = [], []
    for i, (family, level, seed) in enumerate(scenarios):
        lv = generate(family, level, seed, variety)
        name = f"G{i:04d}"
        lumps += map_lumps_for(lv, name)
        entries.append({"map": name, "family": family, "level": level, "seed": seed, "ticks": lv.ticks})
    write_wad(path, lumps)
    return entries
