"""Grid level -> UDMF text. The engine builds the BSP nodes itself, so no external tools are needed.

A level is a grid of square cells. Every open cell belongs to a *region* (one engine sector): rooms,
corridors, doors and the exit alcove. -1 is solid rock. Walls and doors are the edges between cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CELL = 64  # map units per grid cell
VOID = -1
# ZDoom lock numbers: "card or skull" of a colour.
LOCK_RED, LOCK_BLUE, LOCK_YELLOW = 129, 130, 131
LOCK_TEXTURE = {0: "BIGDOOR2", LOCK_RED: "DOORRED", LOCK_BLUE: "DOORBLU", LOCK_YELLOW: "DOORYEL"}


@dataclass
class Region:
    kind: str  # "floor" | "door" | "exit" | "ledge" (too high to climb, can be seen over)
    lock: int = 0
    auto: bool = False  # door also opens when bumped into (easy training stages only)
    floor: int = 0  # floor height in map units


@dataclass
class Thing:
    cx: float  # cell coordinates (centre of cell (i, j) is (i + 0.5, j + 0.5))
    cy: float
    type: int
    angle: int = 0
    ambush: bool = False


@dataclass
class Level:
    grid: np.ndarray  # (H, W) region id or VOID, indexed [y, x]
    regions: list[Region]
    things: list[Thing] = field(default_factory=list)
    switch: tuple[int, int, int, int] | None = None  # exit switch wall: (x, y, dx, dy) of its cell and side
    ticks: int = 2100
    warp: np.ndarray | None = None  # (H + 1, W + 1, 2) corner offsets in map units; None = square grid

    def region_at(self, x: int, y: int) -> int:
        h, w = self.grid.shape
        return int(self.grid[y, x]) if 0 <= x < w and 0 <= y < h else VOID


def _edges(level: Level):
    """Yield (axis, line coordinate, start, key) for every unit edge that needs a linedef.
    key = (front region, back region or VOID, is_switch, front_is_positive_side)."""
    h, w = level.grid.shape
    switch_edge = None
    if level.switch:
        x, y, dx, dy = level.switch
        switch_edge = ("v", x + (dx > 0), y) if dx else ("h", y + (dy > 0), x)
    for axis, outer, inner in (("h", h + 1, w), ("v", w + 1, h)):
        for c in range(outer):
            for s in range(inner):
                if axis == "h":
                    low, high = level.region_at(s, c - 1), level.region_at(s, c)
                else:
                    low, high = level.region_at(c - 1, s), level.region_at(c, s)
                if low == high:
                    continue
                # The front side must be an open region; for doors it must be the non-door side.
                if low == VOID:
                    front_high = True
                elif high == VOID:
                    front_high = False
                else:
                    front_high = level.regions[low].kind == "door"
                front, back = (high, low) if front_high else (low, high)
                yield axis, c, s, (front, back, switch_edge == (axis, c, s), front_high)


def to_udmf(level: Level) -> str:
    verts: dict[tuple[int, int], int] = {}
    sides: list[str] = []
    lines: list[str] = []

    def vertex(x: int, y: int) -> int:
        dx, dy = (0, 0) if level.warp is None else level.warp[y, x]
        return verts.setdefault((x * CELL + int(dx), y * CELL + int(dy)), len(verts))

    def side(sector: int, top="-", mid="-", bottom="-") -> int:
        sides.append(f'sidedef {{ sector = {sector}; texturetop = "{top}"; texturemiddle = "{mid}"; texturebottom = "{bottom}"; }}')
        return len(sides) - 1

    # Merge straight runs of identical edges into one long line (fewer lines = faster ray casting).
    # Warped maps keep lines short, so that the bent corners show as walls at many angles.
    longest = sum(level.grid.shape) if level.warp is None else 2
    runs: list[list] = []
    for axis, c, s, key in _edges(level):
        last = runs[-1] if runs else None
        if last and last[0] == axis and last[1] == c and last[3] == s and last[4] == key and not key[2] and s - last[2] < longest:
            last[3] = s + 1
        else:
            runs.append([axis, c, s, s + 1, key])

    step_face = "STARTAN2" if any(r.floor for r in level.regions) else "-"  # sides of steps and ledges

    for axis, c, s0, s1, (front, back, is_switch, front_high) in runs:
        if axis == "h":  # walking east keeps the southern (low) cell on the right = front
            a, b = (s0, c), (s1, c)
            if front_high:
                a, b = b, a
        else:  # walking north keeps the eastern (high) cell on the right = front
            a, b = (c, s0), (c, s1)
            if not front_high:
                a, b = b, a
        v1, v2 = vertex(*a), vertex(*b)
        fr = level.regions[front]
        if back == VOID:
            if is_switch:
                lines.append(f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {side(front, mid='SW1BRCOM')}; "
                             "blocking = true; special = 243; playeruse = true; }")
            else:
                tex = "DOORTRAK" if fr.kind == "door" else "STARTAN2"
                lines.append(f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {side(front, mid=tex)}; blocking = true; }}")
            continue
        bk = level.regions[back]
        extra, top = "", step_face
        if bk.kind == "door":
            top = LOCK_TEXTURE[bk.lock]
            if bk.lock:
                extra = f" special = 13; arg0 = 0; arg1 = 16; arg2 = 0; arg3 = {bk.lock}; playeruse = true; repeatspecial = true;"
            else:
                extra = " special = 12; arg0 = 0; arg1 = 16; arg2 = 150; playeruse = true; monsteruse = true; repeatspecial = true;"
            if bk.auto:
                extra += " playerpush = true;"
        elif (fr.kind == "exit") != (bk.kind == "exit"):
            extra = " special = 243; playercross = true;"
        lines.append(f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {side(front, top=top, bottom=step_face)}; "
                     f"sideback = {side(back, top=step_face, bottom=step_face)}; twosided = true;{extra} }}")

    out = ['namespace = "zdoom";']
    out += [f"vertex {{ x = {x}.0; y = {y}.0; }}" for x, y in verts]
    for r in level.regions:
        flat = {"floor": "FLOOR4_8", "door": "FLAT5_4", "exit": "GATE3", "ledge": "FLAT5_4"}[r.kind]
        ceil = r.floor if r.kind == "door" else r.floor + 128
        out.append(f'sector {{ texturefloor = "{flat}"; textureceiling = "CEIL1_1"; heightfloor = {r.floor}; '
                   f"heightceiling = {ceil}; lightlevel = 192; }}")
    out += sides + lines
    for t in level.things:
        flags = " ambush = true;" if t.ambush else ""
        out.append(f"thing {{ x = {t.cx * CELL:.1f}; y = {t.cy * CELL:.1f}; angle = {t.angle}; type = {t.type};"
                   f" skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true;{flags} }}")
    return "\n".join(out) + "\n"
