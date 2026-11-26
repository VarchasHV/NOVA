"""Training environment: the official wrapper plus what only training may see.

`TrainEnv` subclasses the unmodified `DoomEnv`, so observations are exactly those of evaluation. It
adds map switching, difficulty and cheat settings, and ground truth (`world_state`) used for
rewards, the critic and auxiliary targets. None of it ever reaches the actor.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np

from doom_env.mapinfo import DOOM_NAMESPACES, load_mapinfo, parse_udmf, udmf_namespace
from doom_env.sensors import KEY_NAMES, PLAYER_HEIGHT
from doom_env.wad import map_lumps, read_wad
from doom_env.wrapper import IWAD, DoomEnv

FREEDOOM_WAD = IWAD
DOOM_EXIT_SPECIALS = {11, 51, 52, 124, 197, 198}
ZDOOM_EXIT_SPECIALS = {243, 244}


@dataclass
class MapMeta:
    """Static facts about one map, from the WAD."""

    info: object  # doom_env MapInfo
    segments: tuple  # (p1, p2, front, back) of sector boundary lines
    door_ids: np.ndarray
    door_locks: np.ndarray
    door_centers: np.ndarray  # (D, 2)
    exits: np.ndarray  # (E, 2) midpoints of exit lines
    num_sectors: int

    def sector_at(self, x: float, y: float) -> int:
        """Which sector contains the point (even-odd rule), or -1."""
        p1, p2, front, back = self.segments
        straddle = (p1[:, 1] > y) != (p2[:, 1] > y)
        if not straddle.any():
            return -1
        a, b = p1[straddle], p2[straddle]
        xs = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
        right = xs > x
        f, k = front[straddle][right], back[straddle][right]
        counts = np.bincount(np.concatenate([f[f >= 0], k[k >= 0]]), minlength=self.num_sectors)
        odd = np.flatnonzero(counts & 1)
        return int(odd[0]) if len(odd) else -1


def _exit_points(lumps: dict) -> np.ndarray:
    points = []
    if "TEXTMAP" in lumps:
        text = lumps["TEXTMAP"].decode("utf-8", "replace")
        specials = DOOM_EXIT_SPECIALS if udmf_namespace(text) in DOOM_NAMESPACES else ZDOOM_EXIT_SPECIALS
        blocks = parse_udmf(text)
        verts = blocks.get("vertex", [])
        for line in blocks.get("linedef", []):
            if int(line.get("special", 0) or 0) in specials:
                a, b = verts[int(line["v1"])], verts[int(line["v2"])]
                points.append(((a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2))
    else:
        hexen = "BEHAVIOR" in lumps
        data, raw = lumps.get("LINEDEFS", b""), lumps.get("VERTEXES", b"")
        size = 16 if hexen else 14
        for i in range(len(data) // size):
            v1, v2 = struct.unpack_from("<HH", data, i * size)
            special = data[i * size + 6] if hexen else struct.unpack_from("<H", data, i * size + 6)[0]
            if special in (ZDOOM_EXIT_SPECIALS if hexen else DOOM_EXIT_SPECIALS):
                a, b = struct.unpack_from("<hh", raw, v1 * 4), struct.unpack_from("<hh", raw, v2 * 4)
                points.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
    return np.array(points, np.float64).reshape(-1, 2)


def load_meta(lumps_by_name: dict, wad: str, map_name: str) -> MapMeta:
    info = load_mapinfo(wad, map_name)
    g = info.lines
    boundary = g.front != g.back
    segments = (g.p1[boundary], g.p2[boundary], g.front[boundary], g.back[boundary])
    ids = np.array(sorted(info.door_sectors), np.int64)
    centers = np.zeros((len(ids), 2))
    for i, sector in enumerate(ids):
        mine = (g.front == sector) | (g.back == sector)
        if mine.any():
            centers[i] = (g.p1[mine] + g.p2[mine]).mean(axis=0) / 2
    return MapMeta(info, segments, ids, np.array([info.door_sectors[i] for i in ids], np.int64), centers,
                   _exit_points(lumps_by_name), info.num_sectors)


class TrainEnv(DoomEnv):
    def __init__(self, wad: str, map_name: str, *, episode_ticks: int = 2100, render_buffers: bool = False):
        super().__init__(wad, map_name, episode_ticks=episode_ticks, render_buffers=render_buffers)
        self._cheats: list[str] = []
        self._meta_cache: dict[tuple[str, str], MapMeta] = {}
        self._objects = []
        self.meta = self._meta(self.wad, self.map_name)

    def _meta(self, wad: str, map_name: str) -> MapMeta:
        key = (wad, map_name)
        if key not in self._meta_cache:
            if len(self._meta_cache) > 512:
                self._meta_cache.clear()
            self._meta_cache[key] = load_meta(map_lumps(read_wad(wad), map_name), wad, map_name)
        return self._meta_cache[key]

    def load(self, wad: str, map_name: str, ticks: int, skill: int = 3, infinite_ammo: bool = False, god: bool = False) -> None:
        """Choose the next episode's map and rules. Call reset() afterwards."""
        wad = os.path.abspath(wad)
        g = self._game
        if wad != self.wad:  # a different WAD file needs an engine restart (~140 ms)
            g.close()
            g.set_doom_scenario_path(wad)
            g.set_doom_map(map_name)
            g.set_doom_skill(skill)
            g.init()
        self.wad, self.map_name = wad, map_name
        g.set_doom_map(map_name)
        g.set_doom_skill(skill)
        g.set_episode_timeout(ticks)
        self.episode_ticks = int(ticks)
        self.meta = self._meta(wad, map_name)
        self.mapinfo = self.meta.info
        self._geometry = None  # rebuilt for the new map on reset
        self._cheats = [f"sv_infiniteammo {int(infinite_ammo)}"] + (["god"] if god else [])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        out = super().reset(seed=seed, options=options)
        for command in self._cheats:  # god mode ends with every episode, so resend
            self._game.send_game_command(command)
        return out

    def _observe(self, state, first: bool = False) -> dict:
        self._objects = state.objects
        return super()._observe(state, first)

    def world_state(self) -> dict:
        """Ground truth for rewards, critic and metrics."""
        s = self.privileged_state()
        meta = self.meta
        floor, ceil = s["sector_floor"], s["sector_ceiling"]
        monsters, keys = [], []
        for o in self._objects:
            if o.category == "Monster":
                monsters.append((o.position_x, o.position_y))
            elif o.name in KEY_NAMES:
                keys.append((o.position_x, o.position_y))
        s["sector"] = meta.sector_at(s["x"], s["y"])
        s["door_open"] = (ceil[meta.door_ids] - floor[meta.door_ids]) >= PLAYER_HEIGHT if len(meta.door_ids) else np.zeros(0, bool)
        s["monsters"] = np.array(monsters, np.float64).reshape(-1, 2)
        s["key_positions"] = np.array(keys, np.float64).reshape(-1, 2)
        s["ammo"] = float(self._last_vars[3])
        s["armor"] = float(self._last_vars[1])
        return s
