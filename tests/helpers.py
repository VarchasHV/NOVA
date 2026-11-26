"""Shared builders for the Streamed Mind tests."""

import numpy as np
import torch

from doom_env.wad import write_wad
from models import obs as O
from training.maps.generator import map_lumps_for
from training.maps.udmf import Level, Region, Thing

FORWARD = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)


def corridor(path, lock: int = 0, key: int | None = None, switch: bool = False) -> str:
    """Player -> (key) -> door -> exit, walking east. Writes a WAD and returns its path."""
    grid = np.full((5, 16), -1)
    grid[1:3, 1:6], grid[1:3, 6], grid[1:3, 7:12] = 0, 1, 2
    regions = [Region("floor"), Region("door", lock), Region("floor")]
    sw = (11, 1, 1, 0) if switch else None
    if not switch:
        regions.append(Region("exit"))
        grid[1:3, 12:14] = 3
    things = [Thing(1.5, 1.6 if switch else 2.0, 1, 0)]
    if key:
        things.append(Thing(3.5, 2.0, key))
    write_wad(path, map_lumps_for(Level(grid, regions, things, sw), "MAP01"))
    return str(path)


def random_obs(batch: int, use_depth: bool = False, entities: int = 5) -> dict:
    out = {}
    for name, (shape, dtype) in O.obs_spec(use_depth).items():
        if dtype == np.int64:
            high = O.NUM_HIT_TYPES if name == "ray_type" else O.NUM_ENTITY_TYPES
            out[name] = torch.randint(0, high, (batch, *shape))
        else:
            out[name] = torch.rand(batch, *shape)
    out["ent_mask"] = torch.zeros(batch, O.MAX_ENTITIES)
    out["ent_mask"][:, :entities] = 1
    return out
