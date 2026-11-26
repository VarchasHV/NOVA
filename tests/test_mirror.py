import math

import numpy as np

from doom_env import make_env
from doom_env.wad import write_wad
from models import obs as O
from training.maps.generator import map_lumps_for
from training.maps.udmf import Level, Region, Thing
from training.mirror import mirror_buttons, mirror_motion, mirror_obs


def test_mirror_twice_is_identity():
    rng = np.random.default_rng(0)
    obs = {}
    for name, (shape, dtype) in O.obs_spec(True).items():
        obs[name] = (rng.integers(0, 6, shape) if dtype == np.int64 else rng.random(shape)).astype(dtype)
    twice = mirror_obs(mirror_obs(obs))
    for name in obs:
        np.testing.assert_allclose(twice[name], obs[name], err_msg=name)
    buttons = rng.integers(0, 2, 9)
    np.testing.assert_array_equal(mirror_buttons(mirror_buttons(buttons)), buttons)
    motion = rng.random(3)
    np.testing.assert_allclose(mirror_motion(mirror_motion(motion)), motion)


def test_mirrored_action_gives_mirrored_world(tmp_path):
    """In a left/right symmetric room, mirrored buttons must produce mirrored observations."""
    grid = np.full((10, 12), -1)
    grid[1:9, 1:11] = 0
    level = Level(grid, [Region("floor")], [Thing(3.0, 5.0, 1, 0), Thing(10.0, 5.0, 2007)], None)
    path = tmp_path / "sym.wad"
    write_wad(path, map_lumps_for(level, "MAP01"))
    a, b = make_env(path), make_env(path)
    oa, _ = a.reset(seed=1)
    ob, _ = b.reset(seed=1)
    script = [[1, 0, 0, 0, 1, 0, 0, 0, 0]] * 6 + [[1, 0, 1, 0, 0, 0, 0, 0, 0]] * 5 + [[0, 0, 0, 0, 0, 1, 0, 0, 0]] * 4
    for act in script:
        act = np.array(act, np.int8)
        oa, *_ = a.step(act)
        ob, *_ = b.step(mirror_buttons(act))
        pa, pb = a.privileged_state(), b.privileged_state()
        assert abs(pa["x"] - pb["x"]) < 1.0 and abs((pa["y"] - 320) + (pb["y"] - 320)) < 1.0
        assert abs(math.sin(math.radians(pa["angle"] + pb["angle"]))) < 0.02
        ma = mirror_obs(O.preprocess(oa, np.zeros(9), False))
        cb = O.preprocess(ob, np.zeros(9), False)
        np.testing.assert_allclose(ma["ray_dist"], cb["ray_dist"], atol=0.01)
        np.testing.assert_allclose(ma["vitals"][14:16], cb["vitals"][14:16], atol=0.2)
        np.testing.assert_allclose(ma["ent_feat"][:1], cb["ent_feat"][:1], atol=0.05)
    a.close()
    b.close()
