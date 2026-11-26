import numpy as np
import pytest

from training.env import TrainEnv
from training.maps import FAMILIES, HELDOUT_SEED_START, NUM_LEVELS, generate, validate_level, write_pack
from training.maps.udmf import LOCK_RED


@pytest.mark.parametrize("family", FAMILIES)
def test_generated_levels_are_solvable(family):
    for level in range(NUM_LEVELS):
        for seed in list(range(40)) + [HELDOUT_SEED_START + 5]:
            lv = generate(family, level, seed)
            assert validate_level(lv) == [], (family, level, seed)
            assert 1000 <= lv.ticks <= 4200


def test_generation_is_deterministic():
    a, b = generate("gauntlet", 4, 123), generate("gauntlet", 4, 123)
    np.testing.assert_array_equal(a.grid, b.grid)
    assert a.things == b.things and a.switch == b.switch and a.ticks == b.ticks


def test_removing_a_key_makes_the_level_unsolvable():
    lv = generate("progression", 2, 3)
    keys = {13, 38, 5, 40, 6, 39}
    lv.things = [t for t in lv.things if t.type not in keys]
    assert validate_level(lv)


def test_packed_maps_load_in_the_engine(tmp_path):
    scenarios = [(f, l, 11) for f in FAMILIES for l in range(NUM_LEVELS)]
    entries = write_pack(tmp_path / "pack.wad", scenarios)
    env = TrainEnv(str(tmp_path / "pack.wad"), entries[0]["map"])
    for e in entries:
        env.load(str(tmp_path / "pack.wad"), e["map"], e["ticks"])
        env.reset(seed=0)
        ws = env.world_state()
        lv = generate(e["family"], e["level"], e["seed"])
        start = next(t for t in lv.things if t.type == 1)
        assert ws["x"] == pytest.approx(start.cx * 64, abs=1) and ws["y"] == pytest.approx(start.cy * 64, abs=1)
        assert env.meta.num_sectors == len(lv.regions)
        assert ws["sector"] == lv.region_at(int(start.cx), int(start.cy))
        assert len(env.meta.door_ids) == sum(r.kind == "door" for r in lv.regions)
        assert len(env.meta.exits) >= 1
        assert len(ws["key_positions"]) == sum(t.type in (13, 38, 5, 40, 6, 39) for t in lv.things)
        env.step(np.zeros(9, np.int8))
    env.close()


def test_locked_door_sets_lock(tmp_path):
    from helpers import corridor

    env = TrainEnv(corridor(tmp_path / "c.wad", lock=LOCK_RED, key=13), "MAP01")
    env.reset(seed=0)
    assert list(env.meta.door_locks > 0) == [True]
    env.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_varied_levels_stay_solvable_and_keep_their_layout(family):
    obstacles = 0
    for level in range(NUM_LEVELS):
        for seed in list(range(40)) + [HELDOUT_SEED_START + 5]:
            plain, varied = generate(family, level, seed), generate(family, level, seed, 1.0)
            assert validate_level(varied) == [], (family, level, seed)
            assert varied.things == plain.things and varied.switch == plain.switch and varied.ticks == plain.ticks
            changed = varied.grid != plain.grid
            obstacles += int(changed.sum())
            assert np.all(plain.grid[changed] >= 0)  # obstacles only ever stand on open floor
            floors = [r.floor for r in varied.regions if r.kind != "ledge"]
            assert max(floors) - min(floors) <= 24 and np.abs(varied.warp).max() <= 12
    assert obstacles > 0


def test_variety_is_off_by_default_and_its_share_is_respected():
    a, b = generate("gauntlet", 3, 9), generate("gauntlet", 3, 9, 0.0)
    assert a.warp is None and b.warp is None and all(r.floor == 0 for r in a.regions)
    share = np.mean([generate("explore", 2, s, 0.6).warp is not None for s in range(300)])
    assert 0.5 < share < 0.7
    x, y = generate("gauntlet", 4, 77, 1.0), generate("gauntlet", 4, 77, 1.0)
    np.testing.assert_array_equal(x.grid, y.grid)
    np.testing.assert_array_equal(x.warp, y.warp)


def test_varied_maps_load_and_ledges_block_movement_but_not_sight(tmp_path):
    scenarios = [(f, l, 11) for f in FAMILIES for l in range(NUM_LEVELS)]
    entries = write_pack(tmp_path / "v.wad", scenarios, 1.0)
    env = TrainEnv(str(tmp_path / "v.wad"), entries[0]["map"])
    ledge_lines = 0
    for e in entries:
        env.load(str(tmp_path / "v.wad"), e["map"], e["ticks"])
        env.reset(seed=0)
        lv = generate(e["family"], e["level"], e["seed"], 1.0)
        ws = env.world_state()
        start = next(t for t in lv.things if t.type == 1)
        assert ws["x"] == pytest.approx(start.cx * 64, abs=1) and ws["y"] == pytest.approx(start.cy * 64, abs=1)
        assert env.meta.num_sectors == len(lv.regions)
        assert ws["sector"] == lv.region_at(int(start.cx), int(start.cy))
        assert len(env.meta.door_ids) == sum(r.kind == "door" for r in lv.regions)
        assert not ws["door_open"].any()
        geo = env._geometry
        for i, r in enumerate(lv.regions):
            if r.kind == "ledge":
                edge = (~geo.one_sided) & ((geo.sec_a == i) | (geo.sec_b == i))
                assert geo.move_block[edge].all() and not geo.sight_block[edge].any()
                ledge_lines += int(edge.sum())
        env.step(np.zeros(9, np.int8))
    env.close()
    assert ledge_lines > 0


def test_switch_share_changes_only_what_it_should():
    seeds = range(200)
    a = [generate("explore", 2, s) for s in seeds]
    assert all(generate("explore", 2, s, 0.0, 0.5).switch == lv.switch for s, lv in zip(seeds, a))
    many = [generate("explore", 2, s, 0.0, 0.85) for s in seeds]
    assert 0.4 < np.mean([lv.switch is not None for lv in a]) < 0.6
    assert 0.75 < np.mean([lv.switch is not None for lv in many]) < 0.95
    assert all(validate_level(lv) == [] for lv in many)
    assert all(generate("explore", 0, s, 0.0, 1.0).switch is None for s in range(20))  # the first level walks in
