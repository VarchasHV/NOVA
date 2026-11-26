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
