import numpy as np
import pytest

from helpers import FORWARD, corridor
from training.env import TrainEnv
from training.maps.udmf import LOCK_RED
from training.rewards import RewardConfig, RewardTracker


def walk(path, use_every: int = 25, steps: int = 200):
    env = TrainEnv(path, "MAP01", episode_ticks=2100)
    env.reset(seed=0)
    ws = env.world_state()
    totals = {"monsters": len(ws["monsters"]), "keys": len(ws["key_positions"]), "locks": int((env.meta.door_locks > 0).sum())}
    tracker = RewardTracker(RewardConfig(), ws, env.meta, totals)
    total = 0.0
    for i in range(steps):
        action = FORWARD.copy()
        action[7] = i % use_every == use_every - 1
        _, _, term, trunc, _ = env.step(action)
        ws = env.world_state()
        total += tracker.step(ws)[0]
        if term or trunc:
            break
    env.close()
    return total, tracker, tracker.summary(ws, "progression")


def test_key_unlock_exit_events(tmp_path):
    total, tracker, summary = walk(corridor(tmp_path / "a.wad", lock=LOCK_RED, key=13))
    r = tracker.totals_by_component
    assert summary["keys"] == 1 and summary["unlocks"] == 1 and summary["doors"] == 0 and summary["exit"] == 1
    assert r["key"] == 2.0 and r["unlock"] == 2.0 and 10.0 < r["exit"] <= 13.0
    assert summary["success"] == 1.0 and r["sector"] > 0 and total > 14


def test_plain_door_and_switch_exit(tmp_path):
    _, tracker, summary = walk(corridor(tmp_path / "b.wad", switch=True))
    assert summary["doors"] == 1 and summary["exit"] == 1 and tracker.totals_by_component["door"] == pytest.approx(0.3)


def test_stuck_player_is_penalised_and_earns_no_exploration(tmp_path):
    total, tracker, summary = walk(corridor(tmp_path / "c.wad", lock=LOCK_RED), steps=150)
    assert summary["exit"] == 0 and summary["stagnant_steps"] > 50
    assert tracker.totals_by_component["grid"] < 0.5 and total < 1.0


def test_finishing_dominates_farming():
    """The best possible repeatable reward must stay below a plain exit plus one key and lock."""
    c = RewardConfig()
    farming = c.kill_cap + c.sector_cap
    finishing = c.exit + c.key + c.unlock
    assert finishing > farming and c.exit > max(c.kill_cap, c.sector_cap)
    # intrinsic: even 300 new grid cells in one episode stay small
    assert c.grid * 300 * 1.0 < finishing + 2


def test_death_costs_more_when_early(tmp_path):
    env = TrainEnv(corridor(tmp_path / "d.wad"), "MAP01")
    env.reset(seed=0)
    ws = env.world_state()
    costs = []
    for tic in (100, 2000):
        tracker = RewardTracker(RewardConfig(), ws, env.meta, {"monsters": 0, "keys": 0, "locks": 0})
        costs.append(tracker.step({**ws, "dead": True, "tic": tic})[2]["death"])
    env.close()
    assert costs[0] < costs[1] < 0


def test_sector_reward_is_capped(tmp_path):
    env = TrainEnv(corridor(tmp_path / "e.wad"), "MAP01")
    env.reset(seed=0)
    ws = env.world_state()
    meta = env.meta
    env.close()
    cfg = RewardConfig()
    meta.num_sectors = 500
    tracker = RewardTracker(cfg, ws, meta, {"monsters": 0, "keys": 0, "locks": 0})
    for s in range(1, 600):
        tracker.step({**ws, "sector": s % 500, "x": ws["x"] + 200 * np.sin(s), "y": ws["y"]})
    assert tracker.totals_by_component["sector"] <= cfg.sector_cap + 1e-9
