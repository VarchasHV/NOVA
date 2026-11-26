import numpy as np
import torch

from doom_env import make_env, spaces
from evaluation.run_heldout import load_agent


def observations(n: int):
    env = make_env("maps/two_rooms.wad")
    obs, _ = env.reset(seed=0)
    out = []
    for _ in range(n):
        out.append(obs)
        obs, *_ = env.step(np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], np.int8))
    env.close()
    return out


def valid(a):
    return a.shape == (9,) and a.dtype == np.int8 and set(np.unique(a)) <= {0, 1}


def test_valid_actions_and_deterministic_per_map():
    agent = load_agent(None, "cpu")
    seq = observations(30)
    first = [agent.act(o) for o in seq]
    agent.reset()
    second = [agent.act(o) for o in seq]
    assert all(valid(a) for a in first) and agent.errors == 0
    np.testing.assert_array_equal(first, second)  # reset() clears everything and reseeds


def test_fallback_when_the_network_raises():
    agent = load_agent(None, "cpu")

    def boom(*a, **k):
        raise RuntimeError("kaboom")

    agent.model.step = boom
    actions = [agent.act(o) for o in observations(45)]
    assert all(valid(a) and a[0] == 1 for a in actions) and agent.errors == 45
    assert actions[0][4] == 1 and actions[25][5] == 1  # alternates turning


def test_survives_nan_observation_and_corrupt_state():
    agent = load_agent(None, "cpu")
    obs = observations(3)
    obs[0]["vitals"][:] = np.nan
    assert valid(agent.act(obs[0])) and agent.errors == 0
    agent.state["h"][:] = float("nan")
    assert valid(agent.act(obs[1])) and agent.errors == 1 and agent.state is None
    assert valid(agent.act(obs[2])) and agent.errors == 1 and torch.isfinite(agent.state["h"]).all()


def test_missing_observation_key_is_handled():
    agent = load_agent(None, "cpu")
    assert valid(agent.act({})) and agent.errors == 1


def test_constructor_uses_official_spaces():
    agent = load_agent(None, "cpu")
    assert agent.action_space.shape == spaces.action_space().shape
