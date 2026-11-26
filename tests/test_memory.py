import math

import pytest
import torch

from helpers import random_obs
from models.actor import StreamedMindActor
from models.config import ModelConfig
from models.odometry import integrate_pose
from models.streamed_memory import StreamedMemory, egocentric


def make(**kw):
    cfg = ModelConfig(slots=4, new_slot_radius=256.0, **kw)
    return cfg, StreamedMemory(cfg)


def step(mem, cfg, state, x, y):
    pose = torch.tensor([[x, y, 0.0]])
    return mem(torch.randn(1, cfg.d_model), torch.randn(1, cfg.core_size), pose, state)


def test_allocates_when_far_and_updates_when_near():
    cfg, mem = make()
    state = mem.initial_state(1, "cpu")
    _, state, info = step(mem, cfg, state, 0.0, 0.0)
    assert state["mem_valid"].sum() == 1
    _, state, info = step(mem, cfg, state, 0.1, 0.0)  # within 0.25 -> same slot
    assert state["mem_valid"].sum() == 1 and state["mem_visits"][0, 0] == 2
    torch.testing.assert_close(state["mem_pos"][0, 0], torch.zeros(2))  # anchors stay put
    _, state, info = step(mem, cfg, state, 1.0, 0.0)
    assert state["mem_valid"].sum() == 2 and info["written_slot"].item() == 1
    assert info["attention"].sum().item() == pytest.approx(1.0)
    assert info["attention"][0, 2:].sum() == 0  # empty slots are never read


def test_full_memory_evicts_least_relevant():
    cfg, mem = make()
    state = mem.initial_state(1, "cpu")
    for i in range(4):
        _, state, _ = step(mem, cfg, state, float(i), 0.0)
    assert state["mem_valid"].sum() == 4
    scores = torch.tensor([[2.0, -3.0, 1.0, 0.5]])
    mem._relevance = lambda content, age, visits: scores
    _, state, info = step(mem, cfg, state, 10.0, 0.0)
    assert info["written_slot"].item() == 1
    torch.testing.assert_close(state["mem_pos"][0, 1], torch.tensor([10.0, 0.0]))


def test_gradients_reach_gate_and_relevance():
    cfg, mem = make()
    state = mem.initial_state(2, "cpu")
    total = 0
    for i in range(3):
        pose = torch.tensor([[i * 1.0, 0, 0], [0, i * 1.0, 1.0]])
        out, state, _ = mem(torch.randn(2, cfg.d_model), torch.randn(2, cfg.core_size), pose, state)
        total = total + out.sum()
    total.backward()
    for p in (mem.gate.weight, mem.relevance[-1].weight, mem.cell.weight_ih):
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_soft_mode():
    cfg, mem = make(memory_mode="soft")
    state = mem.initial_state(2, "cpu")
    for i in range(6):
        out, state, info = mem(torch.randn(2, cfg.d_model), torch.randn(2, cfg.core_size), torch.rand(2, 3), state)
    assert torch.isfinite(out).all() and (state["mem_valid"] <= 1 + 1e-5).all() and (state["mem_valid"] > 0).all()


def test_egocentric_bearing_matches_official_convention():
    """0 straight ahead, positive (sin) to the left - the same as entity angles."""
    pose = torch.tensor([[0.0, 0.0, math.pi / 2]])  # facing +y
    slots = torch.tensor([[[0.0, 1.0], [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0]]])
    _, _, sin, cos = egocentric(slots, pose)[0].unbind(-1)
    torch.testing.assert_close(cos, torch.tensor([1.0, 0.0, 0.0, -1.0]), atol=1e-5, rtol=0)
    torch.testing.assert_close(sin, torch.tensor([0.0, 1.0, -1.0, 0.0]), atol=1e-5, rtol=0)


def test_integrate_pose_turn_and_move():
    pose = torch.zeros(1, 3)
    pose = integrate_pose(pose, torch.tensor([[1.0, 0.0, math.pi / 2]]))  # move east, then turn left
    pose = integrate_pose(pose, torch.tensor([[1.0, 0.5, 0.0]]))  # move north 1, 0.5 to the right (east)
    torch.testing.assert_close(pose[0], torch.tensor([1.5, 1.0, math.pi / 2]), atol=1e-5, rtol=0)


@pytest.mark.parametrize("kw", [{}, {"memory": False}, {"two_timescale": True, "slow_every": 3}, {"memory_mode": "soft"}])
def test_unroll_reproduces_step_by_step(kw):
    """PPO replays sequences with unroll(); it must match what happened while acting."""
    torch.manual_seed(0)
    actor = StreamedMindActor(ModelConfig(**kw)).eval()
    T, B = 12, 3
    seq = [random_obs(B) for _ in range(T)]
    reset = torch.zeros(T, B)
    reset[0] = 1
    reset[5, 1] = reset[8, 2] = 1
    with torch.no_grad():
        state = actor.initial_state(B)
        logits, hidden, poses, states = [], [], [], []
        for t in range(T):
            state = actor.reset_where(state, reset[t])
            states.append(state)
            out = actor.step(seq[t], state)
            state = out.state
            logits.append(out.logits)
            hidden.append(out.hidden)
            poses.append(out.pose)
        obs = {k: torch.stack([o[k] for o in seq]) for k in seq[0]}
        for start in (0, 4):  # from the beginning and from a stored mid-sequence state
            got_l, got_h = actor.unroll({k: v[start:] for k, v in obs.items()}, states[start], reset[start:],
                                        torch.stack(poses[start:]))
            torch.testing.assert_close(got_l, torch.stack(logits[start:]), atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(got_h, torch.stack(hidden[start:]), atol=1e-4, rtol=1e-4)
    assert (poses[5][1] == 0).all()  # a new episode starts at the origin
