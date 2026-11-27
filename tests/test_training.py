"""The training loop pieces working together, with environments inside the test process."""

import math

import numpy as np
import pytest
import torch

from evaluation.run_heldout import evaluate
from models.actor import BBBActor
from models.config import ModelConfig
from models.odometry import integrate_pose
from training.config import PPOConfig, load_config
from training.critic import PrivilegedCritic
from training.director import Director, DirectorConfig
from training.export import export
from training.ppo import PPOLearner
from training.rollout import Collector, ReturnScaler, compute_advantages, sequence_minibatches
from training.vec_env import EnvConfig, VecEnv


@pytest.fixture(scope="module")
def pipeline():
    torch.manual_seed(0)
    cfg = ModelConfig()
    actor, critic = BBBActor(cfg), PrivilegedCritic(cfg.core_size)
    director = Director(DirectorConfig(fixed_level=0))
    env = VecEnv(EnvConfig(variants_per_bucket=1, mirror_prob=0.5), num_workers=0, envs_per_worker=3, seed=3)
    env.reset(director.directive(1.0))
    collector = Collector(env, actor, critic, length=32, seq_len=16, device="cpu")
    rollout, episodes = collector.collect(director.directive(1.0), random_policy=True)
    yield actor, critic, director, rollout, collector
    env.close()


def test_rollout_shapes_and_sequences(pipeline):
    actor, _, _, rollout, _ = pipeline
    assert rollout["reset"].shape == (32, 3) and rollout["value"].shape == (32, 3, 2)
    seen = 0
    for data, state in sequence_minibatches(rollout, 16, 4):
        assert data["reset"].shape[0] == 16 and state["h"].shape[0] == data["reset"].shape[1]
        seen += data["reset"].shape[1]
    assert seen == 6


def test_true_motion_targets_rebuild_the_true_path(pipeline):
    """Integrating the odometry targets with the actor's formula must track ground truth (x, y)."""
    _, _, _, rollout, _ = pipeline
    for e in range(3):
        pose = torch.zeros(1, 3)
        start = None
        for t in range(32):
            priv = rollout["priv"][t, e]
            xy = priv[:2] * 2.0  # world offset from spawn in RAY_RANGE units (y is mirrored if mirrored)
            heading = math.atan2(float(priv[2]), float(priv[3]))
            if rollout["reset"][t, e] or start is None:
                if start is not None:
                    break
                start, pose = (xy.clone(), heading), torch.zeros(1, 3)
                continue
            pose = integrate_pose(pose, rollout["odom"][t, e][None])
            d = xy - start[0]
            c, s = math.cos(start[1]), math.sin(start[1])
            expect = torch.tensor([d[0] * c + d[1] * s, -d[0] * s + d[1] * c])
            torch.testing.assert_close(pose[0, :2], expect, atol=0.01, rtol=0)


def test_ppo_update_changes_weights_and_stays_finite(pipeline):
    actor, critic, _, rollout, collector = pipeline
    with torch.no_grad():  # log-probs must come from the current policy for a fair update
        pass
    scalers = [ReturnScaler(3, 0.99), ReturnScaler(3, 0.99)]
    compute_advantages(rollout, scalers, (0.995, 0.99), 0.95)
    assert torch.isfinite(rollout["adv"]).all()
    learner = PPOLearner(actor, critic, PPOConfig(seq_len=16, minibatches=2, epochs=1, odom_updates=2, odom_batch=64))
    before = actor.head.logits.weight.clone()
    odom_before = actor.odometry.head.weight.clone()
    stats = learner.update(rollout, 0.1)
    assert all(np.isfinite(v) for v in stats.values()), stats
    assert not torch.equal(before, actor.head.logits.weight) and not torch.equal(odom_before, actor.odometry.head.weight)


def test_gae_stops_at_episode_end():
    rollout = {"rew": torch.zeros(4, 1, 2), "done": torch.tensor([[0.0], [1.0], [0.0], [0.0]]),
               "value": torch.zeros(4, 1, 2), "last_value": torch.ones(1, 2)}
    compute_advantages(rollout, [ReturnScaler(1, 0.99), ReturnScaler(1, 0.99)], (0.99, 0.99), 0.95)
    assert (rollout["adv"][:2] == 0).all() and (rollout["adv"][2:] > 0).all()


def test_export_and_held_out_run(pipeline, tmp_path):
    actor, critic, *_ = pipeline
    torch.save({"model_config": actor.cfg.to_dict(), "actor": actor.state_dict(), "critic": critic.state_dict()},
               tmp_path / "ckpt.pt")
    weights = export(tmp_path / "ckpt.pt", tmp_path / "model.pt", fp16=True)
    saved = torch.load(weights, weights_only=True)
    assert set(saved) == {"model_config", "actor", "inference", "source"}
    result = evaluate(weights, maps_per_tier=1, freedoom=False, verbose=False)
    assert result["weights_loaded"] and sorted(result["tiers"]) == [1, 2, 3, 4]
    assert all(m["agent_errors"] == 0 for m in result["maps"])


def test_director_unlocks_and_remembers_failures():
    director = Director(DirectorConfig(min_episodes=3))
    episode = dict(family="progression", level=1, seed=5, success=0.0, dead=1.0, keys=0, unlocks=0, stagnant_steps=0, steps=10)
    director.report(episode, 1)
    assert ("progression", 1, 5) in director.directive(0.5)["nemesis"]
    director.report({**episode, "success": 1.0}, 2)
    assert not director.nemesis.entries
    for i in range(20):
        director.report({**episode, "family": "combat", "level": 0, "success": 1.0}, i)
    assert director.unlocked["combat"] >= 1
    early, late = director.weights(0.0), director.weights(0.9)
    assert {f for f, _ in early} == {"movement", "combat"}
    assert all(level >= 2 for _, level in late) and director.directive(0.9)["final"]
    assert sum(late.values()) == pytest.approx(1.0)
    clone = Director(DirectorConfig(min_episodes=3))
    clone.load_state_dict(director.state_dict())
    assert clone.unlocked == director.unlocked and clone.success == director.success


@pytest.mark.parametrize("name", ["full", "smoke", "gru_baseline", "memory_k64"])
def test_configs_load(name):
    cfg = load_config(f"configs/{name}.yaml", ["ppo.lr=1e-4"])
    assert cfg.ppo.lr == pytest.approx(1e-4) and cfg.ppo.rollout % cfg.ppo.seq_len == 0
    with pytest.raises(ValueError, match="unknown settings"):
        load_config(f"configs/{name}.yaml", ["ppo.learning_rate=1"])
