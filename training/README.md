# Training

Training-only code. Nothing here is imported by `submission/` or `models/`.

| File | Role |
|---|---|
| `train.py` | entry point: odometry warm-up, PPO loop, checkpoints, resume, held-out evaluation |
| `config.py` | YAML files in `configs/` become dataclasses |
| `vec_env.py` | worker processes, shared-memory buffers, map packs |
| `env.py`, `privileged.py` | `TrainEnv` (official wrapper + ground truth) and the critic's input |
| `maps/` | level generator, UDMF writer, solvability check |
| `rewards.py`, `novelty.py` | scorer-aligned reward and episodic exploration bonus |
| `rollout.py`, `ppo.py`, `critic.py`, `aux.py` | experience, GAE, learner, asymmetric critic, auxiliary losses |
| `director.py`, `nemesis.py`, `mirror.py` | curriculum, replay of failed seeds, mirror augmentation |
| `export.py` | checkpoint to submission weights (actor only) |

Setup and running on the GPU laptops: [docs/TRAINING.md](../docs/TRAINING.md).
