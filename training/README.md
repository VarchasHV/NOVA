# Training

Training-only code. Nothing here is imported by `submission/`.

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

## Configs

Each file in `configs/` is one experiment. Everything not set in the file uses the defaults in `config.py`.
Results are in [docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md).

| Config | Run name | What it tests | Status |
|---|---|---|---|
| `full.yaml` | `full` | Streamed Memory (K = 32) with the full curriculum: **the main run, and the source of the submitted weights** | 72.3M decisions in 14 h on the Acer laptop; the submitted weights are its best checkpoint (58.4M) |
| `gru_baseline.yaml` | `gru_baseline` | the same curriculum without spatial memory | not run (no time); the memory ablations in the brief take its place |
| `memory_k64.yaml` | `memory_k64` | 64 slots instead of 32 | not run |
| `smoke.yaml` | `smoke` | the whole pipeline runs end to end within an hour | used for checks before every long run |
