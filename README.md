# BBB (Bictor Bon Boom)

**Budgeted, position-anchored spatial memory for zero-shot DOOM play.**
Our entry for *Yugaantar 2026 · DevForge, Track 1: "Can It Play DOOM?"*

📄 **[2-page project brief (PDF)](docs/brief.pdf)** · [design notes](docs/BRIEF.md) · [environment findings](docs/FINDINGS.md) · [experiment log](docs/EXPERIMENTS.md)

BBB (Bictor Bon Boom) is a 1.07M-parameter recurrent agent. It plays DOOM levels it has never seen, using only the official
ego-centric telemetry: rays, visible entities and vitals. It gets no pixels and no global coordinates. Open-world games keep a fixed budget of
world chunks loaded, streaming chunks in and out as the player moves. Our agent manages its memory the same way. It has **32 memory slots**, each
**anchored to a pose that the agent estimates for itself** with a learned odometry network. When the agent reaches unmapped space, a new slot
is created; when all 32 are in use, the **least relevant slot (by a learned score) is overwritten**. Slots are read by
attention from the agent's current point of view. This lets the agent remember "a locked door behind me and to the left,
three rooms ago" without being given a map.

![Architecture](docs/brief/architecture.png)

## Highlights

- **Our own recurrent PPO** (no RL library) with an **asymmetric privileged critic**, separate extrinsic and intrinsic
  value heads, and a reward built to match the official scoring categories.
- **Our own procedural map factory**: room graphs, doors, one to three key-and-lock chains, switch or walk-over exits,
  uneven floors. A solver checks every level, and Freedoom maps add variety.
- **Curriculum Director**: trains most on scenarios the agent solves about half the time, replays seeds it failed ("nemesis"
  episodes) and mirrors episodes left to right at random.
- **Fast enough to spare**: 1.7 ms per decision on one CPU thread (the budget is 40 ms), with 5.1 MB of weights.
- **Every action comes from the network.** The code around it only checks inputs and outputs. It contains no rules, pathfinding or aim code.

## Results

Evaluation used 243 generated maps that were never used in training, plus Freedoom maps, across all four tiers (249 episodes, fixed seeds). Each policy ran through the real
`submission/agent.py` and was scored by our local copy of the scorer (tiers weighted 1:2:3:4). The organisers' exact weights are not public, so the
absolute numbers are estimates.

| Tier | 40.5M decisions | Main run, 62.0M | Aim run, 61.0M |
|---|---|---|---|
| 1 Combat | 139 · exit 88% | 140 · exit 88% | 140 · exit 88% |
| 2 Exploration | 114 · exit 60% | 115 · exit 58% | **127 · exit 67%** |
| 3 Progression | 145 · exit 58% | **160 · exit 72%** | 153 · exit 68% |
| 4 Gauntlet | 97 · exit 33% | 107 · exit 39% | 107 · exit 42% |
| **Weighted** | 118.8 | 127.5 (+9.0, 95% CI 0.5 to 17.7) | **128.3** (+9.9, 95% CI 0.6 to 19.0) |

**Does the memory help?** Yes. Wiping all memory at every step costs **50 points** (95% CI 36 to 65). Zeroing only
the slot readout costs **10 points** (95% CI 1 to 20), mostly in exploration and progression, which is where remembering
places should matter.

![Memory ablation](docs/brief/memory_ablation.png)

**Known weaknesses:** switch exits (31–52% exit rate, compared with 61–100% for walk-over exits), deaths in tier 4 (about 45% of episodes),
and a low kill rate (about 15% of enemies).

## Train, evaluate, submit

```bash
uv sync
uv run python tools/check_setup.py                            # GPU, speed, suggested worker settings
uv run python -m training.train --config configs/smoke.yaml   # whole pipeline, up to 1 hour
uv run python -m training.train --config configs/full.yaml    # real run; run again to resume
uv run tensorboard --logdir runs
uv run python -m evaluation.run_heldout --ckpt runs/full/best.pt
uv run python -m training.export --ckpt runs/full/best.pt     # writes submission/weights/model.pt
uv run python -m evaluation.profile_latency
uv run python tools/viz_topdown.py --tier 3 --out episode.png
uv run python tools/demo.py all                              # demo pages in runs/demo/ (replay, killcam, deaths, curves)
uv run python validate.py
```

Add `--set key=value` to override any setting, for example `--set max_hours=12 --set num_workers=10`.
Driving the GPU laptops from the Mac: [docs/TRAINING.md](docs/TRAINING.md).

## Repository layout

| Folder | Contents |
|---|---|
| `submission/` | `DoomAgent` (the evaluation entry point) with its safety checks, and the exported weights (Git LFS) |
| `submission/models/` | the actor only: encoders, odometry, Streamed Memory, GRU core, action heads |
| `training/` | PPO, rewards, map generator, curriculum ([details](training/README.md)) |
| `evaluation/` | held-out suite, local scorer, latency profile |
| `configs/` | one YAML file per experiment ([which is which](training/README.md#configs)) |
| `tools/` | live training dashboard, demo pages, top-down episode viewer, remote-training helpers |
| `docs/` | brief, design notes, environment findings, experiment log, and the official contract and rules |
| `env/`, `validate.py`, `maps/` | official template files, unchanged |

## Official documents

| Doc | What's inside |
| --- | --- |
| [docs/CONTRACT.md](docs/CONTRACT.md) | Observation and action spaces and the `DoomAgent` interface |
| [docs/RULES.md](docs/RULES.md) | Ground rules |
| [docs/SUBMISSION.md](docs/SUBMISSION.md) | Submission checklist, Git LFS, size limit |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Map tiers, scoring categories, timeouts |

This repository was created from the official
[devforge-doom-template](https://github.com/Reinforce-SST/devforge-doom-template) during the hackathon.
