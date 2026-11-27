# BBB (Bictor Bon Boom)

**Budgeted, position-anchored spatial memory for zero-shot DOOM play.**
Our entry for *Yugaantar 2026 · DevForge, Track 1: "Can It Play DOOM?"*

📄 **[Full notes, training guide & model history](architecture.md)** · **[2-page brief (PDF)](docs/brief.pdf)** · [design notes](docs/BRIEF.md) · [environment findings](docs/FINDINGS.md) · [experiment log](docs/EXPERIMENTS.md)

BBB (Bictor Bon Boom) is a 1.07M-parameter recurrent agent. It plays DOOM levels it has never seen, using only ego-centric telemetry (rays, visible entities, and vitals) — no pixels and no global coordinates. It maintains **32 spatial memory slots**, each anchored to a pose estimated by a learned odometry network. Unmapped space allocates a new slot; once full, the least relevant slot is overwritten and slots are read by attention from the agent's current point of view.

![Architecture](docs/brief/architecture.png)

## Highlights

- **Custom recurrent PPO** (no RL library) with an asymmetric privileged critic, separate extrinsic/intrinsic value heads, and a reward matched to the scoring categories.
- **Procedural map factory** with room graphs, doors, 1–3 key-and-lock chains, switch or walk-over exits, uneven floors, and Freedoom maps for variety.
- **Curriculum Director** focusing training on ~50% solve-rate scenarios, replaying failed "nemesis" seeds, and randomly mirroring episodes left-to-right.
- **Fast inference**: under 1 ms per decision on one CPU thread (budget: 40 ms), with 4.3 MB of weights.
- **Purely learned policy**: every action comes from the network, with no rule-based aiming or pathfinding.

## Results

Evaluated across two held-out sets of 249 episodes each (240 generated maps + 3 Freedoom maps across all four tiers, fixed seeds) using `submission/agent.py` (tiers weighted 1:2:3:4; estimated local weights):

| Tier | Earlier checkpoint (40.5M) | **Submitted (58.4M)** |
|---|---|---|
| 1 Combat | 143 · exit 92% | 142 · exit 93% |
| 2 Exploration | 128 · exit 70% | 129 · exit 70% |
| 3 Progression | 150 · exit 65% | **163 · exit 75%** |
| 4 Gauntlet | 102 · exit 35% | 107 · exit 39% |
| **Weighted** | 125.7 | **131.3** (+5.7, 95% CI 0.2 to 11.3) |

For memory ablations, known weaknesses, training details, and older checkpoints, see [architecture.md](architecture.md).

## Installation (DEMO)

### 1. Make sure Git LFS downloads the real model weights

All `*.pt` files are tracked with Git LFS. The latest submitted model (`submission/weights/model.pt`, 58.4M decisions, ~4.2 MB) is stored via Git LFS:

```bash
git lfs install
git clone https://github.com/VarchasHV/NOVA.git && cd NOVA
git lfs pull
```

> If `submission/weights/model.pt` is still a tiny pointer file (< 200 B), `submission/agent.py` will warn and fall back to an untrained network.

### 2. Install dependencies

Requires Python `>=3.13`. Install `vizdoom`, `torch`, and the local `doom-env` package with `uv`:

```bash
uv sync
```

*(Optional headless check)*:

```bash
uv run python validate.py
```

### 3. Run the latest model in a ViZDoom window

`tools/watch.py` opens a live 800×600 ViZDoom window with the HUD at real-time speed (35 tics/sec) on held-out maps:

```bash
uv run python tools/watch.py --ckpt submission/weights/model.pt
```

Useful options:
- `--mode greedy` — always picks the most likely action (steadier play; default `--mode sample` re-exports at `temperature=1.0`, whereas `submission/weights/model.pt` stores `temperature=0.5`)
- `--tier 1` (or `2`, `3`, `4`) — filter maps by tier (`1` Combat, `2` Exploration, `3` Progression, `4` Gauntlet)
- `--variety` — include maps with uneven shapes, steps, and ledges
- `--speed 2` — playback speed multiplier (`1.0` is real time)
- `--episodes 5` — stop after `5` maps (default loops until `Ctrl+C` or closing the window)

> **Note:** `tools/watch.py` passes `--ckpt` through `training/export.py`, which reads a training checkpoint's `counters["steps"]` key. When running the already-exported `submission/weights/model.pt`, it will print `trained for 0.0M decisions` in the terminal even though it is running the 58.4M-step model.


## Setup

On macOS the default shell (zsh) does not ignore comments you paste; run the first line once per terminal.

```bash
setopt interactivecomments                                     # macOS zsh only: allow the comments below
git lfs install                                                # once per computer, so model files download
git clone https://github.com/VarchasHV/NOVA.git && cd NOVA     # get the code and the model
git lfs pull                                                   # fetch the model file if it is only a pointer
uv sync                                                        # install Python packages (CUDA build on Linux and Windows)
uv run python validate.py                                      # official checks on the model in submission/
```

If any output says `using an UNTRAINED network`, the model file did not download: run `git lfs pull` again.
An untrained network plays very badly.

## Training guide

Training needs an NVIDIA GPU. Our runs took 14 hours on an RTX 3050 laptop (about 72M decisions); `configs/full.yaml`
runs for 16 hours unless you set `max_hours`.

### 1. Check the computer

```bash
uv run python tools/check_setup.py                             # GPU, speed test, suggested worker settings
uv run python -m training.train --config configs/smoke.yaml    # short run of the whole pipeline, up to 1 hour
```

### 2. Train

```bash
uv run python -m training.train --config configs/full.yaml                      # main run; saves to runs/full/
uv run python -m training.train --config configs/full.yaml                      # the same command again resumes it
uv run python -m training.train --config configs/full.yaml --set max_hours=14   # total hours, including time already trained
uv run python -m training.train --config configs/full.yaml --set num_workers=10 # use the value check_setup.py suggests
uv run tensorboard --logdir runs                                                # training curves at http://localhost:6006
```

`Ctrl+C` saves a checkpoint and stops. If you started with `--set` options, pass the same ones when resuming.

| File in `runs/<name>/` | What it is |
|---|---|
| `ckpt_latest.pt` | newest state; resuming uses this |
| `best.pt` | best score so far on the 27-game test that runs during training |
| `ckpt_XXXXXM.pt` | hourly snapshots, named by millions of decisions |
| `eval.jsonl` | every test score |

### 3. Continue from a trained checkpoint

Copy it into a new run folder, so the original stays untouched. `max_hours` must be larger than the hours the
checkpoint has already trained.

```bash
mkdir -p runs/more && cp runs/full/best.pt runs/more/ckpt_latest.pt                              # start a copy
uv run python -m training.train --config configs/full.yaml --run-name more --set max_hours=16    # keep training it
```

Only training checkpoints can be continued. The model files in git hold the network alone, without the optimiser or critic.

### 4. Other experiments

Each file in `configs/` is one experiment ([which is which](training/README.md#configs)).

```bash
uv run python -m training.train --config configs/gru_baseline.yaml   # no spatial memory (not run by us)
uv run python -m training.train --config configs/memory_k64.yaml     # 64 memory slots (not run by us)
```

### 5. Train on another laptop over SSH

Set up SSH first ([docs/TRAINING.md](docs/TRAINING.md)). Replace `acer` with your SSH alias.

```bash
tools/remote.sh setup acer                        # copy the code, install packages, check speed
tools/remote.sh start acer configs/full.yaml      # start or resume training in tmux
tools/remote.sh status acer                       # GPU load and the last log lines
tools/remote.sh stop acer                         # save a checkpoint and stop
tools/remote.sh fetch acer                        # copy checkpoints back to runs/acer/
python3 tools/dashboard.py --hosts acer           # live page at http://localhost:8765
uv run python tools/watch.py acer --best          # watch that laptop's best checkpoint play
```

## Using different models

The agent always loads `submission/weights/model.pt`. Earlier versions of that file are kept in git history.

| Commit | Model | Decisions | File hash starts |
|---|---|---|---|
| `c6dbfd5` | best checkpoint (**submitted**) | 58.4M | `a5b6f43e` |
| `9e9a299` | earlier checkpoint | 40.5M | `824eefb4` |
| `f00a8f1` | earlier checkpoint | 24.9M | `80b3fc46` |

All three come from the same training run and sample their actions at temperature 0.5.

### Get an earlier model

This needs the full history (not a `--depth 1` clone).

```bash
mkdir -p runs/models                                                                      # a folder git ignores
git cat-file --filters f00a8f1:submission/weights/model.pt > runs/models/main_24M.pt      # the 24.9M model
git cat-file --filters 9e9a299:submission/weights/model.pt > runs/models/main_40M.pt      # the 40.5M model
shasum -a 256 runs/models/*.pt                                                            # compare with the table
```

### Test a model without installing it

```bash
uv run python -m evaluation.run_heldout                                                   # the installed model, 27 games
uv run python -m evaluation.run_heldout --weights runs/models/main_40M.pt                  # another model file
uv run python -m evaluation.run_heldout --weights runs/models/main_40M.pt --maps-per-tier 12 --episodes 3   # more games, steadier score
uv run python -m evaluation.run_heldout --weights runs/models/main_40M.pt --variety 1      # maps with steps and ledges
uv run python -m evaluation.run_heldout --ckpt runs/full/best.pt --temperature 0.5        # a training checkpoint
uv run python tools/viz_topdown.py --weights runs/models/main_40M.pt --tier 3 --out episode.png   # top-down picture of one game
uv run python tools/watch.py --ckpt runs/models/main_40M.pt                                # watch it play (uses temperature 1.0)
```

Compare models on the same options only. A 27-game score can swing by about 20 points, so small gaps mean little.

### Install a model

```bash
cp runs/models/main_40M.pt submission/weights/model.pt                       # use a downloaded model
uv run python -m training.export --ckpt runs/full/best.pt --temperature 0.5 # or export a checkpoint (the default temperature is 1.0)
uv run python validate.py                                                   # official checks
uv run python validate.py --map path/to/map.wad:MAP01                       # try it on your own map
uv run python -m evaluation.profile_latency                                 # decision speed on one CPU thread
git checkout -- submission/weights/model.pt                                 # undo: go back to the committed model
```

To publish it, commit `submission/weights/model.pt` (Git LFS handles the file) and push. The scorer reads the
last commit on the `main` branch.

## Other tools

```bash
uv run pytest                          # test suite
uv run python tools/demo.py all        # demo pages in runs/demo/ (replay, killcam, deaths, curves)
uv run python validate.py --docker     # official checks inside Docker (build once: docker build -t devforge-doom .)
```
## Repository layout

| Folder | Contents |
|---|---|
| `submission/` | `DoomAgent` entry point, model definitions (`submission/models/`), and exported weights |
| `training/` | PPO, rewards, map generator, curriculum ([details](training/README.md)) |
| `evaluation/` | Held-out suite, local scorer, latency profiler |
| `configs/` | Experiment YAML configs |
| `tools/` | Live viewer (`watch.py`), dashboard, top-down visualizer, demo pages |
| `docs/` | Project brief, design notes, findings, experiment log, and official rules/contract |
| `env/`, `validate.py`, `maps/` | Official template files |

This repository was created from the official [devforge-doom-template](https://github.com/Reinforce-SST/devforge-doom-template) during the hackathon.
