# Streamed Mind

Our entry for **Yugaantar 2026 · DevForge — Track 1: "Can It Play DOOM?"**: a recurrent PPO agent with
a budgeted, position-anchored spatial memory. Design: [docs/BRIEF.md](docs/BRIEF.md). Facts about the
environment: [docs/FINDINGS.md](docs/FINDINGS.md).

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

| Folder | Contents |
|---|---|
| `models/` | actor only: encoders, odometry, streamed memory, core, action heads |
| `submission/` | `DoomAgent` with its safety shell, and exported weights (Git LFS) |
| `training/` | PPO, rewards, map generator, curriculum ([details](training/README.md)) |
| `evaluation/` | held-out suite, local scorer, latency profile |
| `configs/` | one YAML file per experiment |

The rest of this file is the original template guide.

---

# devforge-doom-template

Starter template for **Yugaantar 2026 · DevForge — Track 1: "Can It Play DOOM?"**

> For decades, *"Can it run DOOM?"* was the ultimate hardware test. Now: **can your neural network
> actually play DOOM?** Train an agent that plays unseen levels **zero-shot**: survive ambushes,
> navigate multi-room layouts, find keycards, unlock doors and reach the exit.

No dataset is provided. Your agent learns by interacting with the engine.

## Quickstart

1. Click **"Use this template"** on GitHub to create your team's (public) repository.
2. Set up the environment (Python 3.13, [uv](https://docs.astral.sh/uv/)):

   ```bash
   uv sync                 # installs vizdoom, gymnasium and the official wrapper (env/) in editable mode
   uv run python validate.py
   ```

   Or use the official image, which is exactly what the scorer runs:

   ```bash
   docker build -t devforge-doom .            # or: docker pull <official image tag, announced>
   python validate.py --docker --image devforge-doom   # installs requirements.txt, then runs sandboxed like the scorer
   ```

3. Replace the random baseline in `submission/agent.py` with your trained agent.
4. Put weights in `submission/weights/` (Git LFS) and training code in `training/`.
5. Run `python validate.py --docker` and submit your repo URL.

**Platforms:** Linux (x86-64/ARM), macOS on Apple Silicon (macOS 15+) and Windows (x86-64) all
work with `uv sync` and `validate.py`. Intel Macs have no `vizdoom` 1.3.3 wheel: use Docker or a
Linux machine. Evaluation runs on Linux, so `validate.py --docker` is the closest check.

## Repository layout

```
env/                  official ViZDoom Gymnasium wrapper (doom_env): spaces, sensors, protocol, limits
maps/                 one small test map (NOT an evaluation map); bring your own for training
submission/agent.py   your DoomAgent: the evaluation entry point
submission/weights/   model weights (Git LFS)
training/             your training pipeline
validate.py           run before submitting
tests/                tests for the wrapper and validator (uv run pytest)
docs/                 contract, rules, submission guide, evaluation details
```

## Using the environment

```python
from doom_env import make_env

env = make_env("maps/two_rooms.wad")     # test map; use your own maps for training
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Observations are a dict of `rgb`, `depth`, `vitals`, `rays` and `visible_entities`, all
ego-centric and partially observable. See [docs/CONTRACT.md](docs/CONTRACT.md).
The wrapper's `reward` is a basic default for convenience. Shape your own reward for training.

**Compute tip:** training only on raw pixels is expensive. The `rays`, `visible_entities` and
`vitals` signals run thousands of steps per second on a CPU. Hybrid pixel + spatial models are welcome.

## Hardware at evaluation

One NVIDIA **RTX 3090 (24 GB)** shared between evaluation workers. Each agent gets **2 GB VRAM,
8 GB RAM and 1 CPU thread**, so keep models compact and inference fast
(see [hardware](docs/EVALUATION.md#hardware) and [timeouts](docs/EVALUATION.md#timeouts)).
`validate.py` runs your agent under the same CPU/RAM limits and reports GPU memory.

## Documentation

| Doc | What's inside |
| --- | --- |
| [docs/CONTRACT.md](docs/CONTRACT.md) | Observation/action spaces and the `DoomAgent` interface |
| [docs/RULES.md](docs/RULES.md) | Ground rules: what's allowed, what gets you disqualified |
| [docs/SUBMISSION.md](docs/SUBMISSION.md) | Submission checklist, Git LFS, 1 GB limit, `requirements.txt` |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Map tiers, scoring categories, timeouts, replays |

## Rules in one minute

- Actions must come from **trained neural parameters**: no rule trees, no `A*`/Dijkstra, no `atan2` aimbots,
  no unmodified off-the-shelf SB3.
- Change anything you want **during training**. Evaluation uses the **vanilla engine**, the
  official wrapper and **unseen maps**.
- Create the repo during the hackathon. Libraries are fine; the core architecture and training must be yours.
