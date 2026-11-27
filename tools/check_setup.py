"""Run this first on the training machine: checks the GPU and the game, and measures speed.

    uv run python tools/check_setup.py
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def available_ram_gb() -> float | None:
    """Memory that can be used without swapping (Linux only; None elsewhere)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def main() -> int:
    ok = True
    print(f"python {platform.python_version()} on {platform.system()} {platform.machine()}, {os.cpu_count()} CPU threads")
    if sys.version_info[:2] != (3, 13):
        print("  FAIL: Python 3.13 is required (uv sync installs it)")
        ok = False

    import torch

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"torch {torch.__version__}: CUDA OK, {p.name}, {p.total_memory / 2**30:.1f} GB")
        x = torch.randn(512, 512, device="cuda")
        (x @ x).sum().item()  # fails here if the driver is too old for this build
    else:
        print(f"torch {torch.__version__}: NO CUDA. Training will be very slow.")
        print("  Check `nvidia-smi`, update the driver, or see docs/TRAINING.md (older CUDA builds).")
        ok = False

    lfs = shutil.which("git-lfs") or subprocess.run(["git", "lfs", "version"], capture_output=True).returncode == 0
    print("git lfs: OK" if lfs else "git lfs: MISSING (needed to commit the weights)")

    from models.actor import BBBActor
    from training.config import load_config
    from training.critic import PrivilegedCritic
    from training.director import Director
    from training.ppo import PPOLearner
    from training.rollout import Collector, ReturnScaler, compute_advantages
    from training.vec_env import VecEnv

    cfg = load_config(ROOT / "configs" / "full.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpus = os.cpu_count() or 4
    workers_options = sorted({max(1, cpus // 2), max(1, cpus - 2)})
    candidates = [(w, e) for w in workers_options for e in (8, 16)]
    best = None
    actor = BBBActor(cfg.model).to(device)
    critic = PrivilegedCritic(cfg.model.core_size).to(device)
    learner = PPOLearner(actor, critic, cfg.ppo)
    director = Director(cfg.director)
    for workers, per_worker in candidates:
        # Each environment runs its own game process. Skip settings that would not fit in RAM.
        need_gb = (workers * per_worker * 55 + workers * 120 + 2500) / 1024
        free_gb = available_ram_gb()
        if free_gb is not None and need_gb > free_gb:
            print(f"{workers:3d} workers x {per_worker:2d}: skipped, needs about {need_gb:.1f} GB RAM, {free_gb:.1f} GB free")
            continue
        env = VecEnv(cfg.env, workers, per_worker, seed=0)
        try:
            collector = Collector(env, actor, critic, cfg.ppo.rollout, cfg.ppo.seq_len, device)
            env.reset(director.directive(0.2))
            collector.collect(director.directive(0.2))  # warm up
            t = time.time()
            rollout, _ = collector.collect(director.directive(0.2))
            collect = time.time() - t
            scalers = [ReturnScaler(env.num_envs, 0.99), ReturnScaler(env.num_envs, 0.99)]
            compute_advantages(rollout, scalers, (0.995, 0.99), 0.95)
            t = time.time()
            learner.update(rollout, 0.0)
            update = time.time() - t
            steps = env.num_envs * cfg.ppo.rollout
            sps = steps / (collect + update)
            print(f"{workers:3d} workers x {per_worker:2d} = {env.num_envs:3d} envs: collect {steps / collect:7.0f}/s, update {update:5.1f}s, "
                  f"overall {sps:6.0f} decisions/s = {sps * 3600 / 1e6:.1f}M per hour")
            if best is None or sps > best[0]:
                best = (sps, workers, per_worker)
        except torch.cuda.OutOfMemoryError:
            print(f"{workers:3d} workers x {per_worker:2d}: out of GPU memory")
        finally:
            env.close()
    if device == "cuda":
        print(f"peak GPU memory during training test: {torch.cuda.max_memory_reserved() / 2**30:.2f} GB")
    if best is None:
        print("\nNOT READY: no setting fitted in memory; close other programs or lower ppo.rollout in configs/full.yaml")
        return 1
    sps, workers, per_worker = best
    print(f"\nrecommended: --set num_workers={workers} --set envs_per_worker={per_worker}")
    for hours in (8, 14, 18):
        print(f"  {hours:2d} h of training ~ {sps * 3600 * hours / 1e6:5.0f}M decisions")
    print("\nREADY" if ok else "\nNOT READY: fix the items above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
