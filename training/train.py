"""Train BBB (Bictor Bon Boom).

    python -m training.train --config configs/full.yaml          # run again to resume
    python -m training.train --config configs/full.yaml --set ppo.lr=2e-4 --run-name try2

Progress through the curriculum and the schedules is max(steps / total_steps, hours / max_hours).
Ctrl+C saves a checkpoint and exits.

PyTorch is imported inside main(): on Windows and macOS the environment worker processes re-import
this module, and they must stay light.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_hash() -> str:
    try:
        run = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, timeout=10).stdout.strip()
        return run("rev-parse", "HEAD") + (" (modified)" if run("status", "--porcelain", "-uno") else "")
    except OSError:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--run-name")
    ap.add_argument("--device", default="auto", help="cuda | cpu | auto")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint for this run")
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from models.actor import BBBActor

    from .config import load_config, to_dict
    from .critic import PrivilegedCritic
    from .director import Director
    from .ppo import PPOLearner
    from .rollout import Collector, ReturnScaler, compute_advantages
    from .vec_env import VecEnv

    cfg = load_config(args.config, args.overrides)
    if args.run_name:
        cfg.name = args.run_name
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if cfg.num_workers < 0:
        cfg.num_workers = max(1, (os.cpu_count() or 4) - 2)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 2)) if device == "cuda" else max(1, (os.cpu_count() or 4) // 2))
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    run_dir = ROOT / "runs" / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = run_dir / "ckpt_latest.pt"
    resume = None
    if latest.exists() and not args.fresh:
        resume = torch.load(latest, map_location=device, weights_only=False)  # our own file
        saved, wanted = resume["model_config"], cfg.model.to_dict()
        if saved != wanted:
            raise SystemExit(f"{latest} was trained with different model settings.\n saved:  {saved}\n config: {wanted}\n"
                             "Use --run-name for a new run, or --fresh to overwrite.")

    actor = BBBActor(cfg.model).to(device)
    critic = PrivilegedCritic(cfg.model.core_size, cfg.ppo.critic_detach).to(device)
    learner = PPOLearner(actor, critic, cfg.ppo)
    director = Director(cfg.director, cfg.seed)
    gammas = (cfg.ppo.gamma_ext, cfg.ppo.gamma_int)
    num_envs = max(cfg.num_workers, 1) * cfg.envs_per_worker
    scalers = [ReturnScaler(num_envs, g) for g in gammas]
    counters = {"steps": 0, "update": 0, "elapsed": 0.0, "warmup_done": 0, "episodes": 0}

    if resume:
        actor.load_state_dict(resume["actor"])
        critic.load_state_dict(resume["critic"])
        learner.opt.load_state_dict(resume["opt"])
        learner.odom_opt.load_state_dict(resume["odom_opt"])
        director.load_state_dict(resume["director"])
        for scaler, s in zip(scalers, resume["scalers"]):
            scaler.load_state_dict(s)
        counters.update(resume["counters"])
        print(f"resumed {latest} at {counters['steps']:,} decisions, {counters['elapsed'] / 3600:.2f} h")
    elif cfg.init_from:
        source = torch.load(cfg.init_from, map_location=device, weights_only=False)
        missing = actor.load_state_dict(source["actor"], strict=False)
        if "critic" in source:
            critic.load_state_dict(source["critic"], strict=False)
        print(f"initialised from {cfg.init_from} (new or unused tensors: {missing})")

    n_params = sum(p.numel() for p in actor.parameters())
    print(f"run '{cfg.name}' on {device}: {cfg.num_workers} workers x {cfg.envs_per_worker} envs, "
          f"actor {n_params / 1e6:.2f}M parameters, git {git_hash()[:12]}")
    (run_dir / "config.json").write_text(json.dumps({**to_dict(cfg), "git": git_hash(), "device": device}, indent=2))

    writer = SummaryWriter(run_dir / "tb")
    log_file = open(run_dir / "log.jsonl", "a")
    episode_file = open(run_dir / "episodes.jsonl", "a")
    env = VecEnv(cfg.env, cfg.num_workers, cfg.envs_per_worker, cfg.seed + counters["update"])
    collector = Collector(env, actor, critic, cfg.ppo.rollout, cfg.ppo.seq_len, device)

    def progress() -> float:
        return min(1.0, max(counters["steps"] / cfg.total_steps, counters["elapsed"] / (cfg.max_hours * 3600)))

    def save(path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save({
            "model_config": cfg.model.to_dict(), "config": to_dict(cfg), "git": git_hash(),
            "actor": actor.state_dict(), "critic": critic.state_dict(),
            "opt": learner.opt.state_dict(), "odom_opt": learner.odom_opt.state_dict(),
            "director": director.state_dict(), "scalers": [s.state_dict() for s in scalers], "counters": dict(counters),
        }, tmp)
        os.replace(tmp, path)  # never leave a half-written checkpoint

    evaluator: subprocess.Popen | None = None

    def start_evaluation() -> None:
        nonlocal evaluator
        if evaluator is not None and evaluator.poll() is None:
            return
        snapshot = run_dir / "ckpt_eval.pt"
        save(snapshot)
        out = open(run_dir / "eval.log", "a")
        evaluator = subprocess.Popen(
            [sys.executable, "-m", "evaluation.run_heldout", "--ckpt", str(snapshot), "--device", "cpu",
             "--maps-per-tier", str(cfg.eval_maps_per_tier), "--run-dir", str(run_dir),
             "--step", str(counters["steps"]), "--quiet"],
            cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)

    try:
        env.reset(director.directive(progress()))

        # ------------------------------------------------ stage 0: odometry from random walking
        warm = cfg.odom_warmup_updates if cfg.model.memory else 0
        while counters["warmup_done"] < warm:
            t0 = time.time()
            rollout, _ = collector.collect(director.directive(0.0), random_policy=True)
            stats = learner.train_odometry(rollout, cfg.ppo.odom_updates * 8)
            counters["warmup_done"] += 1
            counters["elapsed"] += time.time() - t0
            for k, v in stats.items():
                writer.add_scalar(f"warmup/{k}", v, counters["warmup_done"])
            if counters["warmup_done"] % 10 == 0 or counters["warmup_done"] == warm:
                print(f"odometry warm-up {counters['warmup_done']}/{warm}: "
                      f"error {stats['odom_translation_error_units']:.1f} units, {stats['odom_turn_error_deg']:.2f} deg per decision")
                save(latest)
        env.reset(director.directive(progress()))
        collector.reset_state()

        # ------------------------------------------------ main loop
        recent: dict[str, deque] = {}
        last_save = last_snapshot = last_eval = time.time()
        while progress() < 1.0:
            t0 = time.time()
            p = progress()
            actor.eval()
            rollout, episodes = collector.collect(director.directive(p))
            t_collect = time.time() - t0
            compute_advantages(rollout, scalers, gammas, cfg.ppo.lam)
            actor.train()
            stats = learner.update(rollout, p)
            counters["update"] += 1
            steps = cfg.ppo.rollout * num_envs
            counters["steps"] += steps
            for ep in episodes:
                director.report(ep, counters["steps"])
                recent.setdefault(ep["family"], deque(maxlen=200)).append(ep)
                episode_file.write(json.dumps({"step": counters["steps"], **ep}) + "\n")
            counters["episodes"] += len(episodes)
            dt = time.time() - t0
            counters["elapsed"] += dt

            if counters["update"] % cfg.log_every == 0:
                stats.update(fps=steps / dt, fps_collect=steps / t_collect, progress=p, hours=counters["elapsed"] / 3600,
                             mean_reward_ext=float(rollout["rew"][..., 0].mean()), mean_reward_int=float(rollout["rew"][..., 1].mean()))
                for k, v in stats.items():
                    writer.add_scalar(f"train/{k}", v, counters["steps"])
                summary = {}
                for family, eps in recent.items():
                    for key in eps[0]:
                        if key in ("family", "seed", "level"):
                            continue
                        summary[f"{family}/{key}"] = float(np.mean([e[key] for e in eps]))
                    summary[f"{family}/level_open"] = float(director.unlocked.get(family, 0))
                for k, v in summary.items():
                    writer.add_scalar(f"episode/{k}", v, counters["steps"])
                log_file.write(json.dumps({"step": counters["steps"], "update": counters["update"], **stats, **summary}) + "\n")
                log_file.flush()
                episode_file.flush()
                success = " ".join(f"{f[:4]}:{np.mean([e['success'] for e in eps]):.2f}/L{director.unlocked.get(f, '-')}"
                                   for f, eps in sorted(recent.items()))
                print(f"upd {counters['update']:5d} | {counters['steps'] / 1e6:7.2f}M | {steps / dt:6.0f} sps | "
                      f"{counters['elapsed'] / 3600:5.2f}h {100 * p:5.1f}% | kl {stats.get('kl', 0):.4f} | "
                      f"odom {stats.get('odom_turn_error_deg', 0):.1f}deg | {success}", flush=True)

            now = time.time()
            if now - last_save > cfg.save_minutes * 60:
                save(latest)
                last_save = now
            if cfg.snapshot_minutes and now - last_snapshot > cfg.snapshot_minutes * 60:
                save(run_dir / f"ckpt_{counters['steps'] // 1_000_000:05d}M.pt")
                last_snapshot = now
            if cfg.eval_minutes and now - last_eval > cfg.eval_minutes * 60:
                start_evaluation()
                last_eval = now
        save(latest)
        if cfg.eval_minutes:
            if evaluator is not None:
                evaluator.wait()
            start_evaluation()
            evaluator.wait()
        best = run_dir / "best.pt"
        print(f"finished: {counters['steps']:,} decisions. Next: python -m training.export --ckpt {best if best.exists() else latest}")
    except KeyboardInterrupt:
        print("\ninterrupted: saving checkpoint")
        save(latest)
    finally:
        env.close()
        writer.close()
        log_file.close()
        episode_file.close()
        if evaluator is not None and evaluator.poll() is None:
            evaluator.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
