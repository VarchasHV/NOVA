"""Run the real `submission/agent.py` on the held-out suite and report the local score.

    python -m evaluation.run_heldout                               # the exported weights
    python -m evaluation.run_heldout --ckpt runs/full/ckpt_latest.pt
    python -m evaluation.run_heldout --ckpt ... --mode greedy --episodes 3

Same interface and 4-tic cadence as the scorer: one agent, reset() before every map.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from doom_env import spaces as S
from training.env import TrainEnv
from training.rewards import RewardConfig, RewardTracker

from .local_scorer import aggregate, score_map
from .suite import TIERS, build_suite

ROOT = Path(__file__).resolve().parents[1]


def load_agent(weights: str | Path | None, device: str):
    spec = importlib.util.spec_from_file_location("submission_agent", ROOT / "submission" / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if weights:
        module.WEIGHTS_PATH = Path(weights)
    agent = module.DoomAgent(S.observation_space(), S.action_space(), device)
    if weights and not agent.weights_loaded:
        raise RuntimeError(f"the agent could not load {weights}")
    return agent


def play(agent, env: TrainEnv, entry: dict, seed: int, on_step=None) -> dict:
    env.load(entry["wad"], entry["map"], entry["ticks"], skill=3)
    obs, _ = env.reset(seed=seed)
    ws = env.world_state()
    totals = {"monsters": len(ws["monsters"]), "keys": len(ws["key_positions"]),
              "locks": int((env.meta.door_locks > 0).sum())}
    tracker = RewardTracker(RewardConfig(), ws, env.meta, totals)
    t0 = time.perf_counter()
    agent.reset()
    reset_ms = 1000 * (time.perf_counter() - t0)
    errors_before, latencies, done, steps = agent.errors, [], False, 0
    while not done:
        t = time.perf_counter()
        action = agent.act(obs)
        latencies.append(1000 * (time.perf_counter() - t))
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ws = env.world_state()
        tracker.step(ws)
        steps += 1
        if on_step:
            on_step(agent, ws, steps)
    m = tracker.summary(ws, entry["family"])
    m = {k: v for k, v in m.items() if not k.startswith("r_")}
    m.update(
        tier=entry["tier"], family=entry["family"], map=entry["map"], level=entry["level"], steps=steps,
        health_frac=min(max(ws["health"], 0.0), 100.0) / 100.0,
        time_left_frac=max(0.0, 1.0 - ws["tic"] / entry["ticks"]),
        latency_mean_ms=float(np.mean(latencies)), latency_p99_ms=float(np.percentile(latencies, 99)),
        first_act_ms=latencies[0], reset_ms=reset_ms, agent_errors=agent.errors - errors_before,
    )
    m.update(score_map(m))
    return m


def evaluate(weights, device: str = "cpu", maps_per_tier: int = 6, episodes: int = 1, mode: str | None = None,
             temperature: float | None = None, freedoom: bool = True, verbose: bool = True,
             variety: float = 0.0) -> dict:
    agent = load_agent(weights, device)
    if mode:
        agent.mode = mode
    if temperature is not None:
        agent.temperature = temperature
    tmp = tempfile.mkdtemp(prefix="streamed-mind-eval-")
    try:
        suite = build_suite(tmp, maps_per_tier, freedoom, variety)
        env = TrainEnv(suite[0]["wad"], suite[0]["map"], render_buffers=agent.model.cfg.use_depth)
        results = []
        for entry in suite:
            for ep in range(episodes):
                m = play(agent, env, entry, seed=1000 + ep)
                results.append(m)
                if verbose:
                    print(f"tier {m['tier']} {m['family']:11s} {m['map']:6s} L{m['level']} score {m['score']:6.1f} "
                          f"exit {m['exit']:.0f} keys {m['keys']:.0f} unlocks {m['unlocks']:.0f} kills {m['kills']:.0f} "
                          f"sectors {m['sector_frac']:.0%} {'DIED' if m['dead'] else ''}")
        env.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = aggregate(results)
    out.update(maps=results, mode=agent.mode, temperature=agent.temperature, weights_loaded=agent.weights_loaded)
    return out


def format_table(result: dict) -> str:
    keys = ("score", "exit", "progression", "keys", "unlocks", "sector_frac", "kill_frac", "dead", "damage",
            "stagnant_steps", "latency_mean_ms")
    lines = ["| metric | " + " | ".join(f"T{t} {TIERS[t]}" for t in result["tiers"]) + " |",
             "|---" * (len(result["tiers"]) + 1) + "|"]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(f"{result['tiers'][t][k]:.2f}" for t in result["tiers"]) + " |")
    lines.append(f"\nTier-weighted local score: **{result['score']:.2f}**")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--weights", help="exported weights file (default: submission/weights/model.pt)")
    src.add_argument("--ckpt", help="training checkpoint (exported to a temporary file first)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--maps-per-tier", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=1, help="seeds per map")
    ap.add_argument("--mode", choices=("sample", "greedy"))
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--no-freedoom", action="store_true")
    ap.add_argument("--variety", type=float, default=0.0,
                    help="share of generated maps with uneven shapes, steps and ledges (1 = all); not comparable with 0")
    ap.add_argument("--json", help="write the full result here")
    ap.add_argument("--run-dir", help="append to eval.jsonl there and keep the best checkpoint as best.pt")
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.variety and args.run_dir:
        ap.error("--run-dir keeps scores on the plain suite only")

    with tempfile.TemporaryDirectory() as tmp:
        weights = args.weights
        snapshot = None
        if args.ckpt:
            from training.export import export

            # Copy first: training may overwrite the checkpoint while we are still evaluating it.
            snapshot = shutil.copy(args.ckpt, Path(tmp) / "snapshot.pt")
            weights = export(snapshot, Path(tmp) / "model.pt", mode=args.mode or "sample")
        result = evaluate(weights, args.device, args.maps_per_tier, args.episodes, args.mode, args.temperature,
                          not args.no_freedoom, verbose=not args.quiet, variety=args.variety)
        print(format_table(result), flush=True)
        if args.json:
            Path(args.json).write_text(json.dumps(result, indent=2))
        if args.run_dir:
            run = Path(args.run_dir)
            best_file = run / "best.json"
            best = json.loads(best_file.read_text()) if best_file.exists() else None
            key = (args.maps_per_tier, args.episodes, not args.no_freedoom)
            record = {"step": args.step, "score": result["score"], "suite": list(key), "tiers": result["tiers"]}
            with open(run / "eval.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
            if snapshot and (best is None or best["suite"] != list(key) or result["score"] > best["score"]):
                shutil.copy(snapshot, run / "best.pt")
                best_file.write_text(json.dumps(record, indent=2))
                print(f"new best checkpoint at step {args.step:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
