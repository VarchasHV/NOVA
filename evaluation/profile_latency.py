"""Time the submitted agent the way the sandbox runs it: one CPU thread.

    python -m evaluation.profile_latency [--device cuda]

Targets: mean act() < 15 ms and p99 < 100 ms (the hard budget averages 40 ms per decision).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Must be set before numpy / torch load.
from doom_env.limits import AGENT_MS_PER_STEP, MAX_VRAM_MB, THREAD_ENV

os.environ.update(THREAD_ENV)

import numpy as np  # noqa: E402

from .run_heldout import ROOT, load_agent  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args(argv)

    from doom_env import make_env

    t = time.perf_counter()
    agent = load_agent(args.weights, args.device)
    init = time.perf_counter() - t
    env = make_env(ROOT / "maps" / "two_rooms.wad")
    obs, _ = env.reset(seed=0)
    t = time.perf_counter()
    agent.reset()
    reset = time.perf_counter() - t
    times = []
    for _ in range(args.steps):
        t = time.perf_counter()
        action = agent.act(obs)
        times.append(1000 * (time.perf_counter() - t))
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
            agent.reset()
    env.close()
    rest = np.array(times[1:])
    print(f"weights loaded: {agent.weights_loaded}   device: {agent.device}   agent errors: {agent.errors}")
    print(f"init {init:.2f}s | reset {1000 * reset:.2f}ms | first act {times[0]:.1f}ms")
    print(f"act mean {rest.mean():.2f}ms | p99 {np.percentile(rest, 99):.2f}ms | max {rest.max():.1f}ms "
          f"(budget {AGENT_MS_PER_STEP:.0f}ms)")
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"peak RAM {peak / (1e6 if sys.platform == 'darwin' else 1e3):.0f} MB (includes the game here)")
    except ImportError:
        pass
    if agent.device.type == "cuda":
        import torch

        print(f"peak VRAM {torch.cuda.max_memory_reserved() / 2**20:.0f} MB (limit {MAX_VRAM_MB} MB)")
    ok = rest.mean() < 15 and np.percentile(rest, 99) < 100 and agent.errors == 0
    print("PASS" if ok else "CHECK: above target or the agent raised errors")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
