"""Top-down picture of one episode: walls, the true path, the estimated path and the memory slots.

    python tools/viz_topdown.py --weights submission/weights/model.pt --tier 3 --out episode.png

Slot size shows the attention it received on the last step; colour shows learned relevance. Ground
truth (walls, true path) is used for drawing only; the agent never sees it.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights")
    ap.add_argument("--tier", type=int, default=3)
    ap.add_argument("--index", type=int, default=1, help="which held-out map of that tier")
    ap.add_argument("--out", default="episode.png")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from evaluation.run_heldout import load_agent, play
    from evaluation.suite import build_suite
    from training.env import TrainEnv

    agent = load_agent(args.weights, "cpu")
    tmp = tempfile.mkdtemp(prefix="bbb-viz-")
    try:
        entry = [e for e in build_suite(tmp, args.index + 1, freedoom=False) if e["tier"] == args.tier][args.index]
        env = TrainEnv(entry["wad"], entry["map"], render_buffers=agent.model.cfg.use_depth)
        true_path, estimated, start = [], [], []

        def record(agent, ws, step):
            true_path.append((ws["x"], ws["y"]))
            if agent.last_pose is not None:
                estimated.append(agent.last_pose[0].cpu().numpy().copy())

        env.load(entry["wad"], entry["map"], entry["ticks"])
        env.reset(seed=1000)
        s = env.world_state()
        start = (s["x"], s["y"], math.radians(s["angle"]))
        result = play(agent, env, entry, seed=1000, on_step=record)
        geometry = env.meta.info.lines

        def to_world(p):  # the agent's frame starts at the spawn point, facing the spawn direction
            p = np.atleast_2d(p) * 1024.0
            c, s = math.cos(start[2]), math.sin(start[2])
            return np.stack([start[0] + p[:, 0] * c - p[:, 1] * s, start[1] + p[:, 0] * s + p[:, 1] * c], axis=1)

        fig, ax = plt.subplots(figsize=(10, 10))
        doors = env.meta.info.door_sectors
        for p1, p2, front, back in zip(geometry.p1, geometry.p2, geometry.front, geometry.back):
            lock = max(doors.get(int(front), -1), doors.get(int(back), -1))
            colour = "tab:red" if lock > 0 else "tab:orange" if lock == 0 else "black" if back < 0 else "0.85"
            ax.plot(*zip(p1, p2), color=colour, lw=2 if lock >= 0 else 0.8)
        ax.plot(*zip(*true_path), color="tab:blue", lw=1.2, label="true path")
        if estimated:
            ax.plot(*to_world(np.array(estimated)[:, :2]).T, color="tab:green", lw=1, ls="--", label="estimated path")
        if len(env.meta.exits):
            ax.scatter(*env.meta.exits.T, marker="*", s=200, color="tab:purple", label="exit")
        if agent.state is not None and "mem_pos" in agent.state:
            valid = agent.state["mem_valid"][0].cpu().numpy() > 0.05
            pos = to_world(agent.state["mem_pos"][0].cpu().numpy()[valid])
            rel = agent.last_info["relevance"][0].cpu().numpy()[valid]
            att = agent.last_info["attention"][0].cpu().numpy()[valid]
            dots = ax.scatter(*pos.T, c=rel, s=40 + 900 * att, cmap="viridis", edgecolors="black", vmin=0, vmax=1, label="memory slots", zorder=3)
            fig.colorbar(dots, ax=ax, label="learned relevance", shrink=0.7)
        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        ax.set_title(f"tier {args.tier} {entry['family']} L{entry['level']}: score {result['score']:.0f}, "
                     f"exit {result['exit']:.0f}, keys {result['keys']:.0f}")
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"wrote {args.out}")
        env.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
