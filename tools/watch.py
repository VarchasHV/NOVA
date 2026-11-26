"""Watch the agent play in a game window, using the newest checkpoint from a training laptop.

    uv run python tools/watch.py acer                  # newest Acer checkpoint, one map after another
    uv run python tools/watch.py asus --best           # the best-scoring ASUS checkpoint
    uv run python tools/watch.py acer --tier 1         # only combat maps (2 explore, 3 keys, 4 gauntlet)
    uv run python tools/watch.py acer --mode greedy    # no randomness in the choice of actions
    uv run python tools/watch.py --ckpt path/to.pt     # a checkpoint already on this computer

Run it on a computer with a screen, not on a training laptop. Training is not touched: this copies
the checkpoint (about 15 MB, only when it has changed) and plays it here. Maps are held-out maps.
Before every map the newest checkpoint is fetched again. Ctrl+C quits.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import vizdoom as vd  # noqa: E402

from evaluation.run_heldout import load_agent  # noqa: E402
from evaluation.suite import TIERS, build_suite  # noqa: E402
from training.env import TrainEnv  # noqa: E402
from training.export import export  # noqa: E402

TICS_PER_SECOND = 35


class RealTimeGame:
    """The game, except that each action is shown tic by tic at playing speed instead of skipped through."""

    def __init__(self, game, speed: float):
        self._real = game
        self._seconds_per_tic = 1.0 / (TICS_PER_SECOND * speed)
        self._next = 0.0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def make_action(self, buttons, tics: int = 1) -> None:
        self._real.set_action(buttons)
        for _ in range(tics):
            time.sleep(max(0.0, self._next - time.perf_counter()))
            self._next = max(self._next, time.perf_counter()) + self._seconds_per_tic
            self._real.advance_action(1, True)  # update every tic, so every frame is drawn
            if self._real.is_episode_finished():
                break


class WindowEnv(TrainEnv):
    """TrainEnv with a visible window. Observations are the same as in evaluation."""

    def __init__(self, wad: str, map_name: str, speed: float):
        super().__init__(wad, map_name, render_buffers=False)  # the agent does not use the pixels
        game = self._game
        game.close()
        game.set_window_visible(True)
        game.set_screen_resolution(vd.ScreenResolution.RES_800X600)
        game.set_render_hud(True)
        game.init()
        self._game = RealTimeGame(game, speed)


def newest_on_laptop(host: str, name: str, destination: Path) -> Path | None:
    """Copy the most recently written runs/*/<name> from the laptop; None if it cannot be reached."""
    ssh = ["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", host]
    try:
        listing = subprocess.run([*ssh, f"ls -t nova-train/runs/*/{name}"], capture_output=True, text=True, timeout=20)
        if listing.returncode != 0 or not listing.stdout.strip():
            print(f"could not list checkpoints on {host}: {listing.stderr.strip()}")
            return None
        remote = listing.stdout.splitlines()[0].strip()
        # rsync -t skips the copy when the file has not changed since the last one.
        subprocess.run(["rsync", "-tz", "-e", "ssh -o ConnectTimeout=8 -o BatchMode=yes", f"{host}:{remote}",
                        str(destination)], check=True, timeout=120)
    except (subprocess.SubprocessError, OSError) as error:
        print(f"could not copy from {host}: {error}")
        return None
    return destination


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("host", nargs="?", help="laptop alias from ~/.ssh/config, for example acer or asus")
    ap.add_argument("--ckpt", help="play this local checkpoint instead of fetching one")
    ap.add_argument("--best", action="store_true", help="fetch best.pt instead of the newest checkpoint")
    ap.add_argument("--tier", type=int, choices=sorted(TIERS), help="only maps of this tier")
    ap.add_argument("--speed", type=float, default=1.0, help="1 is real time, 2 is twice as fast")
    ap.add_argument("--mode", choices=("sample", "greedy"), default="sample",
                    help="greedy always takes the most likely action, which looks steadier")
    ap.add_argument("--variety", action="store_true", help="show maps with uneven shapes, steps and ledges")
    ap.add_argument("--episodes", type=int, default=0, help="stop after this many maps (default: never)")
    args = ap.parse_args(argv)
    if bool(args.host) == bool(args.ckpt):
        ap.error("give either a laptop name or --ckpt")
    if args.speed <= 0:
        ap.error("--speed must be positive")

    env = None
    with tempfile.TemporaryDirectory(prefix="streamed-mind-watch-") as tmp:
        suite = build_suite(tmp, variety=1.0 if args.variety else 0.0)
        if args.tier:
            suite = [e for e in suite if e["tier"] == args.tier]
        random.shuffle(suite)
        local = Path(tmp) / "ckpt.pt"
        agent, loaded, played = None, None, 0
        try:
            while not args.episodes or played < args.episodes:
                if args.ckpt:
                    source = Path(args.ckpt)
                else:
                    source = newest_on_laptop(args.host, "best.pt" if args.best else "ckpt_latest.pt", local)
                    source = source or (local if local.exists() else None)
                if source is None:
                    return 1
                stamp = source.stat().st_mtime
                if stamp != loaded:
                    weights = export(source, Path(tmp) / "model.pt", mode=args.mode)
                    agent, loaded = load_agent(weights, "cpu"), stamp
                    if agent.model.cfg.use_depth:
                        raise SystemExit("this checkpoint uses depth images, which the viewer does not provide")
                    saved = time.strftime("%H:%M", time.localtime(stamp))
                    steps = torch.load(weights, weights_only=True)["source"]["steps"]
                    print(f"\nagent saved at {saved}, trained for {steps / 1e6:.1f}M decisions")

                entry = suite[played % len(suite)]
                if env is None:
                    env = WindowEnv(entry["wad"], entry["map"], args.speed)
                env.load(entry["wad"], entry["map"], entry["ticks"], skill=3)
                seed = random.randrange(1_000_000)
                obs, _ = env.reset(seed=seed)
                agent.reset()
                agent.rng.manual_seed(seed)
                print(f"map {played + 1}: {TIERS[entry['tier']]} {entry['map']} ... ", end="", flush=True)
                done = False
                while not done:
                    obs, _, terminated, truncated, _ = env.step(agent.act(obs))
                    done = terminated or truncated
                ws = env.world_state()
                outcome = "reached the exit" if ws["exit_reached"] else "died" if ws["dead"] else "time ran out"
                print(f"{outcome} (kills {int(ws['kills'])}, keys {int(ws['keys'].sum())})")
                played += 1
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nstopped")
        except vd.ViZDoomUnexpectedExitException:
            print("\nthe game window was closed")
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:  # noqa: BLE001 - the engine may already be gone
                    pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
