"""Desk-review demo assets, made from real episodes of the real submission agent.

    uv run python tools/demo.py all            # everything, picking checkpoints from runs/
    uv run python tools/demo.py replay   --late runs/acer/full/best.pt
    uv run python tools/demo.py killcam  --late ...
    uv run python tools/demo.py deaths   --early A.pt --mid B.pt --late C.pt
    uv run python tools/demo.py curves   --runs runs/acer/full runs/asus/full_s2

Output goes to runs/demo/ (open index.html; works offline). Maps are held-out maps the agent never
trained on. True positions, walls and enemies are used only for drawing: the agent sees nothing but
the official observation. Which episodes were selected, and from how many, is printed on each page.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from doom_env.spaces import ENTITY_TYPES  # noqa: E402
from evaluation.run_heldout import load_agent  # noqa: E402
from evaluation.suite import build_suite  # noqa: E402
from models import obs as O  # noqa: E402
from models import policy as P  # noqa: E402
from models.actor import BBBActor  # noqa: E402
from models.config import ModelConfig  # noqa: E402
from training.env import TrainEnv  # noqa: E402
from training.export import export  # noqa: E402
from training.rewards import RewardConfig, RewardTracker  # noqa: E402

TEMPLATES = Path(__file__).resolve().parent / "demo_templates"
FULL_SUITE = [6, 1, True]  # the held-out suite used during training: maps per tier, episodes, Freedoom
NEAR_DOOR = 384.0  # map units: a memory slot written this close to a locked door counts as "at the door"
KEY_COLOURS = {"BlueCard": "blue", "BlueSkull": "blue", "RedCard": "red", "RedSkull": "red",
               "YellowCard": "yellow", "YellowSkull": "yellow"}


# --------------------------------------------------------------------------- checkpoints
def checkpoint_info(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)  # our own files
    steps = data.get("counters", {}).get("steps", data.get("source", {}).get("steps", 0))
    return {"path": str(path), "steps": int(steps), "run": path.parent.name}


def label(info: dict) -> str:
    trained = "untrained (warm-up only)" if info["steps"] == 0 else f"{info['steps'] / 1e6:.1f}M decisions"
    return f"{info['run']}, {trained}"


def loadable(path: Path) -> bool:
    """False for checkpoints from an older version of the network."""
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        BBBActor(ModelConfig.from_dict(data["model_config"])).load_state_dict(data["actor"])
        return True
    except (RuntimeError, KeyError, ValueError):
        return False


def discover(runs_root: Path) -> dict[str, Path]:
    """late = best held-out score; early = least trained; mid = about a third of the way to late."""
    scored = []
    for best in runs_root.rglob("best.json"):
        record = json.loads(best.read_text())
        if (best.parent / "best.pt").exists() and record.get("suite") == FULL_SUITE and loadable(best.parent / "best.pt"):
            scored.append((record["score"], best.parent / "best.pt"))
    if not scored:
        raise SystemExit(f"no best.pt under {runs_root}: run tools/remote.sh fetch <host> first, or pass --late")
    late = max(scored)[1]
    others = [checkpoint_info(p) for p in runs_root.rglob("ckpt_*.pt") if p.name != "ckpt_eval.pt" and loadable(p)]
    if not others:
        return {"early": late, "mid": late, "late": late}
    early = min(others, key=lambda c: c["steps"])
    goal = checkpoint_info(late)["steps"] / 3
    same_run = [c for c in others if c["run"] == late.parent.name and 0 < c["steps"]]
    pool = same_run if any(c["steps"] < 2 * goal for c in same_run) else [c for c in others if c["steps"] > 0]
    pool += [checkpoint_info(p) for _, p in scored if p != late]
    mid = min(pool, key=lambda c: abs(c["steps"] - goal)) if pool else early
    return {"early": Path(early["path"]), "mid": Path(mid["path"]), "late": late}


class Checkpoint:
    def __init__(self, path: Path, tmp: Path, mode: str):
        self.info = checkpoint_info(path)
        self.label = label(self.info)
        weights = export(path, tmp / f"{abs(hash(str(path)))}.pt", mode=mode)
        self.agent = load_agent(weights, "cpu")


# --------------------------------------------------------------------------- recording
def geometry(env: TrainEnv) -> dict:
    lines = env.meta.info.lines
    door_index = {int(s): i for i, s in enumerate(env.meta.door_ids)}
    out = []
    for p1, p2, front, back in zip(lines.p1, lines.p2, lines.front, lines.back):
        door = door_index.get(int(front), door_index.get(int(back), -1))
        if door >= 0:
            kind = "locked" if env.meta.door_locks[door] > 0 else "door"
        else:
            kind = "wall" if back < 0 else "edge"
        out.append([round(float(p1[0])), round(float(p1[1])), round(float(p2[0])), round(float(p2[1])), kind, door])
    keys = [[round(o.position_x), round(o.position_y), KEY_COLOURS[o.name]] for o in env._objects if o.name in KEY_COLOURS]
    xs = [v for l in out for v in (l[0], l[2])]
    ys = [v for l in out for v in (l[1], l[3])]
    return {"lines": out, "keys": keys, "exits": [[round(float(x)), round(float(y))] for x, y in env.meta.exits],
            "bounds": [min(xs), min(ys), max(xs), max(ys)]}


class Episode:
    """Steps the agent through one map and notes everything worth drawing."""

    def __init__(self, checkpoint: Checkpoint, env: TrainEnv, entry: dict, seed: int):
        self.agent, self.env, self.entry, self.seed = checkpoint.agent, env, entry, seed

    def run(self, on_step=None, tail: int = 0) -> dict:
        agent, env, entry = self.agent, self.env, self.entry
        env.load(entry["wad"], entry["map"], entry["ticks"], skill=3)
        obs, _ = env.reset(seed=1000 + self.seed)
        ws = env.world_state()
        totals = {"monsters": len(ws["monsters"]), "keys": len(ws["key_positions"]),
                  "locks": int((env.meta.door_locks > 0).sum())}
        tracker = RewardTracker(RewardConfig(), ws, env.meta, totals)
        geo = geometry(env)
        agent.reset()
        agent.rng.manual_seed(self.seed)  # a different, repeatable sample of the policy per attempt
        start = (ws["x"], ws["y"], math.radians(ws["angle"]))
        recent: deque = deque(maxlen=tail)
        events, step, done = [], 0, False
        while not done:
            before = (obs, agent.prev_action.copy(), agent.state, ws)
            action = agent.act(obs)
            counts = dict(tracker.counts)
            keys = ws["keys"].copy()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ws = env.world_state()
            tracker.step(ws)
            if ((ws["keys"] > 0.5) & (keys < 0.5)).any():
                events.append({"t": step, "type": "key", "text": "picked up a key"})
            if tracker.counts["unlocks"] > counts["unlocks"]:
                events.append({"t": step, "type": "unlock", "text": "unlocked a locked door"})
            if tracker.counts["doors"] > counts["doors"]:
                events.append({"t": step, "type": "door", "text": "opened a door"})
            if ws["dead"]:
                events.append({"t": step, "type": "death", "text": "died"})
            if ws["exit_reached"]:
                events.append({"t": step, "type": "exit", "text": "reached the exit"})
            if tail:
                recent.append((*before, action.copy()))
            if on_step:
                on_step(agent, before[3], action, step, start)
            step += 1
        summary = tracker.summary(ws, entry["family"])
        return {"geo": geo, "events": events, "steps": step, "start": start, "recent": list(recent),
                "dead": bool(ws["dead"]), "exit": bool(ws["exit_reached"]), "summary": summary}


def to_world(estimate: np.ndarray, start) -> np.ndarray:
    """The agent's frame (spawn point, spawn heading, units of 1024) -> map coordinates."""
    p = np.atleast_2d(estimate) * O.RAY_RANGE
    c, s = math.cos(start[2]), math.sin(start[2])
    return np.stack([start[0] + p[:, 0] * c - p[:, 1] * s, start[1] + p[:, 0] * s + p[:, 1] * c], axis=1)


def record_replay(checkpoint: Checkpoint, env: TrainEnv, entry: dict, seed: int) -> dict:
    frames, anchors, previous = [], {}, {}
    env_locked = None

    def on_step(agent, ws, action, step, start):
        nonlocal env_locked
        if env_locked is None:
            env_locked = env.meta.door_centers[env.meta.door_locks > 0]
        pose = agent.last_pose[0].numpy()
        ex, ey = to_world(pose[None, :2], start)[0]
        slots, near_attention, near_count, count = [], 0.0, 0, 0
        if "mem_pos" in agent.state:
            pos = agent.state["mem_pos"][0].numpy()
            valid = agent.state["mem_valid"][0].numpy()
            att = agent.last_info["attention"][0].numpy()
            rel = agent.last_info["relevance"][0].numpy()
            world = to_world(pos, start)
            for k in np.flatnonzero(valid > 0.05):
                k = int(k)
                if k not in previous or not np.allclose(previous[k], pos[k]):  # (re)written at a new place
                    anchors[k] = (ws["x"], ws["y"])
                    previous[k] = pos[k].copy()
                tx, ty = anchors[k]
                slots.append([k, round(float(world[k, 0])), round(float(world[k, 1])), round(tx), round(ty),
                              round(float(att[k]), 3), round(float(rel[k]), 2)])
                count += 1
                if len(env_locked) and np.hypot(*(env_locked - (tx, ty)).T).min() < NEAR_DOOR:
                    near_attention += float(att[k])
                    near_count += 1
        frames.append({
            "p": [round(ws["x"]), round(ws["y"]), round(ws["angle"])], "e": [round(float(ex)), round(float(ey))],
            "h": round(ws["health"]), "k": int(ws["keys"].sum()), "b": "".join(str(int(v)) for v in action),
            "o": np.flatnonzero(ws["door_open"]).tolist(), "m": [[round(float(x)), round(float(y))] for x, y in ws["monsters"]],
            "s": slots, "na": round(near_attention, 3), "ns": round(near_count / max(count, 1), 3),
        })

    result = Episode(checkpoint, env, entry, seed).run(on_step=on_step)
    result.update(frames=frames)
    return result


def interest(result: dict) -> float:
    kinds = [e["type"] for e in result["events"]]
    return 6 * ("key" in kinds) + 8 * ("unlock" in kinds) + 10 * result["exit"] + result["summary"]["sector_frac"]


def attention_report(result: dict) -> dict:
    """Does attention favour memory slots written near locked doors? Compare with the share of slots there."""
    frames = result["frames"]
    key_at = next((e["t"] for e in result["events"] if e["type"] == "key"), None)
    report = {"key_step": key_at}
    for name, part in (("before", frames[:key_at] if key_at is not None else frames),
                       ("after", frames[key_at:] if key_at is not None else [])):
        part = [f for f in part if f["ns"] > 0]
        if part:
            report[name] = {"attention": round(float(np.mean([f["na"] for f in part])), 3),
                            "slots": round(float(np.mean([f["ns"] for f in part])), 3), "frames": len(part)}
    return report


def attention_study(late: Checkpoint, suite: list[dict], env: TrainEnv, episodes: int) -> dict:
    """The single replay is one anecdote. Measure over many episodes with a key pickup: is the attention
    on locked-door memories higher than their share of slots?"""
    candidates = [e for e in suite if e["tier"] in (3, 4)]
    rows, tried = [], 0
    while tried < episodes:
        entry = candidates[tried % len(candidates)]
        report = attention_report(record_replay(late, env, entry, seed=100 + tried // len(candidates)))
        tried += 1
        if "before" in report and "after" in report:
            rows.append(report)
    study = {"tried": tried, "used": len(rows)}
    for part in ("before", "after"):
        if len(rows) > 1:
            attention = np.array([r[part]["attention"] for r in rows])
            slots = np.array([r[part]["slots"] for r in rows])
            difference = attention - slots
            study[part] = {"attention": round(float(attention.mean()), 3), "slots": round(float(slots.mean()), 3),
                           "difference": round(float(difference.mean()), 3),
                           "margin": round(float(1.96 * difference.std(ddof=1) / math.sqrt(len(rows))), 3)}
    return study


def entity_influence(checkpoint: Checkpoint, state, compact: dict) -> tuple[list[float], list[float]]:
    """How much would the action probabilities change if one visible thing were removed? (KL divergence)
    Returns (influence per entity slot, action probabilities [attack, use, forward])."""
    model = checkpoint.agent.model
    present = np.flatnonzero(compact["ent_mask"] > 0.5)
    batch = len(present) + 1
    obs = {k: torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(0).repeat(batch, *([1] * v.ndim)) for k, v in compact.items()}
    for row, slot in enumerate(present, start=1):
        obs["ent_mask"][row, slot] = 0.0
    with torch.inference_mode():
        base = model.initial_state(1) if state is None else state
        states = {k: v.repeat(batch, *([1] * (v.dim() - 1))) for k, v in base.items()}
        logits = model.step(obs, states).logits
        parts = [torch.log_softmax(x, dim=-1) for x in logits.split(P.HEAD_SIZES, dim=-1)]
        change = sum((lp[:1].exp() * (lp[:1] - lp)).sum(-1) for lp in parts)
        probabilities = [float(parts[3][0, 1].exp()), float(parts[4][0, 1].exp()), float(parts[0][0, 1].exp())]
    influence = [0.0] * O.MAX_ENTITIES
    for row, slot in enumerate(present, start=1):
        influence[slot] = max(0.0, float(change[row]))
    return influence, probabilities


# --------------------------------------------------------------------------- pages
def write_page(out: Path, template: str, data: dict) -> Path:
    page = (TEMPLATES / template).read_text()
    page = page.replace("/*COMMON*/", (TEMPLATES / "common.js").read_text())
    page = page.replace("/*STYLE*/", (TEMPLATES / "style.css").read_text())
    page = page.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    path = out / template
    path.write_text(page)
    print(f"  wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def make_replay(late: Checkpoint, suite: list[dict], env: TrainEnv, out: Path, attempts: int, study_episodes: int,
                tiers=(3, 4)) -> None:
    candidates = [e for t in tiers for e in suite if e["tier"] == t]
    best = None
    for i in range(attempts):
        entry = candidates[i % len(candidates)]
        result = record_replay(late, env, entry, seed=i // len(candidates))
        result.update(entry=entry, attempt=i + 1)
        print(f"  attempt {i + 1:2d} tier {entry['tier']} {entry['map']}: " +
              (", ".join(sorted({e['type'] for e in result['events']})) or "nothing"), flush=True)
        if best is None or interest(result) > interest(best):
            best = result
        if interest(result) >= 24:  # key, unlock and exit: nothing better to find
            break
    entry = best["entry"]
    report = attention_report(best)
    print(f"  chosen: attempt {best['attempt']}; attention near locked doors {report}")
    study = attention_study(late, suite, env, study_episodes)
    print(f"  across episodes: {study}")
    write_page(out, "replay.html", {
        "checkpoint": late.label,
        "map": f"held-out tier {entry['tier']} {entry['family']} map {entry['map']}, difficulty {entry['level']}",
        "selection": f"Episode chosen as the most eventful of those tried (number tried: {i + 1}).",
        "geo": best["geo"], "frames": best["frames"], "events": best["events"], "report": report, "study": study,
        "radius": NEAR_DOOR, "result": {"exit": best["exit"], "dead": best["dead"], "steps": best["steps"]},
    })


def make_killcam(late: Checkpoint, suite: list[dict], env: TrainEnv, out: Path, attempts: int, seconds: float) -> None:
    candidates = [e for t in (4, 1, 3) for e in suite if e["tier"] == t]
    tail = max(10, round(seconds * 35 / 4))
    chosen, deaths = None, 0
    for i in range(attempts):
        entry = candidates[i % len(candidates)]
        path = []
        result = Episode(late, env, entry, seed=i // len(candidates)).run(
            on_step=lambda a, ws, act, t, s: path.append([round(ws["x"]), round(ws["y"])]), tail=tail)
        if not result["dead"]:
            continue
        deaths += 1
        seen = np.mean([float(r[0]["visible_entities"]["mask"].sum() > 0) for r in result["recent"]])
        if chosen is None or seen > chosen[0]:
            chosen = (seen, result, entry, path, i + 1)
        if seen > 0.8 or deaths >= 5:
            break
    if chosen is None:
        print(f"  no death in {attempts} attempts: killcam skipped")
        return
    _, result, entry, path, attempt = chosen
    frames = []
    for obs, previous, state, ws, action in result["recent"]:
        compact = O.preprocess(obs, previous, late.agent.model.cfg.use_depth)
        influence, probabilities = entity_influence(late, state, compact)
        things = [[ENTITY_TYPES[int(compact["ent_type"][k])], round(float(compact["ent_feat"][k, 0]) * 1024),
                   round(float(compact["ent_feat"][k, 1]), 3), round(float(compact["ent_feat"][k, 2]), 3),
                   round(influence[k], 4)] for k in np.flatnonzero(compact["ent_mask"] > 0.5)]
        frames.append({
            "p": [round(ws["x"]), round(ws["y"]), round(ws["angle"])], "h": round(ws["health"]),
            "rd": [round(float(v), 3) for v in compact["ray_dist"]], "rt": compact["ray_type"].tolist(),
            "en": things, "b": "".join(str(int(v)) for v in action), "pr": [round(v, 2) for v in probabilities],
            "o": np.flatnonzero(ws["door_open"]).tolist(),
        })
    write_page(out, "killcam.html", {
        "checkpoint": late.label,
        "map": f"held-out tier {entry['tier']} {entry['family']} map {entry['map']}, difficulty {entry['level']}",
        "selection": f"A death picked for having enemies in view; found by attempt {attempt}. Last {len(frames)} decisions.",
        "geo": result["geo"], "path": path, "frames": frames,
    })


def make_deaths(stages: list[tuple[str, Checkpoint]], suite: list[dict], env: TrainEnv, out: Path,
                episodes: int, which: tuple[int, int]) -> None:
    entry = [e for e in suite if e["tier"] == which[0]][which[1]]
    panels, geo = [], None
    for stage, checkpoint in stages:
        runs = []
        for i in range(episodes):
            path = []
            result = Episode(checkpoint, env, entry, seed=i).run(
                on_step=lambda a, ws, act, t, s: path.append([round(ws["x"]), round(ws["y"])]) if t % 2 == 0 else None)
            geo = geo or result["geo"]
            end = env.world_state()
            path.append([round(end["x"]), round(end["y"])])
            runs.append({"path": path, "end": "death" if result["dead"] else "exit" if result["exit"] else "time",
                         "sectors": round(result["summary"]["sector_frac"], 3),
                         "kills": int(result["summary"]["kills"])})
        tally = {k: sum(r["end"] == k for r in runs) for k in ("death", "exit", "time")}
        explored = float(np.mean([r["sectors"] for r in runs]))
        print(f"  {stage:6s} {checkpoint.label}: {tally}, explored {explored:.0%}, kills {np.mean([r['kills'] for r in runs]):.1f}")
        panels.append({"stage": stage, "label": checkpoint.label, "runs": runs, "tally": tally,
                       "explored": round(explored, 3), "kills": round(float(np.mean([r["kills"] for r in runs])), 1)})
    data = {"map": f"held-out tier {entry['tier']} {entry['family']} map {entry['map']}, difficulty {entry['level']}",
            "episodes": episodes, "geo": geo, "panels": panels}
    write_page(out, "deaths.html", data)
    draw_deaths_png(data, out / "deaths.png")


def draw_deaths_png(data: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(data["panels"]), figsize=(5.4 * len(data["panels"]), 5.8), squeeze=False)
    for ax, panel in zip(axes[0], data["panels"]):
        for x1, y1, x2, y2, kind, _ in data["geo"]["lines"]:
            if kind != "edge":
                ax.plot([x1, x2], [y1, y2], color={"wall": "#191919", "door": "#f47230", "locked": "#ec4647"}[kind], lw=1)
        for run in panel["runs"]:
            xs, ys = zip(*run["path"])
            ax.plot(xs, ys, color="#257de0", alpha=0.12, lw=1)
        for end, marker, colour in (("death", "X", "#ec4647"), ("exit", "*", "#008a17"), ("time", "o", "#666666")):
            points = [r["path"][-1] for r in panel["runs"] if r["end"] == end]
            if points:
                ax.scatter(*zip(*points), marker=marker, c=colour, s=70 if end != "time" else 18, zorder=3,
                           edgecolors="white", linewidths=0.6, label={"death": "died", "exit": "reached exit", "time": "time ran out"}[end])
        t = panel["tally"]
        ax.set_title(f"{panel['stage']}: {panel['label']}\ndied {t['death']}, exit {t['exit']}, timed out {t['time']} · "
                     f"explored {panel['explored']:.0%}", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(f"{data['episodes']} attempts per checkpoint on one unseen map ({data['map']})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"  wrote {path.name}")


def make_curves(run_dirs: list[Path], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for d in run_dirs:
        file = d / "eval.jsonl"
        if file.exists():
            records = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
            records = [r for r in records if r.get("suite") == FULL_SUITE]  # only comparable tests
            records = sorted({r["step"]: r for r in records}.values(), key=lambda r: r["step"])
            if records:
                runs.append((d.name, records))
    if not runs:
        print("  no eval.jsonl found: curves skipped")
        return
    panels = [("Weighted score (all tiers)", lambda r: r["score"])]
    names = {"1": "Tier 1: combat", "2": "Tier 2: exploration", "3": "Tier 3: keys and locks", "4": "Tier 4: gauntlet"}
    panels += [(names[t], lambda r, t=t: r["tiers"][t]["score"]) for t in names]
    panels.append(("Exit rate (mean of tiers)", lambda r: float(np.mean([m["exit"] for m in r["tiers"].values()]))))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True)
    colours = ("#257de0", "#f47230", "#00b894")
    for ax, (title, value) in zip(axes.flat, panels):
        for (name, records), colour, marker in zip(runs, colours, ("o", "s", "^")):
            ax.plot([r["step"] / 1e6 for r in records], [value(r) for r in records], color=colour, lw=2,
                    marker=marker, ms=6, markeredgecolor="white", label=name)
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(color="#e6e6e4", lw=0.8)
        ax.set_ylim(bottom=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("training decisions (millions)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Held-out maps over training (our estimate of the scoring; one episode per map, so points are noisy)", fontsize=12)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out / f"learning_curves.{suffix}", dpi=170)
    plt.close(fig)
    print("  wrote learning_curves.png and .svg:", ", ".join(f"{n} ({len(r)} tests)" for n, r in runs))


def write_index(out: Path, notes: dict) -> None:
    items = [("replay.html", "Replay: path, memory slots and attention"), ("killcam.html", "Killcam: the last seconds before a death"),
             ("deaths.html", "All deaths: early, middle and late checkpoints"), ("deaths.png", "All deaths (still image)"),
             ("learning_curves.png", "Learning curves by tier")]
    links = "".join(f'<li><a href="{f}">{t}</a></li>' for f, t in items if (out / f).exists())
    image = '<img src="learning_curves.png" alt="Score per tier over training">' if (out / "learning_curves.png").exists() else ""
    used = "".join(f"<li>{k}: {v}</li>" for k, v in notes.items())
    style = (TEMPLATES / "style.css").read_text()
    (out / "index.html").write_text(
        f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>BBB demo</title><style>{style} img{{max-width:100%;background:#fff;border-radius:10px}}</style>"
        f"<main><h1>BBB: demo</h1><ul>{links}</ul><div class='card'><b>Checkpoints used</b><ul>{used}</ul></div>{image}</main>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("all", "replay", "killcam", "deaths", "curves"))
    ap.add_argument("--early")
    ap.add_argument("--mid")
    ap.add_argument("--late")
    ap.add_argument("--runs", nargs="*", help="run folders for the curves (default: every folder in runs/ with eval.jsonl)")
    ap.add_argument("--out", default=str(ROOT / "runs" / "demo"))
    ap.add_argument("--mode", choices=("sample", "greedy"), default="sample")
    ap.add_argument("--attempts", type=int, default=36, help="episodes to search for the replay and killcam")
    ap.add_argument("--episodes", type=int, default=30, help="attempts per checkpoint in the death overlay")
    ap.add_argument("--death-map", default="4:1", help="held-out tier:index for the death overlay")
    ap.add_argument("--killcam-seconds", type=float, default=5.0)
    ap.add_argument("--study-episodes", type=int, default=60, help="episodes for the memory-attention measurement")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.what in ("all", "curves"):
        dirs = [Path(r) for r in args.runs] if args.runs else sorted(p.parent for p in (ROOT / "runs").rglob("eval.jsonl"))
        print("learning curves")
        make_curves(dirs, out)
    notes = {}
    if args.what != "curves":
        given = {k: getattr(args, k) for k in ("early", "mid", "late")}
        found = {} if all(given.values()) or (args.what in ("replay", "killcam") and given["late"]) else discover(ROOT / "runs")
        paths = {k: Path(v) if v else found.get(k) for k, v in given.items()}
        tmp = Path(tempfile.mkdtemp(prefix="bbb-demo-"))
        try:
            suite = build_suite(str(tmp), maps_per_tier=6, freedoom=False)
            late = Checkpoint(paths["late"], tmp, args.mode)
            notes["late"] = late.label
            env = TrainEnv(suite[0]["wad"], suite[0]["map"], render_buffers=late.agent.model.cfg.use_depth)
            if args.what in ("all", "replay"):
                print(f"replay with {late.label}")
                make_replay(late, suite, env, out, args.attempts, args.study_episodes)
            if args.what in ("all", "killcam"):
                print(f"killcam with {late.label}")
                make_killcam(late, suite, env, out, args.attempts, args.killcam_seconds)
            if args.what in ("all", "deaths"):
                print("death overlay")
                stages = [(k, Checkpoint(paths[k], tmp, args.mode)) for k in ("early", "mid")] + [("late", late)]
                notes.update({k: c.label for k, c in stages})
                tier, index = (int(v) for v in args.death_map.split(":"))
                make_deaths(stages, suite, env, out, args.episodes, (tier, index))
            env.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if args.what != "all" and (out / "index.html").exists():
        return 0  # keep the full index when rebuilding a single page
    write_index(out, notes)
    print(f"open {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
