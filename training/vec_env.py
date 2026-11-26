"""Many environments in parallel: worker processes with shared-memory observation buffers.

Each worker owns a few environments and a "pack" WAD of generated maps. The main process writes
actions into shared memory, tells workers to step, and reads observations back. Works with the
"spawn" start method, so Linux, Windows and macOS behave the same. No PyTorch in the workers.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import random
import shutil
import tempfile
import traceback
from dataclasses import dataclass, field

import numpy as np

from models import obs as O

from .env import FREEDOOM_WAD, TrainEnv
from .maps.generator import HELDOUT_SEED_START, NUM_LEVELS, write_pack
from .mirror import mirror_buttons, mirror_motion, mirror_obs
from .privileged import PRIV_DIM, privileged_vector
from .rewards import RewardConfig, RewardTracker

AUX_DIM = 4  # key ever seen, locked door ever seen, exit nearby, stagnating
FREEDOOM_HELDOUT = ("MAP01", "MAP11", "MAP20")
FREEDOOM_TRAIN = tuple(f"MAP{i:02d}" for i in range(1, 31) if f"MAP{i:02d}" not in FREEDOOM_HELDOUT)
STEP_TIMEOUT_S = 300.0


@dataclass
class EnvConfig:
    use_depth: bool = False
    variants_per_bucket: int = 6  # generated maps per (family, level) in a worker's pack
    rotate_episodes: int = 150  # worker episodes before its pack is regenerated
    mirror_prob: float = 0.5
    freedoom_slots: int = 1  # environments per worker that switch to Freedoom maps later in training
    reward: RewardConfig = field(default_factory=RewardConfig)


def buffer_spec(use_depth: bool) -> dict[str, tuple[tuple[int, ...], type]]:
    spec = {f"obs/{k}": v for k, v in O.obs_spec(use_depth).items()}
    spec.update({
        "action": ((O.NUM_BUTTONS,), np.float32),
        "rew_ext": ((), np.float32),
        "rew_int": ((), np.float32),
        "done": ((), np.float32),
        "reset": ((), np.float32),  # 1 = the current observation starts a new episode
        "priv": ((PRIV_DIM,), np.float32),
        "aux": ((AUX_DIM,), np.float32),
        "odom": ((3,), np.float32),  # true motion since the previous observation (RAY_RANGE units, radians)
    })
    return spec


class ScenarioPool:
    """A worker's generated maps, regenerated now and then so there is always fresh content."""

    def __init__(self, cfg: EnvConfig, rng: random.Random, directory: str):
        self.cfg, self.rng, self.dir = cfg, rng, directory
        self.entries: dict[tuple[str, int], list[dict]] = {}
        self.nemesis: list[dict] = []
        self.generation = 0
        self.episodes = 0

    def rebuild(self, directive: dict) -> None:
        buckets = sorted({f for f, _ in directive["weights"]})
        scenarios = [(f, l, self.rng.randrange(HELDOUT_SEED_START))
                     for f in buckets for l in range(NUM_LEVELS) for _ in range(self.cfg.variants_per_bucket)]
        replay = [tuple(s) for s in directive["nemesis"]]
        path = os.path.join(self.dir, f"pack{self.generation}.wad")
        entries = write_pack(path, scenarios + replay)
        self.entries, self.nemesis = {}, []
        for i, e in enumerate(entries):
            e["wad"] = path
            if i < len(scenarios):
                self.entries.setdefault((e["family"], e["level"]), []).append(e)
            else:
                self.nemesis.append(e)
        old = os.path.join(self.dir, f"pack{self.generation - 3}.wad")
        if os.path.exists(old):
            try:
                os.remove(old)
            except OSError:
                pass
        self.generation += 1
        self.episodes = 0
        self.families = set(buckets)

    def choose(self, directive: dict) -> dict:
        wanted = {f for f, _ in directive["weights"]}
        if not self.entries or self.episodes >= self.cfg.rotate_episodes or not wanted <= self.families:
            self.rebuild(directive)
        self.episodes += 1
        if self.nemesis and self.rng.random() < directive["nemesis_prob"]:
            options = [e for e in self.nemesis if (e["family"], e["level"]) in directive["weights"]]
            if options:
                return {**self.rng.choice(options), "nemesis": True}
        buckets = list(directive["weights"])
        bucket = self.rng.choices(buckets, weights=[directive["weights"][b] for b in buckets])[0]
        return self.rng.choice(self.entries[bucket])


def rules_for(entry: dict, final: bool, rng: random.Random) -> dict:
    """Training-time assists for easy levels; everything is vanilla skill 3 in the final stage."""
    level = entry["level"]
    if final:
        return {"skill": 3, "infinite_ammo": False}
    if entry["family"] == "freedoom":
        return {"skill": rng.choice((1, 2, 3, 3)), "infinite_ammo": False}
    skill = rng.choice(((1,), (1, 2), (2, 3), (2, 3, 3, 4), (3,))[level])
    return {"skill": skill, "infinite_ammo": level == 0 or (level == 1 and rng.random() < 0.5)}


class EnvSlot:
    def __init__(self, cfg: EnvConfig, pool: ScenarioPool, rng: random.Random, freedoom: bool):
        self.cfg, self.pool, self.rng, self.freedoom = cfg, pool, rng, freedoom
        self.env: TrainEnv | None = None

    def reset(self, directive: dict, out: dict, i: int) -> None:
        if self.freedoom and directive["freedoom"]:
            name = self.rng.choice(FREEDOOM_TRAIN)
            entry = {"family": "freedoom", "level": 4 if directive["final"] else 3, "seed": int(name[3:]),
                     "map": name, "wad": FREEDOOM_WAD, "ticks": self.rng.randint(2100, 4200)}
        else:
            entry = self.pool.choose(directive)
        self.entry = entry
        rules = rules_for(entry, directive["final"], self.rng)
        if self.env is None:
            self.env = TrainEnv(entry["wad"], entry["map"], episode_ticks=entry["ticks"], render_buffers=self.cfg.use_depth)
        self.env.load(entry["wad"], entry["map"], entry["ticks"], **rules)
        obs, _ = self.env.reset(seed=self.rng.randrange(2**31))
        ws = self.env.world_state()
        self.totals = {
            "monsters": len(ws["monsters"]), "keys": len(ws["key_positions"]),
            "locks": int((self.env.meta.door_locks > 0).sum()),
        }
        self.spawn = (ws["x"], ws["y"])
        self.tracker = RewardTracker(self.cfg.reward, ws, self.env.meta, self.totals)
        self.mirror = self.rng.random() < self.cfg.mirror_prob
        self.prev_action = np.zeros(O.NUM_BUTTONS, np.float32)
        self.seen_key = self.seen_lock = 0.0
        self.steps = 0
        self.vanilla = bool(directive["final"] or (not rules["infinite_ammo"] and rules["skill"] == 3))
        self.ws = ws
        out["reset"][i] = 1.0
        out["odom"][i] = 0.0
        self._write(obs, ws, out, i)

    def _write(self, obs: dict, ws: dict, out: dict, i: int) -> None:
        compact = O.preprocess(obs, np.zeros(O.NUM_BUTTONS), self.cfg.use_depth)
        if self.mirror:
            compact = mirror_obs(compact)
        compact["prev_action"] = self.prev_action  # already in the policy's (mirrored) frame
        for k, v in compact.items():
            out[f"obs/{k}"][i] = v
        self.seen_key = max(self.seen_key, float(np.any((compact["ent_type"] == 9) & (compact["ent_mask"] > 0.5))))
        self.seen_lock = max(self.seen_lock, float(np.any(compact["ray_type"] == 3)))
        exits = self.env.meta.exits
        near = float(len(exits) and np.min(np.hypot(*(exits - (ws["x"], ws["y"])).T)) < 512)
        out["aux"][i] = (self.seen_key, self.seen_lock, near, float(self.tracker.stagnating))
        out["priv"][i] = privileged_vector(ws, self.env.meta, self.totals, self.spawn,
                                           len(self.tracker.novelty.sectors), self.mirror)

    def step(self, directive: dict, out: dict, i: int) -> dict | None:
        action = out["action"][i].copy()
        executed = mirror_buttons(action) if self.mirror else action
        obs, _, terminated, truncated, _ = self.env.step(executed.astype(np.int8))
        ws = self.env.world_state()
        ext, intrinsic, _ = self.tracker.step(ws)
        done = terminated or truncated
        out["rew_ext"][i], out["rew_int"][i], out["done"][i] = ext, intrinsic, float(done)
        self.prev_action = action.astype(np.float32)
        self.steps += 1
        if done:
            episode = self.tracker.summary(ws, self.entry["family"])
            episode.update(family=self.entry["family"], level=self.entry["level"], seed=self.entry["seed"],
                           steps=self.steps, vanilla=float(self.vanilla), nemesis=float(self.entry.get("nemesis", False)))
            self.reset(directive, out, i)
            return episode
        # True motion since the last observation, expressed in the previous heading (odometry target).
        dx, dy = ws["x"] - self.ws["x"], ws["y"] - self.ws["y"]
        th = math.radians(self.ws["angle"])
        turn = (math.radians(ws["angle"] - self.ws["angle"]) + math.pi) % (2 * math.pi) - math.pi
        motion = np.array([(dx * math.cos(th) + dy * math.sin(th)) / O.RAY_RANGE,
                           (dx * math.sin(th) - dy * math.cos(th)) / O.RAY_RANGE, turn], np.float32)
        out["odom"][i] = mirror_motion(motion) if self.mirror else motion
        out["reset"][i] = 0.0
        self.ws = ws
        self._write(obs, ws, out, i)
        return None

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


class WorkerCore:
    """The environments of one worker. Also used directly (no processes) when num_workers is 0."""

    def __init__(self, cfg: EnvConfig, num_envs: int, seed: int):
        self.dir = tempfile.mkdtemp(prefix="streamed-mind-")
        rng = random.Random(seed)
        self.pool = ScenarioPool(cfg, rng, self.dir)
        self.slots = [EnvSlot(cfg, self.pool, random.Random(rng.randrange(2**62)), j < cfg.freedoom_slots)
                      for j in range(num_envs)]
        self.directive: dict | None = None

    def run(self, command: str, directive: dict | None, out: dict, offset: int) -> list[dict]:
        if directive is not None:
            self.directive = directive
        episodes = []
        for j, slot in enumerate(self.slots):
            i = offset + j
            if command == "reset":
                slot.reset(self.directive, out, i)
                out["rew_ext"][i] = out["rew_int"][i] = out["done"][i] = 0.0
                continue
            try:
                episode = slot.step(self.directive, out, i)
            except Exception:  # an engine failure costs one episode, not the training run
                traceback.print_exc()
                slot.close()
                out["rew_ext"][i] = out["rew_int"][i] = 0.0
                out["done"][i] = 1.0
                slot.reset(self.directive, out, i)
                episode = None
            if episode is not None:
                episodes.append(episode)
        return episodes

    def close(self) -> None:
        for slot in self.slots:
            slot.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def _views(raw: dict, spec: dict, n: int) -> dict[str, np.ndarray]:
    return {k: np.frombuffer(raw[k], dtype=dtype).reshape(n, *shape) for k, (shape, dtype) in spec.items()}


def _worker(pipe, raw: dict, cfg: EnvConfig, n: int, offset: int, num_envs: int, seed: int) -> None:
    core = None
    try:
        out = _views(raw, buffer_spec(cfg.use_depth), n)
        core = WorkerCore(cfg, num_envs, seed)
        while True:
            command, directive = pipe.recv()
            if command == "close":
                break
            pipe.send(("ok", core.run(command, directive, out, offset)))
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:  # noqa: BLE001 - report to the parent
        pipe.send(("error", traceback.format_exc()))
    finally:
        if core is not None:
            core.close()


class VecEnv:
    def __init__(self, cfg: EnvConfig, num_workers: int, envs_per_worker: int, seed: int = 0):
        self.num_envs = max(num_workers, 1) * envs_per_worker
        self.workers = num_workers
        spec = buffer_spec(cfg.use_depth)
        ctx = mp.get_context("spawn")
        sizes = {k: int(np.prod((self.num_envs, *shape))) * np.dtype(dtype).itemsize for k, (shape, dtype) in spec.items()}
        raw = {k: ctx.RawArray("b", size) for k, size in sizes.items()}
        self.buf = _views(raw, spec, self.num_envs)
        self.pipes, self.procs = [], []
        self.local = None
        if num_workers == 0:
            self.local = WorkerCore(cfg, envs_per_worker, seed)
            return
        for w in range(num_workers):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, daemon=True,
                               args=(child, raw, cfg, self.num_envs, w * envs_per_worker, envs_per_worker, seed * 1000 + w))
            proc.start()
            child.close()
            self.pipes.append(parent)
            self.procs.append(proc)

    def _run(self, command: str, directive: dict | None) -> list[dict]:
        if self.local is not None:
            return self.local.run(command, directive, self.buf, 0)
        for pipe in self.pipes:
            pipe.send((command, directive))
        episodes = []
        for w, pipe in enumerate(self.pipes):
            if not pipe.poll(STEP_TIMEOUT_S):
                raise RuntimeError(f"environment worker {w} did not answer within {STEP_TIMEOUT_S:.0f}s")
            try:
                status, payload = pipe.recv()
            except EOFError:
                raise RuntimeError(f"environment worker {w} died") from None
            if status == "error":
                raise RuntimeError(f"environment worker {w} failed:\n{payload}")
            episodes += payload
        return episodes

    def reset(self, directive: dict) -> None:
        self._run("reset", directive)

    def step(self, directive: dict | None = None) -> list[dict]:
        """Execute the actions in buf["action"]. Returns summaries of episodes that just ended."""
        return self._run("step", directive)

    def close(self) -> None:
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except (OSError, BrokenPipeError):
                pass
        for proc in self.procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
        if self.local is not None:
            self.local.close()
        self.pipes, self.procs, self.local = [], [], None
