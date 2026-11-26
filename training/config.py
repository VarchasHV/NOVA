"""Every experiment is fully described by one YAML file loaded into these dataclasses."""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from models.config import ModelConfig

from .director import DirectorConfig
from .vec_env import EnvConfig


@dataclass
class PPOConfig:
    rollout: int = 128  # decisions per environment per update
    seq_len: int = 32  # truncated BPTT length
    minibatches: int = 8
    epochs: int = 3
    lr: float = 3e-4
    lr_end_fraction: float = 0.15
    gamma_ext: float = 0.995
    gamma_int: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    value_clip: float = 0.3
    vf_coef: float = 0.5
    ent_start: float = 0.01
    ent_end: float = 0.002
    grad_clip: float = 0.5
    target_kl: float = 0.03
    aux_coef: float = 0.1
    int_coef: float = 0.3  # weight of the intrinsic advantage at the start
    int_anneal_end: float = 0.7  # progress at which it reaches zero
    odom_lr: float = 1e-3
    odom_batch: int = 2048
    odom_updates: int = 8  # odometry gradient steps per update
    critic_detach: bool = False  # True: the critic's gradient does not shape actor features


@dataclass
class TrainConfig:
    name: str = "run"
    hypothesis: str = ""
    seed: int = 1
    total_steps: int = 200_000_000  # training ends at this many decisions ...
    max_hours: float = 16.0  # ... or after this many hours, whichever comes first
    num_workers: int = -1  # environment processes; -1 = CPU cores - 2; 0 = run inside this process
    envs_per_worker: int = 8
    odom_warmup_updates: int = 60  # random-walk rollouts that only train odometry
    log_every: int = 1
    save_minutes: float = 5.0
    snapshot_minutes: float = 60.0
    eval_minutes: float = 30.0  # held-out evaluation in a background process; 0 = never
    eval_maps_per_tier: int = 6
    init_from: str = ""  # checkpoint whose actor and critic weights start this run
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    director: DirectorConfig = field(default_factory=DirectorConfig)


def _build(cls, data: dict, where: str):
    hints = typing.get_type_hints(cls)
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - fields
    if unknown:
        raise ValueError(f"unknown settings in {where}: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        kind = hints[name]
        if dataclasses.is_dataclass(kind):
            value = _build(kind, value or {}, f"{where}.{name}")
        elif kind is float and isinstance(value, (int, str)):
            value = float(value)  # YAML reads 3e-4 as a string
        elif kind is int and isinstance(value, (str, float)):
            value = int(float(value))
        kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None, overrides: list[str] = ()) -> TrainConfig:
    """overrides look like "ppo.lr=1e-4" or "model.slots=64"."""
    data = yaml.safe_load(Path(path).read_text()) if path else {}
    data = data or {}
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} is not key=value")
        node = data
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    cfg = _build(TrainConfig, data, "config")
    cfg.env.use_depth = cfg.model.use_depth
    return cfg


def to_dict(cfg: TrainConfig) -> dict:
    return dataclasses.asdict(cfg)
