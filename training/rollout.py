"""Collecting experience and turning it into sequence minibatches."""

from __future__ import annotations

import numpy as np
import torch

from models import policy as P
from models.actor import StreamedMindActor

from .vec_env import VecEnv

ODOM_INPUTS = ("prev_ray", "prev_cmd", "prev_cmd2", "prev_vel")


class ReturnScaler:
    """Divides a reward stream by the running std of its discounted return (keeps value targets ~1)."""

    def __init__(self, num_envs: int, gamma: float):
        self.gamma = gamma
        self.ret = np.zeros(num_envs)
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        """rewards/dones (T, N)."""
        history = np.empty_like(rewards)
        for t in range(len(rewards)):
            self.ret = self.ret * self.gamma + rewards[t]
            history[t] = self.ret
            self.ret *= 1.0 - dones[t]
        n, mean, var = history.size, history.mean(), history.var()
        total = self.count + n
        delta = mean - self.mean
        self.var = (self.var * self.count + var * n + delta**2 * self.count * n / total) / total
        self.mean += delta * n / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var) + 0.05)

    def state_dict(self) -> dict:
        return {"mean": float(self.mean), "var": float(self.var), "count": float(self.count)}

    def load_state_dict(self, s: dict) -> None:
        self.mean, self.var, self.count = s["mean"], s["var"], s["count"]


class Collector:
    def __init__(self, env: VecEnv, actor: StreamedMindActor, critic, length: int, seq_len: int, device: str):
        if length % seq_len:
            raise ValueError(f"rollout length {length} must be a multiple of seq_len {seq_len}")
        self.env, self.actor, self.critic = env, actor, critic
        self.length, self.seq_len, self.device = length, seq_len, device
        self.state = actor.initial_state(env.num_envs, device)
        self.sticky = torch.zeros(env.num_envs, P.NUM_HEADS, dtype=torch.long)

    def reset_state(self) -> None:
        self.state = self.actor.initial_state(self.env.num_envs, self.device)

    def _tensor(self, name: str) -> torch.Tensor:
        return torch.from_numpy(self.env.buf[name].copy()).to(self.device, non_blocking=True)

    def _obs(self) -> dict:
        return {k[4:]: self._tensor(k) for k in self.env.buf if k.startswith("obs/")}

    def _random_actions(self) -> torch.Tensor:
        """Random but persistent behaviour for the odometry warm-up."""
        n = self.env.num_envs
        fresh = torch.stack([torch.randint(0, s, (n,)) for s in P.HEAD_SIZES], dim=1)
        fresh[:, 0] = torch.where(torch.rand(n) < 0.6, torch.ones(n, dtype=torch.long), fresh[:, 0])
        fresh[:, 5] = 0
        change = torch.rand(n, P.NUM_HEADS) < 0.15
        self.sticky = torch.where(change, fresh, self.sticky)
        return self.sticky

    @torch.no_grad()
    def collect(self, directive: dict, random_policy: bool = False) -> tuple[dict, list[dict]]:
        """Run `length` decisions in every environment. Returns the rollout and finished episodes."""
        data: dict[str, list] = {}
        chunk_states, episodes = [], []
        add = lambda k, v: data.setdefault(k, []).append(v)
        for t in range(self.length):
            obs = self._obs()
            reset = self._tensor("reset")
            state = self.actor.reset_where(self.state, reset)
            if t % self.seq_len == 0:
                chunk_states.append(state)
            out = self.actor.step(obs, state)
            self.state = out.state
            priv = self._tensor("priv")
            if random_policy:
                idx = self._random_actions().to(self.device)
            else:
                idx = P.sample(out.logits)
            for k, v in obs.items():
                add(f"obs/{k}", v)
            for k in ODOM_INPUTS:
                add(k, state[k])
            add("reset", reset)
            add("odom", self._tensor("odom"))
            add("priv", priv)
            add("aux", self._tensor("aux"))
            add("pose", out.pose)
            add("idx", idx)
            add("logp", P.log_prob(out.logits, idx))
            add("value", self.critic(out.hidden, priv))
            self.env.buf["action"][:] = P.to_buttons(idx).cpu().numpy()
            episodes += self.env.step(directive if t == 0 else None)
            add("rew", torch.stack([self._tensor("rew_ext"), self._tensor("rew_int")], dim=-1))
            add("done", self._tensor("done"))
        # Value of the observation that follows the rollout (not stored, state not advanced).
        final = self.actor.step(self._obs(), self.state, reset=self._tensor("reset"))
        rollout = {k: torch.stack(v) for k, v in data.items()}
        rollout["last_value"] = self.critic(final.hidden, self._tensor("priv"))
        rollout["states"] = {k: torch.cat([s[k] for s in chunk_states]) for k in chunk_states[0]}
        return rollout, episodes


def compute_advantages(rollout: dict, scalers: list[ReturnScaler], gammas: tuple[float, float], lam: float) -> None:
    """Generalised Advantage Estimation, separately for each reward stream. Adds "adv" and "ret"."""
    rew, done, value = rollout["rew"], rollout["done"], rollout["value"]
    scaled = torch.empty_like(rew)
    for s, scaler in enumerate(scalers):
        scaler.update(rew[..., s].cpu().numpy(), done.cpu().numpy())
        scaled[..., s] = (rew[..., s] / scaler.std).clamp(-10, 10)
    gamma = rew.new_tensor(gammas)
    adv = torch.zeros_like(rew)
    running = torch.zeros_like(rew[0])
    next_value = rollout["last_value"]
    for t in reversed(range(len(rew))):
        alive = (1.0 - done[t]).unsqueeze(-1)
        delta = scaled[t] + gamma * next_value * alive - value[t]
        running = delta + gamma * lam * alive * running
        adv[t] = running
        next_value = value[t]
    rollout["adv"], rollout["ret"] = adv, adv + value


def sequence_minibatches(rollout: dict, seq_len: int, count: int):
    """Shuffle whole sequences (never single timesteps). Yields (data (L,S,...), initial state (S,...))."""
    length, envs = rollout["reset"].shape
    chunks = length // seq_len
    order = torch.randperm(chunks * envs, device=rollout["reset"].device)
    steps = torch.arange(seq_len, device=order.device)[:, None]
    for part in order.tensor_split(min(count, len(order))):
        chunk, env = part // envs, part % envs
        t = chunk * seq_len + steps  # (L, S)
        data = {k: v[t, env] for k, v in rollout.items() if k not in ("states", "last_value")}
        yield data, {k: v[part] for k, v in rollout["states"].items()}


def odometry_pairs(rollout: dict, size: int) -> dict | None:
    """Random consecutive-step pairs for supervising odometry."""
    valid = (rollout["reset"] < 0.5).nonzero()
    if len(valid) == 0:
        return None
    pick = valid[torch.randint(len(valid), (min(size, len(valid)),), device=valid.device)]
    t, e = pick[:, 0], pick[:, 1]
    return {
        "prev_ray": rollout["prev_ray"][t, e], "ray": rollout["obs/ray_dist"][t, e],
        "cmd": rollout["obs/prev_action"][t, e], "prev_cmd": rollout["prev_cmd"][t, e],
        "prev_cmd2": rollout["prev_cmd2"][t, e],
        "vel": rollout["obs/vitals"][t, e][:, 14:16], "prev_vel": rollout["prev_vel"][t, e],
        "motion": rollout["odom"][t, e],
    }
