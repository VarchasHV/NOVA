"""Proximal Policy Optimisation with truncated back-propagation through time."""

from __future__ import annotations

import torch
from torch import nn

from models import policy as P
from models.actor import BBBActor

from .aux import aux_loss, odometry_loss
from .config import PPOConfig
from .rollout import odometry_pairs, sequence_minibatches


class PPOLearner:
    def __init__(self, actor: BBBActor, critic: nn.Module, cfg: PPOConfig):
        self.actor, self.critic, self.cfg = actor, critic, cfg
        odom_ids = {id(p) for p in actor.odometry.parameters()}
        self.params = [p for p in actor.parameters() if id(p) not in odom_ids] + list(critic.parameters())
        self.opt = torch.optim.Adam(self.params, lr=cfg.lr, eps=1e-5)
        self.odom_opt = torch.optim.Adam(actor.odometry.parameters(), lr=cfg.odom_lr)

    def train_odometry(self, rollout: dict, updates: int) -> dict:
        stats = {}
        for _ in range(updates):
            batch = odometry_pairs(rollout, self.cfg.odom_batch)
            if batch is None:
                break
            loss, stats = odometry_loss(self.actor.odometry, batch)
            self.odom_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.odometry.parameters(), 1.0)
            self.odom_opt.step()
            stats["odom_loss"] = float(loss.detach())
        return stats

    def update(self, rollout: dict, progress: float) -> dict:
        c = self.cfg
        lr = c.lr * (1.0 - (1.0 - c.lr_end_fraction) * progress)
        for group in self.opt.param_groups:
            group["lr"] = lr
        ent_coef = c.ent_start + (c.ent_end - c.ent_start) * progress
        int_coef = c.int_coef * max(0.0, 1.0 - progress / c.int_anneal_end)
        totals: dict[str, float] = {}
        batches = skipped = 0
        for _ in range(c.epochs):
            epoch_kl = []
            for data, state in sequence_minibatches(rollout, c.seq_len, c.minibatches):
                obs = {k[4:]: v for k, v in data.items() if k.startswith("obs/")}
                logits, hidden = self.actor.unroll(obs, state, data["reset"], data["pose"])
                logp = P.log_prob(logits, data["idx"])
                entropy = P.entropy(logits)
                adv = data["adv"][..., 0] + int_coef * data["adv"][..., 1]
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)
                log_ratio = logp - data["logp"]
                ratio = log_ratio.exp()
                policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - c.clip, 1 + c.clip) * adv).mean()
                value = self.critic(hidden, data["priv"])
                clipped = data["value"] + (value - data["value"]).clamp(-c.value_clip, c.value_clip)
                value_loss = 0.5 * torch.max((value - data["ret"]) ** 2, (clipped - data["ret"]) ** 2).mean()
                extra = aux_loss(self.actor.aux(hidden), data["aux"])
                loss = policy_loss + c.vf_coef * value_loss - ent_coef * entropy.sum(-1).mean() + c.aux_coef * extra

                self.opt.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                loss.backward()
                norm = nn.utils.clip_grad_norm_(self.params, c.grad_clip)
                if not torch.isfinite(norm):
                    skipped += 1
                    continue
                self.opt.step()

                with torch.no_grad():
                    kl = float(((ratio - 1) - log_ratio).mean())
                    epoch_kl.append(kl)
                    stats = {
                        "policy_loss": float(policy_loss), "value_loss": float(value_loss), "aux_loss": float(extra),
                        "kl": kl, "clip_fraction": float(((ratio - 1).abs() > c.clip).float().mean()),
                        "grad_norm": float(norm),
                        "value_error_ext": float((value[..., 0] - data["ret"][..., 0]).abs().mean()),
                        "value_error_int": float((value[..., 1] - data["ret"][..., 1]).abs().mean()),
                        **{f"entropy_{n}": float(entropy[..., i].mean()) for i, n in enumerate(P.HEAD_NAMES)},
                    }
                for k, v in stats.items():
                    totals[k] = totals.get(k, 0.0) + v
                batches += 1
            if c.target_kl and epoch_kl and sum(epoch_kl) / len(epoch_kl) > c.target_kl:
                break
        if skipped and not batches:
            raise FloatingPointError("every PPO minibatch produced a non-finite loss or gradient")
        out = {k: v / max(batches, 1) for k, v in totals.items()}
        out.update(self.train_odometry(rollout, c.odom_updates))
        out.update(lr=lr, ent_coef=ent_coef, int_coef=int_coef, skipped_minibatches=float(skipped))
        return out
