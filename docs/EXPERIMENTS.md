# Experiment log

Every run writes `runs/<name>/config.json` (full settings, seed, git hash), `log.jsonl`,
`episodes.jsonl`, TensorBoard logs and `eval.jsonl` (held-out results). Change one thing per ablation
and compare on the same held-out suite.

## How we compare policies

- Every policy plays through the real `submission/agent.py`, exactly as the scorer would run it.
- **Held-out suites:** two independent sets, each with 240 generated maps that were never used in training (plain and uneven geometry) plus 3 Freedoom maps, across tiers 1–4 (249 episodes per set, fixed seeds).
- **Score:** our local scorer (`evaluation/local_scorer.py`) mirrors the four official categories and weights the tiers 1:2:3:4. The official weights are hidden, so absolute numbers are estimates.
- **Uncertainty:** a paired bootstrap (4,000 resamples) that resamples maps within each tier. A difference counts as real only when its 95% interval excludes zero.

## Runs

| Run | Config | Hypothesis | Result |
|---|---|---|---|
| full | `configs/full.yaml` | the main candidate for submission | 72.3M decisions in 14 h. The best checkpoint (58.4M) is submitted: pooled weighted score 131.3, compared with 125.7 at 40.5M (**+5.7**, CI 0.2 to 11.3). The largest gain is in tier 3 (exit rate 65% → 75%). |
| gru_baseline | `configs/gru_baseline.yaml` | no memory loses on tiers 3 and 4 | not run (no time); replaced by the memory ablations below |
| memory_k64 | `configs/memory_k64.yaml` | 64 slots help the largest maps | not run |

### Held-out results by tier (score per map · share of episodes reaching the exit, both sets pooled)

| Tier | 40.5M | 58.4M (submitted) |
|---|---|---|
| 1 Combat | 143 · 92% | 142 · 93% |
| 2 Exploration | 128 · 70% | 129 · 70% |
| 3 Progression | 150 · 65% | 163 · 75% |
| 4 Gauntlet | 102 · 35% | 107 · 39% |
| Weighted | 125.7 | 131.3 |
| Deaths in tier 4 | 46% | 46% |

## Checkpoint choice

| Set | Checkpoint | Weighted | Δ against 58.4M (95% CI) | Exit rate | Deaths |
|---|---|---|---|---|---|
| 1 | 40.5M | 118.8 | −7.3 (−14.9 to −0.2) | 59% | 17% |
| 1 | **58.4M** | 126.1 | — | 62% | 16% |
| 1 | 62.0M | 127.5 | +1.4 (−6.3 to +9.1) | 63% | 14% |
| 2 | 40.5M | 132.5 | −4.1 (−12.6 to +4.4) | 69% | 13% |
| 2 | **58.4M** | 136.6 | — | 74% | 14% |
| 2 | 72.3M (final) | 135.4 | −1.1 (−8.6 to +6.6) | 72% | 13% |

The 58.4M checkpoint scored best on the 27-game test run during training. The 62.0M and 72.3M checkpoints are tied with it.
Set 1 uses generator seeds offset by 500 from the training-time test, set 2 by 900.

## Memory ablations

Measured on `full` at 34.8M decisions: 81 maps × 2 map variants × 2 decoding settings (`runs/final_check/`).

| Variant | Δ weighted score | 95% CI | Exit rate |
|---|---|---|---|
| Full agent | — | — | 68–72% |
| Slot-memory readout set to zero (GRU kept) | −10.1 | −20.3 to −1.1 | 57–69% |
| Recurrent state and slots wiped every step | −50.5 | −64.8 to −36.4 | 12–28% |

Score per tier (averaged over the four settings):

| Tier | full | readout zeroed | wiped |
|---|---|---|---|
| 1 Combat | 145 | 142 | 86 |
| 2 Exploration | 129 | 110 | 45 |
| 3 Progression | 164 | 155 | 112 |
| 4 Gauntlet | 74 | 66 | 43 |

Zeroing the slot readout hurts most in exploration and progression and barely matters in combat. That is the pattern expected if the slots store places.

## Decoding

Two earlier checkpoints of `full` (24.9M and 34.8M) on both map variants:

| Comparison | Δ score | 95% CI |
|---|---|---|
| sampling at T = 1.0 − sampling at T = 0.5 | −3.9 | −9.5 to +0.9 |
| greedy − sampling at T = 0.5 | −7.0 | −11.6 to −3.0 |

Greedy decoding scores lower; the two sampling temperatures are not clearly different. We submit sampling at T = 0.5.

## Failure analysis (240 unseen maps, 40.5M checkpoint)

| Family | Switch exit reached | Walk-in exit reached |
|---|---|---|
| Exploration | 40% | 100% |
| Progression | 52% | 91% |
| Gauntlet | 31% | 61% |

Switch exits are the main failure: the agent has to press USE, as a fresh press, while facing the switch. The kill rate is about 15–20%, and about 10% of shots hit. The agent tends to hold the fire button, which spreads pistol shots.

## Development checks on the Mac (CPU, not results to report)

| Date | What | Outcome |
|---|---|---|
| 2026-11-26 | 330k decisions on level-0 movement and combat | arena switch-exit rate rose from 40% to 70% of episodes; stuck decisions per episode fell from about 35 to 1; resume and background evaluation worked |
| 2026-11-26 | latency of an exported model, one thread | mean 0.74 ms, p99 0.84 ms |
| 2026-11-26 22:14 | uneven map geometry | 4,000 maps, 1.33M cell-to-cell steps: a 32 x 32 player box was never blocked; plain maps unchanged byte for byte |
