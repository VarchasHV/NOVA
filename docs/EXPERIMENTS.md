# Experiment log

Every run writes `runs/<name>/config.json` (full settings, seed, git hash), `log.jsonl`,
`episodes.jsonl`, TensorBoard logs and `eval.jsonl` (held-out results). Change one thing per ablation
and compare on the same held-out suite.

## How we compare policies

- Every policy plays through the real `submission/agent.py`, exactly as the scorer would run it.
- **Held-out suite:** 243 generated maps that were never used in training, with plain and uneven geometry, plus Freedoom maps, across tiers 1–4 (249 episodes, fixed seeds).
- **Score:** our local scorer (`evaluation/local_scorer.py`) mirrors the four official categories and weights the tiers 1:2:3:4. The official weights are hidden, so absolute numbers are estimates.
- **Uncertainty:** a paired bootstrap (4,000 resamples) that resamples maps within each tier. A difference counts as real only when its 95% interval excludes zero.

## Runs

| Run | Config | Hypothesis | Result |
|---|---|---|---|
| full | `configs/full.yaml` | the main candidate for submission | 62.0M decisions in 11.5 h. Weighted score 127.5, compared with 118.8 at 40.5M (**+9.0**, CI 0.5 to 17.7). The largest gain is in tier 3 (exit rate 58% → 72%). |
| aim_s5 (ASUS) | `configs/asus_aim.yaml`, resumed from `full` at 49.8M | a damage reward improves aiming; more switch practice improves switch exits | 61.0M decisions. Weighted 128.3, statistically tied with `full` (+0.9, CI −8.5 to 10.0). Better in tier 2 (127 vs 115) and slightly better in tier 4; worse in tier 3. |
| variety_s4 (ASUS) | `configs/asus_variety.yaml`, resumed from `full` at 32.6M | uneven practice maps and more Freedoom improve survival and exits on unfamiliar maps | rejected: 8.8 points below `full` (CI 1.7 to 16.2) |
| fight_s3 (ASUS) | `configs/asus_fight.yaml`, resumed from `full` | combat-only exit and kill rewards raise kills without more deaths or weaker navigation | rejected: slightly more kills but fewer exits (plain 115.9 vs 124.1, uneven 102.4 vs 113.9, 81 games each) |
| gru_baseline | `configs/gru_baseline.yaml` | no memory loses on tiers 3 and 4 | not run (no time); replaced by the memory ablations below |
| memory_k64 | `configs/memory_k64.yaml` | 64 slots help the largest maps | not run |

### Held-out results by tier (score per map · share of episodes reaching the exit)

| Tier | `full` 40.5M | `full` 62.0M | `aim_s5` 61.0M |
|---|---|---|---|
| 1 Combat | 139 · 88% | 140 · 88% | 140 · 88% |
| 2 Exploration | 114 · 60% | 115 · 58% | 127 · 67% |
| 3 Progression | 145 · 58% | 160 · 72% | 153 · 68% |
| 4 Gauntlet | 97 · 33% | 107 · 39% | 107 · 42% |
| Weighted | 118.8 | 127.5 | 128.3 |
| Deaths in tier 4 | 51% | 45% | 41% |

Source: `runs/pick0445/compare.json` (04:45, 27 Nov).

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

All three checkpoints (`full` at 34.8M, its best checkpoint, and `variety_s4`) on both map variants:

| Comparison | Δ score | 95% CI |
|---|---|---|
| cooled sampling − sampling at T = 0.5 | +2.0 | −2.2 to +6.4 |
| greedy − sampling at T = 0.5 | −0.4 | −5.2 to +4.5 |

The differences are within noise. We submit sampling at T = 0.5, which avoids the loops that greedy decoding can fall into.

## Failure analysis (240 unseen maps, submitted 40.5M model)

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
