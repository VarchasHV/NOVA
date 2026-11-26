# Experiment log

Every run writes `runs/<name>/config.json` (full settings, seed, git hash), `log.jsonl`,
`episodes.jsonl`, TensorBoard logs and `eval.jsonl` (held-out results). Change one thing per ablation
and compare on the same held-out suite.

## Planned runs

| Run | Config | Hypothesis | Weighted score | Notes |
|---|---|---|---|---|
| full | `configs/full.yaml` | submission candidate | | |
| gru_baseline | `configs/gru_baseline.yaml` | no memory loses on tiers 3 and 4 | | |
| memory_k64 | `configs/memory_k64.yaml` | 64 slots help the largest maps | | optional |
| fight_s3 (ASUS) | `configs/asus_fight.yaml`, resumed from the `full` checkpoint | combat-only exit and kill rewards raise kills without more deaths or weaker navigation | | compare with `full` at the same training time |
| variety_s4 (ASUS) | `configs/asus_variety.yaml`, resumed from the `full` checkpoint at 32.6M or later | uneven practice maps and more Freedoom improve survival and exits on unfamiliar maps | | replaces fight_s3 |
| greedy vs sample | `evaluation.run_heldout --mode` | sampling gets stuck less | | pick before export |

## Development checks on the Mac (CPU, not results to report)

| Date | What | Outcome |
|---|---|---|
| 2026-11-26 | 330k decisions on level-0 movement and combat | arena switch-exit rate rose from 40% to 70% of episodes; stuck decisions per episode fell from about 35 to 1; resume and background evaluation worked |
| 2026-11-26 | latency of an exported model, one thread | mean 0.74 ms, p99 0.84 ms |
| 2026-11-26 22:15 | `fight_s3` (32.3M) against `full` (32.6M), 81 games each, sampling | plain suite 115.9 against 124.1, uneven maps 102.4 against 113.9; `fight_s3` killed slightly more but reached fewer exits |
| 2026-11-26 22:14 | uneven map geometry | 4,000 maps, 1.33M cell-to-cell steps: a 32 x 32 player box was never blocked; plain maps unchanged byte for byte |
