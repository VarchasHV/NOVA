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
| greedy vs sample | `evaluation.run_heldout --mode` | sampling gets stuck less | | pick before export |

## Development checks on the Mac (CPU, not results to report)

| Date | What | Outcome |
|---|---|---|
| 2026-11-26 | 330k decisions on level-0 movement and combat | arena switch-exit rate rose from 40% to 70% of episodes; stuck decisions per episode fell from about 35 to 1; resume and background evaluation worked |
| 2026-11-26 | latency of an exported model, one thread | mean 0.74 ms, p99 0.84 ms |
