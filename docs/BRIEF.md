# BBB (Bictor Bon Boom): a budgeted, position-anchored spatial memory for DOOM

## The idea

Open-world games (GTA: San Andreas, Metroid Prime, Jak & Daxter) keep a fixed budget of world chunks
in memory, streaming new ones in by priority and evicting irrelevant ones. BBB gives a
neural agent the same discipline: a fixed number of memory slots, each anchored to a position the
agent estimates for itself, holding learned content, evicted by learned relevance and read by
attention from the agent's current point of view. It lets the agent recall that a locked door was
behind and to the left, several rooms ago, without any global coordinates.

## Architecture (inference path)

```
official observation
  ├─ 64 rays ── circular 1-D CNN ───────────────┐
  ├─ ≤32 entities ── set attention + pooling ───┼─ fusion (256) ──┬──────────────┐
  └─ vitals + previous action ── MLP ───────────┘                 │              │
rays + velocity + last 3 actions ── neural odometry ── pose ─▶ Streamed Memory    │
                                                  K = 32 slots ── readout (128) ─┤
                                                                      GRU core (256)
                                                     factorised action heads ◀────┘
                                          safety shell (validate, idle on error) ─▶ 9 buttons
```

About 1.1 M parameters; under 1 ms per decision on one CPU thread.

## Streamed Memory

| Step | Mechanism | Learned? |
|---|---|---|
| Ego-motion | network predicts (forward, right, turn) each decision; summed into a pose in the spawn frame | motion: yes (supervised in training only) |
| Stream in | no slot within 320 units: allocate one at the estimated pose | bookkeeping |
| Write | `content ← content + gate · (GRUCell(input, content) − content)` | gate and content: yes |
| Stream out | memory full: overwrite the lowest-relevance slot | relevance: yes |
| Read | multi-head attention; keys include each slot's distance and bearing from the current pose; relevance biases attention | yes |

Relevance learns from the policy gradient: because it biases what is read, slots that help are rated
higher and therefore survive. A fallback mode replaces position-based slot choice with learned soft
addressing.

## Training pipeline (our own code, no off-the-shelf RL library)

- **Recurrent PPO** with sequence minibatches, truncated back-propagation through time, GAE and
  separate extrinsic and intrinsic value heads.
- **Asymmetric critic** sees ground truth (pose, keys, doors, exit, enemies). It exists only in
  training; the actor rejects any non-official input.
- **Scorer-aligned reward**: exit, keys, unlocks, doors, true-sector coverage, kills, damage, death,
  time and stagnation, with caps so that finishing always outweighs farming.
- **Procedural map factory**: room graphs with corridors, doors, one to three key and lock chains,
  switch or walk-over exits; every level is checked to be solvable. Freedoom maps add variety.
- **Curriculum Director**: favours scenarios with about 50 % success, unlocks harder levels, replays
  failed seeds ("nemesis" scenarios) as fresh episodes, and ends on vanilla rules at skill 3.
- **Also**: neural-odometry warm-up, auxiliary memory targets, episode-level left/right mirroring.
- **Model selection** uses a fixed held-out suite run through the real submission agent.

## Results

Held-out scores, ablations and the final model selection are in [EXPERIMENTS.md](EXPERIMENTS.md) and the
[2-page brief](brief.pdf). Local scores are our estimate: the official weights are not public.

## Limits

Odometry drifts over long episodes, so slot positions are approximate. Door and key colours are not
observable from rays or entities. A hierarchical Manager/Worker policy was designed but not built.
