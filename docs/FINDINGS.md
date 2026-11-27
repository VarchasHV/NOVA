# Phase 0 findings: what the template actually provides

Audited on 2026-11-26 at commit `ac443c8` (contract `1.0.0-draft`). Line numbers refer to files under
`env/doom_env/`, which this project does not modify.

## Observation

| Key | Shape / dtype | Range and meaning | Evidence |
|---|---|---|---|
| `rays.distance` | (64,) float32 | hit distance / 1024; 1.0 = nothing | `spaces.py:14-15`, `sensors.py:267-270` |
| `rays.hit_type` | (64,) int | 0 none, 1 wall, 2 door, 3 locked door, 4 enemy, 5 item | `spaces.py:16`, `sensors.py:25` |
| `visible_entities.features` | (32, 5) float32 | distance/1024, sin, cos of bearing, relative velocity forward and right (units/tic) | `spaces.py:19`, `sensors.py:289-295` |
| `visible_entities.entity_type` | (32,) int | 12 classes, 0 = padding; includes projectiles, keys, barrels | `spaces.py:21-34` |
| `visible_entities.mask` | (32,) int8 | 1 = real; nearest first | `sensors.py:286-297` |
| `vitals` | (17,) float32 | health, armor, weapon slot, selected ammo, 4 ammo pools, 6 key flags, velocity forward/right/up | `spaces.py:36-54` |
| `rgb` / `depth` | (120,160,3) / (120,160) uint8 | zeros when `render_buffers=False` | `spaces.py:11-12`, `wrapper.py:189-194` |

- **Rays cover 360°**, index 0 straight ahead, counter-clockwise (index 16 = left) (`sensors.py:190`).
  So circular padding is correct, and mirroring maps ray `i` to `(64 - i) % 64`.
- **Entities** are limited to a 90° field of view with line of sight, up to 4096 units (`sensors.py:22-23`).
- **No heading, position or angular velocity.** Velocity is in the player's frame (`wrapper.py:182-187`).
- **Door and key colours are not observable** from rays or entities. The memory can remember "a locked
  door", not "a red locked door".
- Key flags are inferred when a key item disappears from the level (`sensors.py:302-320`).

## Actions and timing

- `MultiBinary(9)`: forward, back, strafe left, strafe right, turn left, turn right, attack, use,
  next weapon (`spaces.py:58-68`). Anything that is not shape (9,) with values 0/1 is invalid and ends
  the agent's participation in that map (`remote.py:209-217`).
- `USE` and next-weapon fire on button-down; pressing USE on a moving door reverses it (`docs/CONTRACT.md` §3).
- The wrapper applies the 4-tic frame skip (`spaces.py:70`, `wrapper.py:130`).
- Turning with buttons is slower during the first 6 tics a turn key is held (vanilla Doom behaviour), so
  odometry receives the last three actions.

## Episodes, information available in training, map loading

- `terminated` = exit reached or death; `truncated` = tick cap (`wrapper.py:134-139`).
- `info` holds only `tic` and `exit_reached` (`wrapper.py:199-200`).
- `privileged_state()` gives position, angle, health, kills, items, damage, keys, death, exit, sector
  floor and ceiling heights and door lock numbers (`wrapper.py:203-227`). It does **not** give the
  player's sector, monster, key or exit positions; `training/env.py` computes them.
- The scorer measures coverage with sector polygons (`wrapper.py:229-238`), so the exploration reward
  uses true sectors.
- Any WAD loads with `make_env(wad, map)`. Skill is fixed at 3 by the wrapper (`wrapper.py:26`);
  `TrainEnv` changes it for training. Seeds are set per `reset(seed=)`.
- Limits (`limits.py`): init 60 s, reset 10 s, first act 10 s, other acts 5 s, 40 ms average per
  decision, 2 GB VRAM, 8 GB RAM, 1 CPU thread.

## Measured on the development Mac (Apple M3)

| Measurement | Value |
|---|---|
| Single environment, small map | ~5,100 decisions/s (rendering on or off) |
| Single environment, Freedoom MAP01 | ~870 decisions/s |
| Switching maps inside one WAD | ~2 ms |
| Switching WAD files (engine restart) | ~140 ms |
| Door (use), locked door with card and skull keys of all three colours, bump-open door, walk-over exit, switch exit | all verified in generated maps |
| Engine cheats | `god` lasts one episode; `sv_infiniteammo` persists until reset to 0 |
| `DoomGame.save` / `load` | present in the API, **not tested** (Go-Explore not built) |

## Template tests on macOS

`tests/test_remote.py` fails mostly with `AF_UNIX path too long` and one case
(`test_hostile_replies_become_protocol_failures[deep_json]`) hangs;
`tests/test_validate.py::test_cpu_and_ram_limits_are_emulated` needs `os.sched_getaffinity` (Linux).
These files are unchanged from the template. On a Mac, run:

```bash
uv run pytest --ignore=tests/test_remote.py \
  --deselect tests/test_validate.py::test_cpu_and_ram_limits_are_emulated
```

## Fixed bookkeeping in the memory (declared for reviewers)

All actions come from the network, and what is written to memory, how strongly, what is kept and what is read
are learned. Two steps are fixed code:

1. **Slot choice.** The pose is estimated from observations by a neural network. A new slot is created when no slot
   is within 320 units of it; otherwise the nearest slot is written. With `model.memory_mode: soft` the slot choice
   is learned attention instead (not used by the submitted model).
2. **Dead reckoning.** The pose is the sum of the network's per-step motion predictions, using fixed trigonometry.

## Changes from the original plan

| Plan | Built | Reason |
|---|---|---|
| ACS hooks for door and key events | Python, from `privileged_state()` and sector heights | no ACS compiler needed |
| `eval/` | `evaluation/` | `eval` shadows a Python builtin |
| Running mean/std reward normalisation | divide each stream by the std of its return | subtracting reward means changes the task |
| Observation normalisation by running statistics | fixed scaling | identical in training and submission, nothing to export |
| Depth encoder on | off by default (`model.use_depth`) | rays give 360° geometry at lower cost |
| Manager/Worker, behaviour cloning, all stretch items | not built | 24-hour time box |
