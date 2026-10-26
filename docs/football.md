# Football protocol and adapter

The optional adapter supports the four BPTA scenarios. Its Python interface is
`FootballEnv({'scenario': name, 'seed': seed})`; the shared trainer accepts
`environment="football"`. Importing the adapter requires only NumPy. Creating
a real backend requires the private Football environment.

| Scenario | Controlled left players | Configured game duration |
| --- | ---: | ---: |
| academy_3_vs_1_with_keeper | 3 | 400 |
| academy_counterattack_hard | 4 | 400 |
| academy_corner | 10 | 400 |
| 5_vs_5 | 4 | 3000 |

The native clock starts at −1. A full 5v5 match therefore takes 3,001 API steps
for its configured 3,000 ticks. Academy scenarios can end earlier on possession,
score or out-of-play events. We preserve the engine's ending unchanged.

The [rollout and optimizer audit](football_optimization_audit.md) compares the
initial learning pilot with the accepted settings and records the native gate
for matched feed-forward/GRU controls using 15 epochs and two minibatches.

The pinned author release uses `simple115v2`, no frame stack, and
`scoring,checkpoints` rewards. Each actor receives its own115-dimensional vector.
Training uses the sum of rewards across controlled players, matching the legacy
shared-reward wrapper. Raw player rewards remain in each step's information.
The redesigned shared critic receives all local vectors concatenated. This
differs from the release's separate critics, each receiving its own115 features.
All19 default actions remain available, as in the released wrapper.

Source evidence is the [author wrapper](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/envs/football/Football_Env.py),
[entry point](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/scripts/train/train_football.py),
and scenario launchers under that commit's `bta/scripts/temporal/football/lumi/`.
The selected simulator is official Google Research Football2.10.2, tag commit
`ba9952130898bde2879b01ff2b49d3d48d26ce81`. The original paper's installed engine
version is unavailable; this choice is recorded rather than asserted as fact.
The official source archive SHA256 is
`1b0fdcfa78b7fadc3730585ee7f0f412ba825c27e422b2b85ea0cf7ba57800b6`.

## Corrections and evaluation

The legacy wrapper's `seed()` changes Python's global RNG only. The new adapter
derives an explicit native engine seed for every reset, sets
`game_engine_random_seed` and its associated team-processing order, and keeps
Python/NumPy streams private around backend calls. Constructor-only draws are
discarded. Reseeding restarts both the seed stream and the scenario episode
counter; the latter controls alternating kickoff in5v5. Its initial phase
matches the backend constructor. These calls use the pinned native configuration
API. Real-engine checks reproduced complete observation/reward traces exactly
after reseeding and in a fresh instance on all four scenarios, including a full
3,001-step5v5 match.

Native goal, possession, out-of-play and match-horizon endings terminate an
episode. An optional shorter `episode_limit` is an external diagnostic cap and
truncates, preserving the final observation for bootstrapping. Native termination
takes precedence if both happen on the same step. No automatic reset occurs.

Evaluation completes each match and determines winning from the final score.
The legacy evaluator instead stops after200 rollout steps and leaves unfinished
entries as zeros; its win check uses the last step's `score_reward`. That can
differ from final match outcome, especially in5v5. These errors are not retained.
The author [confirmed on 2026-10-25](author_clarifications.md) that the5v5
figure plots evaluation return. Its exact historical evaluation reduction and
code version remain unresolved: this pinned200-step evaluator cannot finish a
native3,001-step match. Keep raw returns and scores separate; do not interpret
the figure using a training-reward reduction or convert it into a win rate.

## Current installation and checks

Use Linux x86_64, micromamba and uv. From the repository root:

```bash
# Set BTA_MICROMAMBA/BTA_UV if those executables are not on PATH.
bash scripts/setup_football.sh
export PYTHONNOUSERSITE=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export LD_LIBRARY_PATH="$PWD/.venv-football/lib:${LD_LIBRARY_PATH:-}"
.venv-football/bin/python scripts/check_environments.py --profile football --strict --imports
.venv-football/bin/python scripts/smoke_football.py --output results/football-native-check.json
```

The installer pins130 native packages by URL/checksum and23 Python package
versions, including Python3.10.22, NumPy1.23.5, Gym0.23.1, GCC12.3 and CPU
PyTorch2.6. The explicit `libopengl` package prevents dependence on a system
OpenGL library that was present on compute nodes but absent on the login node.
Locks are under `configs/environment_locks/football-*`. Use
`BTA_FOOTBALL_ENV_NAME=.venv-football-verify` for a separate fresh installation.
Its compiler, runtime libraries, downloads, build tree and caches stay inside
the project. Setup records provenance under `.native/football-build/ENV_NAME/`.
The source archive is hash checked; Python wheels are version pinned but not
individually hash locked. Native compilation uses two cores unless Slurm
assigns a different `SLURM_CPUS_PER_TASK`. Only build parallelism and private
dependency search paths are patched; simulator physics and scenarios are intact.

Native build/replay jobs and all-method integration jobs are recorded in
`experiments/jobs.jsonl`. The first build failed because compute nodes lacked
a compiler; the private compiler fixed it. The first replay check exposed the
duration/step distinction above. Both failures remain recorded.

A separate installation in `.venv-football-verify` now passes all package pins,
imports, dependency checks and native replay. Its four complete trace hashes
match the original build exactly, including5v5. The original prefix also passes
imports on the login node after adding the private OpenGL runtime. Verification
artifacts are under `.native/football-build/.venv-football-verify/`.

The controlled-backend tests verify all four shapes, shared rewards, native
seed propagation, RNG isolation, final-score win metrics, action copying,
termination precedence, validation and cleanup. All four algorithms also pass
shared-trainer integration checks. Native availability and seed replay are
verified separately; none of these checks establishes benchmark performance.
