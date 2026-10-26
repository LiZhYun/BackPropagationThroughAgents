# Choosing a configuration

The command depends on the configuration file type.

| File type | Contents | How to use it |
| --- | --- | --- |
| Matrix settings | Flat fields such as `entropy_coef` and `action_embed_dim` | `python -m bta_repro.train --config FILE --algorithm METHOD --game GAME` |
| Single simulator run | Flat fields including `environment`, `algorithm`, `steps` and `env_config` | `python -m bta_repro.simulator_train --config FILE --output NEW_DIRECTORY` |
| Simulator experiment plan | `base_config`, `variants`, `seeds`, `label`, `scope` | `python scripts/prepare_simulator.py FILE` expands and freezes individual runs |
| Environment runtime diagnostic | `environment`, `env_config`, `seeds`, `decisions_per_environment`, `repeats`, `timeout_seconds` | `python scripts/profile_environment_processes.py --config FILE --output NEW_DIRECTORY`; see [runtime measurements](../docs/simulator_runtime.md#standalone-process-comparison) |
| Paper targets and environment locks | Source measurements, extraction provenance or dependency pins | Used for comparisons or installation; not training configurations |

A simulator plan is not accepted by the single-run trainer's `--config` option.
After preparation, each run's flat configuration and output path are listed in
the generated `manifest.json`. Preparation copies the current source into the
experiment directory and prints that path. It does not submit or run anything.
Use that frozen source when launching a generated run or replaying a checkpoint;
see [the simulator guide](../docs/simulator_training.md).

For confirmation of a selected experiment, add
`--source-manifest results/SELECTED/manifest.json`. Preparation verifies and
copies that experiment's exact source instead of the current working tree,
recording the parent manifest hash. Use fully resolved settings from the
selected run and validate compatibility with the copied trainer before launch;
the option does not translate new configuration fields into old schemas.

Matrix preparation supports the same `--source-manifest` option:
`python scripts/prepare_matrix.py --overrides FILE --algorithms agentmixer
--games climbing --seeds 2000 2001 --steps 200000 --workers 1` prepares runs
without launching them. An explicit `--steps` takes precedence over the JSON
budget; otherwise the JSON budget is used, or 100k if neither supplies one.
`--wall-time-minutes` sets the generated submission limit. A separate frozen
study plan defines any independent-validation claim.

## Starting points

| Question | Configuration | Evidence and scope |
| --- | --- | --- |
| Run BPPO or its matched AR-PPO baseline on a matrix game | `matrix_projected_entropy01.json` | The selected setting passes its frozen 100k check on ten unused seeds per method/game: optimal greedy actions and expected returns above95% of the ceiling. [Validation](../docs/matrix_confirmation.md#completed-100k-validation) |
| Inspect AgentMixer's correlated discrete redesign | `matrix_joint_lr_fast.json` | At 200k, all three development Climbing seeds succeed; all three Penalty seeds fail |
| Inspect the diagnosed actor-rate control | `matrix_joint_lr_faster_float64.json` | All three development seeds reach each game's greedy optimum at 200k; central/local sampling gaps remain. [Independent Climbing validation](../docs/agentmixer_confirmation.md) finishes with 8/10 stochastic-threshold successes and 10/10 greedy optima |
| Compare learned and identity modifiers | `matrix_agentmixer_learned1m.json`, `matrix_agentmixer_identity1m.json` | Same selected setting extended to 1M, differing only in whether the modifier learns. AgentMixer's archived Climbing budget is 200k. Public examples use seed 0; the [paired study](../docs/matrix_modifier_ablation.md) uses fresh seeds 4000–4004 on both games and is queued |
| Try the shared trainer with only core dependencies | `prey_example.json` | Flat single-run example: 10,240 training decisions, 20 evaluation episodes and 100×200 evaluation windows; not a benchmark result |
| Compare BPPO and AR-PPO on HalfCheetah 2×3 | `mujoco_hc2x3_bppo_ar10m.json` | All six development runs complete: BPPO5980.37±118.29 versus AR5817.39±210.91. [Full comparison](../docs/halfcheetah_confirmation.md) |
| Confirm the selected HalfCheetah protocol on unused seeds | `mujoco_hc2x3_confirmation10m.json` | Ten frozen runs, seeds1000–1004, separate evaluation seed; exact development source retained. See [the study](../docs/halfcheetah_confirmation.md) |
| Check transfer to Ant and Walker | `mujoco_ant_walker_transfer500k.json` | Eleven 500k pilots queued; distinct Ant observation protocols and matched controls |
| Compare continuous AgentMixer scale maps | `mujoco_hc2x3_modifier_scale1m.json` | All nine runs complete; neither modifier improves the final three-seed mean over identity |
| Reproduce the Prey batch controls | `prey_batch1m.json` | Sixteen completed runs at 1M; neither penalty setting clears the selected paper comparison |
| Test restricted Prey modifier ranges | `prey_modifier_bound1m.json` | Eight matched 1M controls queued after distribution diagnostics; no results yet |
| Run the longer SMAC or Football learning checks | `smac_terran5_learning1m.json`, `football_3v1_learning1m.json` | Development plans following the shorter runtime and learning pilots |
| Check private GRU actors in native SMAC | `smac_recurrent_batch_integration10k.json` | Five-method integration queued; one 10,000-transition rollout per run |
| Compare feed-forward and GRU actors at a larger SMAC batch | `smac_terran5_batch1m.json` | Ten-run plan frozen but unsubmitted pending native correctness, runtime and memory checks |
| Measure native SMAC runtime | `smac_runtime_profile4096.json` | Two short diagnostics with a separate timing wrapper; see [the profiling guide](../docs/simulator_runtime.md). Rewards are not learning evidence |

These files describe development and explicitly frozen confirmation studies;
none is a certified final reproduction recipe yet.
The [current status](../STATUS.md) records job completion and acceptance gaps;
the [experiment tracker](../refine-logs/EXPERIMENT_TRACKER.md) retains earlier
controls and negative results. Every result must be associated with its actual
frozen manifest, not merely a similarly named configuration in this directory.

## Interpreting plan fields

A simulator run combines `base_config`, then that variant's overrides, then the
listed training seed. Each `env_config` override replaces the entire nested
object; it is not recursively merged. The resulting flat configuration is
validated before launch. Unspecified fields use the trainer's dataclass
defaults, which are written into each run's recorded configuration.

Training `steps` counts joint environment transitions. Evaluation interactions
are recorded separately. Budgets, evaluation episode counts and evaluation
seeds are part of the protocol; changing them creates a new comparison.

For Predator–Prey, the author selected the released 100-window × 200-decision
reduction as the primary comparison. Keep `eval_windows=100` and
`eval_window_steps=200`; use `fixed_window_reward_per_step` in curve reports.
Completed-episode return and capture rate are additional diagnostics. This
choice is recorded separately from the historical figure labels in
`paper_targets.json` and `curve_estimates_other.json`.

Optional simulators have private dependency locks under `environment_locks/`.
The plotting lock belongs to a separate plotting environment. Matrix and Prey
runs need only the core package; simulator installations are independent.

The Football plans `football_3v1_batch_integration10k.json` and `football_3v1_batch1m.json` use the released50×200 rollout,15 epochs and two minibatches, with matched feed-forward/GRU controls. The native integration must pass before submitting the1M development plan; see the [protocol audit](../docs/football_optimization_audit.md).

`prey_recurrent_protocol_check10k.json` and `prey_recurrent_release1m.json` test the released Prey optimization counts with matched feed-forward/GRU and AgentMixer/identity controls. The full-window replay/runtime gate passes; the 1M study is queued with a measured nine-hour worker cap. See the [study](../docs/prey_recurrent_control.md).
