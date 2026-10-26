# Choosing a configuration

Configuration files serve three purposes. The command depends on the file type.

| File type | Contents | How to use it |
| --- | --- | --- |
| Matrix settings | Flat fields such as `entropy_coef` and `action_embed_dim` | `python -m bta_repro.train --config FILE --algorithm METHOD --game GAME` |
| Simulator experiment plan | `base_config`, `variants`, `seeds`, `label`, `scope` | `python scripts/prepare_simulator.py FILE` expands and freezes individual runs |
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

## Starting points

| Question | Configuration | Evidence and scope |
| --- | --- | --- |
| Run BPPO or its matched AR-PPO baseline on a matrix game | `matrix_projected_entropy01.json` | Both methods reach the optimum on both games in development seeds 0–2 at 100k steps |
| Inspect AgentMixer's correlated discrete redesign | `matrix_joint_lr_fast.json` | At 200k, all three development Climbing seeds succeed; all three Penalty seeds fail |
| Compare BPPO and AR-PPO on HalfCheetah 2×3 | `mujoco_hc2x3_bppo_ar10m.json` | Six development runs at the 10M paper budget; still in progress |
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
