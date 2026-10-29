# Community release status

Updated 29 October 2026. The priority is a clear, faithful repository that readers can run and understand. Full-budget runs, exhaustive experiments and new tuning sweeps are not release requirements. Performance beyond the papers is not claimed.

## Code and method fidelity

- Shared implementations of MAPPO, AR-PPO, BPPO and AgentMixer use explicit objectives, common rollout/evaluation code and saved configuration/source provenance.
- [Method fidelity](docs/method_fidelity.md) maps paper mechanisms to modules and tests. BPPO objective repairs and the discrete AgentMixer mode-preserving correction are explicit. Optional experimental variants are labeled.
- The [README](README.md) leads with short reference examples and a code-reading map. [Runtime guides](docs/environment_versions.md) document the separate optional simulator environments.

## Validation

The exact-source community wheel (`e5c4116`) passes 1,093 tests with one CUDA-only skip, all fourteen build/install/CLI/replay/report commands, and source/README/license checks. The original validation prefix and dependencies are restored. [Evidence and limitations](docs/clean_checkout.md).

Initial native integrations cover MuJoCo/SMAC and Football; the Predator–Prey adapter matches 10,240 pinned-release transitions. These integration checks do not establish benchmark performance. An optional local-scale historical-prefix comparison failed and remains documented; exact historical trajectory equivalence is unproven. [Failure record](experiments/local_scale_native_prefix_failure_20261029.json).

## Experiments are stopped

All five earlier full-budget arrays and all five remaining optional research arrays are stopped, together with their obsolete collectors. No live `bpta-agentmixer-*` jobs remain. Original configurations, sources, metrics and existing checkpoints are retained; archived copies match original bytes. The unrelated reproduction was not changed.

[Full-budget preservation](experiments/reduced_budget_stop_and_preservation_20261029.json); [remaining-sweep preservation](experiments/community_release_stopped_sweeps_20261029.json). The 54 prepared continuous learning runs remain unsubmitted. Do not restart sweeps as a release gate.

## Publication

All four public README commands pass, including the complete Predator–Prey episode/window replay. The unified code targets the user-specified `LiZhYun/BackPropagationThroughAgents` repository. Publication preserves its existing commit history and uses a normal fast-forward push. No further experiment is needed for the public-code release. [Current scope](experiments/community_release_scope_20261029.json).

The [experiment tracker](refine-logs/EXPERIMENT_TRACKER.md), preserved dated status files and [source audit](docs/source_audit.md) retain research history. Benchmark coverage percentages are not release-readiness percentages.
