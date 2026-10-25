# Reproduction status

Goal active. Full reproduction is not ready; no improved benchmark claim is supported yet.

Completed:

- Isolated project directory, source archive hashes, and accepted source extraction.
- CPU environment: Python 3.11.15, PyTorch 2.6.0+cpu, NumPy 2.4.6.
- Discrete independent/AR/BPPO/AgentMixer implementations and shared matrix training/evaluation.
- Initial correctness suite: 42 tests passed on 2026-10-25; continuous policies, temporal rollouts, masks and stable normalization have since passed additional targeted checks.
- MAPPO integration smoke: 2,048 matrix steps, output in results/smoke/mappo_20261025. This is not a performance result.

Current work:

- All four 2,048-step integration smoke runs passed. Initial matrix array 21192344 completed all 24 runs with zero final greedy optima: a failed learning check, retained in full.
- Stable advantage normalization fixed a float32 constant-signal bug. Frozen rerun 21192446 still yields suboptimal non-AgentMixer outcomes; the fix alone does not resolve learning. Exploration and matched action-projection controls are next.
- Source audit and 45 confirmed benchmark targets are recorded in `configs/paper_targets.json`; plot-only targets remain unresolved.
- Gaussian policies, temporal rollouts, termination/truncation handling, agent death masks and atomic checkpoints are implemented.
- Full integrated suite passed277 tests after normalized-control, independent-environment-seed, normalization, and checkpoint refinements.
- Projected AR-PPO and BPPO reached both matrix-game greedy optima in all three development seeds for both entropy settings (jobs21192609,21192611). The matched baseline also succeeds; this is not evidence that peer feedback improves performance.
- Private MuJoCo 2.1.0 / Gym 0.10.8 passed a three-step HalfCheetah smoke (`.native/mujoco-smoke.json`). SMACv2 with StarCraft II 4.10 passed a real 32-step smoke in job21192452 (`results/smoke/smac_21192452.json`). These establish simulator availability only.
- All four methods completed128-step learning/evaluation/checkpoint smoke runs in both real simulators (job21192629). Every one of the12 MuJoCo task/partition pairs now passes mapping and seeded simulator checks, covering21 paper entries.
- MuJoCo curve estimates are extracted with verified PDF axes and reading uncertainty: `configs/curve_estimates_mujoco.json`. These are digitized values, separate from exact numeric paper targets; source aggregation and appendix ambiguities remain explicit.
- Observation/state normalization, nonoverlapping simulator seed streams and an identity-modifier ablation are integrated. The first100k-step HalfCheetah pilot is job21192732, covering five methods/controls with frozen source. Reward/value normalization is not yet implemented.
- MuJoCo adapter and generic simulator trainer are being integrated. Recurrent policies and paper-equivalent preprocessing still need review.
- Run matched-budget baseline and method comparisons, then independent final seeds.

Execution notes:

- Default execution sandbox has a bubblewrap temporary-directory failure. Reviewed `require_escalated` commands work; user authorizes routine low-risk commands.
- Triton scheduler access requires `SLURM_CONF_SERVER=slurmctl,slurmctl2`, as provided by `/etc/profile.d/slurm_conf.sh`.
- The private `.venv-mujoco` conda prefix requires `PYTHONNOUSERSITE=1` to exclude unrelated user packages. Its pinned training dependencies pass `pip check` with this isolation.
- Never alter or cancel the other Codex reproduction's jobs. Record and operate only on this project's own job IDs.
