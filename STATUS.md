# Reproduction status

Full reproduction and performance beyond the papers are not yet established.

## Verified implementation

- Work is isolated to this repository; accepted sources and the unrelated reproduction are preserved.
- Shared MAPPO, AR-PPO, BPPO and AgentMixer training supports matrix games, SMACv2, MA-MuJoCo, Predator–Prey and Football.
- Temporal rollouts preserve terminal/time-limit observations, legal/dead-agent masks, raw policy actions and independent seeds. Evaluation preserves training RNG and normalization.
- Optional critic-target normalization, clipped/Huber losses, initial Gaussian exploration, projected discrete predecessor actions and an experimental two-agent joint modifier are explicit controls.
- Integrated suite: **583 passed, 1 CUDA-only skip**, including checkpoint replay, the SMAC clock guard, recorded Gaussian-noise control, joint-distribution precision controls and matrix/simulator integration. Networks remain float32; optional joint probabilities use float64.
- Continuous BPPO offers an explicit behavior-noise control with analytical gradient checks. Three-seed 500k results are mixed across optimization settings; the fresh-noise default is retained.
- Continuous AgentMixer now has an explicit bounded replacement-scale control. It preserves exact local/central mean equality, undetached local features and shared network initialization; unsupported identity combinations are rejected. Policy, gradient and checkpoint-replay tests pass.
- All 96 integration runs across 15 MuJoCo task/observation variants and nine SMACv2 scenarios pass with four algorithms. Predator–Prey matches 10,240 transitions from the pinned legacy implementation.
- Football2.10.2 builds and exactly replays all four scenarios; all 16 algorithm/scenario integration runs pass. Fresh portable setup passes in a second private prefix; all four native replay traces match the original build. Private OpenGL also fixes login-node imports.
- The author confirmed evaluation return for the Prey and Football 5v5 figures on 2026-10-25; see `docs/author_clarifications.md`. Prey separately logs the released fixed-window reduction, including unfinished episodes. Exact historical normalization and episode handling remain unresolved.
- `scripts/evaluate_simulator.py` restores saved policy weights and observation moments, checks source hashes, and labels checkpoint step/training status. It exactly replays a completed native Prey evaluation with its frozen source; all eight method/action-space combinations also replay in integration tests.
- Native SMAC lifecycle validation passed 512 transitions and 12 planned episode-boundary restarts. All five 200k training retries now complete beyond the previous 129,536-step failure, logging one planned restart per training environment.

## Experiment evidence

- Source inventory covers 45 tasks: 39 digitized plot targets with reading uncertainty and six numeric AgentMixer SMAC values. Historical aggregation ambiguities remain explicit.
- Projected AR-PPO and BPPO solve both matrix games in all three development seeds. Their matched success does not establish a peer-feedback advantage.
- Joint AgentMixer with faster actor updates solves Climbing in all three development seeds at200k but fails Penalty. The faster float64 control solves2/3 seeds in each game; the slower float64 control solves0/3 in both. No robust convergence claim.
- All 48 runs in six newer development suites completed. See `docs/development_results_20261026.md` and its raw CSV with summary/manifest hashes.
- At 3M HalfCheetah steps, combined-protocol BPPO reaches 5228.01 versus AR-PPO 4739.43 (+10.31%) in development seed 0. AgentMixer reaches 2411.14 versus identity 2975.45. Under normalization alone BPPO reaches only 599.40; these settings change several optimization choices together.
- Three-seed BPPO fresh/behavior-noise means at 500k are 1844.05/2853.01 with the small batch and 2547.40/1959.81 with the combined protocol. No estimator wins consistently across settings.
- All sixteen Prey 1M controls complete. Penalty −1 remains unsolved: every fixed-window reward rate is negative, even the higher-entropy condition with 9/20 captures. Evaluation return remains the author-confirmed metric; historical reduction is unresolved.
- Frozen-source Prey replay reproduces all sixteen policies’ original per-episode evaluation returns. Small-batch AgentMixer has a large training/greedy-execution gap: recent stochastic training captures 83–100%, versus 0/20 greedy evaluation captures, with many blocked moves. `docs/prey_behavior_diagnosis.md` records the scope; a separate 100-episode local/central sampling diagnostic is running.
- All five Football 100k controls finish with 0/20 wins. Positive shaped returns do not establish task success.
- Corrected SMAC 200k final win counts are MAPPO 1/32, AR-PPO 4/32, BPPO 0/32, AgentMixer 2/32, identity 0/32. The original failed run is retained.
- Full budgets, independent final training seeds, ablations and release evidence remain outstanding.

## Recorded active/queued jobs

- 21208906: BPPO and matched AR-PPO on HalfCheetah 2×3, 10M steps, development seeds 0–2, 32 deterministic evaluation episodes. Two serial workers, 12h cap. Selected from earlier development results; not independent final confirmation.
- 21209002: AgentMixer residual scale, bounded replacement scale and identity control, 1M steps, development seeds 0–2. Two serial workers, 4h cap, frozen after the 583-test suite passed.
- 21209234: five matched SMAC terran 5v5 controls at 1M steps, seed 0, after 21209002. Same validated runtime and learning settings; two workers, 12h cap.
- 21209248: five matched Football 3v1 controls at 1M steps, seed 0, after 21208906. Complete-match evaluation; two workers, 12h cap.
- Earlier jobs 21196118, 21196495, 21196497, 21198089, 21198090 and 21198169 all completed. Tiny integration 21196515 and native restart validation 21198088 also completed. Failed 21196276 remains preserved.

At most four workers run concurrently, each at most 2CPUs/8GB. Exact manifests and scheduler IDs are in `experiments/jobs.jsonl`; query those IDs before adding jobs. `python scripts/report_progress.py` reads artifact progress. `python scripts/check_paper_targets.py` verifies all 45 target links (six numeric, 39 digitized), not performance.

## Execution notes

- Default sandbox has a bubblewrap temporary-directory failure; reviewed low-risk escalated commands are authorized.
- Scheduler commands use `SLURM_CONF_SERVER=slurmctl,slurmctl2`. Operate only on recorded job IDs.
- Helper agents reached an account usage limit; the main session is continuing from their saved work.
- Optional external memory saving was rejected by automatic review; durable progress stays in this repository.
- Private simulator prefixes require `PYTHONNOUSERSITE=1` and their documented native library paths.
- Do not edit a shell installer while its process is executing. The fresh Football build succeeded, but a live guard-line insertion interrupted the remaining shell reads; the retry recipe is hash recorded and held unchanged.
