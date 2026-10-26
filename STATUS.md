# Reproduction status

Full reproduction and performance beyond the papers are not yet established.

## Verified implementation

- Work is isolated to this repository; accepted sources and the unrelated reproduction are preserved.
- Shared MAPPO, AR-PPO, BPPO and AgentMixer training supports matrix games, SMACv2, MA-MuJoCo, Predator–Prey and Football.
- Temporal rollouts preserve terminal/time-limit observations, legal/dead-agent masks, raw policy actions and independent seeds. Evaluation preserves training RNG and normalization.
- Optional critic-target normalization, clipped/Huber losses, initial Gaussian exploration, projected discrete predecessor actions and an experimental two-agent joint modifier are explicit controls.
- Clean-checkout integrated suite: **693 passed, 1 CUDA-only skip** with the locked Python 3.11.15 / PyTorch 2.6 CPU environment and no native simulator packages. Five additional progress-report portability checks pass. Networks remain float32; optional joint probabilities use float64.
- A separately installed wheel includes all 23 modules, README and MIT notice. Both README learning examples complete their budgets outside the source tree; a 16-step recurrent Prey CLI check exactly replays its checkpoint evaluation. See [the clean-install record](docs/clean_checkout.md); these checks do not establish independent performance.
- Optional private GRU actors support all four discrete methods and the identity-modifier control. Incoming memory is stored explicitly, gradients span one decision, and termination/truncation resets are verified. All 33 recurrent checks pass, including checkpoint replay. The shared critic remains feed-forward; native SMAC integration is queued.
- Learning-curve reports validate source/configuration/budget/protocol consistency and retain every declared seed. Full-budget AUC, mean/median, seed SD and conditional bootstrap intervals are available; all 48 completed native curves pass validation. Standalone data and PNG/PDF figures are in the development report.
- Continuous BPPO offers an explicit behavior-noise control with analytical gradient checks. Three-seed 500k results are mixed across optimization settings; the fresh-noise default is retained.
- Continuous AgentMixer now has an explicit bounded replacement-scale control. It preserves exact local/central mean equality, undetached local features and shared network initialization; unsupported identity combinations are rejected. Policy, gradient and checkpoint-replay tests pass.
- Discrete AgentMixer now exposes the existing optional smooth gap-scale bound in simulator configurations. Fourteen new checks cover initialization, mode invariants, invalid controls, real Prey training and exact episode/window checkpoint replay for both factorized and joint variants. The full suite passes; defaults remain unchanged.
- All 96 integration runs across 15 MuJoCo task/observation variants and nine SMACv2 scenarios pass with four algorithms. Predator–Prey matches 10,240 transitions from the pinned legacy implementation.
- Football2.10.2 builds and exactly replays all four scenarios; all 16 algorithm/scenario integration runs pass. Fresh portable setup passes in a second private prefix; all four native replay traces match the original build. Private OpenGL also fixes login-node imports.
- The author confirmed evaluation return for the Prey and Football 5v5 figures on 2026-10-25; see `docs/author_clarifications.md`. On 2026-10-26 the author selected the released 100×200-decision reduction, including unfinished episodes, for the primary Prey comparison. The historical figure-to-logger mapping and Football reduction remain unverified.
- `scripts/evaluate_simulator.py` restores saved policy weights and observation moments, checks source hashes, and labels checkpoint step/training status. It exactly replays a completed native Prey evaluation with its frozen source; all eight method/action-space combinations also replay in integration tests.
- Native SMAC lifecycle validation passed 512 transitions and 12 planned episode-boundary restarts. All five 200k training retries now complete beyond the previous 129,536-step failure, logging one planned restart per training environment.

## Experiment evidence

- The artifact-coverage reporter indexes 444 runs from 41 manifests, including all 45 targets, with no source/configuration errors in the recorded snapshot. Only four targets have exact paper-budget method artifacts; this includes unsuccessful outcomes and establishes no reproduction success. Sixteen focused report checks pass. See `docs/benchmark_coverage.md`.
- Source inventory covers 45 tasks: 39 digitized plot targets with reading uncertainty and six numeric AgentMixer SMAC values. Historical aggregation ambiguities remain explicit.
- Projected AR-PPO and BPPO solve both matrix games in all three development seeds. Their matched success does not establish a peer-feedback advantage.
- A 40-run independent matrix configuration check uses ten unused seeds at100k; a separate12-run development extension uses the1M paper budget. Both were frozen before launch. The validation reporter verifies source/configuration hashes, complete seed sets, exact budgets and probability-derived rewards; neither study has completed results yet.
- Joint AgentMixer with faster actor updates solves Climbing in all three development seeds at200k but fails Penalty. The faster float64 control solves2/3 seeds in each game; the slower float64 control solves0/3 in both. No robust convergence claim.
- All 48 runs in six newer development suites completed. See `docs/development_results_20261026.md` and its raw CSV with summary/manifest hashes.
- At 3M HalfCheetah steps, combined-protocol BPPO reaches 5228.01 versus AR-PPO 4739.43 (+10.31%) in development seed 0. AgentMixer reaches 2411.14 versus identity 2975.45. Under normalization alone BPPO reaches only 599.40; these settings change several optimization choices together.
- Completed three-seed scale controls at 1M give residual AgentMixer 2722.35±291.72 versus bounded replacement 2546.83±60.04; normalized areas are 1990.36±203.79 and 1609.44±90.11. The bounded map does not improve the mean in this setting; residual remains the default. All nine runs complete: identity reaches 2856.95±566.15 with area 1588.42±248.00. Residual has higher area in all three seeds but wins the endpoint only in seed 0.
- Combined-protocol 3M normalized curve areas are BPPO 3580.56 versus AR-PPO 3290.08 (+8.83%), and AgentMixer 2311.47 versus identity 2326.86 (−0.66%), all in development seed 0. These are area summaries, not independent final results.
- Three-seed BPPO fresh/behavior-noise means at 500k are 1844.05/2853.01 with the small batch and 2547.40/1959.81 with the combined protocol. No estimator wins consistently across settings.
- All sixteen Prey 1M controls complete. Penalty −1 remains unsolved: every fixed-window reward rate is negative, even the higher-entropy condition with 9/20 captures. The author-selected comparison uses the released 100×200-decision reduction. The best zero-penalty rate, 2.142, also remains below the paper reading range [2.19, 2.23].
- Frozen-source Prey replay reproduces all sixteen policies’ original per-episode evaluation returns. Small-batch AgentMixer has a large training/greedy-execution gap: recent stochastic training captures 83–100%, versus 0/20 greedy evaluation captures, with many blocked moves. `docs/prey_behavior_diagnosis.md` records the scope; the full 1,800-episode frozen-policy check (21210161) confirms the gap: central sampling captures 85/100 and 98/100 in the two small-batch policies, versus 0/100 greedy local and 3/100 or 0/100 stochastic local. All visited local/central greedy actions agree exactly. Central sampling is diagnostic only.
- All five Football 100k controls finish with 0/20 wins. Positive shaped returns do not establish task success.
- Corrected SMAC 200k final win counts are MAPPO 1/32, AR-PPO 4/32, BPPO 0/32, AgentMixer 2/32, identity 0/32. The original failed run is retained.
- The first two SMAC1M baselines complete: MAPPO return9.7227 and4/32 wins; AR-PPO return8.5918 and0/32 wins. BPPO and the modifier comparisons remain incomplete. The release audit identifies a256-versus10000 rollout-batch difference and enabled recurrent actors in the released launchers. A named reconstruction adds actor history; larger-batch learning awaits native validation and timing.
- Exact enumeration of all six fast-float64 matrix checkpoints finds inactive numerical floors. The two failed seeds have saturated scales and small useful recovery gradients after switching to losing modes; `docs/matrix_final_diagnosis.md` records saved-Adam controls.
- The first two BPPO HalfCheetah runs finish 10M transitions with returns 5843.80 and 6046.91 over 32 evaluations each. Seed 2 and matched AR-PPO runs remain unfinished; the snapshot reporter refuses group aggregation. Both point values exceed the plot reading range [5660, 5740], but development selection and historical comparison qualifications prevent a reproduction claim.
- Frozen Prey scale diagnosis reconstructs every action in two recorded greedy trajectories per AgentMixer policy. Floors are inactive and hard clamps mostly inactive; small-batch policies have large central/local probability differences. `docs/prey_modifier_control.md` defines the matched bounded-range control.
- Full budgets, independent final training seeds, ablations and release evidence remain outstanding.

## Recorded active/queued jobs

- 21208906: BPPO and matched AR-PPO on HalfCheetah 2×3, 10M steps, development seeds 0–2, 32 deterministic evaluation episodes. Two serial workers, 12h cap. Selected from earlier development results; not independent final confirmation.
- 21209002: completed all nine AgentMixer residual/bounded/identity 1M controls, seeds 0–2; both workers exit 0. Final CSV and validated curves are linked in the development report.
- 21209234: five matched SMAC terran 5v5 controls at 1M steps, seed 0, running after both 21209002 and 21210161 completed. Same validated runtime and learning settings; two workers, 12h cap.
- 21209248: five matched Football 3v1 controls at 1M steps, seed 0, after 21208906. Complete-match evaluation; two workers, 12h cap.
- 21210161: completed all 18 frozen-policy Prey sampling conditions (1,800 episodes); one CPU/4GB. Both dependencies have completed and SMAC is running.
- 21211006: nine matched AgentMixer entropy controls at 1M steps, seeds 0–2, coefficient 0.01 for residual/bounded/identity variants, after native recurrent integration (21213860); two workers, 4h cap.
- 21211813: six matrix joint AgentMixer controls, actor LR 0.0125 at the unchanged 200k budget, after the matching SMAC worker (21209234); two workers at 1CPU/4GB, 30-minute cap. Frozen source is identical to the preceding fast-float64 experiment; only actor LR differs.
- 21212197: eight matched Prey 1M controls, factorized/joint modifiers with default or smooth 0.1–10 scale range, both penalties, seed 0; after entropy job 21211006. Two workers, 4h cap, frozen after all 621 tests passed.
- 21212297: eleven Ant 2×4/Walker 2×3 transfer pilots, 500k, seed 0; after Football 21209248. Ant observation settings stay distinct for the two papers; matched method controls, canonical 1,000-step horizons. Two workers, 8h cap.
- 21213329:40 projected AR-PPO/BPPO matrix validation runs,100k, seeds100–109, after21211813; two1CPU/4GB workers,30-minute cap.
- 21213330:12 projected AR-PPO/BPPO matrix development runs,1M, seeds0–2, after21213329; two1CPU/4GB workers,30-minute cap. The entropy schedule stretches with budget and is not pooled with the100k study.
- 21213860:five recurrent native SMAC integration runs,10k each, using2×5000 rollouts; after21213330. Two2CPU/8GB workers,1h cap. The ten-run `smac_terran5_batch1m` learning plan is frozen but unsubmitted pending correctness, runtime and memory checks.
- Earlier jobs 21196118, 21196495, 21196497, 21198089, 21198090 and 21198169 all completed. Tiny integration 21196515 and native restart validation 21198088 also completed. Failed 21196276 remains preserved.

Queued worker `i` waits for predecessor worker `i`. The two chains are BPPO/AR → Football → Ant/Walker, and SMAC → joint-matrix control → independent matrix check →1M matrix extension → recurrent native check → entropy → Prey, each with indices0 and1. Every changed dependency was verified; see `docs/worker_schedule.md`. These are recorded snapshots, not live scheduler claims.

At most four workers run concurrently, each at most 2CPUs/8GB. Exact manifests and scheduler IDs are in `experiments/jobs.jsonl`; query those IDs before adding jobs. `python scripts/report_progress.py` reads artifact progress. `python scripts/check_paper_targets.py` verifies all 45 target links (six numeric, 39 digitized), not performance.

## Execution notes

- Job IDs and cluster dependencies describe the local development campaign. Use the [simulator guide](docs/simulator_training.md) to run a frozen configuration on another machine.
- Source checkouts exclude local results and reference archives. Missing experiment manifests mean unavailable evidence, not failed training runs.
- Private simulator prefixes require `PYTHONNOUSERSITE=1` and the native library paths in their installation guides. Keep environments and source snapshots fixed while a run is active.
- The repository retains upstream MIT attribution and provides accepted-paper citations. Packaging readiness does not establish benchmark reproduction; performance and independent confirmation remain required.
