# Reproduction status

Full reproduction and performance beyond the papers are not yet established.

## Verified implementation

- Work is isolated to this repository; accepted sources and the unrelated reproduction are preserved.
- Shared MAPPO, AR-PPO, BPPO and AgentMixer training supports matrix games, SMACv2, MA-MuJoCo, Predator–Prey and Football.
- Temporal rollouts preserve terminal/time-limit observations, legal/dead-agent masks, raw policy actions and independent seeds. Evaluation preserves training RNG and normalization.
- Optional critic-target normalization, clipped/Huber losses, initial Gaussian exploration, projected discrete predecessor actions and an experimental two-agent joint modifier are explicit controls.
- Integrated suite: **560 passed, 1 CUDA-only skip**, including checkpoint replay, the SMAC clock guard, recorded Gaussian-noise control, joint-distribution precision controls and matrix/simulator integration. Networks remain float32; optional joint probabilities use float64.
- Continuous BPPO offers an explicit behavior-noise control that preserves the recorded action/noise coupling across update epochs; the existing fresh-noise default is unchanged. An analytic two-epoch scale-gradient check and full trainer integration pass; performance remains to be measured.
- All 96 integration runs across 15 MuJoCo task/observation variants and nine SMACv2 scenarios pass with four algorithms. Predator–Prey matches 10,240 transitions from the pinned legacy implementation.
- Football2.10.2 builds and exactly replays all four scenarios; all 16 algorithm/scenario integration runs pass. Fresh portable setup passes in a second private prefix; all four native replay traces match the original build. Private OpenGL also fixes login-node imports.
- The author confirmed evaluation return for the Prey and Football 5v5 figures on 2026-10-25; see `docs/author_clarifications.md`. Prey separately logs the released fixed-window reduction, including unfinished episodes. Exact historical normalization and episode handling remain unresolved.
- `scripts/evaluate_simulator.py` restores saved policy weights and observation moments, checks source hashes, and labels checkpoint step/training status. It exactly replays a completed native Prey evaluation with its frozen source; all eight method/action-space combinations also replay in integration tests.
- Native SMAC lifecycle validation passed512 transitions and12 planned episode-boundary restarts; scheduler exit0. Longer training beyond the previous failure point remains to be checked.

## Experiment evidence

- Source inventory covers 45 tasks: 39 digitized plot targets with reading uncertainty and six numeric AgentMixer SMAC values. Historical aggregation ambiguities remain explicit.
- Projected AR-PPO and BPPO solve both matrix games in all three development seeds. Their matched success does not establish a peer-feedback advantage.
- Joint AgentMixer with faster actor updates solves Climbing in all three development seeds at200k but fails Penalty. The faster float64 control solves2/3 seeds in each game; the slower float64 control solves0/3 in both. No robust convergence claim.
- Five HalfCheetah controls completed1M steps and seven protocol controls completed500k. At3M, normalized-critic MAPPO reaches3045.06, AR-PPO3599.84, BPPO599.40 and AgentMixer2607.24; its identity control continues. BPPO learns poorly throughout this setting, with no large median KL or gradient spike. A longer matched run of the better500k protocol settings is queued.
- Ten100k Prey runs completed: all methods learn captures without the penalty and struggle with penalty−1. All twelve300k coordination controls completed; penalty−1 still has0–5% capture. Both episodic and fixed-window metrics are retained.
- All five SMAC20k pilots completed with0/8 final wins. Both initial200k workers failed after129,536 recorded steps when SMAC restarted mid-transition, consistent with the native cumulative game-clock limit. Native validation of the episode-boundary restart passed; a fresh matched retry has been released. Failed artifacts remain retained.
- Full budgets, independent final training seeds, ablations and release evidence remain outstanding.

## Recorded active/queued jobs

- 21196118: five3M normalized-critic HalfCheetah controls, two serial workers, running.
- 21196276: SMAC200k failed; MAPPO/AR-PPO stopped and the remaining three methods did not start.
- 21196515: all96 tiny integration runs completed.
- 21196495: sixteen1M Prey controls running; all eight zero-penalty runs completed, penalty−1 runs continue.
- 21198088: native SMAC restart validation completed with12 planned restarts.
- 21196497: five100k Football3v1 controls, after both HalfCheetah and Prey finish; this preserves the worker cap during the earlier SMAC start.
- 21198089: five200k SMAC retries; worker0 released after successful native validation, worker1 after Prey worker0. The gate's scheduler exit0 is recorded before releasing either dependency.
- 21198090: five3M HalfCheetah protocol controls, after the SMAC retries.
- 21198169: twelve500k continuous BPPO noise controls, two optimization settings×two estimators×three development seeds, after Football.

Dependencies enforce at most four workers, each at most2CPUs/8GB. Exact manifests and changes are in `experiments/jobs.jsonl`; use scheduler state before adding jobs. `python scripts/report_progress.py` reads artifact progress without querying or modifying jobs. `python scripts/check_paper_targets.py` verifies all45 target links (six numeric,39 digitized) without treating estimates as exact results.

## Execution notes

- Default sandbox has a bubblewrap temporary-directory failure; reviewed low-risk escalated commands are authorized.
- Scheduler commands use `SLURM_CONF_SERVER=slurmctl,slurmctl2`. Operate only on recorded job IDs.
- Helper agents reached an account usage limit; the main session is continuing from their saved work.
- Optional external memory saving was rejected by automatic review; durable progress stays in this repository.
- Private simulator prefixes require `PYTHONNOUSERSITE=1` and their documented native library paths.
- Do not edit a shell installer while its process is executing. The fresh Football build succeeded, but a live guard-line insertion interrupted the remaining shell reads; the retry recipe is hash recorded and held unchanged.
