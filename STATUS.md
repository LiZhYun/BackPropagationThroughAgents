# Reproduction status

Full reproduction and performance beyond the papers are not yet established.

## Verified implementation

- Work is isolated to this repository; accepted sources and the unrelated reproduction are preserved.
- Shared MAPPO, AR-PPO, BPPO and AgentMixer training supports matrix games, SMACv2, MA-MuJoCo, Predator–Prey and Football.
- Temporal rollouts preserve terminal/time-limit observations, legal/dead-agent masks, raw policy actions and independent seeds. Evaluation preserves training RNG and normalization.
- Optional critic-target normalization, clipped/Huber losses, initial Gaussian exploration, projected discrete predecessor actions and an experimental two-agent joint modifier are explicit controls.
- Integrated suite: **536 passed, 1 CUDA-only skip**, including joint-distribution precision controls and matrix/simulator integration. Networks remain float32; optional joint probabilities use float64.
- SMACv2 and all 12 MA-MuJoCo partitions pass native checks and all-method integration. Predator–Prey matches 10,240 transitions from the pinned legacy implementation.
- Football2.10.2 builds and exactly replays all four scenarios; all 16 algorithm/scenario integration runs pass. Fresh portable setup is being verified with a private OpenGL runtime, not assumed from compute-node success.
- Prey evaluation now separately logs the released fixed-window reduction, including unfinished episodes. The historical plot-to-logger link remains unverified.

## Experiment evidence

- Source inventory covers 45 tasks: 39 digitized plot targets with reading uncertainty and six numeric AgentMixer SMAC values. Historical aggregation ambiguities remain explicit.
- Projected AR-PPO and BPPO solve both matrix games in all three development seeds. Their matched success does not establish a peer-feedback advantage.
- Joint AgentMixer with faster actor updates solves Climbing in all three development seeds at200k but fails Penalty. Matched float64-distribution controls are now queued/running; no convergence claim.
- Five HalfCheetah controls completed1M steps. Seven subsequent500k protocol controls completed; critic normalization alone reached2698.45, BPPO protocol2399.97 and AgentMixer protocol2128.55 in development seed0. These remain below the10M targets.
- Ten100k Prey runs completed: all methods learn captures without the penalty and struggle with penalty−1. Twelve300k coordination controls are running, with separate episodic and fixed-window metrics.
- SMAC20k learning pilots run in roughly four minutes per method. All five completed with0/8 final wins; this short pilot is not a10M benchmark.
- Full budgets, independent final training seeds, ablations and release evidence remain outstanding.

## Recorded active/queued jobs

- 21196038: twelve300k Predator–Prey controls, two serial workers.
- 21196113: five20k SMAC learning controls, two serial workers.
- 21196130: finish fresh Football installation verification after Prey worker1.
- 21196198/21196199: matched float64 joint-distribution matrix controls, one worker each in sequence.
- 21196118: five3M normalized-critic HalfCheetah controls, two workers after Prey and Football verification finish.

Dependencies enforce at most four workers, each at most2CPUs/8GB. Exact manifests and changes are in `experiments/jobs.jsonl`; use scheduler state before adding jobs.

## Execution notes

- Default sandbox has a bubblewrap temporary-directory failure; reviewed low-risk escalated commands are authorized.
- Scheduler commands use `SLURM_CONF_SERVER=slurmctl,slurmctl2`. Operate only on recorded job IDs.
- Helper agents reached an account usage limit; the main session is continuing from their saved work.
- Optional external memory saving was rejected by automatic review; durable progress stays in this repository.
- Private simulator prefixes require `PYTHONNOUSERSITE=1` and their documented native library paths.
- Do not edit a shell installer while its process is executing. The fresh Football build succeeded, but a live guard-line insertion interrupted the remaining shell reads; the retry recipe is hash recorded and held unchanged.
