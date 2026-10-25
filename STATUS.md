# Reproduction status

Goal active. Full reproduction and performance beyond the papers are not yet established.

## Verified implementation

- Isolated repository; accepted sources and unrelated reproduction preserved.
- Shared MAPPO, AR-PPO, BPPO and AgentMixer training for discrete and continuous actions.
- Temporal rollouts, terminal/time-limit bootstrapping, legal/dead-agent masks, independent simulator seeds, frozen evaluation statistics and atomic checkpoints.
- Optional observation/state and critic-target normalization, clipped/Huber critic losses, and initial Gaussian exploration controls. Defaults preserve prior behavior.
- Latest integrated suite: **480 passed, 1 CUDA-only test skipped**.
- SMACv2 and all12 MA-MuJoCo task/partition pairs pass simulator checks; allfour methods completed learning/evaluation smoke runs in both simulators.
- Predator–Prey mechanics match10240 transitions from the pinned release; allfour methods pass shared-trainer integration tests.
- Fresh private SMAC installation verified, including47 package pins and engine discovery. Football adapter/build remains unfinished; its protocol audit is saved under results/football_protocol/.

## Experiment evidence

- Source inventory covers45 benchmark tasks. Separate JSON/CSV files contain39 digitized plot targets with reading uncertainty; six AgentMixer SMAC targets are numeric source values. Missing historical metric/aggregation details remain explicit.
- Projected AR-PPO and BPPO solve both matrix games in allthree development seeds. The matched baseline also succeeds, so peer-feedback superiority is unproven.
- Experimental joint AgentMixer with faster actor updates solves Climbing in allthree development seeds at200k, first sustained greedy optima at61440–71680steps. It fails Penalty; matched factorized and zero-entropy controls do not solve Climbing. No independent confirmation yet.
- Allfive HalfCheetah controls completed1M steps (job21192905). Final mean returns: MAPPO1260.62, AR-PPO2087.35, BPPO2007.54, identity-modifier1980.50, AgentMixer1695.45. These are development results below the10M paper targets.
- Current frozen jobs:21195730 (seven500k HalfCheetah protocol controls),21195734 (ten100k Predator–Prey runs). At mostfour CPUworkers across this project.
- Full SMAC/MuJoCo budgets, Football, independent final seeds, ablations and release evidence remain outstanding.

## Execution notes

- Default sandbox has a bubblewrap temporary-directory failure; reviewed low-risk escalated commands are authorized.
- Scheduler commands use SLURM_CONF_SERVER=slurmctl,slurmctl2. Operate only on IDs in experiments/jobs.jsonl.
- Helper agents reached an account usage limit; their saved work was integrated and tested by the main session.
- Optional external memory saving was rejected by automatic review; durable progress is kept in this repository.
- Private simulator prefixes require PYTHONNOUSERSITE=1. Never change other projects' environments or jobs.
