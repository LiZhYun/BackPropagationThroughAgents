# Experiment tracker

| ID | Stage | Purpose | System | Status | Evidence |
|---|---|---|---|---|---|
| R001 | M0 | Numerical correctness | Shared core + both methods | PASS | 42 tests passed, 2026-10-25 |
| R002 | M0 | Pipeline smoke | MAPPO climbing, 2,048 steps | PASS | results/smoke/mappo_20261025 |
| R003 | M0 | Pipeline smoke | AR-PPO, BPPO, AgentMixer | PASS | results/smoke/{algorithm}_20261025; 2,048 steps each |
| R004 | M1 | Matrix development | Four algorithms × two games × seeds 0–2 | FAIL learning gate | Job21192344; all24 completed, zero final greedy optima |
| R005 | M2 | Simulator availability | Multi-Agent MuJoCo, SMACv2 | PASS smoke only | .native/mujoco-smoke.json; results/smoke/smac_21192452.json |
| R006 | M3 | Full shared benchmarks | Both papers | TODO | configs/paper_targets.json |
| R007 | M3 | Paper-specific suites | Football, Predator-Prey | TODO | configs/paper_targets.json |
| R008 | M4 | Independent confirmation | Frozen final configurations | TODO | At least five disjoint final training seeds |
| R009 | M0 | Continuous and temporal correctness | Gaussian policies, GAE, masks, raw actions | PASS targeted checks | Continuous BPPO14, AgentMixer9, independent3; runner/BPPO54 tests |
| R010 | M1 | Stable normalization rerun | Same24 matrix development controls | RUNNING | Job21192446; results/matrix_normfix_20261024T224628Z; non-AgentMixer outcomes still suboptimal |
| R011 | M1 | Exploration and projection controls | Shared ARPPO/BPPO projection; bounded AgentMixer scales | PREPARING | configs/matrix_*entropy*.json; no final-seed selection |
