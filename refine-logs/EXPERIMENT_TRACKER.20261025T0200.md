# Experiment tracker

| ID | Stage | Purpose | System | Status | Evidence |
|---|---|---|---|---|---|
| R001 | M0 | Numerical correctness | Shared core + both methods | PASS | 42 tests passed, 2026-10-25 |
| R002 | M0 | Pipeline smoke | MAPPO climbing, 2,048 steps | PASS | results/smoke/mappo_20261025 |
| R003 | M0 | Pipeline smoke | AR-PPO, BPPO, AgentMixer | TODO | |
| R004 | M1 | Matrix development | Four algorithms × two games × seeds 0–2 | TODO | Immutable snapshot and job IDs required |
| R005 | M2 | Simulator correctness | Multi-Agent MuJoCo, SMACv2 | TODO | |
| R006 | M3 | Full shared benchmarks | Both papers | TODO | configs/paper_targets.json |
| R007 | M3 | Paper-specific suites | Football, Predator-Prey | TODO | configs/paper_targets.json |
| R008 | M4 | Independent confirmation | Frozen final configurations | TODO | At least five disjoint final training seeds |
