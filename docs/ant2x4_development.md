# Ant 2×4: first full-budget development result

BPPO finishes 10M decisions with a mean evaluation return of **6,562.40**, above the paper curve’s digitized reading range **[4,990, 5,050]**. This is **one development training seed**, not independent confirmation. All three declared global-observation controls are complete; the other 45 runs in the broader coverage study remain visible and unfinished.

| Method | Final evaluation return | Evaluation-episode sample SD | Curve area / 10M |
| --- | ---: | ---: | ---: |
| MAPPO | 5994.52 | 1587.01 | 5322.07 |
| ARPPO | 6410.12 | 364.06 | 5165.40 |
| BPPO | 6562.40 | 105.19 | 5332.08 |

Each endpoint averages 10 complete evaluation episodes using evaluation seed 1,000,000, after training with seed 0. Episode standard deviations do not estimate uncertainty across training seeds. BPPO’s difference from AR-PPO is +152.28 at the endpoint and +166.68 in normalized curve area for this single pair; neither is evidence of a population advantage.

All methods use the same settings apart from the algorithm: Ant-v2 `2x4`, `agent_obsk=None`, 1,000-step horizon, normalized actions and observations, 40×100 rollouts, five epochs, and the shared value-normalization/clipped-Huber setup. The native profile is MuJoCo 2.1.0, Gym 0.10.8, mujoco-py 2.1.2.14, and MAMujoco commit `b212ddd74b258e7cea006ff1d642b5ffada4b99d`, with the exact 20-module frozen training source. Global observations match BPTA’s Ant configuration; this is not the partially observed AgentMixer target.

The [raw audit](../experiments/mujoco_coverage10m_progress_20261029_pm.json) verifies **3,030 evaluation episodes and 7,500 rollout updates** across all three completed runs, including source/config/seed accounting and every evaluation reduction. BPPO job `21272800_2` finishes `COMPLETED/0:0` in 12,280 seconds with 791,960 KiB peak RSS. Final native trajectories have not been independently replayed in this audit.

The paper figure shows a five-seed centerline near 9.904M decisions with unresolved smoothing and train/evaluation convention. Our primary endpoint is the final 10M evaluation. The adjacent 9.9M BPPO evaluation is 6,519.64; it is recorded as a secondary point, not selected as a replacement. The reading range is digitization precision, not a confidence interval. [Target extraction and caveats](curve_targets.md); [complete development evidence](../experiments/ant2x4_global10m_development_20261029.json).

Further training seeds and a separately registered independent comparison remain required. No new learning job is submitted by this result report, and no paper-level reproduction claim is made.
