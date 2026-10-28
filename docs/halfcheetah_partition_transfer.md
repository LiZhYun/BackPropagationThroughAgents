# HalfCheetah partition transfer

The 10M-step BPPO development result on HalfCheetah 2×3 does not establish
performance on the papers' 3×2 and 6×1 partitions. This study transfers the
selected configuration to both partitions, with matched AR-PPO controls.
It reuses the exact 19-module source snapshot of the completed 2×3 study.

| Partition | Methods | Training seed | Pilot budget |
| --- | --- | ---: | ---: |
| HalfCheetah 3×2 | BPPO, AR-PPO | 0 | 500,000 |
| HalfCheetah 6×1 | BPPO, AR-PPO | 0 | 500,000 |

Within each partition, the two configurations differ only in algorithm.
They retain 40 environments × 100 decisions per rollout, five PPO epochs,
64-unit actor/critic layers, actor/critic learning rate 0.0003, initial standard
deviation 0.365529, zero entropy coefficient, observation normalization and
the selected normalized, clipped Huber critic loss. Local observation radius
is zero, actuator coordinates are normalized, and the task horizon is 1,000.
The [adapter guide](mujoco_adapter.md) lists the original actuator groups,
including the unusual middle group in 3×2.

Each pilot evaluates ten complete deterministic episodes at initialization,
every 100k training transitions and the final 500k endpoint. Evaluation seed
1,000,000 and its interactions remain separate from training. Report the
endpoint and complete-curve area for every declared run; do not replace an
endpoint with an earlier peak. One development seed cannot establish an
algorithm ranking across seeds, and 500k is below the papers' 10M budget.

Before releasing those pilots, job **21220888** ran four native checks with
the same source and settings: 12k training transitions, three full rollouts,
and two initial/final evaluation episodes. One worker executes all four checks
serially, using 2 CPUs/8 GB with a 30-minute limit and no automatic requeue.
All four checks completed successfully in 120 seconds. The
[native audit](../experiments/hc_partitions_native_gate_20261026.json) verifies
16 raw evaluation episodes, 12 updates, exact source/configuration, local
dimensions and 1,000-step evaluation truncations. Peak memory was 510,088 KiB
(about 0.49 GiB); total CPU use was 116.37 seconds.

The [study record](../experiments/hc_partitions_transfer_study_20261026.json)
and [eight-configuration preflight](../experiments/hc_partitions_transfer_preflight_20261026.json)
fix the settings and preserve their selected source. The native check follows
matrix identity job 21220500 and precedes recurrent Prey worker 21219719_1;
the [schedule receipt](../experiments/hc_partitions_check_schedule_20261026.json)
verifies this insertion within the four-worker ceiling used at that time.

## Pilot release

Job **21220995** releases the four frozen 500k configurations, one per array
task. Each requests one CPU, 2 GiB and 30 minutes, with at most two tasks active
and no automatic requeue. The longest estimated run takes 20.87 minutes after
a 50% margin; the memory request exceeds four times the observed peak.
The [release plan](../experiments/hc_partitions_transfer_release_plan_20261026.json)
retains the phase measurements, estimate formula, source/configuration checks
and the new dispatch script's hash. The original prepared script is unchanged.

Even/odd task chains follow their matching Ant/Walker worker and precede the
pending 2×3 confirmation worker. The
[release receipt](../experiments/hc_partitions_transfer_release_schedule_20261026.json)
verifies all dependencies while held and after release. That release used
four project workers; the later [eight-worker revision](worker_schedule.md#eight-worker-schedule-2026-10-26)
leaves these A-chain dependencies unchanged. The confirmation keeps its
frozen source, seeds and configurations.

Full 10M runs, independent final training seeds, AgentMixer comparisons and
the other paper targets remain required.


## First two learning pilots, 2026-10-26

Both BPPO pilots complete500k steps. The
[partial audit](../experiments/hc_partitions_transfer500k_progress_20261026T215200Z.json)
verifies120 raw evaluation episodes,250 updates, exact selected source,
configuration/seed accounting, local dimensions and optimizer/ValueNorm counts.
All four declarations remain visible.

| Partition | BPPO final return | BPPO normalized curve area | AR-PPO |
| --- | ---: | ---: | --- |
| 3×2 | 2506.77 | 1060.96 | Not started at snapshot |
| 6×1 | 2174.01 | 934.51 | Not started at snapshot |

No baseline aggregate or algorithm advantage is inferred from this partial
study. All four pilots must finish validly before selecting the10M follow-up;
one development seed also cannot establish independent performance.


## Complete pilots and full-budget development — 2026-10-28

All four 500k pilots pass the [complete audit](../experiments/hc_partitions_transfer500k_complete_20261028.json): 240 raw episodes and 500 updates. Final AR/BPPO returns are 3180.88/2506.77 at 3×2 and 2766.72/2174.01 at 6×1. These short results do not establish the 10M target.

Job **21251961** runs both methods at both partitions with development seeds 0–2: **12 runs at 10M**. Source and optimization settings are unchanged; evaluation uses 32 complete episodes every 500k. Each task requests one CPU/4 GiB/seven hours, at most two active. The worst measured-phase projection is 3.59 hours. [Frozen study](../experiments/followup_studies_20261028.json), [resource plan](../experiments/followup_release_plan_20261028.json). The independent 2×3 BPPO confirmation failed and remains separate; these development runs do not replace it.
