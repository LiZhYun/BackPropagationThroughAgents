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

Before releasing those pilots, job **21220888** runs four native checks with
the same source and settings: 12k training transitions, three full rollouts,
and two initial/final evaluation episodes. One worker executes all four checks
serially, using 2 CPUs/8 GB with a 30-minute limit and no automatic requeue.
Source/configuration, raw evaluation records, update counts, runtime and
memory must pass before the longer jobs are submitted.

The [study record](../experiments/hc_partitions_transfer_study_20261026.json)
and [eight-configuration preflight](../experiments/hc_partitions_transfer_preflight_20261026.json)
fix the settings and preserve their selected source. The native check follows
matrix identity job 21220500 and precedes recurrent Prey worker 21219719_1;
the [schedule receipt](../experiments/hc_partitions_check_schedule_20261026.json)
verifies this insertion within the existing four-worker ceiling.

The 500k manifest is prepared but not submitted. Full 10M runs, independent
final training seeds, AgentMixer comparisons and the other paper targets
remain required. The separate 2×3 independent confirmation keeps its frozen
source, seeds and configurations.
