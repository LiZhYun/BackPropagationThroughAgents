# Ant and Walker transfer pilots

HalfCheetah 2×3 is the first MuJoCo task with longer learning evidence in this
repository. It is not sufficient evidence for the papers' other partitions or
locomotion tasks. `configs/mujoco_ant_walker_transfer500k.json` specifies eleven
matched development runs on three additional task/observation protocols.

| Task and observations | Methods | Related paper targets |
| --- | --- | --- |
| Ant 2×4, `agent_obsk=null` | MAPPO, AR-PPO, BPPO | BPTA Ant 2×4 |
| Ant 2×4, `agent_obsk=0` | MAPPO, AgentMixer, identity modifier | AgentMixer Ant 2×4 |
| Walker 2×3, `agent_obsk=0` | MAPPO, AR-PPO, BPPO, AgentMixer, identity modifier | Both papers' Walker 2×3 |

The two Ant protocols intentionally differ because the papers use different
observation settings. Compare methods within a row; an Ant score difference
between rows does not isolate an algorithm effect. These task/observation
adapters have already passed the all-protocol native integration suite.
The integration tests used a shortened horizon; these learning pilots use the
canonical 1,000-step task limit.

All runs use training seed 0 and exactly 500,000 joint environment transitions.
The transferred settings are 40 environments × 100-step rollouts, two 64-unit
tanh layers, initial Gaussian standard deviation 0.365529, actor/critic LR
0.0003, modifier LR 0.00003, five PPO epochs, maximum gradient norm 10, entropy
coefficient zero, training-feature observation normalization, and normalized
critic targets with clipping 0.2 and Huber delta 10. Actions use the existing
normalized actuator coordinates; rewards are unchanged. AgentMixer retains
the residual continuous scale default, and BPPO retains fresh peer noise.

Ten complete deterministic evaluation episodes use the separate seed 1,000,000,
at initialization, every 100k training transitions, and at the final endpoint.
Evaluation interactions do not count toward the training budget. Because the
rollout batch has 4,000 transitions, these thresholds fall exactly on rollout
boundaries. Per-episode returns, true terminations and time-limit truncations
remain in the raw logs. All methods keep their specified local or ordered
execution rules.

These settings were selected from HalfCheetah development results. The pilot
tests transfer and measures throughput; it cannot establish a 10M result on
Ant or Walker, an algorithm ranking across training seeds, or independent
confirmation. Retain every declared run, including failures, before choosing
follow-up budgets. The full task inventory, other partitions, Humanoid tasks,
matched ablations and at least five disjoint final training seeds remain part
of the release objective.

The plan is prepared with `scripts/prepare_simulator.py`, which freezes source,
expanded configurations and a manifest. It uses at most two serial workers,
each 2CPUs/8GB, with an eight-hour worker limit. Submission must follow the
recorded Football job. The current project ceiling is eight serial workers;
the original submission used four.

Job **21212297** follows the matching Football integration workers **21219475**,
which follow the earlier Football pilot. It uses frozen manifest
`results/mujoco_ant_walker_transfer500k_20261026T112325Z/manifest.json`. Expanded
configurations and all source hashes were verified before submission. The
complete eleven-run comparison remains unfinished.

The current scheduler uses matching worker indices for this dependency; see
the [verified worker schedule](worker_schedule.md). Each successor can start
when its own preceding worker finishes. The later eight-worker revision
leaves these two chains unchanged.


## First completed transfer run

The [partial audit](../experiments/mujoco_transfer500k_progress_20261026.json)
verifies the Ant 2×4 global-observation AR-PPO baseline at 500k decisions:
60 raw evaluation episodes, 125 finite rollout updates, exact frozen source,
configuration, seed derivation and training/evaluation accounting. Its final
10-episode mean return is **877.68**, with normalized curve area **931.33**.
The run takes 792.92 seconds. These are one development seed's pilot results;
they cannot establish performance at the paper's 10M budget or a method ranking.

At this earlier snapshot, worker 1 was continuing and worker 0 was waiting.
The newer audit below supersedes those progress counts; all eleven
declarations remain visible.

The separate [HalfCheetah partition study](halfcheetah_partition_transfer.md)
extends the selected BPPO/AR settings to3×2 and6×1, beginning with a native
full-rollout gate. It does not replace these Ant/Walker controls.


## Eight completed pilots, 2026-10-26

The [new raw-log snapshot](../experiments/mujoco_transfer500k_progress_20261026T215200Z.json)
verifies eight of eleven runs: 480 complete evaluation episodes and1,000 finite
rollout updates, frozen source/configurations, optimizer and ValueNorm counts,
seed derivation and exact500k budgets. Earlier audits remain unchanged.

| Condition | Final return | Normalized curve area | Status |
| --- | ---: | ---: | --- |
| ant2x4_global_mappo | 993.14 | 992.35 | Complete |
| ant2x4_global_arppo | 877.68 | 931.33 | Complete |
| ant2x4_global_bppo | 989.34 | 951.57 | Complete |
| ant2x4_local_mappo | 993.01 | 972.82 | Complete |
| ant2x4_local_agentmixer | 978.27 | 970.45 | Complete |
| ant2x4_local_identity | 1000.55 | 952.66 | Complete |
| walker2x3_local_mappo | — | — | unfinished |
| walker2x3_local_arppo | 365.30 | 342.39 | Complete |
| walker2x3_local_bppo | — | — | not started |
| walker2x3_local_agentmixer | 257.48 | 240.76 | Complete |
| walker2x3_local_identity | — | — | not started |

The six Ant endpoints remain between877.68 and1000.55; the two completed Walker
endpoints are365.30 and257.48. These are single-seed observations below10M.
They do not establish transfer success or an algorithm advantage. The
unfinished Walker controls remain required before follow-up selection; no
partial-study group aggregate or earlier-peak replacement is used.


## All eleven pilots complete — 2026-10-28

The [complete audit](../experiments/mujoco_transfer500k_complete_20261028.json) verifies 660 raw episodes and 1,375 updates. Ant2×4 global final MAPPO/AR/BPPO returns are 993.14/877.68/989.34; local MAPPO/AgentMixer/identity are 993.01/978.27/1000.55. Walker2×3 final MAPPO/AR/BPPO/AgentMixer/identity returns are 251.89/365.30/251.73/257.48/292.51. All declared methods are retained. These seed-0 500k pilots remain far short of the 10M and independent-seed evidence required.
