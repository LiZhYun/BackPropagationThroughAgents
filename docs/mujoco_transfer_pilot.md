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
recorded Football job so that the combined project stays within four workers.

Job **21212297** is queued after Football **21209248**, with frozen manifest
`results/mujoco_ant_walker_transfer500k_20261026T112325Z/manifest.json`. Expanded
configurations and all source hashes were verified before submission. No
learning outcomes are available yet.

The current scheduler uses matching worker indices for this dependency; see
the [verified worker schedule](worker_schedule.md). Each successor can start
when its own preceding worker finishes, within the unchanged four-worker cap.
