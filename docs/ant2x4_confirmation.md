# Ant 2×4: independent confirmation

Ten runs are registered and queued as **21282468**: BPPO and matched AR-PPO on five unused training seeds each. No confirmation run has started at registration. The selected development result, BPPO **6,562.40** at 10M, comes from seed 0 only and remains separate. [Development evidence](ant2x4_development.md).

| Choice | Frozen value |
| --- | --- |
| Task | Ant-v2, 2×4 agents, global observations (`agent_obsk=None`), 1,000-step horizon |
| Methods | BPPO and AR-PPO, matched apart from algorithm |
| Training seeds | 6000–6004, shared as pairs between methods |
| Evaluation seed | 4,000,000; ten deterministic complete episodes per point |
| Budget and checkpoints | 10M decisions; initial evaluation and every 100k; final scheduled checkpoint is primary |
| Training | Selected 20-module source, 40×100 rollouts, five epochs, original selected optimizer and normalization settings |
| Uncertainty | Whole-training-seed 95% percentile bootstrap, 10,000 resamples, private RNG 1729 |

Only the training and evaluation seeds differ from the selected development configurations. The inventory includes **903 earlier declarations**, including 42 prepared but unsubmitted runs. All five training seeds and the evaluation seed are unused there. The 200 derived training-environment streams are unique and disjoint from all 550 prior distinct simulator training streams and recorded evaluation seeds. Both methods deliberately share the new streams for the paired comparison.

The [registered plan](../experiments/ant2x4_confirmation10m_plan_20261029.json) fixes selection, source hashes, all ten resolved configurations and reporting before submission. Selection followed one development seed; it is a reason to test fresh seeds, not an estimate of reliability. No new configuration will be chosen from these final seeds, and unsuccessful runs will remain recorded without seed replacement or pooling with development data.

## What would count as support

All ten declared runs must be valid at 10M. The reconstruction comparison requires both BPPO's five-seed mean and the lower 95% bootstrap mean endpoint to exceed **5,050**, the upper digitized reading bound. A point mean above that bound without interval support is weaker evidence. The historical figure averages five training seeds near 9.904M; its evaluation convention and smoothing remain unresolved. The reading range is not a confidence interval.

The peer-feedback comparison is separate. Support requires the lower 95% bootstrap mean endpoint to be positive for **both** paired BPPO-minus-AR-PPO final return and normalized learning-curve area. A paper-figure improvement can therefore coexist with no demonstrated advantage over AR-PPO. The earlier failed HalfCheetah 2×3 confirmation remains unchanged.

## Execution and final verification

The array has at most two workers, each using one CPU, 4 GiB and seven hours. It waits for the complete Football → Prey native checks → Prey auditor chain, ending at `21282210`. This dependency reserves capacity; Ant does not require a positive Prey result. Existing jobs and dependencies are unchanged. [Verified submission](../experiments/ant2x4_confirmation10m_schedule_20261029.json).

Job **21282782** follows the entire Ant array. It uses a separately frozen reporter and the original training source for native replay. It first audits every declared seed, then replays all ten final checkpoints with their original ten-episode evaluation protocol, checks exact episode-record equality, and verifies that training artifacts remain unchanged. Missing or invalid runs stop acceptance; complete negative results are retained. No follow-up training is submitted automatically.

The [audit preflight](../experiments/ant2x4_confirmation10m_audit_preflight_20261029.json) passes on the previous complete negative confirmation, rejects the unstarted Ant study and six corrupted report cases, and checks the actual Ant BPPO/AR evaluation schema. This is preparation evidence; native replays are pending. [Frozen audit specification](../experiments/ant2x4_confirmation10m_audit_plan_20261029_v2.json); [collector schedule](../experiments/ant2x4_confirmation10m_collector_20261029.json).

The other 45 declarations in the MuJoCo coverage study remain in that study. This confirmation concerns one canonical BPTA target, not the partially observed AgentMixer Ant target or completion of the two papers.
