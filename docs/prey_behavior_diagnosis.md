# Predator–Prey behavior diagnosis — 2026-10-26

The final 1M-step policies were replayed with their frozen training source on the original twenty evaluation episodes. All sixteen conditions exactly reproduce the saved per-episode returns within 1e-12 and the saved capture fractions. Coordinates were logged only for diagnosis, after actions were selected; local actors received the same 28-dimensional observations as before. No checkpoint or training process was changed.

The [raw diagnostic CSV](../experiments/prey_behavior_20261026.csv) includes all sixteen conditions and checkpoint hashes. The table below focuses on penalty −1. Training statistics describe the most recent 100 completed episodes; they are not a frozen-policy evaluation or a matched sample of evaluation starts.

| Variant | Recent training return | Recent training captures | Greedy evaluation return | Greedy captures | Stationary decisions: agent 0 / 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| agentmixer_factorized_small | 5.1931 | 83% | -1.2800 | 0% | 79.8% / 91.3% |
| agentmixer_joint_small | 9.7589 | 100% | -1.2800 | 0% | 50.5% / 77.9% |
| agentmixer_factorized_batch | -0.4674 | 19% | -4.9190 | 5% | 29.7% / 26.3% |
| agentmixer_joint_batch | -0.6062 | 16% | -24.8475 | 20% | 96.6% / 97.8% |
| agentmixer_factorized_batch_entropy | -4.4706 | 52% | -3.4625 | 45% | 41.9% / 9.8% |
| agentmixer_joint_batch_float64 | -1.2995 | 9% | -7.1550 | 10% | 85.4% / 91.2% |
| arppo_projected_batch | -1.5390 | 10% | -0.1420 | 15% | 98.3% / 98.3% |
| bppo_projected_batch | -1.6870 | 12% | -1.3065 | 10% | 91.3% / 93.7% |

The small-batch factorized and joint modifiers have positive recent training return but zero greedy evaluation captures. Their evaluated actions frequently leave predator positions unchanged; many are non-NOOP attempts blocked by a wall or occupied cell. The local policies are not deterministic distributions: their mean per-agent categorical entropies on these trajectories are approximately 1.13/0.75 nats for the factorized policy and 0.46/0.65 for the joint policy. Greedy execution removes that stochastic movement.

This is consistent with the [matrix diagnosis](joint_modifier_diagnosis.md): exact agreement of local and central modes does not guarantee that stochastic centralized return is realized by repeatedly selecting those modes. It does not establish the mechanism by itself. A frozen-policy comparison on shared fresh episode seeds is needed to separate centralized sampling, local sampling and greedy execution from training-policy drift.

The original evaluation metric remains unchanged. Centralized sampling is a diagnostic and cannot replace local execution when claiming reproduction. New evaluation episodes are not new independent training seeds.

Artifacts: `results/prey_behavior_diagnosis_20261026/probe.py`, `summary.json` and `trajectories.jsonl`; the summary records the manifest/probe hashes. The frozen source is `results/prey_batch1m_20261025T090907Z/source`.

The expanded comparison uses 100 separately seeded fresh episodes shared across execution conditions, with diagnostic seed root 2,000,001. A local attempt completed the first greedy condition (factorized small-batch): 0/100 captures, return −1.28 in every episode. It was then interrupted to move the larger evaluation to the batch queue; its partial output and reason remain in `sampling_login_attempt.json`. Job 21210161 runs the unchanged script after an AgentMixer worker releases its slot. The batch comparison has now completed; its results follow.


## Completed frozen-policy sampling comparison

Job 21210161 completed all eighteen conditions: six final policies × three execution modes × 100 fresh episode seeds, or 1,800 complete episodes. Every execution mode uses the same 100 separately seeded initial environments. Local and centralized greedy actions were checked equal at every visited state. The [raw CSV](../experiments/prey_sampling_20261026.csv) retains all metrics, episode counts, checkpoint hashes and within-evaluation episode SDs; those SDs do not describe uncertainty across training seeds.

| Variant | Greedy local return | Local-sampling return | Central-sampling return | Captures / 100: greedy, local sample, central sample |
| --- | ---: | ---: | ---: | ---: |
| agentmixer_factorized_small | -1.2800 | -10.7339 | 5.4541 | 0, 3, 85 |
| agentmixer_joint_small | -1.2800 | -1.4700 | 9.5282 | 0, 0, 98 |
| agentmixer_factorized_batch | -2.3797 | -4.0992 | -0.3381 | 22, 29, 23 |
| agentmixer_joint_batch | -7.9322 | -3.2102 | -1.8653 | 9, 9, 7 |
| agentmixer_factorized_batch_entropy | -1.2821 | -9.0171 | -0.9030 | 59, 53, 70 |
| agentmixer_joint_batch_float64 | -21.1806 | -2.4684 | -1.1500 | 8, 20, 16 |

The successful stochastic training behavior survives a fresh frozen-policy check: the small-batch factorized modifier obtains 85/100 captures with centralized sampling, and the joint modifier 98/100. Their greedy local execution captures 0/100. Local sampling also fails to recover the centralized result (3/100 and 0/100). Thus changing greedy evaluation to stochastic local sampling would not resolve this observed gap, and centralized evaluation cannot be substituted for the paper’s decentralized execution.

This isolates a deployment gap under the current corrected parameterizations. It is not an infrastructure failure, a changed model during evaluation, or a numerical violation of mode agreement on the visited states. It does not prove whether actor optimization, local information, parameterization barriers or their interaction causes the failure. The known matrix fixed-mode barriers remain a concrete next diagnostic; stable decentralized performance is still a release requirement.

Complete artifact: `results/prey_behavior_diagnosis_20261026/sampling_summary.json`, with per-episode data, frozen-source verification, seed scheme, manifest hash and the exact probe-script hash. Central sampling is diagnostic only; no benchmark-success claim is made from its higher scores.

The [modifier-range diagnosis and control](prey_modifier_control.md) reconstructs
the first two saved greedy trajectories for all twelve AgentMixer policies. It
finds inactive numerical floors and mostly inactive hard clamps, but a large
central/local probability difference in the small-batch policies. The proposed
bound tests restricted modification; it is not presented as a saturation fix.
