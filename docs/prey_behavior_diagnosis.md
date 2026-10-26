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
