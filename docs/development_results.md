# Development results

These experiments select configurations and diagnose failures. They are not
independent final-seed evidence and do not establish performance beyond either
paper. Frozen manifests retain every declared run, source hashes and raw logs;
`experiments/jobs.jsonl` identifies their scheduler jobs.

## HalfCheetah2x3 optimization controls

All seven controls completed500,000 training transitions in seed0, with the
same five deterministic evaluation episodes, agent observation radius0,
normalized actuator coordinates and1,000-step episodes. Mean evaluation returns:

| Control | Return |
| --- | ---: |
| MAPPO, critic normalization only |2698.45 |
| MAPPO, lower initial action noise only |1871.94 |
| MAPPO, combined protocol settings |2193.79 |
| AR-PPO, combined protocol settings |2191.15 |
| BPPO, combined protocol settings |2399.97 |
| AgentMixer identity modifier, combined settings |1527.66 |
| AgentMixer, combined settings |2128.55 |

The combined settings use40 environments×100 rollout steps, initial standard
deviation0.365529, normalized critic targets, clipped values and Huber loss.
The isolated controls retain8×100 rollouts and change only the named setting.
Thus the table is an optimization diagnosis, not a matched ranking across every
row. All models remain feedforward; the [source audit](mujoco_optimization_audit.md)
records differences from the original recurrent architecture and normalization.

These scores are below the roughly5,040–5,700 plotted10M-step paper endpoints
for this task. The3M follow-up applies critic normalization alone to all five
method/control families, preserving their remaining common settings. It uses
one development training seed and ten evaluation episodes. No evaluation-episode
standard deviation should be treated as uncertainty across training seeds.

Configuration: `configs/mujoco_hc2x3_protocol500k.json`.
Evidence: `results/mujoco_hc2x3_protocol500k_20261025T075442Z/manifest.json`
and its `aggregate.json`; job21195730.

## Predator–Prey initial pilot

All ten100k-step runs completed in development seed0. Final complete-episode
capture rates from20 evaluation episodes were:

| Method | Penalty0 | Penalty−1 |
| --- | ---: | ---: |
| MAPPO |95% |0% |
| AR-PPO |90% |5% |
| BPPO |100% |0% |
| AgentMixer identity modifier |100% |0% |
| AgentMixer |95% |0% |

This motivates explicit coordination controls rather than treating the harder
task as reproduced. The300k follow-up compares projected AR-PPO/BPPO actions,
factorized/joint AgentMixer, and matched faster actor/modifier rates on both
penalties. It additionally reports20 fixed200-step windows, separately from
episode returns. The released logger used100 windows. The remaining ambiguity
between that logger and the paper's plot is documented in the
[Prey protocol audit](predator_prey_protocol.md).

Configuration: `configs/prey_pilot100k.json`.
Evidence: `results/prey_pilot100k_20261025T075454Z/manifest.json`; job21195734.

## Integration evidence

All16 Football algorithm/scenario combinations complete128-step training and
full-match evaluation. Separate native checks exactly replay all four scenarios
after reseeding and in fresh instances. The5v5 native duration corresponds to
3,001 API steps. These establish availability, execution semantics and
repeatability, not learning performance.

SMAC learning starts with20k-step terran5v5 pilots to measure throughput and
stability before increasing the budget. Matrix results and their unsuccessful
controls remain in [matrix development](matrix_development.md) and the
[joint-modifier diagnosis](joint_modifier_diagnosis.md).
