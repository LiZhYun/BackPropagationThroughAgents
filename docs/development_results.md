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

## Completed coordination follow-up

At300k, all six Prey controls still fail to learn reliable capture with penalty−1:
the projected AR-PPO evaluation captures1/20 episodes and the other controls0/20.
The default AgentMixer variants settle on avoiding the penalty, with−0.01 reward
per step. Faster actor updates and predecessor-action projection do not solve
this setting. Without the penalty, projected AR-PPO/BPPO reach fixed-window
rates2.1975/2.185, factorized AgentMixer1.9875, and joint AgentMixer1.76. These
use20 windows and one training seed, not the released100-window final protocol.

The next1M controls compare8×64 and the released50×200 rollout batch size,
retaining the redesigned feedforward model. Additional entropy and distribution
precision conditions are separately named. Every condition uses both penalties
and100 fixed200-step evaluation windows, plus separate completed episodes.

For the matrix precision control at200k, float64 joint distributions with the
faster actor rate solve Climbing2/3 and Penalty2/3 development seeds. The matched
float32 condition solved Climbing3/3 and Penalty0/3. With the slower actor rate,
float64 solves neither game in any seed. Thus precision affects optimization
but does not supply a robust solution. Some greedy-optimal policies still have
negative expected return when sampling their independent local policies; both
metrics remain reported. Mode consistency concerns greedy execution and does
not imply matching stochastic distributions or an equilibrium theorem.

## Completed longer runs and follow-up controls

All five normalized-critic and all five combined-protocol HalfCheetah runs have
completed 3M steps. The twelve BPPO noise controls, sixteen Prey 1M controls,
five Football 100k pilots and five corrected SMAC 200k runs are also complete.
The [2026-10-26 report](development_results_20261026.md) records all 48 results,
paired noise-control changes, seed uncertainty, negative controls and next steps.
The report's raw CSV links every row to a hashed summary and frozen manifest.

The combined HalfCheetah settings give BPPO 5,228.01 versus matched AR-PPO
4,739.43 in seed 0, but AgentMixer 2,411.14 remains below its identity control
2,975.45. A three-development-seed BPPO/AR comparison now runs to 10M, and a
separate three-seed experiment isolates AgentMixer's modifier scale map.
Neither result establishes final multi-seed paper superiority.

The first two 200k SMAC runs stopped at 129,536 recorded transitions on an
unexpected native restart; three controls did not start. Those failed artifacts
remain in `results/smac_terran5_learning200k_20261025T085510Z`. The corrected
retry completes all five methods and logs one planned episode-boundary restart
per training environment. Final wins are still only 0–4 out of 32 episodes.

All 96 short integration runs across 15 MuJoCo observation variants and nine
SMAC scenarios completed, including 17-agent continuous cases. Job 21196515
verifies native execution and gradient shapes, not learning performance.
