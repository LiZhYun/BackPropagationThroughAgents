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

## Longer runs and native-clock correction

The normalized-critic HalfCheetah experiment has completed three of its five
3M-step runs in development seed0. Final evaluation returns over ten episodes
are3045.06 for MAPPO,3599.84 for AR-PPO and599.40 for BPPO. BPPO's curve remains
poor throughout: its largest recorded evaluation mean is825.37 at1M. Median
approximate KL stays below0.002 in each of the three inspected training periods,
and the median gradient norm stays below1. These diagnostics do not support an
explanation based on exploding updates. They also do not establish the cause
of poor learning from a single seed. AgentMixer and its identity control are
still running. Evidence: job21196118 and
`results/mujoco_hc2x3_valuenorm3m_20261025T083844Z`.

At500k, the larger-batch protocol settings already gave BPPO2399.97 and
AgentMixer2128.55, compared with684.46 and800.11 under the normalization-only
setting. A matched five-method3M protocol follow-up therefore uses the existing
40×100 batch, lower initial action noise, and clipped Huber normalized critic.
This is a configuration comparison; several settings change together.
Configuration: `configs/mujoco_hc2x3_protocol3m.json`; queued job21198090.

The first two200k SMAC runs both stopped after129,536 recorded transitions;
their remaining three controls were not started. The adapter rejected an
unexpected native restart, preserving the error instead of training on a false
terminal transition. This is consistent with StarCraft II's cumulative clock
limit. The [SMAC lifecycle correction](smac.md) restarts at episode boundaries
before expiry, with native validation21198088 and fresh matched retry21198089.
The original failed runs remain in
`results/smac_terran5_learning200k_20261025T085510Z`.

All96 short integration runs across15 MuJoCo observation variants and nine SMAC
scenarios have completed, including the17-agent continuous cases. This verifies
native execution and gradient shapes, not learning performance; job21196515.

An additional estimator control tests whether continuous BPPO benefits from
retaining the Gaussian noise associated with the recorded action. The
[implementation note](bppo_continuous.md) explains the distinct scale gradient
and the paper's underspecified trajectory/noise coupling. The existing
fresh-noise default is preserved. Job21198169 compares both choices under both
existing optimization settings in development seeds0–2 at500k steps each.
Analytic and integration checks pass; no learning result is available yet.
