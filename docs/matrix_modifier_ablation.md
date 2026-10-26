# AgentMixer matrix modifier ablation

The paired study asks whether learning the joint modifier improves local
execution compared with fixing the same initialized modifier at identity.
It uses **both matrix games, 1M decisions and five fresh training seeds**.
It is queued; no result is available yet. The earlier 200k Climbing study's
failed stochastic-return criterion remains separate evidence.

## One controlled difference

Set `modifier_enabled` to `false` to freeze the modifier at its initial log
scale of zero. Local actors still learn through the same PPO objective,
sampling path and numerical guards. The modifier is constructed in both arms,
preserving initialization and RNG consumption, but frozen parameters are
excluded from the identity arm's optimizer. No distillation or extra loss is
introduced. The default is `true`.

This is an identity control within the selected AgentMixer architecture. It
is not a separately tuned MAPPO baseline. The joint distribution's float64
arithmetic and finite-precision guards remain in both arms; small rounding
differences between central and local probabilities can remain at identity.

The public configurations use development seed 0. For example:

```bash
python -m bta_repro.train --config configs/matrix_agentmixer_learned1m.json \
  --game climbing --seed 0 --output results/my_learned_climbing
python -m bta_repro.train --config configs/matrix_agentmixer_identity1m.json \
  --game climbing --seed 0 --output results/my_identity_climbing
```

Use `--game penalty` for the second game. Outputs must be new directories.

## Frozen paired study

The [paired plan](../experiments/matrix_agentmixer_ablation1m_plan.json) links
the two source/configuration manifests. Every pair differs only in
`modifier_enabled`. The source snapshot is identical between arms.

| Setting | Both arms |
| --- | --- |
| Games | Climbing and Penalty |
| Training seeds | 4000–4004, absent from 582 earlier declarations across 53 manifests |
| Budget | Exactly 1,000,000 joint decisions per run |
| Local actor / modifier / critic rates | 0.0125 / 0.0005 / 0.0005; modifier rate unused at identity |
| Batch / PPO epochs / hidden width | 1024 / 4 / 64 |
| Entropy coefficient | Constant 0.01 |
| Joint probabilities | Float64; networks remain float32 |
| Primary comparison | Final local expected return, learned minus identity, paired by training seed |
| Secondary comparisons | Local curve area, greedy payoff and threshold-success counts |

Evaluations enumerate all nine joint actions exactly. Central expected returns
remain diagnostics and cannot replace local execution scores. All 20 runs must
complete and pass their integrity audits before either game receives a paired
aggregate. Missing and failed seeds remain visible; there are no replacements,
selected earlier peaks or pooling with development results.

The reporter uses 10,000 bootstrap draws of whole paired-seed differences with
private seed 1729. Its 95% intervals are pointwise and descriptive. Five seeds
give limited precision; the results cannot establish a general method advantage
or family-wise statistical significance across games and metrics.

```bash
python scripts/report_matrix_ablation.py \
  experiments/matrix_agentmixer_ablation1m_plan.json \
  --output results/my_matrix_ablation_report.json
```

Matching the paper's 1M budget does not remove the documented differences in
architecture, parameterization and historical evaluation. The earlier setting
was selected using Climbing development results; poor Penalty results are
retained. This study tests that fixed setting on both games without retuning it.

## Verification and scheduling

Twelve new checks cover identity initialization, RNG consumption, frozen
modifier weights, actor learning, optimizer ownership and exact checkpoint
replay for factorized and joint policies. Ten reporter checks reject missing
or duplicate seeds, altered plans, changed configurations and results that
predate the paired plan. The [fresh installed wheel](../experiments/matrix_identity_wheel_verification_20261026.json)
passes **899 tests**, with one CUDA-only skip, plus the console/replay checks.

The default path exactly matches the selected older matrix snapshot across
12 short cases: both games and all six policy/precision choices. All 1,596
checkpoint tensors and 18,424 scalar values match, including Adam and RNG
state. The [parity record](../experiments/matrix_identity_default_parity_20261026.json)
states the comparison scope; these short tests do not prove learning success.

Jobs **21220499** (learned) and **21220500** (identity) each run ten configurations
serially on one CPU/4GB, with a three-hour cap and automatic requeue disabled.
The cap scales the slowest completed 200k run by five, then ten runs, and adds
50% margin. Runtime can differ by game and control; timeouts remain evidence.

The jobs follow their matching existing Prey workers and precede the pending
recurrent Prey workers. The [schedule proof](../experiments/matrix_agentmixer_ablation1m_schedule_20261026.json)
records verified holds, dependencies, resources and releases. The unchanged
four-chain structure limits the project to four concurrent workers.
