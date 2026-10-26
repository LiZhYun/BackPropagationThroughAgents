# Predator–Prey modifier-range control

The completed frozen-policy comparison finds a large training/execution gap:
small-batch factorized and joint AgentMixer capture 85/100 and 98/100 fresh
episodes under central sampling, but neither captures an episode with local
greedy execution. Local sampling captures only 3/100 and 0/100. All visited
local and central greedy modes agree. See the
[behavior diagnosis](prey_behavior_diagnosis.md).

A second diagnostic checks whether the numerical saturation found in the
matrix games also appears here. It reconstructs observations from the saved
coordinates and decision indices of each policy's first two original greedy
evaluation episodes. It imports the frozen environment and policy source,
verifies its hashes, and reproduces every recorded greedy action exactly.
There are no new simulator episodes, optimizer steps or checkpoint changes.

## Measured distributions

All twelve AgentMixer policies from the 1M batch control are retained in the
[raw CSV](../experiments/prey_scale_diagnosis_20261026.csv). The penalty −1 rows
are shown below. State counts reflect the recorded episode lengths, not
independent samples or a common state distribution across policies.

| Variant | States | Nonmode scales at a hard clamp | Mean KL(central joint ∥ local joint) | Mean total variation | Mean central probability of the greedy tuple |
| --- | ---: | ---: | ---: | ---: | ---: |
| factorized, small batch | 256 | 0% | 2.331 | 0.481 | 0.385 |
| joint, small batch | 256 | 0.358% | 3.654 | 0.372 | 0.578 |
| factorized, released batch size | 256 | 0% | 0.149 | 0.185 | 0.162 |
| joint, released batch size | 256 | 0% | 0.177 | 0.183 | 0.157 |
| factorized, released batch, entropy 0.1 | 138 | 0% | 0.041 | 0.079 | 0.077 |
| joint, released batch, float64 probabilities | 256 | 0% | 0.082 | 0.148 | 0.086 |

The positive-energy numerical floor is inactive on every recorded state for
all twelve policies. Thus the widespread hard-clamp/floor explanation from
the failed matrix checkpoints is not supported here. The small-batch policies
still make large probability changes: their observed nonmode scales range
from 0.00112 to 85.95 and from 0.000335 to 91.24, respectively. Most of these
outputs have nonzero derivatives through the modifier.

Identical modes alone permit substantial differences between distributions.
A tiny gap scale can make a locally unlikely action almost as probable as the
local greedy action during central sampling. This helps explain how central
sampling and local sampling can behave differently, but it does not prove why
the local greedy policy failed to learn. These observations come from two
greedy trajectories per policy; they are not the central training-state
occupancy or an estimate over all possible states.

## Matched next experiment

`configs/prey_modifier_bound1m.json` specifies eight development runs:
factorized and joint AgentMixer, each with the default range and a smooth bound,
for penalties 0 and −1. All use seed 0, exactly 1M training transitions, 8×64
rollouts, five PPO epochs, the existing optimizer settings and normalized
clipped Huber critic loss. All eight controls use one common frozen source.

The new simulator configuration field is `modifier_log_scale_bound`. With
`b = log(10)`, the existing policy implementation computes

```text
log_scale = b * tanh(raw_modifier_output / b)
scale = exp(log_scale)  # between 0.1 and 10
```

This limits how much the modifier can shrink or enlarge local action gaps.
The head still initializes at zero, every scale starts at one, network weights
and RNG use are shared with the default control, and all local mode sets remain
preserved. The actor, modifier and PPO losses are unchanged. This is a test of
restricted probability modification, not an asserted numerical bug fix or a
convergence guarantee. The default remains `null`, using the previous hard
log-scale clamp of −8 to 8.

The option requires discrete AgentMixer with its modifier enabled. Continuous
policies, other algorithms and the identity control reject non-null values.
The bound is saved in configuration/checkpoint files and changes the method
label to `mode_consistent_bounded_gap_modifier` or
`experimental_joint_bounded_gap_scaling`. It uses the same optional bound
already tested in matrix policies; no additional policy parameterization is
introduced.

Evaluation follows the author's selected protocol: 100 independent windows of
200 decisions, including auto-resets and unfinished rewards, using only local
greedy actors. Twenty complete-episode evaluations also retain return and
capture diagnostics. The zero-penalty variants check whether any gain on the
hard task compromises the easier task. No centralized sampling score is used
as the benchmark result. The fixed endpoint and every declared run must be
reported; further development seeds and separate final seeds remain necessary.

## Evidence and validation

The full diagnostic is `results/prey_scale_diagnosis_20261026/summary.json`,
with `probe.py` alongside it. It records hashes of the source, trace,
checkpoints, probe and reconstructed observations, plus raw scale ranges,
clamp/floor counts and distribution summaries. Original checkpoint hashes are
verified before and after analysis.

`tests/test_discrete_modifier_control.py` checks shared initialization,
unchanged initial likelihoods, active bounded modification, local/central
mode agreement, unsupported configurations, and real Prey training followed
by exact checkpoint evaluation replay for both policy families, including
fixed-window evaluation. Existing policy tests cover masking, ties and extreme
values. These checks establish implementation behavior, not learning success.

The focused integration suite passes 182 tests; the full suite passes 621 with
one CUDA-only skip (2026-10-26). Job **21212197** queues the eight frozen runs
at `results/prey_modifier_bound1m_20261026T111603Z`, after entropy job 21211006.
No learning result is available yet.
