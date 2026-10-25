# Continuous AgentMixer

The accepted source, [`references/agentmixer/aaai25.tex`](../references/agentmixer/aaai25.tex), defines continuous Individual-Global-Consistency by retaining the local Gaussian means while allowing the centralized policy modifier to determine standard deviations. [`src/bta_repro/agentmixer_continuous.py`](../src/bta_repro/agentmixer_continuous.py) implements that construction directly.

Each agent has an independently parameterized actor whose local observation produces a mean vector and diagonal standard deviations. The centralized modifier embeds each agent's mean and standard deviations, concatenates global state features, and applies the same alternating agent/channel mixing blocks as the discrete implementation. Its output is a residual to the local log standard deviations:

```text
local_std = exp(clamp(local_log_std, -5, 2))
joint_mean = local_mean
joint_std = exp(clamp(log(local_std) + modifier(local_mean, local_std, state), -5, 2))
```

The bounds are numerical and exploration choices, not constants specified by the theorem. They give strictly positive standard deviations between approximately `0.0067` and `7.389`. The modifier's final layer initializes to zero, so the initial centralized and local distributions agree. The modifier controls exploration without any path from the state into the means. Means are unbounded. Since a nonsingular Gaussian's mode equals its mean, local and centralized **raw Gaussian** modes agree exactly.

The local actors and modifier receive PPO gradients through the centralized distribution. The modifier receives mean and standard deviation tensors without detaching them. Its hidden layers begin learning after its zero-initialized output layer changes. `local_actions(obs)` returns only local means; `sample(..., decentralized=True)` samples the local policies without invoking the modifier or using a state.

## Raw policy and actuator commands

`sample` returns raw Normal actions `[batch, agents, action_dim]`, together with log probabilities summed over action coordinates `[batch, agents]`. `evaluate_actions` accepts those same raw actions and also returns coordinate-summed differential entropies. It does not sum across agents.

An environment adapter can clip the raw action to the actuator's legal interval. The rollout must keep the original raw action for PPO scoring. Clipping is a deterministic environment-facing map: the return is observed after applying that map, while the sampled random variable optimized by the score-function estimator remains the raw action. Scoring the clipped action as though it were the sampled Normal action gives the wrong likelihood ratio. No tanh transformation or change-of-variables correction is applied here.

Clipping can introduce probability mass at actuator boundaries. The equality of Gaussian modes is therefore a statement about the raw distributions, not a guarantee about the modes of the resulting clipped actuator distributions. At deterministic evaluation, applying the same clipping map to the identical local and centralized means still produces identical actuator commands.

## Scope and verification

Conditioned on observations and state, the implementation uses independent Normal noise across agents and action coordinates. State-dependent standard deviations alone do not create conditionally correlated samples. Mode consistency does not establish convergence or performance superiority.

The implementation currently supports homogeneous continuous action dimensions and feedforward observations. Heterogeneous actuator sets need explicit padding/masking or another adapter convention; recurrence is a separate extension. The environment adapter owns actuator bounds and simulator details.

`tests/test_agentmixer_continuous.py` verifies exact mean equality, state-independent local execution, local information isolation, bounded positive standard deviations, coordinate-summed likelihoods and entropies, raw action handling, deterministic means, actor and modifier gradients, and checkpoint restoration. Those checks establish implementation properties, not reproduction of benchmark returns.
