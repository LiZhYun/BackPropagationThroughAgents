# Continuous BPPO

`src/bta_repro/bppo_continuous.py` adds homogeneous continuous action spaces to
the corrected BPPO surrogate described in [bppo.md](bppo.md). The accepted BPTA
source specifies the reparameterization trick for continuous actions in its
proof appendix, and evaluates continuous control on Multi-Agent MuJoCo. It
does not uniquely specify the Gaussian architecture or action-bound handling
implemented here. These choices are explicit reproduction parameterizations,
not claims of exact equivalence to an unspecified original implementation.

## Distribution and action representation

Each actor owns a two-hidden-layer tanh mean network and one trainable log
standard deviation per action coordinate. The standard deviations are
state-independent and initialized to one unless `initial_std` is specified.
Effective log standard deviations
are clamped to `[-5, 2]`, giving positive finite scales in
`[exp(-5), exp(2)]`; gradients through the scale clamp are zero outside those
bounds. There is no parameter sharing between agents.

Agent `i` sees its own observation and all recorded raw predecessor action
vectors. Its conditional action distribution is a diagonal normal, represented
as `Independent(Normal(mean, std), 1)`. Therefore log probability and entropy
are **sums over the action coordinates**, each producing one scalar per agent
and transition. PPO ratios are computed only after that sum. Treating action
coordinates as independent PPO samples or averaging their log probabilities
would implement a different objective.

`sample` returns raw Gaussian actions and their event log probabilities.
Deterministic sampling returns the conditional mean. Neither path clips or
squashes actions. The environment adapter may clip a separate copy to its
physical bounds, but the rollout must preserve the raw sample used to compute
its log probability. Successors also condition on that raw sample. This
defines a latent raw-action policy followed by a deterministic actuator map;
the Gaussian log probability is the density of the latent sample, not of a
clipped boundary action. It is valid to score the stored latent sample in the
policy-gradient estimator. Scoring a clipped action with its raw-sample log
probability is inconsistent and is rejected by the behavior snapshot check.

## Reverse update and peer feedback

As in discrete BPPO, one frozen behavior snapshot is taken at update entry,
and the caller supplies one optimizer per actor and one shared advantage per
transition. All epochs for the last agent run before updating its predecessor.
Calls must use fresh rollouts whose behavior policy is still the current
policy. Advantage normalization and the critic remain caller responsibilities.

The continuous implementation reuses the discrete module's successor-ratio
derivative and scalar objective helpers. For predecessor action vector `x`,
the ratio is the product of updated-to-behavior successor event densities.
Both densities retain their derivative with respect to `x`, even though the
behavior parameters are frozen. Other recorded actions are held constant.
The exact first derivative is evaluated at the recorded raw predecessor
action and detached before constructing the peer objective.

By default, the peer surrogate uses a fresh normal reparameterization
`g_i = mean_i + std_i * epsilon` with `epsilon ~ Normal(0, I)`. Its dot product
with the detached successor-ratio derivative is summed over coordinates and
averaged over transitions. The selected PPO ratio and advantage are detached
on this path. The own scalar PPO term is reduced separately, and the same
documented clipping-branch convention and advantage correction as discrete
BPPO apply. No second-order gradients or successor-parameter updates pass
through the peer coefficients. This does not assert that the finite-rollout,
clipped BPPO surrogate is an unbiased estimator of the exact policy gradient.

## Recorded-noise control

The simulator option `continuous_peer_noise="behavior"` reconstructs the
Gaussian noise that produced each recorded raw action:

```
epsilon_recorded = (action_recorded - behavior_mean) / behavior_std
g_current = current_mean + current_std * stop_gradient(epsilon_recorded)
```

The behavior distribution conditions on the same recorded predecessor actions
as the original rollout. Its parameters remain frozen, and the reconstructed
noise stays fixed across that actor's update epochs. At the first epoch,
`g_current` equals the recorded action up to floating-point error. Later epochs
move it using the current parameters. The successor derivative still uses the
recorded action, as required by the existing first-order surrogate.

This is a controlled change to the sampling estimator. With fresh independent
noise, a fixed peer coefficient times the log-standard-deviation derivative
has zero expectation over the newly sampled noise. Reusing the recorded noise
retains its correlation with the action-dependent coefficient and advantage.
For example, with zero-mean action `a = sigma * epsilon` and coefficient `c=a`,
the coupled derivative of `E[c*g]` with respect to `log(sigma)` at the behavior
policy is `sigma**2`; independent fresh noise gives zero. The mean derivative
does not have this distinction for a state-independent Gaussian scale.

The paper's proof writes `a = g(theta, epsilon)`, while its algorithm and
surrogate do not completely specify how a stored trajectory is coupled to
update-time noise. We therefore expose both choices and preserve `"fresh"` as
the existing default. Both retain reverse updates, conditional successor
feedback and local actor parameter ownership. Neither choice establishes
unbiasedness of the complete clipped surrogate or a performance improvement.
The option is accepted only for continuous BPPO; discrete Gumbel sampling and
other algorithms are unchanged. Raw actions, rather than actuator-clipped
actions, are essential for reconstructing this noise.

`configs/mujoco_hc2x3_peer_noise500k.json` compares both choices under the
normalization-only and combined protocol settings, using development seeds
0, 1 and 2. These are diagnostic 500k budgets, not final paper-budget results.

When there are no successors, the ratio is one and the peer term is zero.
The implementation then matches standard Gaussian PPO, including event-summed
entropy. No unnecessary normal sample is drawn for the zero-peer last agent.

## Checks and remaining scope

`tests/test_bppo_continuous.py` covers sample/evaluation likelihood identity,
coordinate-summed densities and entropy, raw-action preservation, bounded
scales, causal conditioning, an analytic Gaussian successor-ratio derivative,
finite differences across multiple successors, behavior-derivative
cancellation, isolated peer gradients, single-agent PPO equivalence,
coordinate and batch reductions, reverse update order, and finite small
updates. These are local correctness checks, not evidence of benchmark gains.

This module currently assumes equal observation and action dimensions across
agents, a fixed order, feed-forward actors, and full-rollout updates. It does
not implement recurrent policies, heterogeneous action dimensions, action
squashing, adaptive order, or benchmark-specific tuning. Non-finite importance
ratios and gradients raise errors rather than being silently clamped.
