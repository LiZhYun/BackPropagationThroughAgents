# AgentMixer implementation and discrete correction

The canonical accepted source is [`references/agentmixer/aaai25.tex`](../references/agentmixer/aaai25.tex), especially “Policy Modifier,” “Implementation of IGC,” and equations `mixer`, `igc`, and `cate_joint`. The implementation is [`src/bta_repro/agentmixer.py`](../src/bta_repro/agentmixer.py).

Each agent has its own actor and parameters. Its logits depend only on its local observation. During centralized training, the policy modifier embeds those logits and the global state, concatenates the features, and applies two blocks of alternating agent mixing and channel mixing MLPs. Agent mixing communicates across agents; channel mixing transforms each agent's features. The modifier changes exploration while retaining each local actor's preferred actions. Greedy deployment uses `local_actions(obs, available_actions)` and requires no state or modifier.

## Why the discrete parameterization changes

The source describes learned, action-specific additive Gumbel offsets before a softmax. Arbitrary additive offsets can change the maximizing action. For example, local logits `[2, 1]` and offsets `[0, 3]` produce `[2, 4]`, reversing the preferred action. A positive temperature, including its limit toward zero, does not repair this reversal. We therefore preserve the paper's mode consistency objective with an explicitly different discrete parameterization. This is a corrected AgentMixer variant, not a byte-for-byte reproduction of the published implementation.

For agent `i`, let `z[i,a]` denote its local logit, `m[i]` the maximum **over available actions**, and `r[i,a]` the modifier output. Define:

```text
s[i,a] = exp(clamp(r[i,a], -8, 8))
joint_logit[i,a] = -s[i,a] * (m[i] - z[i,a])
```

All original maximizing actions have joint logit zero, including ties. Every other available action has a strictly negative logit because its gap and scale are positive. Unavailable actions receive negative infinity. Thus the entire set of individual modes is preserved, and the Cartesian product of these modes is the set of modes of the factorized joint policy. The maximum participates in differentiation; detaching it would give incorrect gradients to the local maximizing logit when scales differ by action.

The implementation additionally caps unscaled gaps at `1e4 * exp(8)` before multiplication and effective gaps at `1e4` to avoid overflow from extreme finite inputs. Nonzero effective gaps have a floor of `8 * machine_epsilon`, preventing new ties from underflow and categorical normalization. These guards modify extremely concentrated or nearly tied distributions. Inputs in half precision use float32 arithmetic. Nonfinite logits/scales and masks without any valid action raise an error. The final modifier head initializes to zero, making scales exactly one and the initial categorical probabilities equal to the local probabilities up to the documented floating point guards. Its hidden layers begin receiving gradients after the output head moves from zero.

## What the model does and does not establish

The modifier depends on all local policy parameters and the full state. Given those inputs, each agent's action is sampled from its own categorical distribution independently. Shared deterministic features do **not** make these samples conditionally correlated. This implementation does not claim arbitrary correlated joint distributions or a shared stochastic latent variable. It also does not inherit the paper's correlated equilibrium convergence theorem: mode consistency alone is not a practical convergence guarantee.

This module handles homogeneous discrete action spaces with feed-forward actors by default. Separate modules provide optional [private recurrent actors](recurrent_actors.md) and the [continuous Gaussian branch](agentmixer_continuous.md); heterogeneous actions remain unsupported. The shared trainer is responsible for PPO losses, advantage estimation, rollout accounting, and evaluation. Decentralized stochastic evaluation must be requested explicitly with `decentralized=True`; greedy local evaluation uses the same maximizing action as the centralized policy.

## Verification

`tests/test_agentmixer.py` checks randomized mode consistency, tied modes, masked maxima, single valid actions, extreme scales and finite logits, subnormal gaps, differentiation through the maximum, double precision gradient checks away from ties, initial identity probabilities, sampling/evaluation agreement, decentralized information isolation, cross-agent and state dependence in the modifier, and gradients to both local actors and the modifier. These correctness checks are not evidence of benchmark performance.

## Optional experimental local distillation

The simulator trainer supports `agentmixer_distillation_coef` (default 0) for factorized discrete policies. It adds a detached current-central-to-local KL to the actor objective, with explicit experimental run labels. The [diagnosis and control specification](prey_distillation.md) gives its gradient, mask reduction, limitations, compatibility checks, completed native gates and the now-submitted 24-run study. No learning improvement is established.
