# Experimental joint-gap AgentMixer

[`JointAgentMixerPolicy`](../src/bta_repro/agentmixer_joint.py) is a named, experimental **two-agent discrete redesign**. It retains the accepted work's central ideas: independent local actors, a centralized policy modifier conditioned on local policy parameters and state, exact Individual-Global-Consistency of modes, and local deterministic execution. It replaces factorized exploration with an explicitly correlated categorical distribution over action pairs. It is not the accepted paper's implementation and inherits no correlated-equilibrium theorem or performance guarantee.

## Distribution and mode consistency

For each local actor, take the maximum over available actions and form nonnegative gaps:

```text
m_i = max over available actions of z_i(a)
g_i(a) = m_i - z_i(a)
s(a,b) = exp(r(a,b)) > 0
E(a,b) = s(a,b) * (g_0(a) + g_1(b))
pi(a,b) = exp(-E(a,b)) / sum over legal pairs of exp(-E)
```

The modifier uses the same alternating agent/channel mixing blocks as the factorized version, then outputs one log scale per action pair. Its final layer initializes to zero. Consequently, initial scales are one and `exp(-E)` factorizes into the original local categorical distributions, up to numerical guards.

Every legal pair of local maximizing actions has energy zero. Every other legal pair has a positive gap sum and a strictly positive scale, hence positive energy. The complete joint mode set is therefore **exactly** the Cartesian product of the local mode sets, including ties. Masks exclude illegal pairs. A dead agent with only a noop action is a valid special case.

Unlike a sum of independent per-agent energies, arbitrary pair scales can create conditional correlation. For example, local gaps `[0,1]` for both agents and pair scales `[[1,10],[10,0.05]]` give energies `[[0,10],[10,0.1]]`. The coordinated alternative `(1,1)` has probability approximately `0.475`, while `(0,0)` remains the unique mode with probability approximately `0.525`. This example establishes representational capability, not successful learning.

For two agents, unrestricted positive pair scales can represent any strictly positive joint distribution with the required mode set: for each nonmode pair, choose its target negative-log probability gap divided by the local gap sum. Finite scale bounds restrict this family in the implementation.

## Differentiation and numerical guards

The masked maxima remain attached to autograd. The construction is differentiable away from changes in the maximizing actions; ties use PyTorch's maximum subgradient. If every local action is tied, IGC requires every joint action to be a mode, forcing a uniform joint distribution. The modifier therefore cannot introduce correlation at that exact point, and its scale gradients vanish there.

By default, log scales are clamped to `[-8,8]`. Optional `log_scale_bound=b` uses `b*tanh(raw/b)`, preserving a unit derivative and identity value at zero while smoothly limiting scales. Positive gaps and energies are bounded to avoid overflow. Nonzero energies receive a small machine-precision floor to prevent underflow from creating extra modes; these guards can change extreme or nearly tied distributions. Half precision inputs use float32 energy calculations.

The optional `joint_distribution_float64=true` training configuration computes
only the enumerated joint energies, probabilities, entropy and likelihoods in
float64. Networks, optimizer state and local execution retain their original
dtype. This reduces the near-tie region affected by the numerical floor without
changing the formula or introducing a substitute gradient. For example, a0.001
local gap at scale`exp(-8)` is floored in float32, erasing its derivative; the
float64 distribution preserves its exact nonzero derivative and unique mode.
Both matrix and Predator–Prey trainers retain the double-precision behavior
likelihoods through PPO. This is an explicit numerical development control;
its effect on learning remains empirical. Default behavior is unchanged.

Nonnegative penalties added to an already fixed independent energy can mainly concentrate the current mode. Pair scaling can also reduce the energy of coordinated alternatives. Nevertheless, the modifier can still sharpen a bad mode or improve stochastic return without moving local greedy actions. Exact IGC alone does not resolve these optimization failures. Evaluate local greedy performance, joint and local distributions, and modifier scales separately.

## API and integration boundary

The constructor matches the factorized policy and rejects `num_agents != 2`. Local APIs are `local_logits`, `local_actions`, and `local_actors`; `modifier_log_scales` returns `[batch,K,K]` and `joint_logits` returns unnormalized energy logits of that same shape.

Centralized `sample` returns actions `[batch,2]` and log probabilities `[batch,1]`. `evaluate_actions` returns log probabilities and exact entropies `[batch,1]`: these are **one joint distribution component**, not fictitious per-agent scores. Enumeration costs `O(batch*K²)` and computes exact normalization, sampling, likelihoods, and entropy. Entropy gradients include every joint action.

Decentralized sampling ignores state and the modifier, samples the independent local policies, and returns their per-agent log probabilities `[batch,2]`. Deterministic centralized sampling explicitly chooses the local argmax tuple, which is a global joint MAP. Greedily choosing marginal modes in an ancestral sampler would not provide that guarantee.

This variant is intended for the matrix trainer's explicit joint-ratio objective. The generic simulator runner currently requires per-agent score shapes and does not support this single-component interface. No tree generalization, simulator integration, or Gaussian extension is implemented here.

## Verification

The tests compare whole joint mode sets against products of masked local mode sets, cover ties and single legal pairs, verify identity probabilities and normalization, demonstrate nonfactorization, compare exact score gradients with enumerated expected-return derivatives, check entropy derivatives, test likelihood agreement and local execution isolation, and exercise extreme finite values. These are mathematical and implementation checks; benchmark evidence must come from separately recorded development and held-out evaluations.
