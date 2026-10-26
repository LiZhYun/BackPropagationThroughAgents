# Discrete BPPO implementation

`src/bta_repro/bppo.py` implements a readable, discrete-action BPPO variant. Its
reference is the accepted AAAI source
`references/bpta/Formatting-Instructions-LaTeX-2024.tex`, specifically the
conditional policy-gradient theorem, equation labelled `eq:bppo`, and
Algorithm 1. The source is preserved unchanged. This is a documented correction
and concrete parameterization of that objective, not a claim of bit-for-bit
equivalence to the original code or a completed benchmark reproduction.

## Policy and update order

Agent `i` has its own two-hidden-layer tanh MLP. Its input is its observation
and the concatenated one-hot actions of agents `0, ..., i-1`. No parameters are
shared. Sampling follows this order; likelihood evaluation uses the recorded
preceding actions. Availability masks apply before the categorical distribution
is constructed. A row with no valid action is an error rather than an arbitrary
fallback action.

The update takes a frozen copy of the entire rollout policy on entry and checks
its likelihoods against the supplied old log probabilities. Call it once per
fresh rollout: calling it repeatedly on stale rollouts cannot reconstruct the
behavior policy from log probabilities alone. Updates run in reverse agent
order. All epochs for agent `i` complete before updating `i-1`; its successors
then remain fixed. Each actor has an independent optimizer. A shared critic,
advantage estimation, and any advantage normalization belong to the caller.

## Optional action projection

`AutoregressivePolicy(..., action_embed_dim=0)` preserves the original raw
one-hot inputs, sequential actor modules, parameter names, and seeded
initialization. A positive embedding width enables a separate learned,
bias-free linear action projection inside each successor actor. The same
projection is applied to that successor's predecessor slots; different
successors have independent projections. Projected vectors are concatenated
with the actor's own observation and passed to its usual two-layer tanh MLP.
The first actor has no predecessor and therefore no projection. Projections
use orthogonal initialization with gain one, with no payoff-dependent inputs
or initialization.

The projection belongs to its successor actor's optimizer. It is computed
inside the conditional forward pass, so both raw action probes and projection
parameters retain their derivatives. AR-PPO and BPPO can use exactly the same
policy class and projection width for matched comparisons.

The accepted source reports a high-dimensional projection ablation in which
AR-MAPPO improves on the matrix games, but does not specify the exact projection
module in the supplied LaTeX. This implementation is an explicit reconstruction
ablation. A linear projection followed by an MLP changes parameterization and
optimization; it is not claimed to strictly enlarge the policy's function
class. For BPPO, it can also change the learned continuous extension between
one-hot actions and hence the action-input derivative used by the peer term.
No performance gain is assumed until measured with matched controls.

## Own objective and peer objective

For one transition, let `r_i` be the current-to-behavior conditional probability
ratio of the recorded action of agent `i`. Let

```
M_i(x) = product over j > i of
         current_pi_j(recorded a_j | recorded predecessors with a_i = x)
         / behavior_pi_j(recorded a_j | recorded predecessors with a_i = x)
c_i = gradient_x M_i(x), evaluated at the recorded one-hot a_i
```

All other predecessor actions remain recorded constants during this
derivative, including intermediate successor actions. The implementation
directly differentiates the full product via its log probabilities, so all
successors and skip connections contribute without manually indexed ratio
caches.

Behavior **parameters** are frozen, but its dependence on the action `x`
remains differentiable. Thus the numerator and denominator both contribute to
`c_i`. Before a successor changes, its ratio is identically one and its peer
derivative is zero. Treating a stored behavior log probability as constant in
this derivative would produce a different estimator; that convention is not
used here. The resulting `M_i` and `c_i` are detached before the actor update.

The own objective, which is maximized, is

```
mean_batch[stop_gradient(M_i) * min(r_i * A, clip(r_i, 1-eps, 1+eps) * A)]
```

The peer objective is

```
mean_batch[stop_gradient(selected_ratio) * A
           * sum_action_coordinates(stop_gradient(c_i) * g_i)]
```

`selected_ratio` is the raw or clipped ratio selected by the scalar own PPO
surrogate for that transition, including the sign of the advantage. The peer
gradient remains active on a clipped branch, as implied by the detached clipped
ratio in the paper. The reparameterized action `g_i` is a straight-through
Gumbel-softmax sample from the current actor with temperature 1 by default.
Its forward value is one-hot, and its backward derivative is that of the soft
sample. Fresh Gumbel noise is sampled for each actor epoch. This is a biased
gradient estimator for discrete actions; no unbiasedness claim is made.

Both objectives are reduced to scalars separately and then added with mean
categorical entropy. In particular, the scalar own objective is not broadcast
across action coordinates, and the peer dot product is **summed** across those
coordinates before averaging over transitions. The detached peer coefficients
prevent second-order derivatives and updates to successor parameters. The
last agent has `M=1` and `c=0`, reducing exactly to ordinary clipped PPO.

## Gradient diagnostics

Discrete `bppo_update` also reports `own_grad_norm`, `peer_grad_norm`, and
`own_peer_grad_cosine`. For each actor epoch, these are norms and cosine of
the actual own and peer parameter gradients before entropy is added and
before gradient clipping. `torch.autograd.grad` measures the components
without accumulating into parameter `.grad` or performing extra optimizer
steps. Computing these diagnostics requires additional backward traversals.
The ordinary `grad_norm` remains the norm of the combined gradient, including
entropy, before clipping.

`successor_ratio_abs_change` reports the active-sample mean of `abs(M-1)`;
`peer_coefficient_norm` remains the active-sample mean action-derivative norm.
The latter alone does not measure the effective peer parameter gradient,
because reparameterization can suppress components of the action derivative.
All diagnostics are averaged over actor epochs that actually update. The
last actor contributes zero peer norm and zero cosine. A zero norm on either
gradient defines its cosine as zero, so this aggregated cosine includes those
zero cases; it is not the cosine between globally concatenated mean gradients.
All-inactive batches return zero diagnostics and skip actor optimizers.

## Explicit corrections and scope

The printed `eq:bppo` omits the advantage multiplier from its own-learning
term, although it appears in the peer term. We include it in both: otherwise
the last-agent/no-peer limit is not PPO and its update cannot distinguish
positive from negative advantage. The source also leaves the reduction of the
action derivative and the branch choice for the combined own/peer expression
underspecified. We use the explicit dot product and choose clipping from the
scalar own PPO surrogate before adding the peer surrogate. This separates
first-order own and peer gradients and avoids letting the numerical value of a
gradient surrogate alter the clipping branch. These are methodological
choices to validate and ablate, not silently asserted identities to the
typeset formula.

This initial implementation uses feed-forward actors, fixed agent order,
homogeneous discrete action spaces, full-rollout updates, and a fixed number of
epochs. Separate modules provide [continuous-action actors](bppo_continuous.md)
and optional [recurrent discrete actors](recurrent_actors.md). The latter use
the same BPPO update with recorded incoming hidden states and one-decision
gradient chunks. Heterogeneous action spaces and minibatching are not implemented
here. The sum of log likelihood ratios is exponentiated without an
undocumented clamp; non-finite ratios or gradients raise an error. Training
budgets and gradient-norm clipping still need benchmark tuning.

Optional `active_agents[batch, agents]` masks let environments mark dead agents.
Each actor's own, peer, entropy, and diagnostic reductions use only its active
transitions, with their own active count. The successor derivative is still
computed per transition, with no batch averaging applied to that derivative.
An actor with no active transitions skips its optimizer entirely. An all-dead
batch returns zero metrics and leaves every optimizer state unchanged. Discrete
environment adapters should give an inactive agent a single legal no-op; its
conditional log probability is then zero, its ratio is one, and its successor
feedback derivative is zero. An inactive predecessor's recorded no-op remains
an input to any still-active successors.

## Verification

`tests/test_bppo.py` checks sampling/evaluation likelihood identity, legal
actions, causal conditioning, finite-difference agreement for the full
successor-ratio derivative, cancellation with unchanged successors, isolated
peer gradient paths, exact reduction to single-agent PPO, action-coordinate
and repeated-batch scaling, reverse update order, empty masks, and rejection
of stale behavior snapshots or incorrect optimizer ownership. These checks
establish local implementation properties, not published performance.
