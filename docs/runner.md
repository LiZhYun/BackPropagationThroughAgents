# Temporal runner

`runner.collect_rollout` and `runner.update_rollout` provide the shared temporal
training path for discrete and continuous environments. They use a synchronous
list of environments, a feed-forward or optionally recurrent actor policy, and
a feed-forward centralized scalar critic.
There are no simulator imports, hidden worker processes,
automatic checkpoint writes, or environment ownership changes in this module.
The caller constructs and eventually closes the environments.

## Environment contract

An environment exposes `num_agents`, `obs_dim`, `state_dim`, `is_discrete`, and
either `num_actions` or `action_dim`. Environments in one collection call must
have matching dimensions and action type. The interface is:

```
reset(seed=None) -> (observation, info)
step(raw_action_array) -> (next_observation, team_reward, terminated, truncated, info)
```

It must not autoreset. Observations contain `obs[agents, obs_dim]` and
`state[state_dim]`, optionally `available_actions[agents, num_actions]` for
discrete policies and `active_agents[agents]`. Missing masks become all true.
Every agent needs a legal action when a policy is sampled, including inactive
agents, which should have one legal no-op. Final observations may have empty
availability masks because only their state value is needed before reset.
Rewards are finite scalar team rewards; terminal and truncation flags are
booleans. The adapter is responsible for distinguishing true termination from
time-limit truncation.

The runner sends each environment a private numpy copy of its sampled action.
Continuous actions are raw Gaussian samples and are preserved in the rollout,
even if the environment adapter clips its received array in place. Policies
are sampled under `torch.no_grad()`. Collection does not alter policy weights.

## GAE and episode boundaries

For every transition, the runner records both the current critic value and
the value of the actual next state returned by `step`, before any reset.
GAE uses these values with explicit termination and truncation flags:

```
delta_t = reward_t + gamma * (not terminated_t) * next_value_t - value_t
adv_t = delta_t + gamma * lambda * (not (terminated_t or truncated_t)) * adv_next
```

A true terminal removes the next-state bootstrap. A truncation retains that
bootstrap but stops recursion into the reset episode. A rollout boundary
bootstraps its actual next state and does not reset an unfinished episode.
`gamma` and `gae_lambda` default to `0.99` and `0.95` and are explicit optional
collection arguments.

The returned next observations are an `ObservationCarry`, which behaves as a
list and retains unfinished episode returns, lengths and optional actor memory. Pass this object
back to the next collection call. Converting it to an ordinary list discards
that metadata and any recurrent memory. A carry is checked against the identities and order
of its original environments. Completed episode records contain environment
index, full return, full length, termination/truncation flags, and final info.
Episodes may span arbitrarily many collection calls.

To choose initial seeds, reset each environment with the desired seed and
pass the resulting observation list to the first collection call. The runner
does not reseed subsequent resets. With no initial observations, it calls
`reset()` itself. Supplying an ordinary list starts episode accounting from
that point; it is intended for initial reset observations.

## Tensor layout

Collection starts with `[T, E, ...]` tensors. All training fields flatten the
first two axes in time-major order, so sample `b = t * E + e` always denotes
time `t` in environment `e`. Observation, state, raw action, old log probability,
advantage, return, availability, and active-agent masks share that order.
`RolloutBatch` extends the common `Batch` without modifying it, and records
`rollout_shape=(T, E)`. Its audit fields `rewards`, `values`, `next_values`,
`terminated`, and `truncated` retain `[T, E]` shape.

## Update interface

`update_rollout` accepts the policy, critic, optimizer dictionary, batch,
algorithm name, and action-space type. Optimizer keys match the existing
matrix trainer: `critic` and `actor_0`, ... for BPPO; `critic` and `policy` for
MAPPO, autoregressive PPO, or AgentMixer. BPPO calls the dedicated discrete or
continuous update. Other algorithms use the common PPO objective. AgentMixer
uses the joint product-distribution ratio; independent and autoregressive PPO
baselines use per-agent ratios.

The experimental two-agent discrete [joint modifier](agentmixer_joint.md)
instead emits one categorical score per joint transition. Its
`joint_event=True` marker selects `[T×E, 1]` behavior likelihoods; PPO scores
that event once. Collection and updates require every agent to be active for
this variant. The simulator CLI restricts it to Predator–Prey.

Discrete BPPO and the PPO baselines support inactive-agent masks. Each BPPO
actor averages own, peer, and entropy terms over its active transitions and
skips its optimizer when none exist. For AgentMixer, inactive factors are
removed before forming the joint ratio, and all-inactive transitions are
excluded. Continuous BPPO currently requires all agents to remain active.
Team advantages are normalized over transitions with at least one active
agent by default using the common helper's float64 moments. Identical active
advantages become exact zeros, avoiding spurious score gradients from
float32 reduction roundoff. Inactive transitions become zero after
normalization; `normalize_advantages=False` preserves the original scale.

The critic minimizes `value_coef * mean((value - return)**2)`, with
`value_coef=0.5` by default. It runs the requested number of epochs after the
policy update, with its own optimizer and gradient clipping. Setting
`value_coef=0` skips critic optimizer steps. Diagnostics include weighted value
loss, unweighted mean squared error, update counts, and active-agent fraction.

`num_minibatches=1` preserves full-rollout updates and consumes no shuffle RNG.
Larger values shuffle every epoch and split the rollout into balanced, nonempty
minibatches, retaining every transition exactly once. Unequal sizes are allowed;
the number cannot exceed the rollout size, including the last partial rollout.
Advantage normalization and ValueNorm statistics use the full rollout once.
Old likelihoods, raw values and return targets stay fixed. Each BPPO actor
finishes all of its epochs and minibatches before its predecessor starts; one
behavior snapshot covers the entire rollout. Successor feedback is computed
after successors finish and remains fixed during that actor's optimization.
Policy and critic epochs use separate shuffles. Actor minibatches with no
active transitions skip optimization. Diagnostics average executed optimizer
steps; update counts therefore reflect actual minibatches, and BPPO counts
each actor's steps separately.

Optional [recurrent actors](recurrent_actors.md)
pack incoming hidden state after each local observation in `batch.obs`; raw
environment dimensions and centralized `batch.state` remain unchanged. The
encoder and GRU are recomputed during every likelihood evaluation while the
incoming state stays detached, giving one-decision gradient chunks. Each
environment's memory resets on termination or truncation and otherwise carries
across updates. Minibatch selection keeps each observation together with its
recorded incoming memory. Longer gradient chunks, recurrent critics,
asynchronous environments, heterogeneous agent shapes, and
distributed training remain outside this module. Tests use deterministic fake
environments to establish boundary handling, raw-action retention, likelihood
identity, masking, and finite small updates; they do not establish simulator
performance.
