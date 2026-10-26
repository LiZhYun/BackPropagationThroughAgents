# Optional recurrent actors

Set `recurrent_actor: true` in a discrete simulator configuration to give each
actor a private GRU memory. MAPPO, AR-PPO, BPPO and factorized AgentMixer use the
same actor construction. AgentMixer's identity-modifier control is supported.
The default remains feed-forward. Continuous policies and the experimental
enumerated joint modifier currently reject this option.

[`recurrent.py`](../src/bta_repro/recurrent.py) contains the actor and three thin
policy subclasses. Each actor encodes its own current observation with a linear
layer and tanh, updates a GRU cell, then computes logits with a tanh action head.
Hidden width is `hidden_dim`. For AR-PPO and BPPO, the head also receives current
predecessor actions or their optional learned projections. Those actions do not
enter the recurrent cell. No agent's observation or hidden state enters another
agent's memory. Parameters are independent across agents.

The runner explicitly carries a `[environments, agents, hidden_dim]` tensor.
Before each action, it appends the incoming memory to the raw observation and
stores that packed input in the rollout. It then advances memory using the same
actor parameters. Termination and truncation both zero the relevant environment's
memory; other environments keep theirs. An unfinished episode keeps its memory
across rollout boundaries and policy updates. The policy object itself stores
no mutable inference state.

During PPO, the incoming rollout memory is detached and held fixed. Each actor's
encoder, GRU and head are recomputed on every likelihood evaluation, so all
their parameters receive gradients. Gradients span one decision, matching the
released configuration's `data_chunk_length=1` convention. This is truncated
recurrent training, not backpropagation through complete episodes or recomputing
old hidden states after every optimizer update. BPPO's behavior copy uses the
same recorded incoming states; its successor ratio and action derivative retain
their existing definitions. Each BPPO actor optimizer owns its encoder, GRU,
head and optional predecessor projection.

AgentMixer's modifier still consumes the raw centralized state and local policy
logits during training. Local evaluation uses only each actor's observation and
private memory. Positive gap scaling preserves its local greedy modes as before.
Evaluation starts fresh memory for every episode and every fixed Prey window,
and resets it at episode boundaries within a window. Evaluation does not consume
or alter the training carry. Saved-policy replay reconstructs the recurrent
architecture from the checkpoint configuration; native simulator resume remains
unsupported.

This is a documented architecture reconstruction, not an exact copy of the
release. The centralized critic stays feed-forward. The recurrent actor uses
tanh and the existing compact action-conditioning scheme; it does not reproduce
the release's ReLU/LayerNorm encoders, recurrent critic or action-feature mixing
network. The [SMAC audit](smac_optimization_audit.md) records those differences.
No performance benefit is assumed before matched experiments finish.

`tests/test_recurrent.py` checks independent terminal/truncation resets, split
rollout identity, exact behavior-likelihood replay, local memory isolation,
optimizer ownership, encoder/GRU gradients, a finite-difference successor-action
derivative, AgentMixer mode consistency, unsupported settings, and real Prey
training/checkpoint replay through episode and fixed-window evaluation. The
full suite passed 691 tests with one CUDA-only skip after integration. Native
SMAC validation is a separate scheduled experiment. Two additional evaluator
trace checks explicitly verified fresh memory at episode boundaries and resets
inside fixed windows; all 33 recurrent-focused tests passed afterwards.
