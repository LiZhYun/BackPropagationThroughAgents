# Optional recurrent actors

Set `recurrent_actor: true` to give each actor a private GRU memory. Discrete
MAPPO, AR-PPO, BPPO and factorized AgentMixer share one actor construction.
Continuous AgentMixer has a Gaussian actor construction; its identity control
is also supported. The default remains feed-forward. Continuous MAPPO, AR-PPO
and BPPO, and the experimental enumerated joint modifier, reject this option.

[`recurrent.py`](../src/bta_repro/recurrent.py) contains the actors and four thin
policy subclasses. Each discrete actor encodes its own current observation with
a linear layer and tanh, updates a GRU cell, then computes logits with a tanh action head.
Hidden width is `hidden_dim`. For AR-PPO and BPPO, the head also receives current
predecessor actions or their optional learned projections. Those actions do not
enter the recurrent cell. No agent's observation or hidden state enters another
agent's memory. Parameters are independent across agents.

Continuous AgentMixer keeps the existing two-layer tanh observation encoder,
then applies a GRU. Gaussian mean and observation-dependent scale heads read
its output; the state-independent scale option uses a separate learned vector.
The centralized Gaussian retains these exact local means. Its modifier receives
current local Gaussian parameters and global state, with no separate recurrent
state. [Continuous memory control](agentmixer_continuous.md#optional-local-history).

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

Discrete AgentMixer's modifier still consumes the centralized state and local
policy logits during training; the continuous modifier consumes local means
and scales with the centralized state. Local evaluation uses only each actor's
observation and private memory. Discrete policies preserve greedy modes through
positive gap scaling; continuous policies retain identical local and centralized means.
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

## Completed native gate and learning release

All five recurrent SMAC terran5v5 integrations now complete at 10,000 decisions.
The [full audit](../experiments/smac_recurrent_integration_complete_20261026.json)
checks 40 raw evaluation episodes, five finite rollout updates, source bytes,
resolved settings, budgets and final summaries. Each uses a 2×5000 rollout and
five PPO epochs. Run times range 148.41–168.50s; batch peak resident memory is
3.97GiB. The earlier partial record remains available.

The previously frozen ten-run feed-forward/GRU comparison is now queued as
job **21220264**, with **one 1M-step run per task**, two active tasks at most,
and a six-hour cap per task. The [release plan](../experiments/smac_batch1m_release_plan_20261026.json)
records exact source/configuration reuse and matched architecture settings.
The largest observed collection/update duration, plus scaled 32-episode
evaluations, estimates 4.47h per run. A 25% margin rounds to 6h. Five such runs in
one task would need 27.93h with the margin, exceeding the previous 24h header.
The new dispatch layout leaves the frozen configurations and source unchanged.

Even-numbered tasks follow the preceding even task; odd-numbered tasks follow
the preceding odd task. Their first tasks were initially queued behind recurrent Prey workers
21219719_0/1. The verified [eight-worker revision](worker_schedule.md#eight-worker-schedule-2026-10-26)
removes those scheduling-only dependencies after the native gate completed.
These are now two independent serial chains. Automatic requeue is disabled,
failures remain recorded, and the six-hour estimate is not a guarantee.

The [native process diagnostic](simulator_runtime.md#completed-native-process-diagnostic)
shows a modest call-time improvement under normal restart settings, with exact
trace parity, but does not validate training-state or optimizer equivalence.
This learning study retains the existing sequential backend. The 1M budget and
single development seed remain below the paper-level evidence requirements.


## Continuous AgentMixer control, 29 October 2026

The [new verification](../experiments/continuous_recurrent_verification_20261029.json) adds 41 continuous-memory checks. The full suite passes 1,044 tests and one CUDA-only skip. All 29 existing policy configurations in the compatibility audit remain bitwise unchanged, including discrete recurrence and projected autoregressive actors. Synthetic continuous training and saved-policy replay pass for all six modifier/local-scale combinations.

Twelve native 44k checks are queued as **21284086**, with complete-family auditor **21284087**. The frozen observer passes bitwise neutrality checks in both runtimes, rejects incorrect resets and other injected errors, and passes a synthetic full-horizon checkpoint replay. The native checks must still establish those properties in MuJoCo and measure runtime/memory before learning. Eighteen recurrent 1M development runs remain frozen and unsubmitted. [Execution and evidence](agentmixer_continuous.md#optional-local-history). No native continuous recurrence result is claimed. The original discrete native validations above remain evidence for those exact frozen studies.
