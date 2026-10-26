# SMAC training audit, 2026-10-26

The active feed-forward Terran 5v5 pilot has low evaluation win rates near
750k decisions. That observation does not identify a single cause. The accepted
appendix and pinned released launcher reveal two concrete differences worth
testing: rollout batch size and actor history.

## Paper, release and current control

| Setting | Accepted BPTA appendix | Pinned released launchers | Current 1M pilot |
| --- | --- | --- | --- |
| Rollout layout | 50 × 200 | No-modifier: 50 × 200; modifier: 25 × 400 | 2 × 128 |
| Joint transitions per update | 10,000 | 10,000 | 256 |
| PPO epochs / minibatches | 5 / 1 | 5 / 1 | 5 / 1 |
| Actor/critic LR | 5e-4 | 5e-4 defaults | 5e-4 |
| Entropy coefficient | 0.01 | 0.01 default | 0.01 |
| Actor history | Chunk length 1 in common settings | GRU enabled; chunk length 1 | None |
| Nonlinearity and normalization | ReLU in common settings | ReLU, input and hidden LayerNorm | Two tanh hidden layers |
| Critic | See source architecture | Separate recurrent critics | Shared scalar feed-forward critic |
| Training budget | 10M | 10M | 1M development |
| Evaluation | Historical complete convention unspecified | 32 episodes in launchers | 32 complete deterministic episodes |

The accepted table is in
`references/bpta/Formatting-Instructions-LaTeX-2024.tex:538–571`. The public
implementation is pinned to commit
`13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0`. Its active
[no-modifier command](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/scripts/temporal/smac/lumi/train_smac_ar.sh)
and [modifier command](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/scripts/temporal/smac/lumi/train_smac.sh)
both collect 10,000 transitions per update. Their filenames alone do not identify
the training algorithm; the temporal runner's dispatch remains relevant.

The actual [argument declarations](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/config.py)
set `use_recurrent_policy=True` with a disabling `store_false` flag, and set
`data_chunk_length=1`. Neither command disables recurrence.
[R_Actor](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/algorithms/bta/algorithm/r_actor_critic.py)
applies the GRU to local observation features before adding action-conditioning
features. The [recurrent layer](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/algorithms/utils/rnn.py)
uses a GRU and output LayerNorm. The
[buffer](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/utils/separated_buffer.py)
stores actor/critic hidden states and supplies the stored state at each sampled
chunk's start. These are verified release-code settings, not proof that this
exact revision generated a historical figure.

The released `layer_N=1` does not mean just one linear hidden transform:
[`MLPLayer`](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/algorithms/utils/mlp.py)
first applies `fc1`, then one cloned hidden block. Both use ReLU and LayerNorm.
The new optional [recurrent actors](recurrent_actors.md) preserve the redesigned
tanh networks and shared feed-forward critic; they test actor history without
silently claiming the complete released architecture.

## Environment differences retained explicitly

The launchers select `StarCraft2v2`. The
[training entry point](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/scripts/train/train_smac.py)
routes that name to
[`SMACv2_modified.py`](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/envs/starcraft2/SMACv2_modified.py).
It disables observation last-actions and requests an agent-specific centralized
state through `get_state_agent`. The alternate
[`SMACv2.py`](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/envs/starcraft2/SMACv2.py)
repeats the shared state instead. The current adapter exposes the pinned official
backend's common global state to a shared critic. Local observation and capability
settings remain fully recorded; exact equivalence to the modified historical
backend is not established. Both wrappers repeat the same scalar team reward
per agent; the redesigned runner stores that team reward once.

## Evidence and next control

At the inspected snapshot, MAPPO's evaluation win rates at about 300k, 450k,
600k and 750k were 4/32, 2/32, 1/32 and 1/32. AR-PPO had 1/32, 1/32 and 2/32
through 600k. Recent update diagnostics had approximate KL around 0.0003–0.0004,
clipping fractions near zero and gradient norms below 1. These points do not
support an exploding-update explanation. They also cannot establish that
larger batches or recurrence will improve learning. The original runs continue
unchanged and their complete outcomes will be retained.

A later [progress audit](../experiments/smac_terran5_1m_progress_20261026.json)
records three completed 1M runs: BPPO wins 6/32 final episodes, MAPPO 4/32 and
AR-PPO 0/32, with mean returns 11.6111, 9.7227 and 8.5918 respectively. The audit
recomputes win counts and return statistics from 768 raw evaluation episodes
across all checkpoints and verifies 11,721 rollout updates. Both unfinished
AgentMixer controls remain listed. This one-seed, partial development study
does not establish a method advantage or results at the 10M paper budget.

The next native integration check uses the recurrent actors with a 10,000-sample
rollout. The matched learning control then compares feed-forward and recurrent
actors at that batch size for all five method/control variants. It uses two
native environments × 5,000 decisions to stay within 2CPUs/8GB per worker.
This matches the paper's samples per update, **not its parallel environment
layout**. Episode termination and time limits still reset the environment and
GRU within each rollout; rollout length does not lengthen a SMAC episode.
The number of environment streams and trajectory grouping are explicit
experimental differences. Performance claims await actual results.

Public source captures used in this audit are retained under
`results/smac_release_audit_20261026/`; references and archived papers were not
modified.

The native check is frozen in
`results/smac_recurrent_batch_integration10k_20261026T125710Z/manifest.json`
and queued as job `21213860`. The larger learning control is frozen in
`results/smac_terran5_batch1m_20261026T125713Z/manifest.json` and remains
unsubmitted until native correctness, runtime and memory have been checked.
The [study record](../experiments/smac_recurrent_study_20261026.json) states the
gates and the [worker schedule](worker_schedule.md) preserves resource limits.

Two recurrent integration runs now complete: AR-PPO in 148.41 seconds and
AgentMixer in 151.60 seconds. Each finishes one rollout of shape `[5000, 2]`,
one finite five-epoch update, and four complete initial/final evaluation
episodes. The [partial audit](../experiments/smac_recurrent_integration_progress_20261026.json)
verifies source/configuration hashes, all 16 raw evaluation episodes and both
rollout budgets, retaining the three pending variants.

Worker `21213860_1` exits 0 after 5m06s. Slurm reports batch-step MaxRSS
3,825,900 KiB (about 3.65 GiB) across its two sequential runs. This is sampled
accounting, not a separate per-method peak or a guarantee for longer training.
The remaining worker follows AgentMixer matrix validation `21218175_0` in the
existing B0 chain. The larger learning study remains unsubmitted until all five
native checks and the full runtime/memory review pass.
