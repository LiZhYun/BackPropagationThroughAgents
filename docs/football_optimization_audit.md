# Football rollout and optimizer audit

The first three completed 1M-step 3v1 pilots have zero final wins. Their positive
shaped returns do not establish task success. The accepted BPTA appendix and
pinned release reveal concrete optimization differences worth testing.

| Setting | Existing 1M pilot | Accepted appendix / release | New development control |
| --- | ---: | ---: | ---: |
| Training environments × rollout length | 8 × 100 | 50 × 200 | 50 × 200 |
| PPO epochs | 5 | 15 | 15 |
| Minibatches per epoch | 1 | 2 | 2 |
| Actor recurrence | Feed-forward | Enabled by default | Matched feed-forward / GRU pairs |
| Value-loss coefficient | 0.5 | 1 in released configuration | 1 |
| Evaluation episodes | 20 | 100 | 100 |
| Training decisions | 1M | 20M | 1M development |

The accepted table is “Common hyper-parameters used in the GRF” in the local
BPTA source, lines 630–665. The pinned
[3v1 launcher](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/scripts/temporal/football/lumi/train_football_3v1_ar.sh)
sets 50 environments, 200 steps and two minibatches; it inherits 15 epochs and
recurrent actors from the
[configuration](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/config.py).
The actual selected algorithm is `temporal`; the launcher's filename alone
does not identify the method. Both existing and new controls use Adam epsilon
1e-5, actor/critic learning rates 5e-4, entropy coefficient 0.01, width 64 and
maximum gradient norm 10.

The [frozen study](../experiments/football_batch_study_20261026.json) compares
five methods within each architecture: MAPPO, AR-PPO, BPPO, AgentMixer and its
identity-modifier control. Each feed-forward/GRU pair differs only in
`recurrent_actor`. Comparisons with the earlier pilot bundle several changes
and cannot isolate the effect of minibatches or rollout size.

Job **21219475** first runs ten native integration configurations, each with
one 10,000-transition rollout, 15 epochs and two minibatches. Every run must
have finite losses, 30 critic steps, 90 BPPO actor steps or 30 steps for the
other methods, and four complete initial/final evaluation episodes. Source,
configuration, raw scores, budgets, runtime and peak memory are checked before
releasing the separate ten-run 1M learning manifest. Each integration worker
requests 2 CPUs/8GB for at most two hours. It follows the corresponding existing
Football worker and precedes Ant/Walker, preserving four serial worker chains.

The learning manifest is **prepared, not submitted**. Its primary development
metric is final complete-match win fraction over 100 deterministic episodes;
shaped return and score difference remain separate diagnostics. Evaluation
runs initially, at 400k and 800k, and at the exact 1M endpoint. No earlier
checkpoint replaces the final endpoint. Seed 0 remains a development seed.

These controls retain reconstructed tanh actors and a shared feed-forward
critic. They do not reproduce the historical ReLU/LayerNorm networks,
recurrent separate critics or action-feature mixing network. Corrected native
seeding and complete-match evaluation also differ from the defective released
evaluator. The [adapter protocol](football.md) documents those corrections;
the historical 5v5 figure reduction remains unresolved. A 1M development result
cannot establish the 20M, five-seed paper target.

The implementation provides [explicit minibatch semantics](runner.md#update-interface).
The default single-batch path exactly matches the pre-change implementation
across 68 cases, including optimizer and RNG state. The clean installed wheel
passes 877 tests with one CUDA-only skip, including 32 minibatch checks and
exact recurrent Prey checkpoint replay. These are correctness checks, not
evidence of Football learning performance.


## Updated pilot evidence

The [partial audit](../experiments/football_3v1_1m_progress_20261026T194000Z.json)
now verifies MAPPO, AR-PPO and BPPO at 1M: 300 raw evaluation episodes and
3,750 finite rollout updates, with exact source/configuration/budget checks.
All three final evaluations have 0 wins in 20 complete episodes. Final shaped
returns are 0.8850, 0.7000 and 0.8850 respectively; normalized shaped-return
curve areas are 0.6410, 0.6707 and 0.6850. The audit retains the earlier AR-PPO
win without substituting it for the final endpoint. AgentMixer and its identity
control remain unfinished at this snapshot. These are one-seed pilots below
the paper budget, with no evidence of a method advantage.


## Partial native integration gate

Worker 21219475_1 completes all five recurrent variants in 411 seconds, with
sampled batch peak memory of 3.48GiB. The
[partial audit](../experiments/football_batch_integration_progress_20261026.json)
verifies 40 raw evaluation episodes and five complete 50×200 rollout updates.
Each has 15 epochs and two minibatches: 30 critic steps, 90 BPPO actor steps
or 30 actor steps for the other methods, with one ValueNorm update per rollout.
Run times range from 74.49 to 85.88 seconds. Source, configuration, native win
flags, interaction counts and finite losses all pass.

These are 10k-step correctness checks with only four episodes per evaluation.
Their scores do not establish learning performance. The remaining five
feed-forward integrations are pending, so the all-method learning-release gate
remains closed.
