# Football rollout and optimizer audit

All five completed 1M-step 3v1 pilots have zero final wins. Their positive
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

Job **21219475** first ran ten native integration configurations, each with
one 10,000-transition rollout, 15 epochs and two minibatches. Every run must
have finite losses, 30 critic steps, 90 BPPO actor steps or 30 steps for the
other methods, and four complete initial/final evaluation episodes. Source,
configuration, raw scores, budgets, runtime and peak memory are checked before
releasing the separate ten-run 1M learning manifest. Each integration worker
requests 2 CPUs/8GB for at most two hours. It follows the corresponding existing
Football worker and precedes Ant/Walker, preserving four serial worker chains.

The learning manifest is released as **21221161** after all ten native gates
passed. Its primary development
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


## Completed 1M pilot evidence

The [complete audit](../experiments/football_3v1_1m_complete_20261026.json)
verifies all five methods at 1M: 500 raw evaluation episodes and 6,250 finite
rollout updates, with exact source/configuration, seed, native score and
interaction-count checks. Both workers exit successfully; peak memory is
about 1.27 GiB.

| Method | Final wins / 20 | Final shaped return | Normalized shaped-return curve area |
| --- | ---: | ---: | ---: |
| MAPPO | 0 | 0.8850 | 0.6410 |
| AR-PPO | 0 | 0.7000 | 0.6707 |
| BPPO | 0 | 0.8850 | 0.6850 |
| AgentMixer | 0 | 1.0950 | 0.9357 |
| Identity modifier | 0 | 1.7450 | 0.9568 |

Earlier evaluations remain in the report and never replace a final endpoint.
Shaped returns must not substitute for completed-match wins. These are one-seed
pilots below the paper budget, without independent evidence of a method
advantage. The [earlier partial audit](../experiments/football_3v1_1m_progress_20261026T194000Z.json)
remains unchanged.


## Completed native integration gate and learning release

Both workers complete successfully in 414/411 seconds, with maximum batch
peak memory of 3.49 GiB. The
[complete audit](../experiments/football_batch_integration_complete_20261026.json)
verifies 80 raw evaluation episodes and ten complete 50×200 rollout updates.
Each has 15 epochs and two minibatches: 30 critic steps, 90 BPPO actor steps
or 30 actor steps for the other methods, with one ValueNorm update per rollout.
Run times range from 74.49 to 89.49 seconds. Source, configuration, native win
flags, interaction counts and finite losses all pass.

These are 10k-step correctness checks with only four episodes per evaluation.
Their scores do not establish learning performance. The
[earlier recurrent-only audit](../experiments/football_batch_integration_progress_20261026.json)
remains unchanged.

The [learning release plan](../experiments/football_batch1m_release_plan_20261026.json)
retains all ten frozen configurations and their 25-module source snapshot.
One array task runs one configuration, using one CPU/8 GiB with a four-hour
limit and no automatic requeue. Array concurrency is capped at two. Even/odd
tasks form serial chains after the matching independent HalfCheetah worker,
within the current [eight-worker project ceiling](worker_schedule.md#eight-worker-schedule-2026-10-26). The initial
[schedule receipt](../experiments/football_batch1m_release_schedule_20261026.json)
verifies every task's dependencies and resource requests before release.

The allocation estimate uses the slowest observed rollout/update and evaluation
costs. It allows 401 decisions for each of the 400 evaluation matches, following
the pinned academy duration and native clock; all 53,577 completed episodes
in the earlier 1M pilots also respect that horizon. With a 50% margin, the
estimate is 3.16 hours per run. Eight GiB exceeds twice the measured peak memory.
These are allocation estimates, not guaranteed runtimes or learning results.
The original prepared two-worker script and all experiment configurations remain
unchanged; a separately hashed dispatch script executes the frozen manifest.

## Completed 1M comparison and 20M follow-up, 2026-10-28

All ten FF/GRU controls pass the [complete audit](../experiments/football_batch1m_complete_20261028.json): 4,000 complete-match evaluation episodes and 1,000 rollout updates, with native score/win flags and optimizer/ValueNorm counts verified. Final wins out of 100, FF/GRU respectively, are MAPPO 0/1, AR-PPO 1/8, BPPO 3/2, AgentMixer 4/4 and identity 3/1. Positive shaped rewards do not replace win rates. One training seed below 20M does not establish an architecture or algorithm advantage.

Job **21251960** extends all five recurrent methods/controls to **20M**, retaining the exact completed source and resolved training/evaluation settings except training budget. GRU is selected consistently for released actor-history alignment, not separately by each method's best 1M endpoint. Evaluation retains 100 complete matches every 400k. The [frozen study](../experiments/followup_studies_20261028.json) and [resource plan](../experiments/followup_release_plan_20261028.json) retain all outcomes: one CPU/8 GiB/60 hours per task, maximum two active, no automatic requeue. The slowest phase-based projection is 44.24 hours, not a guarantee. Full results and independent seeds remain outstanding.
