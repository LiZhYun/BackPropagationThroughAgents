# Predator–Prey distribution diagnosis and optional local distillation

The released-count 1M study still falls short for learned AgentMixer. Its identity control reaches the zero-penalty figure range, but that does not establish AgentMixer reproduction. The new checkpoint diagnosis motivates a controlled auxiliary-loss experiment; it does not show a cause or a learning improvement.

## Exact replay diagnosis

All eight original checkpoints replay their 20 complete diagnostic episodes exactly: **160 episodes and 5,668 decisions**. No optimizer steps occurred, all checkpoint bytes and loaded tensors stayed unchanged, and the original 25-module source and simulator versions were verified. Every visited decision preserved the full set of local/global modes. Neither a modifier hard clamp nor the positive-gap numerical floor was active.

The primary scores below are the original audited **100×200-decision fixed-window evaluation returns**, with auto-reset, unfinished episodes included, and team reward counted once per decision. They were neither recomputed nor replaced by the complete-episode diagnosis. The user selected this released protocol. KL and action probabilities are state-weighted summaries along each policy's own greedy trajectories.

| Penalty | Actor | Modifier | Original window return | Mean episode length | Mean central→local joint KL (nats) | Greedy pair probability, local / central |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 0 | FF | Learned | 1.95200 | 4.05 | 1.8008 | 0.146 / 0.906 |
| 0 | GRU | Learned | 2.05150 | 4.10 | 1.3900 | 0.205 / 0.859 |
| 0 | FF | Identity | 2.21600 | 4.35 | 0.0000 | 0.923 / 0.923 |
| 0 | GRU | Identity | 2.24650 | 4.05 | 0.0000 | 0.944 / 0.944 |
| -1 | FF | Learned | 0.07125 | 61.35 | 0.5147 | 0.083 / 0.294 |
| -1 | GRU | Learned | 0.04055 | 58.55 | 0.3865 | 0.162 / 0.346 |
| -1 | FF | Identity | -0.00875 | 115.65 | 0.0000 | 0.181 / 0.181 |
| -1 | GRU | Identity | 0.26760 | 31.30 | 0.0000 | 0.393 / 0.393 |

The historical figure reading ranges are [2.19, 2.23] for penalty 0 and [0.35, 0.36] for penalty −1. Each condition uses training seed 0. These diagnostic states are dependent, have different occupancies across policies, and do not describe the centralized training trajectory distribution. Probability mass on the greedy tuple is not an evaluation sampling frequency: evaluation always selects the local mode. The difference therefore cannot by itself explain the return gap. Complete [JSON evidence](../experiments/prey_recurrent_distribution_diagnostic_20261029.json) and [CSV](../experiments/prey_recurrent_distribution_diagnostic_20261029.csv) retain all outcomes, raw-trace hashes, and limitations.

## Explicit experimental objective

The accepted source discusses asymmetric distillation in equations `va` and `cad`, and explains its partial-observability limitations. Its PPO implementation does not prescribe the following added loss. We retain the corrected [mode-preserving modifier](agentmixer.md) and introduce an optional simulator configuration field, `agentmixer_distillation_coef`, default **0**. Positive values currently require discrete, factorized AgentMixer with a learned modifier; private GRU actors are supported.

For current local probabilities `p`, current centralized probabilities `q`, and active-agent mask `A`, the auxiliary term is:

```text
L_distill = mean over live states b [sum over active agents i
             sum over available actions a
               stopgrad(q[b,i,a]) * (log(stopgrad(q[b,i,a])) - log(p[b,i,a]))]
L_actor = L_joint_PPO - entropy_coef * H_joint + coefficient * L_distill
```

Both distributions use the same current minibatch and legal-action support. The teacher is recomputed each optimizer step and detached; it is not a saved behavior-policy target. Joint KL sums agent factors, then averages states with at least one active agent, matching the existing joint PPO reduction. Inactive factors contribute zero. All-dead minibatches skip actor updates. Unavailable log terms are zeroed before subtraction to prevent undefined masked arithmetic.

Only local actors receive a direct gradient from the auxiliary term. Joint PPO still updates the modifier. The existing shared gradient-norm clip can indirectly rescale modifier gradients when the actor contribution changes. Rollouts remain centralized; greedy evaluation remains decentralized. Mode consistency continues to hold for the current policy after parameter updates, but the preferred action can change during learning.

This is a heuristic under centralized rollout occupancy with a changing teacher. It does not inherit an identifiability or convergence guarantee from the paper. It may reduce exploration or hurt returns. Runs receive the explicit `_experimental_local_distillation` method suffix, record the objective in metadata, and log unweighted KL, weighted auxiliary loss, and total actor loss. The coefficient is preserved in checkpoints.

## Correctness and planned comparison

At introduction, the full source suite passed **1,003 tests with one CUDA-only skip**. Analytic categorical gradients cover unavailable actions, single legal actions, partial/all-dead agents, and float32/float64. FF/GRU policies route the auxiliary gradient only into actors; modes still agree after an update. Runner updates match an independent coefficient/reduction reference. Short real Prey training and exact episode/window checkpoint replay pass for FF and GRU. These 16-step checks are not learning evidence.

Across 21 default policy cases, pre-change and current source produce bitwise-identical policy/critic tensors, gradients, Adam states, metrics, actions, likelihoods, and final Torch RNG after two epochs with unequal minibatches. The zero-coefficient path does not compute the auxiliary term. [Verification](../experiments/agentmixer_distillation_verification_20261029.json); [default compatibility](../experiments/agentmixer_distillation_default_parity_20261029.json). The earlier installed-wheel report applies to the prior source.

The [frozen plan](../experiments/prey_distillation_study_20261029.json) contains eight full-rollout native gates and 24 subsequent 1M development runs. The gates are now **queued** as 21282173; the 24 learning runs remain **unsubmitted**. It compares coefficients 0, 0.01, and 0.1 plus the identity control, with GRU actors, both penalties, and development seeds 0–2. Other learning settings match the released-count study: 50×200 rollouts, 15 epochs, one minibatch, the same optimizer/value settings, and the complete 10k-spaced evaluation grid.

All eight gates must pass exact final replay. The four coefficient-zero gates must also match their original seed-0 first-update records. Native resource measurements and a verified free scheduling allocation are required before learning release. The native gates reserve the current Football array’s slots after its entire completion; the 1M study has no allocation, and existing queued work retains priority. All 24 outcomes must be audited before aggregation. The same nonzero coefficient must beat both controls and the figure upper reading in both penalties, without reducing mean curve area against coefficient zero, to proceed to separately registered independent confirmation. If both qualify, select 0.01. Otherwise retain the failed result; no per-seed winners or peak substitution.


## Native launch and complete-family audit

The [execution specification](../experiments/prey_distillation_gate_execution_20261029.json) freezes the launcher, raw-accounting helper, evaluator, auditor, dispatch, and all eight config hashes. Job **21282173** waits for all tasks in Football array **21251960** to end. It uses at most two 1 CPU/4 GiB/30 minute tasks, replacing two 1 CPU/8 GiB Football tasks. Submission overrides the partition to `batch-milan`, matching the original Prey runs on `milan1`; the native gate still verifies exact default-update equality. No existing job or dependency was changed. [Submission verification](../experiments/prey_distillation_gate_schedule_20261029.json).

Before submission, the raw auditor checked all **808 historical evaluations: 16,160 episodes and 80,800 windows**, and rejected 48 corrupted copies covering reward algebra, denominator, derived seeds, unfinished returns, capture reduction, and episode endings. It also rejected the unstarted gate family. The existing private core runtime passed its read-only environment check. These checks validate the auditor and reused runtime; they do not establish new native gate completion. [Preflight evidence](../experiments/prey_distillation_gate_preflight_20261029.json).

Job **21282210** runs the full-family auditor automatically after all eight gates end, using their frozen `PYTHONPATH`. It requests one CPU/2 GiB for five minutes. Missing or invalid conditions fail the audit; no partial-family success is reported, and no learning jobs are released automatically. Its expected new output is `experiments/prey_distillation_native_complete_20261029.json`. Resource measurements remain a separate release requirement. [Collector verification](../experiments/prey_distillation_gate_collector_20261029.json).
