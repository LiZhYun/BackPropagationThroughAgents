# Experiment plan

Date: 2026-10-25. Status: active; no full benchmark reproduction claimed.

## Claim map

| Claim | Minimum evidence |
|---|---|
| C1: the redesign preserves the two papers' core mechanisms | Exact toy references, causal gradient checks, AgentMixer mode consistency, execution-information isolation, and documented equation-to-code mapping |
| C2: the shared implementation exceeds published performance under comparable conditions | Matched task/observation/reward/step protocols, independent final seeds, all-run metrics and uncertainty, plus ablations isolating peer feedback and policy modification |

Anti-claims: improvement only comes from changed simulator physics, more environment steps, selected successful seeds, or an algorithm silently replacing the papers' ideas. No LLM or other unrelated component is part of these methods.

## Required experiment blocks

1. **Numerical correctness.** Exact matrix payoffs/gradients, scalar PPO reductions, likelihood identity, causal masks, legal actions, termination/truncation, and checkpoint restoration. Failure stops algorithm benchmarking. Unit and integration tests provide evidence; they do not prove convergence.
2. **Development matrix games.** Independent MAPPO, AR-PPO, BPPO, and AgentMixer on climbing and penalty. Start with training seeds 0, 1, 2 and 100k joint steps per run. Track exact expected reward, greedy reward, optimum probability, first threshold attainment, and all seed outcomes. These are one-step development games, matching pinned released environment termination. Legacy training logs multiply step reward by200; the exact historical figure-to-logger mapping remains unverified.
3. **Shared simulator benchmarks.** Multi-Agent MuJoCo and SMACv2, with distinct observation protocols where the papers differ. Use `configs/paper_targets.json`; never replace missing numeric thresholds with invented values. Run baseline smoke checks, method pilots, then paper-matched budgets. Compare final performance and learning-curve area; preserve the paper's mean/median statistic and additionally publish seed-level scores.
4. **Paper-specific coverage and ablations.** BPTA Football, AgentMixer Predator-Prey, no-peer BPPO/AR-PPO, and identity-modifier AgentMixer/MAPPO comparisons. Keep architecture and budget differences explicit. Required for full coverage, with lower priority than shared suites.
5. **Independent confirmation and release.** Freeze configurations and code snapshots before at least five final training seeds disjoint from development seeds. Record evaluation episode seeds, environment versions, raw returns, all failures, checkpoints, and exact commands. Final seed IDs will be fixed before launch. Add confidence intervals and distinguish point-score improvements from statistically established improvements.

## Execution order and resource limits

| Stage | Gate | Initial resource use |
|---|---|---|
| M0 correctness | All relevant tests and all-method smoke runs pass | Single-threaded CPU checks |
| M1 matrix development | Stable training and interpretable across-seed results | CPU array, at most two concurrent tasks |
| M2 simulator pilot | Action/observation semantics and metric parity verified | One smoke job per new simulator |
| M3 full benchmarks | Development configuration frozen; protocol resolved | Start with at most two training jobs; size resources from pilot timings |
| M4 release | All required tasks and release artifacts validated | Independent final seeds and documentation checks |

After the100k HalfCheetah pilot (136–186seconds perrun), allow at most four concurrent CPU workers across this project: two for the longer MuJoCo check and two for short matrix/SMAC diagnostics. Each requests at most2CPUs/8GB, so the ceiling is8CPUs/32GB. Record job dependencies to enforce this limit. This supersedes the initial two-worker ceiling above.

The current independent two-worker arrays use matching-index dependencies: successor worker `i` waits for predecessor worker `i`. Four sequential chains preserve the same worker/resource ceiling while releasing work as each predecessor finishes. The verified graph is in [the worker schedule](../docs/worker_schedule.md); future submissions must preserve that cap.

Do not invent GPU-hour estimates before measuring simulator throughput. Known budgets are in the target manifest. Full reproduction spans dozens of tasks and hundreds of seed runs; measured pilot costs determine scheduling. No paid external compute is assumed. Preserve the concurrent unrelated reproduction.

## Risks and interpretation

- BPPO's printed scalar/vector expression and advantage placement require documented corrections; numerical tests must match the chosen surrogate, not an unsupported unbiasedness assertion.
- AgentMixer's original discrete offsets do not enforce the stated mode constraint. Positive gap scaling is a labeled corrected parameterization; the paper's theorem is not automatically inherited.
- Plot-only thresholds need extraction with uncertainty or author-provided raw data. Environment migration is a separate comparison track unless equivalence is established.
- Matrix return ceilings are 11 and 10. A strict return improvement above a ceiling is impossible; report reliability and sample efficiency instead.
- Simulator initialization and scheduler failures are infrastructure outcomes, not negative algorithm results.

## Completion checklist

- [x] Isolated code and source provenance.
- [x] Initial numerical correctness suite.
- [x] All-method end-to-end checks on matrix, SMACv2 and MA-MuJoCo.
- [ ] Development matrix sweep and diagnosis.
- [ ] Continuous/recurrent method coverage where required.
- [ ] All benchmark protocols and numeric/plot targets resolved.
- [ ] Full baseline and method evidence.
- [ ] Independent final seeds and ablations.
- [ ] Reproducible community release with verified performance claims.

## HalfCheetah independent confirmation, 2026-10-26

The first completed10M BPPO development group has mean5980.37±118.29 across
seeds0–2. Freeze five unused training seeds1000–1004 and evaluation seed2000000
for BPPO and matched AR-PPO, retaining the exact selected source and all other
resolved settings. The study tests one pinned reconstruction; all historical
comparison qualifications and the remaining45-target scope stay in force.

The [study record](../experiments/hc2x3_confirmation10m_plan.json) fixes the
endpoint, complete-group rule, paired comparisons and uncertainty before launch.
Both workers follow matching Ant/Walker workers21212297_i. Measured runtime
predicts13.68hours per serial worker; request18hours,2CPUs/8GB each within the
existing four-chain ceiling. No replacement seeds, early selection, or pooling
with development is allowed. See [the study guide](../docs/halfcheetah_confirmation.md).


## Paired AgentMixer matrix ablation, 2026-10-26

The completed 200k Climbing configuration check has 10/10 optimal greedy
policies but only 8/10 local stochastic returns above its frozen threshold.
Keep that failed criterion unchanged. Test the same selected setting at the
longer 1M budget on both Climbing and Penalty, pairing a learned modifier
with its frozen-identity control on five unused seeds 4000–4004. All 20 runs
must be valid before paired aggregation; prior development is not pooled.

The [paired plan](../experiments/matrix_agentmixer_ablation1m_plan.json) fixes
settings, seeds, source, endpoint and uncertainty before launch. Each arm has
one CPU/4GB and a three-hour cap, derived from measured runtimes with 50%
margin. Jobs 21220499/21220500 follow matching existing Prey workers and
precede recurrent Prey, preserving four serial project chains. Independent
simulator evidence and every other benchmark remain required.


The [budget correction](../experiments/matrix_budget_interpretation_correction_20261026.json)
clarifies that AgentMixer's archived Climbing budget is 200k; BPTA's matrix
budget is 1M. The paired AgentMixer study is a longer-budget extension, with
Penalty as an additional task. Its endpoint cannot certify the 200k target.


## HalfCheetah partition transfer, 2026-10-26

Transfer the selected2×3 BPPO/AR configuration and exact source to3×2 and6×1.
A short native gate uses three40×100 rollouts per condition and the canonical
1000-step horizon. Job21220888 follows matrix21220500_0 and precedes recurrent
Prey21219719_1, preserving four serial worker chains. Four500k development
pilots are frozen but remain unsubmitted until correctness/runtime/memory
checks pass. All methods and both partitions remain declared, with no
early-peak selection. Full10M runs, independent training seeds and AgentMixer
coverage remain required. See the [study](../docs/halfcheetah_partition_transfer.md).

All four native checks subsequently passed:16raw evaluation episodes,12full
updates,120seconds and0.49GiB peak memory. Job21220995 releases the four
500k pilots,one run/task,1CPU/2GiB and30minutes,no requeue. A0 takes tasks0→2
and A1 takes1→3,between matching Ant/Walker and pending2×3 confirmation
workers. The measured slowest estimate is20.87minutes including50%margin.
See the [release plan](../experiments/hc_partitions_transfer_release_plan_20261026.json).
The four-worker ceiling and frozen confirmation configurations remain fixed.


## Football released-count learning, 2026-10-26

All ten native FF/GRU integration checks pass, including80raw evaluation
episodes,10full rollouts and expected optimizer/ValueNorm counts. Job21221161
releases the unchanged ten-condition1M study,one configuration per task,
1CPU/8GiB and4hours,no requeue. Even/odd task chains append after matching
independent HalfCheetah workers21214776_i;four project workers remain the cap.
The measured3.16-hour estimate includes50%margin and all400evaluations at
the native401-decision horizon. The previous five small-batch1M pilots all
finish with0/20wins. Retain all new variants andseed0;do not substitute
shaped returns or early peaks for final complete-match wins. This remains
below20M and does not supply independent final training seeds. See the
[release plan](../experiments/football_batch1m_release_plan_20261026.json).
