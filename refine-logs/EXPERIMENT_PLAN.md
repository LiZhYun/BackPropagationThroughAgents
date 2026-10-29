# Experiment plan

Updated: 2026-10-29. Status: active work; no full benchmark reproduction claimed. Earlier dated plans and allocations remain as history.

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

Current ceiling: **ten workers/sixteen CPUs/104 GiB**, with twelve CPUs scheduled at maximum. See the [latest allocation graph](../docs/worker_schedule.md#measured-mujoco-concurrency-revision-29-october-2026). Dated allocations below preserve their historical limits.

| Stage | Gate | Initial resource use |
|---|---|---|
| M0 correctness | All relevant tests and all-method smoke runs pass | Single-threaded CPU checks |
| M1 matrix development | Stable training and interpretable across-seed results | CPU array, at most two concurrent tasks |
| M2 simulator pilot | Action/observation semantics and metric parity verified | One smoke job per new simulator |
| M3 full benchmarks | Development configuration frozen; protocol resolved | Start with at most two training jobs; size resources from pilot timings |
| M4 release | All required tasks and release artifacts validated | Independent final seeds and documentation checks |

After the100k HalfCheetah pilot (136–186seconds perrun), allow at most four concurrent CPU workers across this project: two for the longer MuJoCo check and two for short matrix/SMAC diagnostics. Each requests at most2CPUs/8GB, so the ceiling is8CPUs/32GB. Record job dependencies to enforce this limit. This supersedes the initial two-worker ceiling above.

On2026-10-26, after completed native gates and measured development runs,
the pilot ceiling is revised to eight serial worker chains, at most16CPUs/64GiB
total and2CPUs/8GiB per worker. This is an autonomous scheduling choice under
the user's standing authorization, not a separately requested resource count.
The [verified schedule](../docs/worker_schedule.md#eight-worker-schedule-2026-10-26)
separates already-gated AgentMixer10M and SMAC studies from scheduling-only
predecessors. No experimental runs or budgets are added. That revision replaces the earlier
four-worker limit; the later measured revisions below supersede its eight-chain ceiling.

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


## AgentMixer HalfCheetah full-budget development, 2026-10-26

The complete matched entropy comparison validates all18 runs. Bounded+0.01
has the strongest1M method endpoints. Job21221384 extends bounded,residual
and identity to10M on all three development seeds0–2,using the exact selected
source and training settings. Only budget and evaluation cadence change.
Require all nine valid endpoints before aggregation,retain failed outcomes,
and report each seed,mean/median/sample SD and full-grid curve area. The
bounded/identity contrast changes the scale map and initial centralized scale;
residual/identity is the isolated modifier switch. This is development,with
independent final confirmation still required. See the
[study guide](../docs/agentmixer_halfcheetah10m.md).


## Remaining MuJoCo coverage, 2026-10-26

Freeze37 seed0 conditions across11 task/observation groups: nine without
learning pilots and the missing HC3×2/6×1 MAPPO/AgentMixer/identity controls.
Use the exact20-module Ant/Walker source and selected training settings;
changes from the selected runtime configuration are algorithm,modifier switch
and task/observations only. All74 gate/pilot configurations validate with the
frozen trainer. Job21221614 runs37 native12k checks before any500k pilot
release. Require complete raw-log,optimizer/normalization,source/seed and
scheduler resource audits; the pilot allocation remains provisional until
measured. Two2CPU/8GiB/two-hour gate workers insert into A0/B0 without changing
the eight-worker ceiling. These are development studies below10M,not final
confirmation. See [the coverage guide](../docs/mujoco_remaining_transfer.md).

## 2026-10-28 measured follow-ups

All eleven earlier arrays completed. The independent HalfCheetah confirmation fails the paper-reading criterion and is retained. Full-budget Football and HalfCheetah partition development studies follow completed pilots; SMAC first tests the released25×400 collection layout. See [frozen studies](../experiments/followup_studies_20261028.json) and [resource plan](../experiments/followup_release_plan_20261028.json).

The aggregate ceiling remains eight workers,16 CPUs and64GiB. One30-minute native SMAC gate may request32GiB, replacing the earlier8GiB per-worker limit for this diagnostic only. With two4GiB MuJoCo pilots,two4GiB HC runs and two8GiB Football runs,maximum concurrency is seven workers/eight CPUs/64GiB. This is an autonomous bounded allocation under standing authorization; no GPU or paid service. SMAC learning remains unsubmitted pending all five native checks and measured memory.

## SMAC measured memory allocation,2026-10-28

All five25×400 native checks pass; sampled peak21.24GiB and worst1M phase projection8.48h. Release the five frozen1M controls with two2CPU/32GiB/14h workers. The total memory ceiling rises64→96GiB,while eight workers and16CPUs remain the ceiling;actual scheduled maximum8workers/10CPUs/96GiB. This measured,autonomous allocation runs the same declared study faster. No GPU,paidcompute,unrelatedjob changes or extra experimental trials. [Release record](../experiments/smac_layout25_learning_release_20261028.json). Both tasks follow diagnostic21252362 so the temporary2GiB diagnostic cannot overlap the full96GiB reservation.


## Full-budget entropy control after final-policy diagnosis

All nine AgentMixer final policies exactly replay under their frozen source. The complete diagnostic identifies scale/clamp behavior but does not establish causality. Test entropy0 versus0.01 at the same10M budget/source/configuration/grid for all three variants and development seeds0–2. Job21252423 reuses the two MuJoCo coverage-worker slots after21251756 completes;the8worker/16CPU/96GiB ceiling is unchanged. All18 old/new outcomes are required before paired interpretation. This does not replace independent final seeds or the failed BPPO confirmation. See [frozen plan](../experiments/hc2x3_entropy0_10m_plan_20261028.json).


## Full-budget coverage continuation, 29 October 2026

The 37 remaining MuJoCo 500k conditions and all five SMAC 25×4001M controls passed complete raw audits. SMAC shows no consistent layout gain. Extend all five methods to 10M without other setting/source changes:21272799,2 CPU/ 32 GiB/60 h each, max 2. Use 25×400 for released modifier-layout alignment, not pilot winner selection; architecture and non-modifier 50×200 differences remain.

Extend all 48 audited Ant/Walker/remaining MuJoCo pilot conditions to 10M with every control retained and only budget changed:21272800,1 CPU/4 GiB/30 h each, max 2, after the entire HC partition array 21251961. Exact 20-module source. Both studies use seed 0 development first; no final confirmation claim. All declarations for each task must complete and validate before task-level comparisons, all study outcomes remain visible, no early-peak replacements. Current independent BPPO failure remains unchanged; later selected configurations require new final seeds.

The two arrays reuse existing slots, preserving 8workers/16 CPU/96 GiB; scheduled maximum 8workers/10 CPU/96 GiB. No requeue or unrelated-job changes. Linear worst projections 37.81 h SMAC and 17.03 h MuJoCo have at least 50% wall-time headroom; full batches take several days. [Evidence and all endpoints](../docs/progress_20261029.md).


## Fresh HC3×2 confirmation and completed entropy control, 29 October morning

All12 partition development runs are valid. All3 HC3×2 BPPO endpoints exceed the figure reading upper bound5030, while paired AR comparisons remain mixed. Freeze ten independent runs with seeds5000–5004 and evaluation seed3000000, absent from the782-run inventory. Exact19-module source and every other resolved setting remain unchanged. Primary final10M mean; paired effects, AUC, mean/median/SD and whole-seed bootstrap intervals reported separately. No replacement/pooling. Job21279129 reuses completed entropy-array slots, at most two1CPU/4GiB/7h tasks. A revised held registration selects all six3×2 runs instead of mixed-task environment signatures; no training started before validation passed.

All18 entropy-paired10M runs are valid. Zero entropy lowers all three final means and AUCs; it does not resolve the modifier gap. Retain this negative result; do not expand that sweep without a new evidence-based hypothesis. Current long-budget coverage studies continue.

A complete-study exporter now verifies and packages frozen source, all configurations, full logs, summaries and checkpoints with portable operational paths. The actual12-run HC study passes relocation and identical-curve checks. Native replay21279857 waits for entire confirmation21279129 and reuses a freed slot. No active jobs or training source change. [Current evidence](../docs/progress_20261029_am.md).


## Local Gaussian scale control, 2026-10-29

After the complete entropy-zero comparison lowered all three variant means and AUCs, isolate observation dependence of continuous AgentMixer's local uncertainty. The released local scale is state-independent; the new optional vector retains the current exp/clamp map, bounds, initial std and all mean/modifier initial tensors. Default observation-dependent behavior is bitwise unchanged in 24 cases through one Adam step. Do not bundle recurrence, LayerNorm, activation or released sigmoid-local-map changes.

Freeze residual/bounded/identity × observation/state-independent scale at positive entropy 0.01. Gate: six seed-0 native 12k runs, true 1,000-step episode limit, two episodes at initial/final evaluations and two final checkpoint replay episodes per run. Require exact default-prefix agreement against completed positive-entropy seed-0 runs on common numerical fields; source/config/raw accounting, all replay episodes and measured resources must pass before learning. Native job 21280689 waits after 21279857 (bundle replay), which waits after the entire fresh HC3×2 confirmation. It reuses those two slots at 1 CPU/4 GiB/30 minutes each, max two workers; ceiling remains 8 workers/16 CPUs/96 GiB.

The 18-run 1M plan (six conditions × development seeds 0–2) is frozen but unsubmitted. Keep all outcomes; final 1M local-mean return is primary, normalized 0–1M curve area and within-variant paired scale differences are secondary. No peak replacement or independent-confirmation claim. Require measured native timings/RSS before releasing its 2-hour-per-run envelope. Full 10M and fresh final seeds remain necessary after any selection. Plans: `experiments/hc2x3_local_scale_study_20261029.json`; schedule: `experiments/hc2x3_local_scale_native_schedule_20261029.json`.


## Remaining recurrent discrete coverage, 2026-10-29

Three Football scenarios and eight SMAC maps have short native integrations but lack the recurrent training-layout gate used for current learning. Freeze and queue all remaining shapes with five controls each (MAPPO, AR-PPO, BPPO, AgentMixer, identity), seed0, one10k rollout, four complete initial/final episodes and exact replay of four final episodes. Source and settings match the completed reference cohorts, apart from scenario/gate budget/evaluation/checkpoint cadence. Football retains50×200/15epochs/2minibatches,25source modules; SMAC retains25×400/5epochs/1minibatch,23modules. All55frozen parsers pass; no learning or performance inference from gates. Native audit and per-task memory/time review are required before separate learning pilots.

Football21281001 waits for entire local-scale gate21280689 and uses one1CPU/8GiB/45min worker, reusing both4GiB confirmation-chain slots. **The unsubmitted18-run local-scale1M study must now wait for21281001 or use another verified free chain.** SMAC21281002 waits for entire currentSMAC10M21272799 and uses at most two2CPU/32GiB/60min workers. Existing jobs unchanged; ceiling8workers/16CPUs/96GiB unchanged. Later resource decisions must use actual native RSS/time, not assume the new agent/observation shapes fit. Plan `experiments/remaining_discrete_native_study_20261029.json`; guide `docs/discrete_remaining_coverage.md`.


## 2026-10-29: optional discrete local-distillation control

The complete released-count Prey diagnosis is in `docs/prey_distillation.md`; own-occupancy probability differences do not establish a causal return mechanism. An explicitly experimental detached central-to-local actor KL is implemented with coefficient0 by default. 1003 source tests and21 exact default-update comparisons pass. The frozen `experiments/prey_distillation_study_20261029.json` registers8 native gates and24 matched1M development runs, all unsubmitted. Preserve prior protocol/endpoints and all outcomes. Run the8 full-rollout gates with exact episode/window replay and4 coefficient0 first-update matches before any learning release, then measure native runtime/RSS and reserve a verified free dependency chain. No extra capacity is currently reserved; active/pending studies and continuous local-scale controls retain priority. Complete-study selection uses one coefficient across both penalties and requires separately registered unused final training/evaluation seeds.


### Prey native execution release, 2026-10-29

All8 native gates are queued as21282173 after the wholeFootball20M array21251960, with2×1CPU/4GiB/30min and no requeue. The partition overridebatch-milan matches the original Prey CPU family; exact prefixes remain a hard check. Full-family auditor21282210 follows the entire gate array,1CPU/2GiB/5min with frozenPYTHONPATH. See `experiments/prey_distillation_gate_execution_20261029.json`, schedule and collector receipts. All808historical evaluations pass the raw checker;48corruptions and the unstarted family are rejected. The24-run1M plan remains unsubmitted and unallocated until all8 native results,4exact default prefixes,resource measurements and a new free allocation are verified. Existing queued studies and the18-run continuous-local-scale learning plan retain priority. No active or pending existing job was modified.

The completed seed0 global Ant2×4 BPPO10M result is a promising candidate (6562.40; figure reading upper5050), with all3global controls now audited. It still needs additional training seeds and separately registered independent confirmation; no Ant follow-up has been submitted. Preserve the current48-run coverage campaign and report episode SD separately from training-seed uncertainty. See `docs/ant2x4_development.md`.


## Continuous AgentMixer local history, 29 October 2026

The release audit identifies recurrent local actors as another architecture difference. The current source now supports optional private GRU Gaussian actors with the same IGC means and scale maps. This tests a specific hypothesis after the completed learned modifiers and entropy-zero control failed to resolve the gap. Modifier/critic recurrence, ReLU/LayerNorm and released local scale bounds remain separate differences.

The [registered study](../experiments/hc2x3_recurrent_study_20261029.json) freezes twelve 44k native checks and eighteen GRU 1M development runs matched to all eighteen earlier feedforward local-scale runs. Native execution needs an audited memory observer, replay checks and complete-family auditor before submission; learning also requires measured resources and a free allocation. All thirty frozen configuration parsers pass, the full source suite passes 1,044 tests/one CUDA skip, and 29 existing behavior cases are bitwise unchanged. These are correctness results, not native MuJoCo validation or learning gains. Existing queued studies and prior local-scale learning retain priority. No new job is submitted by this preparation.


### Continuous-memory native execution, 29 October 2026

The observer and complete-family auditor are now frozen in `experiments/continuous_recurrent_gate_execution_20261029.json`. All twelve synthetic observed/unobserved conditions are bitwise identical in both the core and unchanged private MuJoCo runtimes; each rejects 47 corrupted cases and verifies a synthetic 44k-step run with forty resets and exact checkpoint replay. The raw reducer also accepts 9,090 historical native evaluation episodes. No new native gate has completed.

Native array **21284086** follows both the Ant final auditor **21282782** for capacity and all local-scale gates **21280689** for reference data. At most two batch-milan workers each use one CPU/4 GiB/30 min. Complete-family collector **21284087** follows the entire new array with one CPU/2 GiB/5 min. Both were submitted held, verified, and released with no requeue; no existing job was modified. Frozen helper/source/config hashes are checked before execution. The collector rejects missing or invalid conditions and cannot submit learning.

The eighteen recurrent learning runs remain unsubmitted until all twelve native conditions pass, the original six local-scale gates pass their own full audit, actual native resources are reviewed, and a free allocation is reserved. Prior eighteen-run feedforward learning retains priority after 21281001 or another free chain. Prey learning now waits for 21284087 or another verified free allocation. All sixty prepared learning runs remain excluded from the ledger coverage count. The new snapshot indexes 883 runs/74 manifests across 45 targets, with zero basic provenance errors; ten targets have exact-budget method artifacts including failures, not ten reproduced targets.


### HC 3×2 complete-study verification, 29 October 2026

The registered 5000–5004 paired BPPO/AR study retains its original nineteen-module source, ten 10M runs, evaluation seed3000000 and32complete episodes every500k. Two of ten runs are complete and valid; all group, paired and figure decisions remain withheld. Final auditor21284586 is queued after the entire remaining Football native array21281001, reusing the HC-chain allocation with one CPU/4GiB/30min and no requeue. No existing job changed. Prior18-run feedforward local-scale learning now waits for21284586 or another verified free allocation; it retains priority over the later recurrent and Prey learning studies.

The auditor first requires all ten strict records, then replays all ten final native checkpoints with the original32episodes, up to320k additional evaluation interactions. Every original config, metrics file, training-episode file, summary and checkpoint hash must remain unchanged. Negative results remain valid data; missing or invalid runs prevent acceptance. Preflight accepts the completed negative HC2×3 study, rejects nine corrupted reports and the incomplete current family, verifies all six real HC3×2 development schemas, and imports the exact evaluator in the unchanged private runtime. No final native replay is claimed. All126 development evaluation timings, never confirmation outcome scores, inform the 17.06min margin estimate.

MuJoCo coverage now has four of48runs raw-audited:4040episodes/10000updates. New Ant local MAPPO endpoint1437.34 is a single-seed baseline; its matched local methods remain incomplete. All883ledger-indexed runs/74manifests and45targets retain zero basic provenance errors, including negative outcomes. There are still60prepared unsubmitted learning runs.


### Current core wheel verification, 29 October 2026

An exact committed export of f7f9de3 now passes1044installed tests/oneCUDA-only skip in54.93s with source-path injection disabled. This includes28discrete-distillation and41continuous-memory cases, with synthetic continuous checkpoint replays. All25installed module bytes, README and license match. Both matrix commands, recurrent Prey and an additional positive-distillation recurrent Prey CLI run/replay pass outside the export. The tiny64/16-step budgets are integration evidence.

The inactive prior CPU validation prefix was warm-reused with the identical dependency lock and inventory. Only bta-repro was reinstalled without dependencies; the original wheel, modules and README were restored exactly, and no active training environment changed. Fourteen command logs and archive/wheel/JUnit/source hashes are retained in `experiments/core_wheel_controls_verification_20261029.json`. This is not a fresh-environment, hosted-CI or native-engine claim. The earlier975-test fresh install remains separate evidence. Native performance and all independent confirmation requirements remain open; no new study or learning job is submitted by this package check.


### Measured coverage acceleration, 29 October 2026

The existing 48-run MuJoCo10M array21272800 now permits four concurrent workers rather than two. All48 complete prior pilots,48unchanged config hashes,20source modules,manifest and frozen dispatcher pass preflight; source and per-task limits remain unchanged. One `scontrol update JobId=21272800 ArrayTaskThrottle=4` is applied. The original two active raw handles/start times are preserved. Two already registered Walker2×3 MAPPO/AR tasks start in the added slots and match pilot native versions, training streams and first-rollout accounting. No new job or scientific trial is submitted.

This measured autonomous revision supersedes the earlier eight-worker/96GiB policy with a ceiling of ten workers/sixteen CPUs/104GiB. Current four allocation branches sum to ten workers/twelve CPUs/104GiB. The original study estimates277.96worker-hours; ideal worker-hour division changes138.98→69.49h for the whole original campaign, not a forecast of remaining time. Queueing, task imbalance and host load remain relevant. No paid service is added and the other reproduction is untouched. Existing independent confirmation criteria, all-run reporting and learning-gate priorities remain unchanged.

Current resources and coverage are recorded in `experiments/live_resource_check_20261029_acceleration.json` and `experiments/benchmark_coverage_20261029_acceleration.provenance.json`. The same883runs/74manifests cover45targets,0basic provenance errors;60prepared learning runs remain unsubmitted. The first helper's syntax error occurred before any execution; its source is retained and successfulv2records the actual scheduler change.
