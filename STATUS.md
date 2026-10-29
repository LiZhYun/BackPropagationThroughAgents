# Reproduction status

Updated 2026-10-29. **Full reproduction and performance beyond the papers are not established.** The independent HalfCheetah BPPO confirmation failed its frozen criterion; learned AgentMixer modifiers also underperformed identity in the completed 10M study. Both findings remain recorded.

## Current scope: shorter budgets, 29 October 2026

The user removed full-budget execution as a release requirement and requested stopping already-submitted full-budget jobs. All five long-study arrays are stopped; completed and partial artifacts are preserved byte-for-byte. The two full-study collectors whose complete-group prerequisites can no longer pass are retired. Ten existing short native/artifact checks retain their original scientific dependencies, configurations and resource limits. The shorter 24-run Predator–Prey development study continues. No new learning is submitted.

Readiness now depends on method correctness, readable code and runnable examples, supported native integrations, selected predeclared multi-seed comparisons at explicit shorter budgets, independent evidence for claimed improvements, and portable artifacts with accurate limitations. Completing all 45 paper budgets is not required. The earlier 10/45 (22%) count describes historical exact-budget artifacts and does not measure readiness. [Scope](experiments/reproduction_scope_reduced_budget_20261029.json); [preserved outcomes](experiments/reduced_budget_stop_and_preservation_20261029.json).

The dated progress and job descriptions below retain earlier observations; this scope change supersedes their full-budget completion requirements.

## Evidence and observations before the scope change

- **Ant 2×4 BPPO completes 10M:** final evaluation return 6,562.40 for development seed 0, above figure reading [4,990, 5,050]. All three global-observation controls are audited (3,030 episodes/7,500 updates). Five fresh paired seeds are registered; the first two BPPO runs are training in an independent allocation, followed by a full audit and native replay. [Result and caveats](docs/ant2x4_development.md); [confirmation rules](docs/ant2x4_confirmation.md).
- **Predator–Prey diagnosis:** all 160 saved diagnostic episodes replay exactly; mode consistency holds throughout, with no active modifier clamp/floor. Learned policies have a central/local probability gap. An optional actor-only distillation objective passes analytic/masked/recurrent checks; all 21 disabled-objective comparisons are bitwise identical. All eight full-rollout native gates and the complete-family audit pass, including exact checkpoint replay and four default-prefix matches. The frozen 24-run 1M development study is running as 21287360 after measured resource review; its first run completes 1M and passes the raw audit (2,020 episodes/10,100 windows). All 100 updates and 101 evaluations repeat the original seed-0 baseline exactly, excluding elapsed time. Family selection remains withheld. Full-family audit and all 24 final replays are queued as 21288041. [Evidence and limitations](docs/prey_distillation.md).
- **HalfCheetah partitions: 12/12 complete and audited.** At 10M, BPPO 3×2 averages 5431.33±294.34 across three development seeds, all above figure reading bound 5030. Its paired advantage over AR-PPO is small and mixed; 6×1 remains below its upper bound.
- **Fresh 3×2 confirmation: 4/10 complete and valid.** The first four BPPO seeds pass the strict artifact audit; the remaining declared runs continue through the scheduler. A complete-study auditor and all ten final checkpoint replays are queued as 21284586. Both method aggregates remain withheld until their complete seed sets are available. The failed 2×3 independent result remains unchanged. [Study and final verification](docs/halfcheetah3x2_confirmation.md); [partial audit](experiments/hc3x2_confirmation10m_progress_20261029_1510.json).
- **Entropy control: 9/9 complete.** Zero entropy lowers all three variant means and curve areas compared with 0.01; it does not fix the AgentMixer gap.
- Football is around 18.7M/20M and SMAC around 6.45M/10M for their first two baseline runs. Four of 48 MuJoCo coverage runs are complete and raw-audited (4,040 episodes/10,000 updates). The existing MuJoCo array now runs four workers: two Ant local controls and two newly dispatched Walker 2×3 global controls. These [recorded counters](experiments/live_resource_check_20261029_sigmoid_native.json) are progress observations, not final results.
- An optional state-independent local Gaussian scale now isolates another documented AgentMixer architecture difference. All 1,093 current source tests pass (one CUDA skip), and 24 default-compatibility cases are bitwise identical through one Adam step. Six native gates are queued; the matched 18-run learning plan is frozen and unsubmitted. See the [control specification](docs/agentmixer_continuous.md#local-uncertainty-control).
- **Continuous AgentMixer now supports optional private GRU memory.** The core Gaussian mode constraint is preserved; all 29 existing-policy compatibility cases are bitwise identical. The memory observer passes bitwise neutrality checks in both core and private MuJoCo runtimes; each rejects 47 corrupted cases. Twelve native 44k checks and their complete-family auditor are now queued. Eighteen recurrent 1M development runs remain unsubmitted pending native results and resource review. No native learning gain is claimed. [Control and limitations](docs/agentmixer_continuous.md#optional-local-history).
- Remaining recurrent native coverage is queued for three Football scenarios and eight SMAC maps: 55 conditions tied to 16 canonical targets, with all five controls retained. No additional learning jobs are released. [Scope and dependencies](docs/discrete_remaining_coverage.md).
- A portable complete-study exporter passes 11 focused tests. The relocated 12-run HalfCheetah bundle reproduces all curves and aggregates; native checkpoint replay is pending separately.

See the [new control verification](experiments/continuous_local_scale_verification_20261029.json), the [morning progress report](docs/progress_20261029_am.md) for all comparisons and limitations, and the [export guide](docs/portable_studies.md) for portable artifacts.

The new optional bounded local Gaussian map preserves the AgentMixer mean constraint and passes analytic/replay tests. All35existing policy modes match bitwise through PPO; six native runs are queued and eighteen development runs remain unsubmitted. The current installed wheel also passes all 1,093 tests with one CUDA-only skip. [Control verification](experiments/continuous_sigmoid_verification_20261029.json).

## Submitted work after the scope change

| Job | Work | Allocation |
| --- | --- | --- |
| 21251960 | Stopped by user: Football 20M controls; artifacts retained | No active allocation |
| 21272799 | Stopped by user: SMAC 10M controls; artifacts retained | No active allocation |
| 21272800 | Stopped by user: MuJoCo 10M coverage; artifacts retained | No active allocation |
| 21279129 | Stopped by user: HC 3×2 independent confirmation; incomplete | No active allocation |
| 21279857 | Native replay of a completed archived study; confirmation stopped | One 1 CPU/2 GiB/10 min task |
| 21280689 | Six local-scale native checks and checkpoint replays, after bundle replay | At most 2 × 1 CPU/4 GiB/30 min |
| 21281001 | Fifteen remaining Football native checks, after local-scale checks | One 1 CPU/8 GiB/45 min worker |
| 21281002 | Forty remaining SMAC native checks, after current SMAC study | At most 2 × 2 CPUs/32 GiB/60 min |
| 21287360 | All 24 frozen Prey distillation/identity 1M development runs; native audit passed | At most 2 × 1 CPU/4 GiB/4 h |
| 21288041 | Complete 24-run Prey curve/raw-protocol audit, final replays and frozen selection rule; after whole learning array | One 1 CPU/4 GiB/45 min task |
| 21282468 | Stopped by user: Ant 2×4 independent confirmation; incomplete | No active allocation |
| 21282782 | Retired: complete Ant study prerequisite cannot pass | No active allocation |
| 21284086 | Twelve continuous-memory native checks; all local-scale reference checks still required | At most 2 × 1 CPU/4 GiB/30 min |
| 21289017 | Corrected full-grid memory-gate audit, after all twelve checks; replaces never-started read-only 21284087 | One 1 CPU/2 GiB/5 min task |
| 21289365 | Six sigmoid-scale native44k gates, after Prey collector and corrected original memory audit | At most 2 × 1 CPU/4 GiB/30 min |
| 21289366 | Complete sigmoid native-family audit, after all six checks | One 1 CPU/2 GiB/5 min task |
| 21284586 | Retired: complete HC study prerequisite cannot pass | No active allocation |
| 21288137 | Complete local-scale and remaining Football native-family audits, after the entire HC branch | One 1 CPU/2 GiB/10 min task |
| 21288138 | Complete remaining SMAC native-family audit, after all 40 checks | One 1 CPU/2 GiB/10 min task |

The unchanged short-check chains and Prey development reserve at most eight workers/ten CPUs/88 GiB, within the prior fourteen-worker/sixteen-CPU/120-GiB ceiling. No sources, configurations or scientific dependencies are edited. The cancelled full-study collectors had capacity-only edges to later native checks; their cancellation does not certify study success. All 54 prepared continuous 1M learning runs remain unsubmitted. The other reproduction is untouched. [Schedule amendment](experiments/reduced_budget_short_checks_schedule_20261029.json).

## What is verified

- Shared explicit implementations of MAPPO, AR-PPO, BPPO and AgentMixer; [BPPO guide](docs/bppo.md) and [AgentMixer guide](docs/agentmixer.md), independent rollout/evaluation seeds, legal-action and termination handling, checkpoint replay and optional private GRU actors.
- All 96 initial MuJoCo/SMAC native integrations and 16 Football integrations pass. Predator–Prey matches 10,240 pinned-release transitions. These short checks do not establish learned performance.
- The current wheel from commit `f311fc2` passes **1,093 tests with one CUDA-only skip**, with source-path injection disabled. All 25 installed modules, README and license match. Package/CLI/replay checks pass; the inactive validation prefix and dependency inventory are restored. This reuses the earlier locked environment. Optional native engines are absent and hosted CI remains unverified. [Installed-wheel evidence](docs/clean_checkout.md).
- Independent BPPO/AR matrix 1M confirmation passes all 20 runs near the games' return ceilings. Both methods succeed; the paired endpoint intervals include zero. [Evidence](docs/matrix_confirmation.md).
- All 45 canonical targets are indexed. The [latest snapshot](experiments/benchmark_coverage_20261029_sigmoid_native.provenance.json) contains 913 runs/76 manifests with zero provenance errors. Ten targets have completed exact-budget method artifacts, including unsuccessful outcomes; 35 still lack them. Artifact counts are not success counts.

## Required before release

Method correctness, readable shared code and examples, supported native integration, selected complete multi-seed studies at declared shorter budgets, independent confirmation and ablations for claimed improvements, qualified historical protocol differences, and portable artifacts with verified claims. Full-budget coverage across all 45 targets is not required. Current failed and interrupted configurations are retained; partial trajectories are not completed studies. Any revised configuration needs new final seeds after development.

Use the [simulator guide](docs/simulator_training.md) to run frozen configurations and the [tracker](refine-logs/EXPERIMENT_TRACKER.md) for the detailed evidence history. Previous status snapshots are preserved under `refine-logs/STATUS_*.md`.
