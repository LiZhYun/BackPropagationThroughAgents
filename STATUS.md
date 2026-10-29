# Reproduction status

Updated 2026-10-29. **Full reproduction and performance beyond the papers are not established.** The independent HalfCheetah BPPO confirmation failed its frozen criterion; learned AgentMixer modifiers also underperformed identity in the completed 10M study. Both findings remain recorded.

## Latest progress

- **Ant 2×4 BPPO completes 10M:** final evaluation return 6,562.40 for development seed 0, above figure reading [4,990, 5,050]. All three global-observation controls are audited (3,030 episodes/7,500 updates). Five fresh paired seeds are now registered and queued, followed by a full audit and native replay. [Result and caveats](docs/ant2x4_development.md); [confirmation rules](docs/ant2x4_confirmation.md).
- **Predator–Prey diagnosis:** all 160 saved diagnostic episodes replay exactly; mode consistency holds throughout, with no active modifier clamp/floor. Learned policies have a central/local probability gap. An optional actor-only distillation objective passes analytic/masked/recurrent checks; all 21 disabled-objective comparisons are bitwise identical. Eight native gates are queued after the current Football array, followed by a complete-family auditor. The 24 matched development runs remain unsubmitted. [Evidence and limitations](docs/prey_distillation.md).
- **HalfCheetah partitions: 12/12 complete and audited.** At 10M, BPPO 3×2 averages 5431.33±294.34 across three development seeds, all above figure reading bound 5030. Its paired advantage over AR-PPO is small and mixed; 6×1 remains below its upper bound.
- **Fresh 3×2 confirmation: 2/10 complete and valid.** The first two BPPO seeds pass the strict artifact audit; the next two runs are around 6.37M/10M. A complete-study auditor and all ten final checkpoint replays are queued as 21284586. Both method aggregates remain withheld until their complete seed sets are available. The failed 2×3 independent result remains unchanged. [Study and final verification](docs/halfcheetah3x2_confirmation.md); [partial audit](experiments/hc3x2_confirmation10m_progress_20261029_1400.json).
- **Entropy control: 9/9 complete.** Zero entropy lowers all three variant means and curve areas compared with 0.01; it does not fix the AgentMixer gap.
- Football is around 17.1M/20M and SMAC around 5.44M/10M for their first two baseline runs. Four of 48 MuJoCo coverage runs are complete and raw-audited (4,040 episodes/10,000 updates); the next two are around 3.56M and 0.23M. These [recorded counters](experiments/live_resource_check_20261029_hc3_audit.json) are progress observations, not final results.
- An optional state-independent local Gaussian scale now isolates another documented AgentMixer architecture difference. All 1,044 current source tests pass (one CUDA skip), and 24 default-compatibility cases are bitwise identical through one Adam step. Six native gates are queued; the matched 18-run learning plan is frozen and unsubmitted. See the [control specification](docs/agentmixer_continuous.md#local-uncertainty-control).
- **Continuous AgentMixer now supports optional private GRU memory.** The core Gaussian mode constraint is preserved; all 29 existing-policy compatibility cases are bitwise identical. The memory observer passes bitwise neutrality checks in both core and private MuJoCo runtimes; each rejects 47 corrupted cases. Twelve native 44k checks and their complete-family auditor are now queued. Eighteen recurrent 1M development runs remain unsubmitted pending native results and resource review. No native learning gain is claimed. [Control and limitations](docs/agentmixer_continuous.md#optional-local-history).
- Remaining recurrent native coverage is queued for three Football scenarios and eight SMAC maps: 55 conditions tied to 16 canonical targets, with all five controls retained. No additional learning jobs are released. [Scope and dependencies](docs/discrete_remaining_coverage.md).
- A portable complete-study exporter passes 11 focused tests. The relocated 12-run HalfCheetah bundle reproduces all curves and aggregates; native checkpoint replay is pending separately.

See the [new control verification](experiments/continuous_local_scale_verification_20261029.json), the [morning progress report](docs/progress_20261029_am.md) for all comparisons and limitations, and the [export guide](docs/portable_studies.md) for portable artifacts.

## Active and queued work

| Job | Work | Allocation |
| --- | --- | --- |
| 21251960 | Five Football 3v1 recurrent 20M controls | At most 2 × 1 CPU/8 GiB/60 h |
| 21272799 | Five SMAC Terran 5v5 recurrent 10M controls | At most 2 × 2 CPU/32 GiB/60 h |
| 21272800 | All 48 MuJoCo pilot conditions at 10M | At most 2 × 1 CPU/4 GiB/30 h |
| 21279129 | Ten fresh-seed HC 3×2 BPPO/AR confirmation runs | At most 2 × 1 CPU/4 GiB/7 h |
| 21279857 | Native replay from relocated evidence, after confirmation | One 1 CPU/2 GiB/10 min task |
| 21280689 | Six local-scale native checks and checkpoint replays, after bundle replay | At most 2 × 1 CPU/4 GiB/30 min |
| 21281001 | Fifteen remaining Football native checks, after local-scale checks | One 1 CPU/8 GiB/45 min worker |
| 21281002 | Forty remaining SMAC native checks, after current SMAC study | At most 2 × 2 CPUs/32 GiB/60 min |
| 21282173 | Eight Prey distillation native checks, after the whole Football 20M array | At most 2 × 1 CPU/4 GiB/30 min |
| 21282210 | Complete Prey gate audit, after all eight checks | One 1 CPU/2 GiB/5 min task |
| 21282468 | Ten fresh-seed Ant 2×4 BPPO/AR confirmation runs, after Prey audit | At most 2 × 1 CPU/4 GiB/7 h |
| 21282782 | Complete Ant audit and ten native checkpoint replays, after Ant confirmation | One 1 CPU/4 GiB/30 min task |
| 21284086 | Twelve continuous-memory native checks, after Ant audit and all local-scale checks | At most 2 × 1 CPU/4 GiB/30 min |
| 21284087 | Complete memory-gate audit, after all twelve checks | One 1 CPU/2 GiB/5 min task |
| 21284586 | Complete HC 3×2 confirmation audit and ten native replays, after remaining Football native array | One 1 CPU/4 GiB/30 min task |

The schedule stays within eight workers/sixteen CPUs/96 GiB. The Football allocation continues through Prey checks, Ant confirmation, continuous-memory checks and their respective audits; [current graph](docs/worker_schedule.md#hc-3x2-final-verification-29-october-2026). The unsubmitted Prey learning study must wait for the memory-gate auditor or another verified free allocation. The unsubmitted local-scale learning study must wait for the HC 3×2 final auditor or another verified free allocation. The [live resource check](experiments/live_resource_check_20261029_hc3_audit.json) confirms all eight active workers below their memory allocations, with linear runtime projections below wall limits; those projections are not guarantees. Other reproduction work is untouched. Batches require several days; query recorded jobs for live state.

## What is verified

- Shared explicit implementations of MAPPO, AR-PPO, BPPO and AgentMixer; [BPPO guide](docs/bppo.md) and [AgentMixer guide](docs/agentmixer.md), independent rollout/evaluation seeds, legal-action and termination handling, checkpoint replay and optional private GRU actors.
- All 96 initial MuJoCo/SMAC native integrations and 16 Football integrations pass. Predator–Prey matches 10,240 pinned-release transitions. These short checks do not establish learned performance.
- The current wheel from commit `f7f9de3` passes **1,044 tests with 1 CUDA-only skip**, with source-path injection disabled. All 25 installed modules match; matrix commands, recurrent Prey and positive-distillation checkpoint replays pass outside the checkout. The inactive locked CPU prefix was reused and its original wheel restored; the earlier fresh-environment check remains separate. Optional native engines are absent and hosted CI remains unverified. [Installed-wheel evidence](docs/clean_checkout.md).
- Independent BPPO/AR matrix 1M confirmation passes all 20 runs near the games' return ceilings. Both methods succeed; the paired endpoint intervals include zero. [Evidence](docs/matrix_confirmation.md).
- All 45 canonical targets are indexed. The [latest snapshot](experiments/benchmark_coverage_20261029_hc3_audit.provenance.json) contains 883 runs/74 manifests with zero provenance errors. Ten targets have completed exact-budget method artifacts, including unsuccessful outcomes; 35 still lack them. Artifact counts are not success counts.

## Required before release

Comparable full-budget results across all 45 targets; successful disjoint final training seeds and ablations; resolved or clearly qualified historical protocol differences; portable release artifacts with verified claims. Current failed configurations are retained. Any revised configuration needs new final seeds after development.

Use the [simulator guide](docs/simulator_training.md) to run frozen configurations and the [tracker](refine-logs/EXPERIMENT_TRACKER.md) for the detailed evidence history. Previous status snapshots are preserved under `refine-logs/STATUS_*.md`.
