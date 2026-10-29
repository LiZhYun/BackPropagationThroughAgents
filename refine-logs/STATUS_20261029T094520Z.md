# Reproduction status

Updated 2026-10-29. **Full reproduction and performance beyond the papers are not established.** The independent HalfCheetah BPPO confirmation failed its frozen criterion; learned AgentMixer modifiers also underperformed identity in the completed 10M study. Both findings remain recorded.

## Latest progress

- **37/37 MuJoCo 500k pilots and 5/5 SMAC 1M layout controls completed and passed raw audits.** Including finished runs from partial studies, 52 newly completed runs verify 10,896 raw episodes and 30,125 updates.
- HalfCheetah partition 10M: **8/12 complete**. Matched AgentMixer entropy-zero 10M: **2/9 complete**. No aggregate from incomplete cohorts.
- Football 20M: **0/5 complete**; first two runs are at about 7.8M. At 7.6M, MAPPO wins 90/100 matches and AR-PPO 61/100. These are partial baseline results; BPPO and AgentMixer are queued.
- SMAC 25×400 ends at 1–2 wins / 32 across five methods, with no consistent improvement over 2×5000. The missing full budget is now being tested for all controls.

See the [current progress report](docs/progress_20261029.md) for all 48 MuJoCo pilot endpoints, matched SMAC results, raw audits and limitations. The [previous completed studies](docs/progress_20261028.md) retain independent-confirmation failures and other negative results.

## Active and queued work

| Job | Work | Allocation |
| --- | --- | --- |
| 21251960 | Five Football 3v1 recurrent 20M controls | At most 2 × 1 CPU/8 GiB/60 h |
| 21251961 | Twelve HC 3×2/6×1 BPPO/AR 10M development runs | At most 2 × 1 CPU/4 GiB/7 h |
| 21252423 | Nine matched AgentMixer HC 2×3 entropy-zero 10M runs | At most 2 × 1 CPU/4 GiB/7 h |
| 21272799 | Five SMAC Terran 5v5 recurrent 10M controls | At most 2 × 2 CPU/ 32 GiB/60 h |
| 21272800 | All 48 MuJoCo pilot conditions at 10M; after 21251961 | At most 2 × 1 CPU/4 GiB/30 h; reuses its worker slots |

Both new studies change only the training budget from their audited pilots and retain every declared method/control. They use development seed 0, not independent final seeds. The batches require several days. Source, configurations, dependency and resources were verified before release; maximum schedule remains eight workers/ten CPUs/96 GiB. Other reproduction work is untouched. Query recorded jobs for live state; this is a dated snapshot.

## What is verified

- Shared explicit implementations of MAPPO, AR-PPO, BPPO and AgentMixer; [BPPO guide](docs/bppo.md) and [AgentMixer guide](docs/agentmixer.md), independent rollout/evaluation seeds, legal-action and termination handling, checkpoint replay and optional private GRU actors.
- All 96 initial MuJoCo/SMAC native integrations and 16 Football integrations pass. Predator–Prey matches 10,240 pinned-release transitions. These short checks do not establish learned performance.
- The clean installed-wheel check passed 899 tests with 1 CUDA-only skip before three later reporting-only budget tests. Training defaults and frozen snapshots were preserved. Hosted CI remains unverified; [clean-install evidence](docs/clean_checkout.md).
- Independent BPPO/AR matrix 1M confirmation passes all 20 runs near the games' return ceilings. Both methods succeed; the paired endpoint intervals include zero. [Evidence](docs/matrix_confirmation.md).
- All 45 canonical targets are indexed. The [latest coverage snapshot](experiments/benchmark_coverage_20261029.provenance.json) contains 782 runs/67 manifests with zero provenance errors. Eight targets have completed exact-budget method artifacts, including unsuccessful outcomes; 37 still lack them. Artifact counts are not success counts.

## Required before release

Comparable full-budget results across all 45 targets; successful disjoint final training seeds and ablations; resolved or clearly qualified historical protocol differences; portable release artifacts with verified claims. Current failed configurations are retained. Any revised configuration needs new final seeds after development.

Use the [simulator guide](docs/simulator_training.md) to run frozen configurations and the [tracker](refine-logs/EXPERIMENT_TRACKER.md) for the detailed evidence history. Previous status snapshots are preserved under `refine-logs/STATUS_*.md`.
