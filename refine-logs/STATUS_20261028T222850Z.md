# Reproduction status

Updated 2026-10-28. **Full reproduction and performance beyond the papers are not established.** The latest independent HalfCheetah BPPO confirmation fails its frozen performance criterion. All results, including failures, remain recorded.

## Latest completed evidence

All 127 runs in the eleven previously queued arrays completed with scheduler exit 0. The simulator audits verify 40,378 raw evaluation episodes, 84,800 Predator–Prey windows and 67,918 rollout updates; the paired matrix reporter validates all 20 matrix runs. Numerical validity does not establish learning success.

| Study | Complete result | Interpretation |
| --- | --- | --- |
| HalfCheetah 2×3 BPPO/AR, 10M, five fresh seeds each | BPPO 5372.06±390.17; AR 5782.67±516.02 | BPPO below figure reading[5660,5740]; paired difference interval includes zero |
| AgentMixer HalfCheetah 2×3, 10M, three development seeds | Bounded 3819.11±2315.14; residual 3560.85±679.86; identity 5569.45±372.68 | Learned modifiers underperform identity; no AgentMixer reproduction claim |
| AgentMixer matrix learned/identity, 1M, five pairs/game | Learned threshold successes 4/5 Climbing, 3/5 Penalty; identity 0/5 | Climbing is 5× the paper budget; Penalty is additional; failed seeds retained |
| Prey released-count FF/GRU, 1M, seed0 | Zero-penalty AgentMixer 1.9520/2.0515; identity 2.2160/2.2465 | Released 100×200-window reward/decision; identity score cannot establish AgentMixer success |
| SMAC Terran5v5 FF/GRU, 1M, seed0 | Final wins 0–5/32 across ten controls | Low win rates; below 10M; collection-layout check follows |
| Football3v1 FF/GRU, 1M, seed0 | Final wins 0–8/100 across ten controls | Below 20M; full-budget recurrent controls now running |
| MuJoCo transfer and native checks | All15 pilots and37 native gates valid | Task coverage and integrity evidence; full budgets/seeds remain |

Uncertainty above is sample SD across training seeds. See the [complete progress report](docs/progress_20261028.md), [independent confirmation](docs/halfcheetah_confirmation.md), [AgentMixer10M comparison](docs/agentmixer_halfcheetah10m.md) and [matrix ablation](docs/matrix_modifier_ablation.md).

## Active work

| Job | Work | Allocation |
| --- | --- | --- |
| 21251756 | 37 frozen MuJoCo500k coverage pilots | At most2 tasks,1CPU/4GiB/90min each |
| 21251961 | 12 HalfCheetah3×2/6×1 BPPO/AR development runs, 10M, seeds0–2 | At most2 tasks,1CPU/4GiB/7h each |
| 21251960 | Five recurrent Football3v1 controls,20M,seed0 | At most2 tasks,1CPU/8GiB/60h each |
| 21252369 | Five recurrent SMAC25×400 collection-layout controls, 1M, seed0 | At most2 tasks,2CPU/32GiB/14h each |
| 21252423 | Nine matched AgentMixer HalfCheetah10M entropy-zero controls | At most2 tasks,1CPU/4GiB/7h each,after21251756 |

All five SMAC native checks passed: 40 raw episodes, five updates, 26m13s and 21.24 GiB peak memory. The [measured learning allocation](experiments/smac_layout25_learning_release_20261028.json) raises total project memory to **96 GiB**, retaining the eight-worker and sixteen-CPU ceilings. Current scheduled maximum is eight workers/ten CPUs/96 GiB. The entropy-zero study reuses the two MuJoCo coverage-worker slots after that array finishes.

The [completed read-only diagnostic](experiments/hc_agentmixer_final_diagnostic_complete_20261028.json) exactly replays all nine AgentMixer final policies: 90 episodes and 90,000 additional evaluation decisions. Its original float32 assertion failure and corrected rerun are both retained; training artifacts were unchanged. Other reproduction jobs, files and environments remain untouched. Query recorded job IDs for live state; artifacts alone do not establish scheduler state.

## What is verified

- Shared explicit implementations of MAPPO, AR-PPO, BPPO and AgentMixer; [BPPO guide](docs/bppo.md) and [AgentMixer guide](docs/agentmixer.md), independent rollout/evaluation seeds, legal-action and termination handling, checkpoint replay and optional private GRU actors.
- All 96 initial MuJoCo/SMAC native integrations and 16 Football integrations pass. Predator–Prey matches10,240 pinned-release transitions. These short checks do not establish learned performance.
- The clean installed-wheel check passed 899 tests with 1 CUDA-only skip before three later reporting-only budget tests. Training defaults and frozen snapshots were preserved. Hosted CI remains unverified; [clean-install evidence](docs/clean_checkout.md).
- Independent BPPO/AR matrix1M confirmation passes all 20 runs near the games' return ceilings. Both methods succeed; the paired endpoint intervals include zero. [Evidence](docs/matrix_confirmation.md).
- All 45 canonical targets are indexed. The [latest coverage snapshot](experiments/benchmark_coverage_20261028.provenance.json) contains715 runs / 63 manifests with zero provenance errors. Seven targets have exact-budget method artifacts, including unsuccessful outcomes;38 still lack them. Artifact counts are not success counts.

## Required before release

Comparable full-budget results across all 45 targets; successful disjoint final training seeds and ablations; resolved or clearly qualified historical protocol differences; portable release artifacts with verified claims. Current failed configurations are retained. Any revised configuration needs new final seeds after development.

Use the [simulator guide](docs/simulator_training.md) to run frozen configurations and the [tracker](refine-logs/EXPERIMENT_TRACKER.md) for the detailed evidence history. The previous status narrative is preserved in `refine-logs/STATUS_20261028_083812.md`.
