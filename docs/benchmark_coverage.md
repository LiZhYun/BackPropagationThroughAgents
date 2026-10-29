# Benchmark evidence coverage

As of 29 October 2026, full-budget coverage is optional and is not a release-readiness percentage. The user requested stopping full-budget studies; all preserved partial outcomes remain indexed. See the [current scope](../experiments/reproduction_scope_reduced_budget_20261029.json).

`python scripts/report_benchmark_coverage.py` lists all 45 paper targets and the
experiment artifacts recorded in `experiments/jobs.jsonl`. It needs only the
Python standard library and reads files without querying or changing scheduler
jobs. It never compares scores or certifies reproduction.

```bash
python scripts/report_benchmark_coverage.py \
  --output results/coverage_snapshot.json --csv results/coverage_snapshot.csv
```

Outputs must be distinct new files. The full JSON retains every indexed run,
configuration, artifact status, source/configuration/summary hashes, related
targets, task-field differences and errors. The CSV is a compact table. Result
directories are not included in the source checkout; absent local artifacts
are reported as missing manifests or unfinished runs, not recreated.

## What the columns establish

| Column | Meaning |
| --- | --- |
| `paper_budget` | Training steps from the target's source configuration, following inheritance |
| `declared_method_runs` | Related runs using BPPO for BPTA or an enabled AgentMixer modifier for AgentMixer; includes unfinished and task-field mismatches |
| `completed_consistent_method_runs` | Completed summaries reach their declared budget, recorded configuration/source agrees with the frozen manifest, source files match their hashes, and the explicit basic task checks pass |
| `max_completed_method_steps` | Largest recorded budget among those consistent method runs; no score-based selection |
| `exact_paper_budget_method_runs` | Number of those runs whose budget exactly equals the source budget; longer runs do not count here |

Identity-modifier, independent and autoregressive baselines remain visible as
controls. A run can relate to both papers when task settings overlap, but it
counts as a method only for its corresponding paper. Counts are **not** numbers
of independent seeds: the same development seed can appear under many settings.
No scores or seeds are pooled across those settings.

The basic checks use task identity and a deliberately limited set of fields:
MuJoCo scenario, partition, observation radius, 1,000-step horizon and normalized
action coordinates; SMAC scenario and game version; Football scenario; Prey
penalty and 128-step horizon; or matrix game name. For example, local-observation
Ant runs cannot count toward BPTA's full-observation setting, and 16-step
MuJoCo integrations do not count as canonical-horizon runs. The JSON exposes
these differences rather than discarding the runs.

These checks do **not** establish complete simulator/observation/reward protocol
equivalence, metric reductions, learning-curve integrity, independent seed
registration, successful performance or a method's theoretical claims. A fully
counted run may have a very poor score. Use the [curve reporter](learning_curve_reports.md)
for metric and curve validation, then audit the source protocol, method controls,
held-out training seeds and actual comparison before making a release claim.

Manifest/source failures remain explicit. Missing frozen source, changed source,
configuration disagreement, wrong final budgets or duplicate output paths
prevent a run from counting as consistent. Duplicate ledger updates and path
aliases to one manifest do not duplicate evidence. The older eight-run smoke
manifest lacks a seed field in each entry; its explicit single-run configuration
supplies the seed, and that origin is recorded. No seed is guessed.

## 2026-10-26 20:46 UTC snapshot

The [coverage CSV](../experiments/benchmark_coverage_20261026T204631Z.csv) and its
[provenance](../experiments/benchmark_coverage_20261026T204631Z.provenance.json) index
596 runs from 55 manifests. All 45 targets are retained, including those with
only short integrations or queued experiments. This snapshot has no manifest
or run-provenance errors.

| Suite | Targets | Targets with any consistent method artifact at the exact paper budget |
| --- | ---: | ---: |
| Matrix | 3 | 3 |
| MA-MuJoCo | 21 | 1 |
| SMACv2 | 15 | 0 |
| Football | 4 | 0 |
| Predator–Prey | 2 | 2 |

These six targets have evidence at the specified training budget. That count
does not establish successful reproduction: it includes the failed AgentMixer
Climbing criterion and unsuccessful Prey outcomes. Both BPTA matrix games now
have complete independent configuration checks, with matched AR-PPO controls
also near the ceilings. The three completed 10M BPPO HalfCheetah seeds remain
development evidence; independent native confirmation is queued. The other
39 targets still lack a completed method run at the exact paper budget.

The [earlier 444-run snapshot](../experiments/benchmark_coverage_20261026.csv)
and its [provenance](../experiments/benchmark_coverage_20261026.provenance.json)
remain unchanged. Snapshot counts must not be read as live scheduler state.

Sixteen focused tests cover observation mismatches, identity controls, shortened
horizons, short budgets, failed/missing runs, provenance changes, explicit older
seed records, path aliases and all supported task families. Training code and
running frozen experiments are unchanged by this report.


## Snapshot 29 October 2026

The [new snapshot](../experiments/benchmark_coverage_20261029.provenance.json) indexes 782 runs/67 manifests, all 45targets, and zero manifest/run-provenance errors. Eight targets have completed exact-budget method artifacts and 37 still lack them. This includes unsuccessful outcomes; it is not eight successful reproductions. The 53 newly queued full-budget follow-ups do not count as complete. [Current progress](progress_20261029.md).

The [12:27 Helsinki snapshot](../experiments/benchmark_coverage_20261029_midday.provenance.json)
indexes 853 runs/71 manifests with zero basic provenance errors. Nine targets
have completed exact-budget method artifacts and 36 still lack them. It includes
six queued continuous local-scale checks and 55 queued [remaining discrete
scenario checks](discrete_remaining_coverage.md); none of those short checks
establishes learned performance. The unsubmitted 18-run local-scale learning
plan is not included in the jobs ledger.
