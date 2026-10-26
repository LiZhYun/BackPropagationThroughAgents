# Benchmark evidence coverage

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

## 2026-10-26 snapshot

The [coverage CSV](../experiments/benchmark_coverage_20261026.csv) and its
[provenance](../experiments/benchmark_coverage_20261026.provenance.json) index
444 runs from 41 manifests. All 45 targets are retained, including those with
only short integrations or queued experiments. This snapshot has no manifest
or run-provenance errors.

| Suite | Targets | Targets with any consistent method artifact at the exact paper budget |
| --- | ---: | ---: |
| Matrix | 3 | 1 |
| MA-MuJoCo | 21 | 1 |
| SMACv2 | 15 | 0 |
| Football | 4 | 0 |
| Predator–Prey | 2 | 2 |

This is **four targets with budget-level artifacts, not four reproduced targets**.
It includes unsuccessful matrix and Prey outcomes. The two completed 10M BPPO
HalfCheetah seeds remain development evidence; their third seed and matched
baselines were unfinished at this snapshot. Independent final confirmation and
full benchmark coverage remain outstanding.

Sixteen focused tests cover observation mismatches, identity controls, shortened
horizons, short budgets, failed/missing runs, provenance changes, explicit older
seed records, path aliases and all supported task families. Training code and
running frozen experiments are unchanged by this report.
