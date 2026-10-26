# AgentMixer Climbing configuration validation

Ten unused training seeds test the selected joint-modifier configuration at
200,000 decisions. All ten runs now finish with valid records. The predeclared
criterion fails: eight seeds meet the local expected-return threshold. This
tests local execution reliability on Climbing. It has no
matched ablation and cannot establish a modifier advantage.

The setting uses actor learning rate 0.0125, modifier rate 0.0005, constant
entropy coefficient 0.01, batch 1024, four PPO epochs, hidden size 64, and
float64 joint probabilities. Networks remain float32. The experimental joint
modifier remains explicit; the factorized default is unchanged.

The [development results](matrix_final_diagnosis.md#completed-actor-rate-control-2026-10-26)
motivated this fixed selection: local Climbing expected returns were 11, 11,
and 10.6879 on seeds 0–2, with optimal greedy actions in all three. One central
expected return was only 9.0941. Penalty had optimal greedy actions but poor
local stochastic returns; those negative results remain part of the evidence.

| Item | Frozen choice |
| --- | --- |
| Training seeds | 2000–2009, absent from all 523 previously prepared runs across 47 manifests |
| Source | Exact eight-module snapshot of the selected faster-actor float64 experiment |
| Configuration | Every resolved field except training seed matches the selected Climbing run |
| Budget | Exactly 200,000 joint decisions per run |
| Primary evaluation | Exact local expected team reward over all nine joint actions; no sampled evaluation seed |
| Success | Every seed has greedy return 11 and local expected return at least 10.45 at the final budget |
| Diagnostics | Central expected return and optimal-action probability, local curve area, sustained threshold crossing |
| Uncertainty | Whole-seed bootstrap with 10,000 draws, private seed 1729, plus a Wilson success interval |
| Failures | Keep all declared seeds; no replacements, development pooling or best-checkpoint selection |

The [plan](../experiments/matrix_agentmixer_confirmation200k_plan.json) includes
source/configuration hashes, software versions, selection evidence and all ten
resolved configurations. The [prior-seed inventory](../experiments/matrix_agentmixer_confirmation_prior_seeds_20261026.json)
includes the eight older smoke configurations whose manifests omitted a seed
field; their declared configurations specify seed 0. The copied trainer
validated every new configuration before the plan was frozen.

The standard-library reporter separately recomputes local and central scores
from their respective probabilities. Central scores cannot substitute for
execution scores. Missing or invalid runs prevent completion, and an empty
or mismatched validation rule cannot pass. The initial
[preflight](../experiments/matrix_agentmixer_confirmation_preflight_20261026.json)
records all ten runs as not started, with no integrity errors.

```bash
python scripts/report_matrix_validation.py \
  experiments/matrix_agentmixer_confirmation200k_plan.json \
  --output results/agentmixer_confirmation_report.json \
  --csv results/agentmixer_confirmation_report.csv
```

Job **21218175** completed in 22m30s on one serial worker, one CPU and 4GB,
within its 45-minute cap. Batch peak resident memory was 375,176KiB.
Measured development Climbing runs took about 1.5 minutes each. It was submitted
held, checked, and inserted between `21209234_0` and the pending
`21213860_0`; the same four-worker ceiling applies. The
[schedule proof](../experiments/matrix_agentmixer_confirmation_schedule_20261026.json)
records the verified dependencies and resources before release.

Even a passing study would validate this reconstruction setting only. The
budget is below the paper's 1M matrix budget, historical protocol differences
remain, and the payoff ceiling prevents return above 11. No cross-paper sample
efficiency, Penalty reliability, population-perfect convergence or algorithm
advantage follows from this study. Future tuning after these seeds are observed
requires a new untouched seed set for independent validation.

## Completed result

The [complete audit](../experiments/matrix_agentmixer_confirmation200k_results_20261026.json)
and [seed-level CSV](../experiments/matrix_agentmixer_confirmation200k_results_20261026.csv)
retain all ten seeds and 210 evaluation points. Source bytes, frozen settings,
software versions, exact budgets and probability-derived scores pass validation.
The statistical criterion **fails**, with 8/10 successes and a Wilson 95% interval
[0.4902, 0.9433]. No seed is replaced and the threshold remains 10.45.

All ten greedy joint actions are optimal, returning 11. The two local expected
returns below threshold are:

| Training seed | Local expected return | Local optimal-action probability | Central expected return | Greedy return |
| --- | ---: | ---: | ---: | ---: |
| 2000 | −2.370520 | 0.673883 | 11.000000 | 11 |
| 2008 | 8.919642 | 0.948357 | 11.000000 | 11 |

Across all ten seeds, local expected return is 9.453557 ± 4.205604 (sample SD),
with median 10.999149 and bootstrap 95% mean interval [6.777780, 10.999079].
Central expected return is 10.625784 ± 0.788938; seeds 2005 and 2007 have central
returns 9.141265 and 9.116576 despite local expected return 11. Local and central
stochastic distributions can differ in either direction while preserving
optimal greedy modes. The results do not establish reliable stochastic local
execution or a modifier advantage. They also cannot replace the separate
Predator–Prey or native-simulator evaluations.
