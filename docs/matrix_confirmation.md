# Matrix configuration validation

The selected configuration uses a 64-dimensional predecessor-action projection
and entropy coefficient 0.1, held for half the budget and linearly reduced to
0.01. Both AR-PPO and BPPO solved both games in all three development seeds at
100,000 decisions. The larger tested coefficient, 0.3, also succeeded; we chose
the smaller coefficient. This is configuration selection, not evidence that
peer feedback outperforms the matched AR-PPO baseline.

Two separate studies were frozen on 2026-10-26, before any of their runs started:

| Study | Methods and games | Seeds | Decisions per run | Purpose |
| --- | --- | --- | --- | --- |
| Independent configuration validation | AR-PPO and BPPO; Climbing and Penalty | 100–109 | 100,000 | Test the selected short-budget configuration on ten unused seeds per method and game |
| Budget extension | Same methods and games | 0, 1, 2 | 1,000,000 | Test the longer entropy schedule on development seeds before selecting a full-budget configuration |

The plans are [100k validation](../experiments/matrix_confirmation100k_plan.json)
and [1M development](../experiments/matrix_budget1m_plan.json). They record every
resolved configuration, source and configuration hashes, the frozen manifest,
software versions, seed inventory audit, metrics and reporting rules. The prior
experiment ledger contained only training seeds 0, 1 and 2. The plans, manifests
and output paths are separate; their results must not be pooled.

The primary score is exact expected execution reward per joint decision,
computed by enumerating all nine actions **only during evaluation**. Greedy
reward and optimal-action probability are reported alongside it. An incidental
optimal argmax at initialization is not evidence of learning. At 100k, the
predeclared BPPO validation criterion requires every seed in each game to
finish, select an optimal greedy action and achieve expected reward at least
95% of the ceiling: 10.45 for Climbing and 9.5 for Penalty. A missing run prevents
a completed validation decision. A failed seed remains in the report and is
not replaced.

Reports include every seed, mean, median, sample standard deviation, bootstrap
uncertainty and paired BPPO-minus-AR-PPO differences. Threshold success counts
use all declared seeds, with Wilson intervals; even ten successes do not imply
perfect population reliability. Full-curve area and the first logged point
after which expected reward stays above 95% are diagnostics. The crossing is
limited to the recorded grid, generally every 10,240 decisions.

The current BPPO and payoff files are byte-identical to the selected development
snapshot. The runner's optimizer extraction preserves the two methods' parameter
ownership and initialization. Its added late learning-rate multiplier defaults
to one, preserving their constant rate. Added AgentMixer and continuous-policy
branches do not apply to these runs. Source hashes distinguish the two snapshots
without silently claiming whole-repository byte identity.

Increasing the budget to 1M changes the entropy hold/decay interval from
50k–100k to 500k–1M. Therefore the longer run is a new development setting, not
an extension of the independent 100k validation claim. Further tuning after
observing validation results must use a new untouched seed set for any later
independent claim.

Neither study alone completes the paper reproduction. The 100k study is below
the BPTA paper budget; the 1M study uses development seeds. Architecture, batch
and epoch settings remain documented reconstruction choices. The historical
matrix figure-to-logger mapping is unresolved, and matrix payoff ceilings
prevent strict improvement in maximum return. Cross-paper sample-efficiency
and algorithm-superiority claims require additional matching and evidence.

## Reporting

The standard-library-only reporter audits the pre-run plan against its job-ledger
hash, source snapshot, manifest, launcher and configuration files. Completed
runs must match their recorded software versions, start after the freeze,
finish at the declared budget, and contain the exact expected evaluation grid.
It recomputes expected reward, greedy payoff and optimal-action probability
from the frozen payoff table and the nine logged action probabilities. Summary
and final curve values must agree. Missing or invalid runs remain visible;
their groups and paired comparisons are not aggregated. A changed study-level
input invalidates the report without removing declared runs.

```bash
python scripts/report_matrix_validation.py \
  experiments/matrix_confirmation100k_plan.json \
  --output results/matrix_confirmation_report.json \
  --csv results/matrix_confirmation_report.csv
```

Use `experiments/matrix_budget1m_plan.json` and different output names for the
development extension. Outputs must be new files. The reporter does not query
the scheduler and its `paper_reproduction` field remains `not_established` even
when the specific independent configuration validation passes. Training
metadata's generic matrix-development label is retained; validation context
comes from the independently frozen study plan.

The initial preflight reports are
`results/matrix_confirmation100k_preflight_20261026.json` and
`results/matrix_budget1m_preflight_20261026.json`, each with a CSV. They recorded
all 40 and 12 runs respectively as not started, with no provenance errors.
Jobs `21213329` and `21213330` use the [recorded serial worker
dependencies](worker_schedule.md); the four-worker resource limit is preserved.
