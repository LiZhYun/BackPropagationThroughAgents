# Reporting an independent simulator confirmation

The current reporter audits the frozen BPPO/AR-PPO episodic-return study in
[HalfCheetah 2×3](halfcheetah_confirmation.md). It uses only Python's standard
library and does not load checkpoints, execute archived source or start engines.

```bash
python scripts/report_simulator_confirmation.py \
  experiments/hc2x3_confirmation10m_plan.json \
  --output results/hc_confirmation_report.json \
  --csv results/hc_confirmation_runs.csv
```

Both output paths must be new. The JSON contains every declared run, integrity
errors, complete-group statistics, paired differences and a qualified figure
comparison. The CSV keeps missing, failed and invalid runs visible. A command
that produces a report is not itself evidence that the study passed: inspect
`validation_decision`, `errors` and each run's `eligible` field.

## Evidence checked

The report verifies the study-plan hash against its pre-run submission record;
the manifest, complete run set, configurations, source files and launchers;
and the hashed pre-freeze seed inventory. Confirmation requires at least five
unused training seeds per method. The two methods must share every resolved
training field except algorithm and training seed. Derived environment streams
must be distinct and separate from evaluation.

Run metadata must match the complete frozen configuration, source and software
versions. Native engine revisions must agree with the selected development
report, whose hash was frozen in the study record. Run creation must follow
submission. Paths are restricted to this repository, including symlink targets.

For completed runs, the report checks every evaluation call against the planned
schedule, including rollout rounding. It recomputes mean return, sample episode
SD, mean length, termination counts and reward per decision from the raw episode
records. Each evaluation must contain the declared number of complete episodes,
using the fixed seed and deterministic autoregressive execution. Initial and
final summary records must exactly match the curve's episode records.

Every recorded rollout update must have the expected number, step coordinate
and interaction shape, including the final partial rollout. These records must
account for the exact training budget. Evaluation interactions are counted
separately and checked against the summary. A final checkpoint must exist; its
hash is recorded. The reporter does **not** decode that checkpoint or establish
policy replay. Use `scripts/evaluate_simulator.py` with the frozen source for
the separate saved-policy replay check.

## Statistics and decisions

A method's mean, median, sample SD and full-budget curve area are computed only
after all its declared seeds are valid and complete. No selected checkpoint,
common prefix, interpolation to fill missing checkpoints, replacement seed or development score
enters the aggregate. Missing endpoints and malformed completed logs block it.
A partially written final line in a live run remains incomplete.

Both methods must complete before the report compares them or the figure.
Pairing uses the training seed, regardless of manifest row order. Differences
are BPPO minus AR-PPO for both final return and normalized area. The uncertainty
unit is the whole training seed, retaining pairs for differences: 10,000 private
RNG resamples with seed 1729 and two-sided 95% percentile intervals. Evaluation
episodes are not treated as independent training seeds.

The figure comparison distinguishes a mean above the upper digitization bound
from a lower bootstrap mean endpoint above that bound. Neither establishes
historical simulator, evaluation or smoothing equivalence. The reporter always
leaves `paper_reproduction` as `not_established`; one study cannot certify the
full release. A figure threshold and a positive paired method difference answer
separate questions.

The implementation is in
[`simulator_confirmation.py`](../src/bta_repro/simulator_confirmation.py), with
the curve reader and bootstrap calculation shared with
[`learning_curves.py`](../src/bta_repro/learning_curves.py). Tests cover valid and
negative results, pairing, missing seeds, changed provenance, raw episode
reductions, rollout accounting and paths outside the checkout.
