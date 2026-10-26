# Learning-curve reports

The final checkpoint and the entire learning curve answer different questions.
`scripts/summarize_learning_curves.py` reports both from a frozen simulator
manifest. It never selects the best checkpoint or silently removes an
unsuccessful seed. Its calculations use only Python's standard library.

```bash
python scripts/summarize_learning_curves.py results/EXPERIMENT/manifest.json \
  --metric return_mean --output results/EXPERIMENT/curves.json
```

The output must be a new file. For a fresh progress snapshot, choose another
output name. Existing logs and reports are never overwritten. The command also
prints compact group summaries; the JSON retains every declared run, its points,
eligibility errors, source/metric hashes and the evaluation denominator.

Available metrics are `return_mean`, `win_rate`, `capture_rate` and
`fixed_window_reward_per_step`. The last is the separately recorded Prey
fixed-window evaluation, including unfinished segments. The author selected this released
reduction for the primary Prey reproduction comparison on 2026-10-26; the exact
historical figure-to-logger mapping remains unverified. Success rates require a success
record for every evaluated episode.

For fixed-window scores, the reporter recomputes total reward and mean reward
per decision from **every recorded window**, including unfinished segments.
It checks the configured window count, length, evaluation seed, deterministic
execution convention, ordered indices, distinct uint32 window seeds and
episode-ending bookkeeping. The window seed list must remain identical across
checkpoints and pooled runs. Initial and final raw window records must match
the saved summary exactly. Missing or inconsistent evidence makes the run
ineligible and prevents its entire seed group from being aggregated.

These checks audit the recorded totals; they cannot independently reconstruct
rewards without transition traces or verify NumPy seed derivation with only the
standard library. Saved-policy replay tests the actual environment, policy and
seed stream. The report also does not turn a 20-window development run into the
author-selected 100-window comparison: its explicit protocol fields retain
that distinction.

The [recorded Prey audit](../experiments/prey_window_audit_20261026.json) applies
these checks to all sixteen completed 1M-step development runs: 80 evaluations,
8,000 windows and 1,600,000 evaluation decisions. Of those windows, 7,480 end
with an unfinished episode whose rewards remain included. The audit also checks
the frozen source bytes and separately recomputes every window seed with the
recorded NumPy scheme; none overlaps a training environment seed. All sixteen
runs pass. The [complete CSV](../experiments/prey_window_audit_20261026.csv)
retains each variant's final score and curve area. Results are unchanged: the
best zero-penalty rate is 2.142, and every penalty −1 rate remains negative.
These are development results, not independent confirmation.

## Area and aggregation

For logged evaluations `(s_i, r_i)`, normalized curve area is

```text
sum((s[i+1] - s[i]) * (r[i+1] + r[i]) / 2) / training_budget
```

The reporter uses the actual training-step coordinates, including rollout
rounding. The first point must be at step zero and the final evaluation must
match both the declared budget and the saved final summary. The integral is a
trapezoidal approximation between recorded evaluations. It does not recover
unobserved learning behavior, smooth the curve, or extrapolate missing steps.

A group is aggregated only when every declared training seed has a valid,
completed curve. Seeds must share training settings (apart from the seed),
simulator versions, the evaluation protocol and the exact evaluation grid.
Unequal grids are reported for inspection; no common-prefix selection or
implicit interpolation across seeds is performed. Failed and missing seeds
remain listed. A partially written final JSON line in a live run is explicitly
flagged and ignored; this never makes the run eligible for a final aggregate.

Each eligible group reports the mean, median and sample standard deviation
across training-seed scores for the final value and normalized area. Curve
points also report these seed statistics. A single seed has no estimated
training-seed SD. Variation among evaluation episodes is not used as a
substitute for independent training seeds.

For groups of at least five seeds, final values and normalized areas also have
95% percentile-bootstrap intervals for their mean and median: 10,000 resamples
of whole training-seed scores, private RNG seed 1729, and linear interpolation
between sorted bootstrap percentiles. The report records these choices. These
intervals describe the supplied seed sample; they do not establish that the
seeds were held out from configuration selection, that the paper protocol is
matched, or that a method is superior. Comparing two separate intervals is not
a paired test of a method difference.

## Figures

The optional plotting command uses a Python environment with Matplotlib
installed; it does not import the training package or simulator engines.
The verified plotting environment uses Python 3.11, NumPy 1.26.4 and Matplotlib
3.7.1. Its complete dependency pins are separate from all training environments:

```bash
python3.11 -m venv .venv-plots
PIP_CACHE_DIR="$PWD/.cache/pip-plots" .venv-plots/bin/python -m pip install \
  -r configs/environment_locks/plotting-requirements.txt
```

Use `.venv-plots/bin/python` for the plotting command below if Matplotlib is not
available in the current environment. `MPLCONFIGDIR="$PWD/.cache/matplotlib"`
keeps its font cache within the repository.

```bash
python scripts/plot_learning_curves.py results/EXPERIMENT/curves.json \
  --title 'HalfCheetah 2×3 — development' --output results/EXPERIMENT/curves.pdf
```

PDF, PNG and SVG outputs are supported. Lines are training-seed means and any
shading is ±1 sample SD, not a confidence band. Single-seed curves have no band.
The plot labels the seed counts and reports how many invalid or incomplete
groups were omitted; their details remain in the input JSON. A sidecar records
the input/reporting hashes, plotting-script hash and installed Matplotlib
version. Neither an existing figure nor its sidecar is overwritten.

The same training budget, metric and environment protocol are necessary for
meaningful comparisons. The reporter exposes these records; it does not
automatically certify the full reproduction checklist.
