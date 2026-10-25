The remaining supplied figures support **18 additional benchmark endpoint
estimates**: BPTA's two matrix games, nine SMACv2 tasks, and four Football tasks;
plus AgentMixer's climbing game and two Predator-Prey tasks. Together with the
[21 MuJoCo estimates](curve_targets.md) and six tabulated AgentMixer SMACv2
results, these cover the existing 45-row benchmark inventory. This is coverage
of source targets, not completion of their reproduction. No canonical target
configuration or original reference was changed by this extraction.

The [structured output](../results/target_extraction_other/curve_targets.json)
also retains 14 secondary curves: nine SMAC training curves, three Football
training curves, and two projected-ARMAPPO ablation curves. Primary and secondary
records are explicitly distinguished. The nine selected PDFs contain 32 selected
centerlines with 1,690 vector vertices in total.

Each endpoint below is `rounded estimate [digitization reading range]`.
Brackets describe an explicit geometric reading allowance, not seed variation,
standard errors, or confidence intervals. Values are last plotted evaluations,
without extrapolation or selection of an earlier best checkpoint.

| Matrix task | Method | Last plotted steps | Plotted episode-return endpoint | Source status |
| --- | --- | ---: | ---: | --- |
| Climbing | BPPO | 1,000,000 | 2,200 [2,180, 2,220] | Active main figure |
| Penalty | BPPO | 1,000,000 | 2,000 [1,960, 2,040] | Active main figure |
| Climbing | AgentMixer | 200,000 | 11.00 [10.81, 11.19] | Commented appendix figure |

BPTA's payoff tables give one-decision optima of 11 and 10. Its matrix chart
plateaus are approximately 200 times these payoffs, and its hyperparameter table
lists `episode length = 200` (lines 529–531). This supports a **scale inference**,
not proof that an environment episode consists of 200 decisions. The number
could also describe rollout length or a logging reduction. AgentMixer's archived
table likewise lists length 200 (lines 1229–1231), but its figure reaches 11.
The extraction preserves original plot units and never silently divides one
paper's results by 200.

Both manuscripts call the matrix game stateless, but do not specify the exact
observation/state tensor, previous-action inputs supplied by the environment, or
time features. Autoregressive within-decision action conditioning is a separate
method feature. A read-only check of the released code at commit
`13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0` resolves the **pinned implementation**:

- [The matrix environment](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/envs/matrix/matrix_env.py#L30-L104)
  sets `length=1`, `use_fixed_obs=True`, and observation `[1, 1]` for every agent.
  Every decision terminates. The active observation path contains neither prior
  actions nor time; an inactive alternative would expose the step counter.
- [The temporal runner](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/runner/temporal/matrix_runner.py#L59-L111)
  collects `episode_length` transitions, counts steps as updates × rollout
  length × parallel environments, and logs mean training reward multiplied by
  rollout length. It copies observations into critic inputs and clears recurrent
  state/masks on each terminal transition. Its evaluation path sums rollout
  rewards and divides by rollout length, with observation-space upper bound 1,
  yielding per-decision rewards (lines 254–319).

Thus a 200-decision **environment episode** is not supported by this pinned
implementation: 200 is a rollout/logging length. The runner dispatches both
sequential BPPO-style training and modifier-based joint training, controlled by
`use_action_attention`. The archived 2,200 versus 11 scales are consistent with
these training versus evaluation logger reductions, but their exact association
with the plotted curves is still an inference. No plotting scripts, run IDs, or
checkpoints connect the figures to particular logger keys. Current example
launchers also include action attention and a 1M budget, rather than establishing
the historical AgentMixer 200k configuration. The local source captures and Git
blob identities are retained in
[legacy protocol provenance](../results/target_extraction_other/legacy_protocol/provenance.json),
with a [structured protocol audit](../results/target_extraction_other/legacy_protocol/audit.json).

This source check supports constant-observation one-step semantics for the
released code; it does not prove complete equivalence to the historical plotted
runs. The legacy Climbing payoff also exchanges the first and third rows of the
BPTA payoff table, so identical action labels do not identify identical outcomes.
In particular, original training versus deterministic evaluation metrics,
batch/update schedules, action-row conventions, and advantage construction must
still be matched before claiming a 100k-versus-230k sample-efficiency improvement.

For a reproducible sample-efficiency diagnostic, the extractor uses 90%, 95%,
and 99% of each positive scale-equivalent optimum: 2,200/2,000 for BPTA and 11 for
AgentMixer. These thresholds are **our analysis choices**, not thresholds stated
in either paper, and are not reward-range normalization. Matrix curves have
10,000-step vertex spacing. A crossing is bracketed by adjacent plotted vertices;
the script does not infer a precise interpolated evaluation time.

| Method/task | Fraction of optimum | Threshold in plot units | First observed crossing | Previous vertex | First crossing whose lower reading envelope clears threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| BPPO Climbing | 90% | 1,980 | 190k | 180k | 200k |
| BPPO Climbing | 95% | 2,090 | 230k | 220k | 240k |
| BPPO Climbing | 99% | 2,178 | 340k | 330k | 560k |
| BPPO Penalty | 90% | 1,800 | 170k | 160k | 180k |
| BPPO Penalty | 95% | 1,900 | 200k | 190k | 220k |
| BPPO Penalty | 99% | 1,980 | 340k | 330k | Not resolved by reading allowance |
| AgentMixer Climbing | 90% | 9.90 | 80k | 70k | 80k |
| AgentMixer Climbing | 95% | 10.45 | 80k | 70k | 80k |
| AgentMixer Climbing | 99% | 10.89 | 80k | 70k | Not resolved by reading allowance |

Horizontal reading allowances expand each bracket by approximately 4.1k steps
on each side for BPPO's main matrix figure and 0.8k for AgentMixer. The JSON
records separate possible and certain crossings under the geometric allowance.
For example, BPPO Climbing's 95% first observed bracket is approximately
[215.9k, 234.1k] after horizontal allowance; vertical allowance permits an earlier
crossing at the 220k vertex and only guarantees a crossing by the 240k vertex.
These allowances are not probabilities about training success.

All three primary matrix centerlines stay above each threshold after their
first observed crossing. The projection ablation illustrates why this must be
checked: projected ARMAPPO Penalty first crosses 95% near 220k, but does not stay
above it at every remaining plotted vertex until about 940k. Its centerline
crosses 99% near 470k and stays above from about 960k. This is curve stability,
not a recovered seed-level success rate. The full
[threshold table](../results/target_extraction_other/matrix_thresholds.csv)
and JSON retain both first and sustained crossings.

| BPTA SMACv2 figure label | Test win-rate endpoint | Training win-rate endpoint |
| --- | ---: | ---: |
| 10gen_protoss 5v5 | 0.502 [0.498, 0.505] | 0.465 [0.461, 0.468] |
| 10gen_protoss 10v10 | 0.444 [0.441, 0.447] | 0.395 [0.392, 0.398] |
| 10gen_protoss 10v11 | 0.090 [0.089, 0.091] | 0.085 [0.084, 0.087] |
| 10gen_zerg 5v5 | 0.469 [0.466, 0.472] | 0.412 [0.409, 0.415] |
| 10gen_zerg 10v10 | 0.425 [0.422, 0.427] | 0.384 [0.381, 0.386] |
| 10gen_zerg 10v11 | 0.238 [0.236, 0.240] | 0.219 [0.217, 0.221] |
| 10gen_terran 5v5 | 0.504 [0.501, 0.507] | 0.466 [0.463, 0.469] |
| 10gen_terran 10v10 | 0.349 [0.347, 0.351] | 0.352 [0.349, 0.355] |
| 10gen_terran 10v11 | 0.124 [0.123, 0.125] | 0.125 [0.123, 0.126] |

All SMAC curves end at approximately **9.76M**, with 40 vertices and approximately
250k-step spacing. They do not provide a point exactly at 10M. Rates are fractions
on a 0–1 scale. `SMAC.pdf` is the active appendix's test figure; `SMAC_incre.pdf`
is the active main training figure. They must not be pooled. The manuscript's
mention of 20v20/20v23 conflicts with the plotted 5v5/10v10/10v11 titles. The six
prose-only 20v20/20v23 tasks have no identified corresponding panels; no numerical
targets are assigned to them.

| BPTA Football task | Test win rate at 20M | Training win rate at 20M |
| --- | ---: | ---: |
| academy_3_vs_1_with_keeper | 0.754 [0.749, 0.759] | 0.796 [0.790, 0.801] |
| academy_counterattack_hard | 0.612 [0.607, 0.617] | 0.638 [0.633, 0.643] |
| academy_corner | 0.644 [0.639, 0.649] | 0.690 [0.685, 0.695] |

The active main `Football_train.pdf` and active appendix `Football.pdf` contain
50 vertices per curve, with approximately 400k-step spacing. The counterattack
panel is labeled generically; its mapping to the hard task follows the prose and
existing benchmark inventory. `Football_5v5.pdf` instead reports **averaged
episode return**, ending at **−0.29 [−0.30, −0.28] at 20M**. Its caption does not
resolve train versus test aggregation by itself. The author
[confirmed evaluation return on 2026-10-25](author_clarifications.md).
Historical normalization and episode handling remain unresolved. It must not
be converted into a win rate.

| AgentMixer Predator-Prey task | Endpoint at 1M, in plotted return units |
| --- | ---: |
| Penalty = 0 | 2.21 [2.19, 2.23] |
| Penalty = −1 | 0.36 [0.35, 0.36] |

`Prey_main.pdf` is referenced only by commented appendix source. Its accompanying
commented text calls the metric average test-episode return and describes reward
10 for capture. The same author clarification confirms evaluation return for
both penalties. The PDF does not determine reward normalization, team versus
agent reduction, or episode length; the extracted 2.21/0.36 values must retain
their plotted scale. AgentMixer's active main text refers readers to additional
Climbing and Predator-Prey results, but their figure inclusion blocks remain
commented in the supplied source. Their aggregation across seeds and their band
statistic are unresolved. The SMAC median caption does not define these curves.

BPTA's global experiments paragraph specifies means and standard deviations over
five seeds. That statement is attached as source context, not verified from
underlying data. Band polygons and midpoint offsets are saved separately; the
extractor does not derive confidence intervals from them. AgentMixer's original
seed-level arrays are likewise unavailable. Both archive inventories were
searched for numerical arrays and plotting scripts; none were present.

[The new script](../scripts/extract_other_curves.py) reuses the immutable geometry
helpers from the [MuJoCo extractor](../scripts/extract_curve_targets.py), without
regenerating MuJoCo outputs. It verifies original archive hashes and byte equality
of every selected PDF/manuscript with its ZIP member. Per-panel metadata include
the legend text and stroke, axis tick/grid positions, scale multiplier, signed
tick repairs, centerline/band path indices, point coordinates, source scope, and
file hashes. Legend matching identifies red `#e41a1c` in all nine figures;
projected ARMAPPO is labeled as that distinct method. AgentMixer's matrix curve
is dashed and overlaps MAT after convergence, but its own path identity and
legend stroke remain distinguishable.

Some negative tick signs are vector glyphs omitted from PDF text extraction.
Their signs were visually checked and are explicitly transcribed in the recipe
for BPPO Penalty, both projection panels, Football 5v5, and AgentMixer Climbing.
The script still requires the observed numeric magnitudes, grid positions, and
linear mapping to agree; it rejects mismatches. This prevents a silent sign
error in the negative-return panels. The maximum tick-fit residual is below
0.000034 PDF points.

The reading convention matches the MuJoCo extraction: 0.001 point for coordinate
precision, 0.25 point for registration, half the 1-point curve stroke, plus the
axis-fit residual. The total is converted using the relevant axis slope.
Endpoints round to 10 plotted returns for BPTA matrix figures, 0.001 for win
rates, and 0.01 for the other returns; interval boundaries round outward. A
range extending above a theoretical payoff ceiling reflects this geometric
allowance, not a claim that a policy can exceed the ceiling.

All nine originals and nine overlays were visually reviewed. Thin blue traces
follow the selected red paths and endpoint circles agree with their final
vertices. The render artifacts are paired below; the original PDFs were never
written.

| Figure | Original | Verified overlay |
| --- | --- | --- |
| BPPO matrix | [image](../results/target_extraction_other/bppo_matrix_original.png) | [image](../results/target_extraction_other/bppo_matrix_verified_overlay.png) |
| ARMAPPO projection | [image](../results/target_extraction_other/bppo_matrix_projection_original.png) | [image](../results/target_extraction_other/bppo_matrix_projection_verified_overlay.png) |
| BPPO SMAC test | [image](../results/target_extraction_other/bppo_smac_test_original.png) | [image](../results/target_extraction_other/bppo_smac_test_verified_overlay.png) |
| BPPO SMAC train | [image](../results/target_extraction_other/bppo_smac_train_original.png) | [image](../results/target_extraction_other/bppo_smac_train_verified_overlay.png) |
| BPPO Football test | [image](../results/target_extraction_other/bppo_football_test_original.png) | [image](../results/target_extraction_other/bppo_football_test_verified_overlay.png) |
| BPPO Football train | [image](../results/target_extraction_other/bppo_football_train_original.png) | [image](../results/target_extraction_other/bppo_football_train_verified_overlay.png) |
| BPPO Football 5v5 | [image](../results/target_extraction_other/bppo_football_5v5_original.png) | [image](../results/target_extraction_other/bppo_football_5v5_verified_overlay.png) |
| AgentMixer matrix | [image](../results/target_extraction_other/agentmixer_matrix_original.png) | [image](../results/target_extraction_other/agentmixer_matrix_verified_overlay.png) |
| AgentMixer Predator-Prey | [image](../results/target_extraction_other/agentmixer_prey_original.png) | [image](../results/target_extraction_other/agentmixer_prey_verified_overlay.png) |

The [endpoint CSV](../results/target_extraction_other/endpoint_summary.csv),
[all vertices](../results/target_extraction_other/digitized_curves.csv), and
[verification record](../results/target_extraction_other/validation.json)
accompany the structured output. Reproduce from the repository root with the
same optional dependency installation used for MuJoCo:

```bash
PYTHONPATH=results/target_extraction/_deps python3 scripts/extract_other_curves.py
```

These target estimates support transparent provisional comparisons. They do not
resolve missing evaluation protocols, establish statistically higher performance,
or replace independent-seed experiments.
