The archived MA-MuJoCo figures support reproducible **estimates of plotted
endpoints**, with the limitations below. They do not supply exact numerical
targets or original evaluation returns. This extraction covers BPPO's nine
panels and AgentMixer's twelve panels; nine task names occur in both sources.
The canonical `configs/paper_targets.json` is unchanged.

Each table entry is `rounded centerline estimate [digitization reading range]`,
in episode-return units. The brackets describe reading precision, **not seed
variation or a confidence interval**. Every selected curve ends at approximately
**9.904 million environment steps**, despite axes extending to 10 million.
The endpoint is the last plotted vertex, without extrapolation or selection of
the best earlier evaluation.

| Shared task name | BPPO | AgentMixer | AgentMixer figure scope |
| --- | ---: | ---: | --- |
| Ant-v2 2x4 | 5,020 [4,990, 5,050] | 1,300 [1,280, 1,320] | Active main figure |
| Ant-v2 4x2 | 5,250 [5,220, 5,280] | 1,600 [1,580, 1,610] | Active main figure |
| Ant-v2 8x1 | 5,160 [5,130, 5,200] | 1,290 [1,280, 1,300] | Active main figure |
| HalfCheetah-v2 2x3 | 5,700 [5,660, 5,740] | 5,040 [5,000, 5,070] | Archived appendix asset; source inclusion commented |
| HalfCheetah-v2 3x2 | 4,990 [4,960, 5,030] | 4,910 [4,870, 4,950] | Archived appendix asset; source inclusion commented |
| HalfCheetah-v2 6x1 | 4,580 [4,550, 4,610] | 3,010 [2,980, 3,030] | Active main figure |
| Walker2d-v2 2x3 | 980 [960, 990] | 1,020 [1,000, 1,040] | Archived appendix asset; source inclusion commented |
| Walker2d-v2 3x2 | 530 [520, 540] | 540 [520, 550] | Archived appendix asset; source inclusion commented |
| Walker2d-v2 6x1 | 640 [630, 650] | 350 [340, 360] | Archived appendix asset; source inclusion commented |

These columns are not a controlled comparison between methods. In particular,
BPTA specifies `agent_obsk=None` for Ant, whereas AgentMixer describes restricting
agents to their own observable elements and its archived parameter table gives
`agent_obsk=0`. The simulator and observation protocol must be matched separately
to each paper. See [the source audit](source_audit.md) and
[benchmark manifest](../configs/paper_targets.json).

The remaining AgentMixer panels are inexpensive to extract from the same assets:

| Task | AgentMixer | Figure scope |
| --- | ---: | --- |
| Ant-v2 2x4d | 1,710 [1,690, 1,730] | Active main figure |
| Humanoid-v2 17x1 | 660 [650, 670] | Active main figure |
| HumanoidStandup-v2 17x1 | 87,600 [86,900, 88,200] | Archived appendix asset; source inclusion commented |

The [BPTA manuscript](../references/bpta/Formatting-Instructions-LaTeX-2024.tex)
states that results use the mean and standard deviation of five seeds (line
276), includes `MuJoCo.pdf` in active source (line 313), and gives the MuJoCo
budget and observation settings at lines 601, 607 and 621. Consequently BPPO's
centerline is labeled as a mean according to that statement; the individual
seed data are unavailable for verification.

The [AgentMixer manuscript](../references/agentmixer/aaai25.tex) includes
`MuJoCo_main.pdf` in active source (line 411) but leaves `MuJoCo_app.pdf` commented
out (line 941). The main MuJoCo text does not explicitly identify the centerline
as a mean or median, or define its shaded band. The median/standard-deviation
caption at line 559 belongs to SMACv2, so it is not applied here. A commented
significance-test paragraph at line 1063 mentions averaging the last five
evaluations over five seeds; this does not identify the MuJoCo plotting
aggregation. Appendix-asset values are therefore archived evidence, not a claim
that those panels appear in the supplied active manuscript.

There is also a statistical inconsistency that the figures cannot resolve.
For example, BPPO Walker2d 2x3 ends near 975.4, with a plotted band from 749.6 to
1,402.8. Its band midpoint is about 100.8 above the centerline, so the geometry
does not match a symmetric mean-plus/minus-standard-deviation construction at
that point. AgentMixer Walker2d 2x3 has a similar midpoint offset of 106.0.
Undocumented smoothing or aggregation could explain this, but no plotting code
is supplied. The output preserves both band boundaries and their midpoint
offset; it does not convert them into standard errors or confidence intervals.

[The extraction script](../scripts/extract_curve_targets.py) verifies the
hashes in [the source manifest](../references/sources.json) and checks that each
selected PDF and manuscript exactly matches its ZIP member. The BPTA archive
contains 21 files and AgentMixer contains 29; neither contains numerical-array
files or plotting scripts. The three selected PDFs have no embedded attachment
files. Their metadata identify Matplotlib PDF backends, and their centerlines
are vector polylines rather than raster images.

The script associates each method label with its adjacent legend stroke. All
three legends identify the requested method by red RGB `#e41a1c`. It then finds
each titled panel's gray frame and selects the unique red centerline and red
translucent polygon inside that frame. Each central curve contains 100 vertices.
Axes are calibrated independently from numeric tick labels and their grid-line
positions, including the displayed `1e7` multiplier. A linear least-squares map
converts PDF coordinates into steps and returns. The largest tick-fit residual
is below 0.000025 PDF points. PDF coordinates use points (1/72 inch), a top-left
origin, and downward-positive y; the fitted y slopes are negative.

For illustration, the first BPPO Ant panel's x=0 and x=10M grid lines are near
75.15621 and 281.34186 PDF points. The red line ends at x=279.36249, which maps to
9,904,000 steps. Using the curve bounds themselves to calibrate a 0–10M axis
would incorrectly move this final evaluation to 10M. The approximately 4,000
first-step value and 100,000-step vertex spacing are recovered from the same
tick calibration; they are not imposed as assumptions.

The stated reading allowance sums 0.001 point for coordinate precision, 0.25
point for grid registration, half the 1-point centerline stroke, and the measured
axis-fit residual. Multiplying by the absolute y-axis slope gives a return-unit
allowance. Estimates are rounded to the nearest 10 returns, or nearest 100 for
HumanoidStandup; range boundaries round outward. This is an explicit conservative
digitization convention, not an empirically calibrated probability bound. It
does not cover unknown evaluation rules, smoothing, seed variability, or
differences in simulator versions. Unrounded coordinates remain in the audit
artifacts solely to make the calculation reproducible.

The JSON also records the last five plotted values, their mean, and their range.
These are an aid to assessing endpoint volatility, not a recovered paper metric.
For example, AgentMixer Ant 2x4 has an endpoint near 1,300 and a last-five-vertex
mean near 1,498; BPPO Walker2d 6x1 has an endpoint near 638 and a last-five mean
near 714. A new benchmark must choose its evaluation aggregation before using
these figures for comparisons.

The outputs are:

- [Structured targets and provenance](../results/target_extraction/curve_targets.json):
  source and archive hashes, PDF metadata, text/grid/path identities, complete
  axis mappings, endpoint ranges, band bounds, and unresolved source questions.
- [Endpoint summary](../results/target_extraction/endpoint_summary.csv) and
  [all 2,100 plotted vertices](../results/target_extraction/digitized_curves.csv).
- [Verification record](../results/target_extraction/validation.json), including
  hashes for the nine numeric/image artifacts that matched a second extraction
  byte for byte.
- BPPO [original](../results/target_extraction/bppo_original.png) and
  [annotated overlay](../results/target_extraction/bppo_verified_overlay.png).
- AgentMixer main [original](../results/target_extraction/agentmixer_main_original.png)
  and [annotated overlay](../results/target_extraction/agentmixer_main_verified_overlay.png).
- AgentMixer appendix [original](../results/target_extraction/agentmixer_app_original.png)
  and [annotated overlay](../results/target_extraction/agentmixer_app_verified_overlay.png).

All six renders were visually reviewed: the thin blue traces follow the selected
red curves, and blue circles mark their last vertices. Titles, numeric axes, and
legend identity agree with the recorded extraction in all 21 panels. The
overlays are generated in memory and rendered at 108 dpi; source PDFs are never
written. Repeating the extraction is deterministic for the pinned renderer.

From the repository root, reproduce the artifacts without modifying the
training environment:

```bash
python3 -m pip install --target results/target_extraction/_deps PyMuPDF==1.26.5
PYTHONPATH=results/target_extraction/_deps python3 scripts/extract_curve_targets.py
```

These estimates can inform provisional engineering targets. Establishing that
new results exceed paper claims still requires matching the benchmark protocol,
resolving the relevant aggregation, and evaluating fresh seeds. Narrow
digitization ranges do not supply missing experimental uncertainty.
