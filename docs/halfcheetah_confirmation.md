# HalfCheetah 2×3 independent confirmation

The three completed BPPO development seeds at 10M steps return 5843.80,
6046.91 and 6050.40. Their mean is 5980.37, with sample SD 118.29; normalized
learning-curve area is 5159.19 ± 150.42. The
[validated all-run report](../experiments/hc2x3_bppo10m_development_20261026.json)
retains all six declared development runs. Only AR-PPO seed 0 has finished
(5726.12), so there is no complete matched baseline aggregate yet.

Every BPPO endpoint exceeds the paper figure's reading range [5660, 5740].
This motivates confirmation, not a final reproduction claim. The original
figure averages five training seeds; these three seeds were used during
development. Its last plotted vertex is about 9.904M, and the historical
evaluation, smoothing and checkpoint-selection details remain unverified.
See the [figure audit](curve_targets.md).

## Frozen confirmation

The [training configuration](../configs/mujoco_hc2x3_confirmation10m.json) and
[study record](../experiments/hc2x3_confirmation10m_plan.json) were frozen before
launch. This study changes only the training and evaluation seeds relative to
the selected development settings:

| Item | Choice |
| --- | --- |
| Task | Legacy HalfCheetah-v2, 2×3 agents, observation radius 0, normalized control units, 1,000-step horizon |
| Methods | Corrected BPPO and architecture-matched AR-PPO, five runs each |
| Training seeds | 1000, 1001, 1002, 1003, 1004; absent from all 511 previously prepared runs |
| Training budget | Exactly 10M joint environment decisions per run |
| Evaluation | 32 deterministic complete episodes, separate seed 2,000,000, frozen training observation statistics |
| Primary endpoint | Mean of the five final training-seed evaluation means |
| Other summaries | Median, sample SD, full-budget curve area, paired BPPO minus AR-PPO differences |
| Uncertainty | 10,000 whole-seed bootstrap resamples, private RNG seed 1729, two-sided 95% percentile intervals; preserve pairing for differences |
| Selection | Final scheduled checkpoint; all declared seeds and failures retained |

The evaluation seed was absent from the prepared-run inventory. The 200
derived training-environment seeds (40 environments × five training seeds)
are distinct and exclude the evaluation seed. Both methods use the same seed
pairs. New evaluation episodes do not replace independent training seeds.

The new preparation option `--source-manifest` copies the selected experiment's
19 source modules after verifying every hash. All ten configurations were
validated using that copied code; every resolved field except training and
evaluation seeds matches the corresponding development run. Both launcher
files are also identical to the selected experiment. Later repository changes,
including optional recurrent actors, therefore do not enter this study.

## Decision and cost

Only complete, valid five-seed groups can support a confirmation statement.
A BPPO mean above 5740 is a point comparison with the digitized figure; a lower
95% bootstrap mean endpoint above 5740 supplies stronger evidence under this
pinned reconstruction. Neither criterion resolves unknown historical protocol
differences. Report paired baseline differences separately: passing a paper
threshold does not establish an advantage from peer feedback. Unsuccessful
confirmation seeds are never replaced or pooled with development seeds.

Measured BPPO runtime is 2.73–2.75 hours per 10M run. Five sequential runs per
worker therefore take approximately 13.68 hours at that throughput. The plan
uses two workers, 2 CPUs and 8GB each, with an 18-hour cap. Each worker follows
the matching Ant/Walker worker `21212297_i`, preserving the existing four
serial chains. Scheduler IDs and verified edges belong to
`experiments/jobs.jsonl`; preparing a manifest alone does not submit a job.

The frozen study was subsequently submitted as **21214776**. Both workers were
held until their respective `afterany:21212297_0` and `afterany:21212297_1`
dependencies and resource limits were verified, then released. They remain
queued behind those predecessors; no active experiment was changed.

This is one target's confirmation study. It does not complete the remaining
MuJoCo, SMAC, Football, Predator–Prey or matrix-game release requirements.

Use the [confirmation reporter](simulator_confirmation.md) to check this frozen
study. It validates the original plan and every declared seed before reporting
group statistics, paired baseline differences or a qualified figure comparison.
