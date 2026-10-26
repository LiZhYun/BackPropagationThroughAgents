# Final float64 matrix checkpoint diagnosis

The faster float64 joint-modifier control finishes at the optimal greedy payoff
in two of three development seeds on each game. The two failures have moved to
poor modes; they are not still at the earlier CC/BB plateaus. The exact frozen
200k checkpoints from `matrix_joint_lr_fast_float64_20261025T084839Z` give:

| Game | Seed | Local greedy action | Greedy payoff | Central expected payoff | Local sampling expected payoff |
| --- | ---: | --- | ---: | ---: | ---: |
| Climbing | 0 | AA | 11 | 11 | -0.1494 |
| Climbing | 1 | AA | 11 | 11 | 8.0542 |
| Climbing | 2 | AB | -30 | -9.50523 | -17.5376 |
| Penalty | 0 | CC | -100 | -26.67329 | -43.3791 |
| Penalty | 1 | CA | 10 | 10 | -19.9357 |
| Penalty | 2 | AC | 10 | 10 | -26.5830 |

Local greedy execution is the matrix benchmark convention. Central sampling
and local sampling are diagnostic distributions; their expected payoffs do not
replace the greedy metric. In particular, an optimal central distribution need
not produce concentrated local probabilities even when their modes agree.

## What happens in the failed seeds

Climbing seed 2 first approaches the CC plateau (central expected payoff near
8), switches to AC by 61,440 steps, and then AB by 143,360 steps. At the final
checkpoint it assigns probability 0.4998725 to AA and 0.5001275 to AB; other
outcomes are essentially suppressed. The second actor's B-over-A logit gap is
1.5201. Its exact objective ascent gradient raises A and lowers B, each with
magnitude 0.00016773. The first actor's gradient is negligible because its
preferred A is already correct.

Penalty seed 0 approaches the BB plateau (central expected payoff near 22/3),
switches to BC by 102,400 steps and CC by 112,640. At the final checkpoint,
AC/CA/CC probabilities are 0.3332968/0.3333097/0.3333935. Both actors' gradients
raise A relative to C, with magnitude about 0.00007907 per logit. Their
C-over-A gaps are 0.8651 and 0.7500. This direction can leave the losing CC
mode, but successful coordination after a mode change is not guaranteed.

The float64 positive-energy floor is inactive in all six checkpoints.
Gradients in both failed seeds are dominated by the direct local-gap path;
holding modifier outputs fixed does not remove their useful direction. The
relevant favorable-action scales have reached the lower hard clamp `exp(-8)`.
These observations support slow recovery with saturated scales, rather than
an absent gradient caused by the numerical floor.

## Saved-Adam counterfactuals

The probe enumerates all nine actions and differentiates
`E[R] / sigma + entropy_coefficient * H`, holding the current population reward
SD `sigma` fixed. At the behavior policy, this is the expected normalized PPO
ascent gradient before clipping activates. It is not the original finite-batch
gradient or a prediction of the later PPO epochs. Global gradient clipping and
saved Adam moments are retained. Updates act on fresh in-memory copies only.

| Failed checkpoint | Actor epsilon | Actor LR | Exact central payoff change after one step |
| --- | ---: | ---: | ---: |
| Climbing seed 2 | 1e-5 | 0.0025 | +0.00004638 |
| Climbing seed 2 | 1e-8 | 0.0025 | +0.00005276 |
| Climbing seed 2 | 1e-5 | 0.0125 | +0.00023103 |
| Penalty seed 0 | 1e-5 | 0.0025 | +0.00004606 |
| Penalty seed 0 | 1e-8 | 0.0025 | +0.00005750 |
| Penalty seed 0 | 1e-5 | 0.0125 | +0.00023039 |

Lower epsilon increases the measured central-payoff movement by approximately
14% and 25%; fivefold actor LR increases it about fivefold. None of these single
steps changes the local greedy action. Fractional parameter steps from 0.001 to
1 also increase the exact objective in these two snapshots. This local evidence
does not establish that a larger rate is stable throughout a fresh run.

The next control therefore changes only actor LR from 0.0025 to 0.0125, retaining
all six game/seed combinations, 200k decisions, four PPO epochs, modifier LR
0.0005, entropy 0.01, and float64 joint probabilities. Adam epsilon stays 1e-5.
The control is reward-independent and does not alter the forward mode invariant.
It must retain every seed and be assessed at the fixed endpoint. A successful
configuration would still need disjoint final seeds and non-matrix validation.

## Completed actor-rate control, 2026-10-26

Job `21211813` completes all six declared 200k-step runs. Source hashes,
software versions and resolved configurations match the preceding float64
control, with actor learning rate **0.0025 → 0.0125 as the only setting change**.
All 126 evaluation points were checked against the frozen payoff tables and
all nine probabilities of both the central and local distributions. The
[result record](../experiments/matrix_faster_actor200k_20261026.json) and
[CSV](../experiments/matrix_faster_actor200k_20261026.csv) retain every seed.

| Game | Seed | Greedy payoff | Central expected payoff | Local sampling expected payoff | First logged stable greedy optimum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Climbing | 0 | 11 | 9.0941 | 11.0000 | 61,440 |
| Climbing | 1 | 11 | 11.0000 | 11.0000 | 30,720 |
| Climbing | 2 | 11 | 11.0000 | 10.6879 | 30,720 |
| Penalty | 0 | 10 | 10.0000 | −29.3461 | 81,920 |
| Penalty | 1 | 10 | 10.0000 | −33.3073 | 122,880 |
| Penalty | 2 | 10 | 10.0000 | 1.4776 | 71,680 |

The last column means every subsequent **logged** greedy payoff is optimal;
it does not infer unobserved behavior between checkpoints. The faster update
recovers the optimal greedy outcome in both previously failed seeds. Local
sampling reaches at least 95% of the Climbing ceiling in all three seeds, but
remains poor on Penalty. Climbing seed 0 also retains central probability
0.4765 on the suboptimal BB action despite nearly deterministic local AA
execution. Its central expected payoff is below 95% of the ceiling.

Thus six optimal greedy endpoints do not establish optimal central and local
stochastic policies across all runs. The results support an independent configuration
check, not a population reliability, paper sample-efficiency or convergence
claim. The correlated modifier remains a named experimental redesign, and the
default configuration is unchanged.

## Provenance and limitations

The [raw table](../experiments/matrix_final_diagnosis_20261026.csv) retains all
six policies and all three counterfactuals. The full diagnostic is
`results/matrix_final_diagnosis_20261026/summary.json`, alongside `probe.py`.
It records the frozen manifest, source, checkpoint and probe SHA-256 hashes,
local logits, probabilities, full/direct gradients, scale saturation, optimizer
motion, and fractional-step results. Original checkpoint hashes are checked
before and after the diagnostic.

The probe imports the original frozen training package and verifies every
source hash. Greedy action replay is exact. Floating-point scalar and
probability replay uses relative/absolute tolerance 1e-5 across CPUs, with
scalar differences retained explicitly. This diagnostic uses no new environment
samples, no extra training, and no evaluation-based checkpoint selection.
