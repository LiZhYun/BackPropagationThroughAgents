# Author clarifications

These statements come from the authors during this reproduction, separately
from the archived papers and pinned released code.

## 2026-10-25: Predator–Prey and Football 5v5 metric

Asked whether the Predator–Prey and Football 5v5 figures plotted evaluation
return or training reward, the author answered **“eval return”**.

This confirms evaluation return for both Predator–Prey penalty settings and
BPTA Football 5v5. Training-reward reductions are therefore not candidate
interpretations for these figures. It does not identify the historical logger
key, code revision, reward normalization, treatment of unfinished episodes, or
cross-seed aggregation. No original logs or plotting-script path accompanied
the clarification.

The [Prey audit](predator_prey_protocol.md) describes the released evaluator's
200-step autoreset windows and per-decision reward reduction. The
[Football audit](football.md) explains why the pinned 200-step evaluator cannot
complete a native 5v5 match. New evaluations retain explicit metrics; the subsequent Prey protocol
selection is recorded below. The clarification does not change digitized
values or their reading uncertainty.

## 2026-10-26: selected Prey reproduction protocol

Asked whether the Prey figure used total reward per complete episode or the
released 100×200-step evaluation-window reduction, the author answered:
**“I don’t recall; use the released evaluation protocol”**.

The primary Prey reproduction comparison therefore uses mean evaluation reward
per decision over 100 separately seeded 200-decision windows, with automatic
resets after captures/timeouts and rewards from unfinished segments included.
Local actors select deterministic actions. Each predator receives the same
reward; average across agents, rather than summing it twice. The denominator
is 20,000 environment decisions. Completed-episode returns, lengths and capture
rates remain separate diagnostic metrics.

This selects the reproduction convention; it does not verify the historical
figure-to-logger mapping, historical seeds or cross-seed aggregation. The
selected metric is recorded separately from source labels in the target files.
Digitized endpoints and their reading ranges are unchanged. No further author
answer is needed to select the Prey metric. This reply does not resolve the
separate Football 5v5 reduction.
