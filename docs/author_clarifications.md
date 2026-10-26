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
complete a native 5v5 match. New evaluations retain explicit metrics while the
historical reduction is resolved. The clarification does not change digitized
values or their reading uncertainty.

## 2026-10-26: pending Prey evaluation reduction

A follow-up asks whether the confirmed evaluation return is summed per complete
episode or averaged per decision over the released 100×200-step evaluation
windows. No reply has been recorded yet. Both metrics remain logged, and no
historical reduction is inferred from the absence of a reply.
