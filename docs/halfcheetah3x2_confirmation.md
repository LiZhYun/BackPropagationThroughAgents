# HalfCheetah 3×2: independent confirmation

Ten runs compare BPPO with matched AR-PPO on five unused training seeds per method. Four BPPO runs are complete and valid; group means, paired comparisons and figure decisions remain withheld. The completed development comparison used seeds 0–2: BPPO averaged 5,431.33 ± 294.34 at 10M, with a small, mixed paired advantage over AR-PPO. Development and confirmation remain separate. The failed HalfCheetah 2×3 confirmation is retained.

| Choice | Frozen value |
| --- | --- |
| Task | HalfCheetah-v2, three agents with two actuators each, `agent_obsk=0`, normalized actions, 1,000-step horizon |
| Methods | BPPO and AR-PPO, otherwise matched |
| Training seeds | Five fresh pairs, 5000–5004 |
| Evaluation | Seed 3,000,000; 32 deterministic, complete autoregressive episodes |
| Budget | Exactly 10M decisions; initial evaluation, every 500k and the final scheduled checkpoint |
| Training source | Original nineteen-module snapshot; 40×100 rollouts, five PPO epochs |
| Uncertainty | Whole-training-seed 95% percentile bootstrap, 10,000 resamples, private RNG 1729 |

The [registered plan](../experiments/hc3x2_confirmation10m_plan_20261029_v2.json) fixes every configuration, source hash, stream, comparison and failure rule. Only training and evaluation seeds differ from the selected development runs. An initial registration preflight rejected mixed 3×2/6×1 development signatures; the task-specific revision passed before training began. No training code or settings were changed by that correction.

The figure comparison uses the upper digitized reading bound **5,030**. A five-seed BPPO mean above it is a point comparison; a lower 95% bootstrap mean endpoint above it is stronger reconstruction evidence. The historical simulator, evaluation and smoothing equivalence remain unresolved. Reading bounds are not confidence intervals. The separate peer-feedback comparison reports every paired BPPO-minus-AR-PPO endpoint and normalized curve area; passing the figure threshold alone does not establish an advantage over AR-PPO. All ten runs must be valid before these decisions, without seed replacements or development pooling.

## Execution and final verification

Training array **21279129** runs at most two one-CPU/4 GiB workers, with a seven-hour limit each. The [current strict report](../experiments/hc3x2_confirmation10m_progress_20261029_1510.json) has four valid completed runs, no integrity errors and no group decisions. The remaining declared runs continue through the scheduler. No configuration is tuned from these confirmation outcomes.

Auditor **21284586** follows the existing HC confirmation → portable replay → local-scale checks → remaining Football native checks chain. This uses the existing allocation; it changes no earlier job or dependency. At one CPU/4 GiB/30 min, it first requires a complete strict ten-run audit, then replays all ten native checkpoints using their original 32 episodes. The replay adds up to 320,000 evaluation interactions and no training. Configurations, raw metrics, training episodes, summaries and checkpoints must retain their original hashes.

Complete negative results remain valid evidence. Missing or malformed runs prevent acceptance; replay failures remain visible. The job submits no follow-up training. The unsubmitted feedforward local-scale study retains priority after this auditor or another verified free allocation. [Current job graph](worker_schedule.md#independent-allocations-and-prey-learning-29-october-2026).

The [preflight](../experiments/hc3x2_confirmation10m_audit_preflight_20261029.json) accepts the complete negative 2×3 study, rejects nine corrupted reports and the incomplete current family, and checks six real 3×2 native evaluation schemas. The exact saved-policy evaluator imports in the unchanged private runtime. Native final replay remains pending. All 126 evaluation-call timings from the six matched development runs give a ten-replay estimate of 17.06 minutes with a 50% margin, below the 30-minute cap. Fresh-process startup and host load may change actual time.

[Frozen audit specification](../experiments/hc3x2_confirmation10m_audit_plan_20261029.json); [verified submission](../experiments/hc3x2_confirmation10m_collector_20261029.json); [reporter behavior](simulator_confirmation.md). This study covers one canonical BPTA target. Full reproduction across both papers remains unfinished.
