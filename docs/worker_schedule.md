# Recorded worker schedule

The current queued arrays use two independent workers each. Worker `i` executes
its assigned runs sequentially: indices `i, i+2, i+4, …` in the frozen manifest.
Each run writes to its own output directory and keeps its own model and RNG.
There is no training-data or checkpoint dependency between workers.

On 2026-10-26, the scheduler dependencies were changed to the matching worker
index. For example, Football worker `21209248_1` waits for `21208906_1` to end.
Every change was restricted to a pending worker, checked with `scontrol`, and
recorded in `experiments/jobs.jsonl`. Active jobs, source snapshots, run order
within each worker, configurations, seeds and budgets were unchanged.

| Sequential chain | Root worker | Successors in order |
| --- | --- | --- |
| A0 | `21208906_0` — BPPO/AR-PPO | `21209248_0` — Football → `21212297_0` — Ant/Walker |
| A1 | `21208906_1` — BPPO/AR-PPO | `21209248_1` — Football → `21212297_1` — Ant/Walker |
| B0 | `21209234_0` — SMAC | `21211813_0` — matrix → `21211006_0` — entropy → `21212197_0` — Prey |
| B1 | `21209234_1` — SMAC | `21211813_1` — matrix → `21211006_1` — entropy → `21212197_1` — Prey |

Every edge uses `afterany`, so the successor can start after its predecessor
ends, including a failed predecessor. Those failures remain separate evidence;
a successor never retries, resumes or replaces the preceding run. The existing
plans used the same terminal-state condition. Results from an earlier job are
not needed to construct these already frozen successors.

Each chain can have at most one running worker. Four chains therefore preserve
the cap of four concurrent workers, each requesting at most 2CPUs/8GB. Matrix
workers request only 1CPU/4GB. A shorter worker can release its successor while
the other chain continues, reducing avoidable idle time without increasing the
resource ceiling.

At verification, the four root workers were running and all ten successors
were pending with the exact dependencies above. This is a scheduling snapshot,
not a statement that these states remain current. Query only recorded job IDs
for current state. The artifact reporter does not query Slurm.

Future jobs must be added to this graph or otherwise checked against the same
cap. The matching-index rule assumes the independent two-worker layout above;
it must not be copied blindly to plans with shared output or different worker
counts. Array-level historical dependency events remain in the ledger, followed
by explicit per-worker updates and the verified scheduler values.
