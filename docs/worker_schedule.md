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

## Matrix validation insertion, 2026-10-26

The [frozen matrix studies](matrix_confirmation.md) were subsequently inserted
between matrix job `21211813` and entropy job `21211006`, for each index `i`:

```
21209234_i → 21211813_i → 21213329_i → 21213330_i → 21211006_i → 21212197_i
SMAC          joint LR      100k check    1M extension   entropy       Prey
```

Job `21213329` has 40 runs on ten unused seeds; job `21213330` has 12 development
runs. Each has two workers requesting 1CPU/4GB for at most 30 minutes. They were
submitted held, assigned matching-index dependencies, and released only after
the scheduler confirmed all six changed edges. Entropy workers were still
pending during the update. The two A chains are unchanged. There are still
four serial chains and at most four concurrent workers. Submission hashes and
every dependency update are in the job ledger.

## Recurrent native integration insertion, 2026-10-26

Job `21213860` was then inserted between `21213330` and `21211006`, again for
each matching worker index. The current B chains are:

```
21209234_i → 21211813_i → 21213329_i → 21213330_i → 21213860_i → 21211006_i → 21212197_i
SMAC          joint LR      100k check    1M matrix     GRU check     entropy       Prey
```

The GRU check contains five native SMAC runs of 10,000 transitions, each with
one complete 2×5,000 rollout and four final evaluation episodes. Its two workers
request 2CPUs/8GB and one hour. It was submitted held and released after the
four changed dependency edges were checked. All affected successors were still
pending. The four-worker ceiling and A chains remain unchanged.

The ten-run `smac_terran5_batch1m` learning manifest is prepared but **not
submitted**. Its integration, measured-runtime and memory gates are recorded in
[`experiments/smac_recurrent_study_20261026.json`](../experiments/smac_recurrent_study_20261026.json).
Passing unit tests alone does not release this native learning study.

## HalfCheetah confirmation appended, 2026-10-26

The [independent confirmation](halfcheetah_confirmation.md), job `21214776`,
extends each A chain after its matching Ant/Walker transfer worker:

```
21208906_i → 21209248_i → 21212297_i → 21214776_i
BPPO/AR       Football      Ant/Walker    five-seed confirmation
```

The new job has ten runs at 10M decisions, two workers with five serial runs
each, 2CPUs/8GB per worker and an 18-hour cap. It was submitted held; both
dependencies and resource limits were verified before release. The B chains
are unchanged, and there are still four serial chains. Existing jobs, source
snapshots and dependencies were not modified. The study record and submission
ledger retain the frozen manifest, source, launcher and plan hashes.
