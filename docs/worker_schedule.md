# Recorded worker schedule

This page retains successive scheduling snapshots. The [latest verified
graph](#smac-learning-with-single-run-tasks-2026-10-26) appends bounded SMAC learning
tasks while retaining four serial worker chains.

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

## Native timing insertion, 2026-10-26

Job `21215880` adds the two short [native runtime profiles](simulator_runtime.md)
between the running SMAC study and the pending joint-matrix control. The latest
B chains are:

```
21209234_i → 21215880_i → 21211813_i → 21213329_i → 21213330_i → 21213860_i → 21211006_i → 21212197_i
SMAC          timing        joint LR      100k check    1M matrix     GRU check     entropy       Prey
```

Each profiling worker requests 2CPUs/8GB for 30 minutes and runs one method for
4,096 decisions. The matrix workers were still pending when held for this
change. All four changed dependency edges and resource limits were verified
before releasing both arrays; the released workers report `PENDING/Dependency`.
The A chains and running jobs are unchanged. Four serial chains still limit
concurrency to four workers. The timing study measures execution cost, not
learning performance.

## Short diagnostics first, 2026-10-26

Once SMAC worker 0 began its last 1M identity-control run, worker 1 was close to
finishing its last run. Waiting for both long workers would delay the completed
timing and matrix comparisons. Nine still-pending workers were therefore held,
their dependencies updated, and all edges and unchanged resources checked
before release. Active jobs, manifests, sources, assigned runs, seeds and
budgets were unchanged. No additional jobs were submitted.

The four serial chains at that update were:

| Chain | Order |
| --- | --- |
| A0/A1 | HalfCheetah development → Football → Ant/Walker → HalfCheetah confirmation, with matching worker indices as before |
| B0 | `21209234_0` SMAC → `21213860_0` GRU check → `21211006_0` entropy → `21212197_0` Prey |
| B1 | `21209234_1` SMAC → both timing workers → both joint-matrix workers → both 100k matrix workers → both 1M matrix workers → `21213860_1` GRU check → `21211006_1` entropy → `21212197_1` Prey |

Within each short study, worker **1 runs before worker 0**. Their array IDs are
`21215880`, `21211813`, `21213329` and `21213330`, respectively. GRU worker 0
also waits for matrix worker `21213330_0`: both that prerequisite and its own
SMAC predecessor must finish. This extra cross-chain prerequisite cannot raise
concurrency beyond four workers. Every edge still uses `afterany`; failures
remain evidence and are never silently replaced.

The [scheduling record](../experiments/short_diagnostic_schedule_20261026.json)
contains all prior states, requested edges, the concurrency argument and a hash
of the hold/update/release verification. The ledger records each array's full
dependency map. Both timing workers subsequently completed with exit 0; the
joint-matrix control began while the remaining native training continued.

## AgentMixer validation inserted, 2026-10-26

The short studies and GRU worker 1 have now completed. A new ten-run
[AgentMixer Climbing validation](agentmixer_confirmation.md), job `21218175`,
occupies one existing B0 position:

```text
21209234_0 → 21218175_0 → 21213860_0 → 21211006_0 → 21212197_0
SMAC         AM check     GRU check     entropy       Prey
```

The new worker runs ten serial 200k configurations using 1 CPU, 4GB and a
45-minute limit. It was submitted held. After the frozen-plan preflight passed,
the still-pending GRU worker 0 was held, its dependency updated, and both workers'
edges and resources verified before release. The former matrix prerequisite
had already completed. Active jobs and frozen run files were unchanged.

The [verification record](../experiments/matrix_agentmixer_confirmation_schedule_20261026.json)
preserves the old and new states. A0/A1 still run Football → Ant/Walker →
HalfCheetah confirmation; B1 runs entropy → Prey after its completed GRU check.
There are still at most four concurrent workers, each at most 2 CPUs and 8GB.


## Raw process diagnostic inserted, 2026-10-26

Job `21218609` measures exact sequential/process SMAC trace parity and native
call time. It requests one 2CPU/8GB worker for at most 30 minutes. The latest B0
chain is:

```text
21209234_0 → 21218175_0 → 21213860_0 → 21218609 → 21211006_0 → 21212197_0
SMAC         AM check     GRU check     process     entropy       Prey
```

The profiler was submitted held after its unit and copied-source preflights.
Only the pending entropy worker 0 was held and rewired; both edges/resources
were verified before release. The [schedule record](../experiments/smac_process_profile_schedule_20261026.json)
retains the old and new states. All edges use `afterany`; a failed diagnostic
remains evidence and does not silently retry or alter the frozen entropy run.
A0/A1 and B1 are unchanged, so the same four-worker ceiling applies.

## Independent full-budget matrix validation inserted, 2026-10-26

The [five-seed 1M study](matrix_confirmation.md#independent-1m-confirmation), job
`21218775`, adds one serial worker to B1 after its active entropy worker:

```text
21211006_1 → 21218775_0 → 21212197_1
entropy       matrix 1M    Prey
```

Twenty configurations run serially within 1CPU/4GB and a 60-minute limit. The
job was submitted held and all twenty declarations passed preflight. Only the
pending Prey worker 1 was held and rewired; both workers' resource/dependency
checks passed before release. The [verification record](../experiments/matrix_confirmation1m_schedule_20261026.json)
retains old and new states. A0/A1 and B0 are unchanged. At most four workers
still run concurrently, each at most 2CPUs/8GB, and no active experiment was changed.

## Football minibatch integration inserted, 2026-10-26

The new native integration job `21219475` follows each matching Football worker
and precedes Ant/Walker. The latest A chains are:

```
21208906_i → 21209248_i → 21219475_i → 21212297_i → 21214776_i
BPPO/AR       Football      batch check   Ant/Walker    HC confirmation
```

Both integration workers request 2 CPUs/8GB for at most two hours. Each runs
five fixed configurations serially. The job was submitted held; only the two
pending Ant/Walker dependencies changed. All four worker dependencies, holds,
resource requests and releases were checked and saved in the
[scheduling proof](../experiments/football_batch_schedule_20261026.json).
The B0/B1 chains remain as recorded in the preceding sections. Four serial
chains still limit this project to four workers.

The separate 1M Football development manifest remains prepared but unsubmitted
until all ten integration runs pass correctness, runtime and memory gates.
See the [optimization audit](football_optimization_audit.md).

## Prey recurrent check and learning release, 2026-10-26

One short Prey protocol check, `21219646_0`, was inserted after AgentMixer matrix
confirmation `21218175_0` and before pending native GRU `21213860_0`. The initial
post-release assertion required a fresh scheduler read; the saved
[proof](../experiments/prey_recurrent_protocol_schedule_20261026.json) confirms
the intended states. No repeated job submission or restart occurred. The check
then completed successfully, including exact checkpoint replay and measured
memory/runtime, and the native GRU worker started.

The subsequent eight-run study `21219719` appends one worker to each existing
Prey chain, after matching `21212197_i`. Each worker runs four fixed configurations
serially with 2CPUs/8GB and a nine-hour cap. The
[release record](../experiments/prey_recurrent_release_schedule_20261026.json)
contains the measured estimate, immutable source/configuration checks and
verified held/released scheduler states. The latest B chains are:

```text
B0: ... → 21218175_0 → 21219646_0 → 21213860_0 → 21218609 → 21211006_0 → 21212197_0 → 21219719_0
B1: ... → 21211006_1 → 21218775_0 → 21212197_1 → 21219719_1
```

The A chains retain Football → Football integration → Ant/Walker → independent
HalfCheetah confirmation. All four chains remain serial; at most four project
workers can run. Every edge remains `afterany`, with failures recorded separately.

## SMAC learning with single-run tasks, 2026-10-26

After all five native recurrent integrations passed and runtime/memory were
reviewed, the ten frozen 1M controls were released as job `21220264`. Each task
executes one manifest index through the existing `run_simulator_array.py`.
The new dispatch script retains the original source and configs, uses 2CPUs/8GB,
a six-hour cap and `--no-requeue`. The original two-worker 24h script remains unchanged.
The [release plan](../experiments/smac_batch1m_release_plan_20261026.json) explains
the measured estimate and retains every configuration hash.

```text
B0: ... → 21219719_0 → 21220264_0 → 21220264_2 → 21220264_4 → 21220264_6 → 21220264_8
B1: ... → 21219719_1 → 21220264_1 → 21220264_3 → 21220264_5 → 21220264_7 → 21220264_9
```

All ten tasks were submitted held, assigned these dependencies, and released
after resource/dependency checks. The
[proof](../experiments/smac_batch1m_release_schedule_20261026.json) records the
scheduler states. Array concurrency is also capped at two. Each task names a
unique output; no stride or repeated configuration is used. Together with the
unchanged A0/A1 chains, at most four project workers can run. This is a
scheduling change, with no changes to the training backend or experiments.


## Paired full-budget matrix study inserted, 2026-10-26

Two bounded serial workers add the missing learned/identity modifier comparison
on both games at 1M. The latest B chains are:

```text
B0: ... → 21212197_0 → 21220499_0 → 21219719_0 → 21220264_0 → _2 → _4 → _6 → _8
B1: ... → 21212197_1 → 21220500_0 → 21219719_1 → 21220264_1 → _3 → _5 → _7 → _9
```

Each new matrix job has one task, one CPU/4GB, ten serial runs, a three-hour
cap and no automatic requeue. Both recurrent Prey workers were still pending
when held. The new jobs were submitted held, all 20 configurations passed the
[paired preflight](../experiments/matrix_agentmixer_ablation1m_preflight_20261026.json),
and all four dependencies/resource requests were checked before release.
The [schedule proof](../experiments/matrix_agentmixer_ablation1m_schedule_20261026.json)
retains the before/after states. Active jobs were unchanged. The A chains
remain Football → Football integration → Ant/Walker → independent HalfCheetah;
at most four workers still run concurrently.
