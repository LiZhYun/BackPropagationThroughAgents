# Recorded worker schedule

This page retains successive scheduling snapshots. The [current graph](#ant-confirmation-and-final-audit-29-october-2026) uses at most eight workers, sixteen CPUs, and 96 GiB. Earlier ceilings and graphs below are historical. Query the recorded job IDs for live state.

The initial arrays used two independent workers each. Worker `i` executes
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


## Paired 1M matrix study inserted, 2026-10-26

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

## HalfCheetah partition check inserted, 2026-10-26

Job `21220888_0` adds four short native checks for HalfCheetah 3×2 and 6×1,
using the selected BPPO/AR-PPO source and full 40×100 rollouts. Its one worker
runs all four configurations serially with 2 CPUs/8 GB, a 30-minute cap and
no requeue. The latest B1 segment is:

```text
21220500_0 → 21220888_0 → 21219719_1 → 21220264_1 → _3 → _5 → _7 → _9
matrix       HC checks     recurrent Prey   SMAC
```

The [receipt](../experiments/hc_partitions_check_schedule_20261026.json)
records the pending successor's hold, the new job's held submission, verified
dependencies/resources and releases. The other three chains are unchanged,
preserving four concurrent workers. The four related 500k pilots remain
unsubmitted until the [native gate](halfcheetah_partition_transfer.md) passes.

## HalfCheetah partition pilots released, 2026-10-26

All four checks in `21220888_0` passed in 120 seconds, with 0.49 GiB peak memory.
The four 500k pilots are released as `21220995`, using one configuration per
task, one CPU/2 GiB, a measured 30-minute limit and no automatic requeue.
The latest A-chain endings are:

```text
A0: ... → 21212297_0 → 21220995_0 → 21220995_2 → 21214776_0
A1: ... → 21212297_1 → 21220995_1 → 21220995_3 → 21214776_1
           Ant/Walker    HC partition pilots       HC2×3 confirmation
```

The [release receipt](../experiments/hc_partitions_transfer_release_schedule_20261026.json)
records all six held and released dependency states. Both confirmation workers
were pending; their source, configurations and seeds remain fixed. The B chains
retain recurrent Prey followed by their even/odd SMAC tasks; B0 first finishes
its entropy, earlier Prey and learned-matrix workers. Each chain remains serial,
so at most four project workers can run. The pilot array also has its own cap
of two concurrent tasks.

## Football batch learning released, 2026-10-26

After all ten native Football checks passed, the frozen 1M FF/GRU study was
released as `21221161`. One configuration runs per task, with one CPU/8 GiB,
a four-hour measured limit and no requeue. The array is capped at two tasks.
It appends these A-chain endings:

```text
A0: ... → 21220995_0 → 21220995_2 → 21214776_0 → 21221161_0 → _2 → _4 → _6 → _8
A1: ... → 21220995_1 → 21220995_3 → 21214776_1 → 21221161_1 → _3 → _5 → _7 → _9
          HC partition pilots      HC confirmation   Football FF/GRU
```

The [release receipt](../experiments/football_batch1m_release_schedule_20261026.json)
records all ten tasks' held and released states. Only the new jobs receive
dependencies; earlier jobs and both B chains are unchanged. Four serial chains
still limit this project to four concurrent workers. The
[runtime plan](../experiments/football_batch1m_release_plan_20261026.json)
includes the complete-match evaluation allowance and preserves the original
source/configurations. This remains development work below the 20M paper budget.


## AgentMixer 10M submission, 2026-10-26

Job21221384 first entered the A chains between independent HalfCheetah and
Football learning, with even/odd task chains. The
[submission receipt](../experiments/hc2x3_agentmixer10m_schedule_20261026.json)
retains that initial four-worker arrangement. The following revision supersedes
its scheduling dependencies; frozen source, configurations and budgets remain
unchanged.

## Eight-worker schedule, 2026-10-26

The initial four-worker pilot ceiling is revised to **eight workers**, each
at most two CPUs and 8 GiB: at most **16 CPUs and 64 GiB** for this project.
This autonomous scheduling decision uses the user's standing authorization
to continue low-risk local work; the user did not separately request eight
workers. It follows completed native correctness/resource gates and the full
entropy comparison. The CPU partitions reported 2,620 idle CPUs in the saved
capacity snapshot. No new experimental run or training budget is added.

The [prepared change](../experiments/eight_worker_schedule_plan_20261026.json)
and [verified receipt](../experiments/eight_worker_schedule_verified_20261026.json)
record all38 jobs,30 serial edges and six changed pending heads. Those heads
were held while ownership, frozen readiness records, every edge, array caps
and unchanged per-task requests were verified, then released. Active jobs
and the other reproduction were untouched. The four newly independent roots
are ready because their scientific gates have already completed; their former
predecessors were scheduling constraints.

| Chain | Root at revision | Pending successors, in order |
| --- | --- | --- |
| A0 | Ant/Walker `21212297_0` | HC partition `21220995_0 → _2` → HC confirmation `21214776_0` → Football `21221161_0 → _2 → _4 → _6 → _8` |
| A1 | HC confirmation `21214776_1` | Football `21221161_1 → _3 → _5 → _7 → _9` |
| B0 | Prey bounds `21212197_0` | Matrix learned `21220499_0` → recurrent Prey `21219719_0` |
| B1 | Recurrent Prey `21219719_1` | None |
| C0 | AgentMixer10M `21221384_0` | `21221384_2 → _4 → _6 → _8` |
| C1 | AgentMixer10M `21221384_1` | `21221384_3 → _5 → _7` |
| D0 | SMAC `21220264_0` | `21220264_2 → _4 → _6 → _8` |
| D1 | SMAC `21220264_1` | `21220264_3 → _5 → _7 → _9` |

Every arrow uses `afterany`; each chain has at most one active task. Each
study still has an array cap of two. AgentMixer tasks use one CPU/4 GiB,
SMAC two CPUs/8 GiB, Football one CPU/8 GiB. New jobs must fit these eight
chains or have an explicit verified scheduling revision. No automatic retries
or timeout-triggered restarts are introduced. Snapshot states are not live
scheduler claims; query the recorded job IDs for current progress.


## Remaining MuJoCo native gates, 2026-10-26

Job21221614 adds37 [native checks](mujoco_remaining_transfer.md), split into
19/18 serial runs. Each worker requests2 CPUs/8 GiB for at most two hours,
without requeue. Its prepared500k learning counterpart remains unsubmitted.
The only changes to the eight-chain graph are:

```
A0: 21212297_0 → 21221614_0 → 21220995_0 → existing A0 successors
B0: 21212197_0 → 21221614_1 → 21220499_0 → existing B0 successors
```

The two existing successor heads were pending when held. The new array was
submitted held, then all four changed dependencies, unchanged successor
requests, gate requests and all32 serial edges were verified before release.
The [receipt](../experiments/mujoco_remaining_gate_schedule_20261026.json)
records40 project-owned jobs and eight active workers at release. Other
chains, active jobs, configurations and the unrelated reproduction remain
unchanged. The project ceiling stays eight workers,16 CPUs and64 GiB.


## Completed arrays and new studies — 2026-10-28

All eleven arrays from the previous schedule finish with exit 0. The [new verified schedule](../experiments/followup_schedule_20261028.json) contains independent arrays capped at two active tasks each: MuJoCo coverage **21251756** (37×500k,1CPU/4GiB/90min), HalfCheetah partitions **21251961** (12×10M,1CPU/4GiB/7h), Football **21251960** (5×20M,1CPU/8GiB/60h). Sources and configurations are frozen before submission.

One native SMAC25-stream gate **21251962** uses 2CPUs/32GiB/30min and one serial worker. This revises the earlier per-worker8GiB limit for that bounded diagnostic only; the total ceiling stays eight workers,16CPUs/64GiB. The scheduled maximum is seven workers,eight CPUs and64GiB. The read-only AgentMixer diagnostic **21252362** (1CPU/2GiB/30min) follows the gate's entire single-element array. Slurm expands that dependency to `afterany:21251962_*`; task0 and throttle1 were verified before release. A future32GiB SMAC learning worker must also follow the diagnostic. No unrelated job was changed.


## Current measured schedule after the SMAC gate

The native gate **21251962** completed0:0 in1573s with21.24GiB peak. Its requested wall-time increase was denied by the cluster; it finished within the original30min cap. Diagnostic **21252012** failed on an overstrict float32 scale assertion; corrected read-only job **21252362** completed0:0 in247s and exactly replayed all nine policies. Neither event changed any training source/configuration/checkpoint.

Both **21252369** SMAC1M workers follow the completed diagnostic. Each uses2CPUs/32GiB/14h,at most2active. The measured memory ceiling is now **96GiB**, while the project remains capped at eight workers/sixteen CPUs. The scheduled maximum is eight workers/ten CPUs/96GiB: SMAC64GiB,Football16GiB,HC partitions8GiB,MuJoCo coverage8GiB. The [release record](../experiments/smac_layout25_learning_release_20261028.json) records current capacity, measured resource estimates and the autonomous allocation decision.

The [nine-run entropy-zero study](../experiments/hc2x3_entropy0_10m_plan_20261028.json), **21252423**, has an `afterany:21251756_*` dependency on the complete MuJoCo coverage array. It then reuses those two1CPU/4GiB slots; the aggregate ceiling does not rise. All held-job settings and dependencies were verified before release.


## Continuation on 29 October 2026

MuJoCo coverage 21251756 (37/37) and SMAC layout 21252369 (5/5) completed 0:0. NewSMAC array **21272799** uses the same two 2 CPU/ 32 GiB slots, now 60 h/task for five full 10M controls. NewMuJoCo array **21272800** uses two 1 CPU/4 GiB/30 h slots only after the entire HC partition array 21251961 finishes. Both held allocations and dependencies were verified before release; no automatic requeue. Other project jobs remain untouched.

The maximum schedule is still 8workers/10 CPUs/96 GiB, below the 16 CPU ceiling. Source-identical plans and resource projections: [SMAC](../experiments/smac_terran5_gru10m_plan_20261029.json), [MuJoCo](../experiments/mujoco_coverage10m_plan_20261029.json). Both are development studies; full batches take several days. [Latest dated state](progress_20261029.md).


## Native controls and automatic Prey audit, 29 October 2026

The current reservations are four separate chains. Array edges wait for the **entire** predecessor array to terminate. Failure permits an independent next check to run; it never authorizes a retry or a performance claim.

| Allocation | Current chain | Maximum reserved resources |
| --- | --- | --- |
| Football | 21251960 (5×20M) → 21282173 (8 Prey native gates) → 21282210 (complete gate audit) | Two workers, two CPUs, 16 GiB; Prey uses 8 GiB total, audit 2 GiB |
| SMAC | 21272799 (5×10M) → 21281002 (40 remaining native checks) | Two workers, four CPUs, 64 GiB |
| MuJoCo coverage | 21272800 (48×10M) | Two workers, two CPUs, 8 GiB |
| HC confirmation and controls | 21279129 (10×10M) → 21279857 (bundle replay) → 21280689 (6 local-scale checks) → 21281001 (15 Football native checks) | Two workers, two CPUs, 8 GiB; Football native checks use one 8 GiB worker |

The sum is eight workers, ten CPUs, and 96 GiB, within the sixteen-CPU ceiling. Unsubmitted 18-run local-scale learning must wait for 21281001 or another verified free allocation. Unsubmitted 24-run Prey learning has no allocation and still requires all native checks plus a resource audit. Other future follow-ups must respect these reservations. No existing job was modified by the Prey addition.

The [Prey gate receipt](../experiments/prey_distillation_gate_schedule_20261029.json), [automatic-audit receipt](../experiments/prey_distillation_gate_collector_20261029.json), and [current status](../STATUS.md) record the constraints and pending work. Native gates and collectors establish correctness evidence, not reproduced performance.


## Ant confirmation and final audit, 29 October 2026

The Ant confirmation extends the existing Football/Prey allocation. Every edge below waits for the entire predecessor array or single job to terminate. Existing jobs, configurations and dependencies are unchanged.

| Allocation | Current chain | Maximum reserved resources |
| --- | --- | --- |
| Football, Prey and Ant confirmation | 21251960 → 21282173 → 21282210 → **21282468** (ten Ant confirmation runs) → **21282782** (full audit and ten native replays) | Two workers, two CPUs, 16 GiB; Ant uses two 4 GiB workers, final audit one 4 GiB worker |
| SMAC | 21272799 → 21281002 | Two workers, four CPUs, 64 GiB |
| MuJoCo coverage | 21272800 | Two workers, two CPUs, 8 GiB |
| HC confirmation and controls | 21279129 → 21279857 → 21280689 → 21281001 | Two workers, two CPUs, 8 GiB |

The combined bound remains eight workers, ten CPUs and 96 GiB, within the sixteen-CPU ceiling. The unsubmitted 18-run local-scale study retains priority after 21281001 or another verified free allocation. The 24-run Prey learning study still requires its complete native gates and a resource audit; it cannot overlap the new Ant reservation and must wait for 21282782 or another verified free allocation.

Ant uses the selected 20-module trainer unchanged. Its final auditor uses a separately pinned reporter plus the same native training source for replay. Preflight checks pass; no final Ant result exists yet. [Training schedule](../experiments/ant2x4_confirmation10m_schedule_20261029.json), [audit schedule](../experiments/ant2x4_confirmation10m_collector_20261029.json), [study rules](ant2x4_confirmation.md).


## Continuous-memory native execution, 29 October 2026

The twelve continuous-memory checks and their auditor extend the existing Football/Prey/Ant allocation. Job **21284086** also waits for all six local-scale gates in **21280689**, which provide the feedforward reference data. This additional dependency reserves no capacity in the HC chain. The scheduler verified both dependencies before release; no existing job or dependency changed.

| Allocation | Current chain | Maximum reserved resources |
| --- | --- | --- |
| Football, Prey, Ant and memory checks | 21251960 → 21282173 → 21282210 → 21282468 → 21282782 → **21284086** → **21284087** | Two workers, two CPUs, 16 GiB; memory checks use two 4 GiB workers, audit one 2 GiB worker |
| SMAC | 21272799 → 21281002 | Two workers, four CPUs, 64 GiB |
| MuJoCo coverage | 21272800 | Two workers, two CPUs, 8 GiB |
| HC confirmation and controls | 21279129 → 21279857 → 21280689 → 21281001 | Two workers, two CPUs, 8 GiB |

Every dependency waits for the entire predecessor array or job to terminate. Native checks use batch-milan, one CPU/4 GiB/30 min each, at most two active; the auditor uses one CPU/2 GiB/5 min. Neither requeues automatically. The maximum remains eight workers, ten CPUs and 96 GiB, below the sixteen-CPU ceiling.

The unsubmitted 18-run feedforward local-scale study retains priority after 21281001 or another verified free allocation. The unsubmitted 24-run Prey learning study must now wait for **21284087** or another verified free allocation. The 18-run recurrent learning study remains unsubmitted. Native results, original local-scale validation and measured resources remain separate requirements; an afterany dependency does not mean a predecessor passed. No auditor automatically releases learning.

[Execution specification](../experiments/continuous_recurrent_gate_execution_20261029.json), [native schedule](../experiments/continuous_recurrent_gate_schedule_20261029.json), [collector schedule](../experiments/continuous_recurrent_gate_collector_20261029.json), [live resource snapshot](../experiments/live_resource_check_20261029_memory_gate.json).


## HC 3×2 final verification, 29 October 2026

Final auditor **21284586** extends the existing HC allocation after all fifteen remaining Football native checks. It audits the complete registered ten-run HC 3×2 study and replays every final checkpoint using the original 32 episodes. It adds no training runs and releases no learning jobs. Existing jobs and dependencies were unchanged at submission.

| Allocation | Current chain | Maximum reserved resources |
| --- | --- | --- |
| Football, Prey, Ant and memory checks | 21251960 → 21282173 → 21282210 → 21282468 → 21282782 → 21284086 → 21284087 | Two workers, two CPUs, 16 GiB |
| SMAC | 21272799 → 21281002 | Two workers, four CPUs, 64 GiB |
| MuJoCo coverage | 21272800 | Two workers, two CPUs, 8 GiB |
| HC confirmation, controls and final verification | 21279129 → 21279857 → 21280689 → 21281001 → **21284586** | Two workers, two CPUs, 8 GiB; final audit one CPU/4 GiB/30 min |

The memory checks additionally depend on all of 21280689 for reference data. Every array edge waits for the entire predecessor to terminate. Maximum project reservations remain eight workers, ten CPUs and 96 GiB within the sixteen-CPU ceiling. The unsubmitted eighteen-run feedforward local-scale learning study retains priority after **21284586** or another verified free allocation. Prey and recurrent learning remain unsubmitted and must satisfy their own gates and capacity review.

All six matched HC 3×2 development runs contribute 126 evaluation-call timings. The maximum 32-episode call is 68.25 s; ten calls with a 50% margin estimate 17.06 min, below the 30-minute cap. Fresh-process startup and host load remain uncertain. The held job's command, ownership, one CPU/4 GiB allocation, no-requeue flag and whole-array dependency were verified before release. [Frozen audit](../experiments/hc3x2_confirmation10m_audit_plan_20261029.json), [preflight](../experiments/hc3x2_confirmation10m_audit_preflight_20261029.json), [submission](../experiments/hc3x2_confirmation10m_collector_20261029.json).


## Measured MuJoCo concurrency revision, 29 October 2026

The existing MuJoCo array **21272800** now permits four active tasks. The sole scheduler mutation was `ArrayTaskThrottle=2 → 4`; all 48 configurations, the twenty-module training source, manifest, dispatch file, seeds, budgets and per-task limits retain their frozen hashes. The two original running task handles and start times were preserved. The dispatcher still records its original two-task header; the explicit scheduler override below supersedes that scheduling field.

| Allocation | Current chain | Maximum reserved resources |
| --- | --- | --- |
| Football, Prey, Ant and memory checks | 21251960 → 21282173 → 21282210 → 21282468 → 21282782 → 21284086 → 21284087 | Two workers, two CPUs, 16 GiB |
| SMAC | 21272799 → 21281002 | Two workers, four CPUs, 64 GiB |
| MuJoCo coverage | **21272800**, max four active tasks | Four workers, four CPUs, 16 GiB |
| HC confirmation, controls and final verification | 21279129 → 21279857 → 21280689 → 21281001 → 21284586 | Two workers, two CPUs, 8 GiB |

The new ceiling is **ten workers, sixteen CPUs and 104 GiB**. This graph reserves at most ten workers/twelve CPUs/104 GiB and supersedes the earlier eight-worker/96-GiB scheduling limit. Each MuJoCo task still uses one CPU/4 GiB/30 h with no requeue. Every existing dependency remains in place; the memory checks also wait for all six local-scale checks in 21280689. Unsubmitted learning studies retain their existing gates and priorities.

All 48 prior 500k pilots and their frozen metadata pass the revision preflight. Native peak 1.58 GiB supports the existing 4-GiB task request. The original pilot-scaled workload estimate is 277.96 worker-hours; dividing it by two versus four workers gives ideal whole-campaign bounds of 138.98 and 69.49 hours. Those figures exclude scheduling imbalance and do not forecast remaining wall time. No new trials, selected outcomes, replacement seeds or additional per-trial steps are introduced.

The two added slots have started Walker 2×3 MAPPO and AR-PPO. Their source/configuration bytes, native versions and training streams match the registered pilots, and each first 4,000-step rollout has finite losses and the expected five actor/critic updates. These are startup checks, not complete benchmark outcomes. All ten live workers remain below their observed memory allocations and linear runtime limits in the [resource snapshot](../experiments/live_resource_check_20261029_acceleration.json).

[Revision plan](../experiments/mujoco_coverage10m_concurrency_plan_20261029.json), [verified scheduler change](../experiments/mujoco_coverage10m_concurrency_schedule_20261029.json), [native startup evidence](../experiments/mujoco_coverage10m_concurrency_startup_20261029.json). The first helper had a syntax error before execution; it changed no scheduler or ledger state. The original file is retained, and the verified action uses `apply_v2.py`.


## Independent allocations and Prey learning, 29 October 2026

The Ant confirmation and Prey native checks had no scientific dependence on Football or each other. Their required selection/reference data was already complete. Two capacity waits were removed after every input/configuration hash and the Ant strict registration passed preflight. Only `Dependency` changed on the pending arrays; source, seeds, per-task limits, partitions, maximum concurrency and acceptance rules remain fixed. [Plan](../experiments/independent_study_capacity_plan_20261029.json), [verified revision](../experiments/independent_study_capacity_schedule_20261029.json).

| Allocation | Current chain | Maximum reserved resources |
| --- | --- | --- |
| Football | 21251960 | Two workers, two CPUs, 16 GiB |
| Prey | 21282173 → 21282210 (both complete), then explicitly released **21287360** (24×1M) | Two workers, two CPUs, 8 GiB |
| Ant and continuous-memory checks | 21282468 → 21282782 → 21284086 → 21284087 | Two workers, two CPUs, 8 GiB |
| SMAC | 21272799 → 21281002 | Two workers, four CPUs, 64 GiB |
| MuJoCo coverage | 21272800, max four active | Four workers, four CPUs, 16 GiB |
| HC confirmation/controls/final verification | 21279129 → 21279857 → 21280689 → 21281001 → 21284586 | Two workers, two CPUs, 8 GiB |

The new ceiling and scheduled maximum are **fourteen workers/sixteen CPUs/120 GiB**. The sixteen-CPU limit is unchanged. Memory checks still require all six local-scale gates in 21280689; all audit/data dependencies remain. The prior eighteen-run local-scale study keeps its HC-chain reservation and priority for eligible uncommitted capacity. Eighteen recurrent development runs also remain unsubmitted.

All eight Prey gates and their family auditor pass, including exact replay and four original prefixes. The measured 1.372-GiB peak and 3.494-h estimate with a 50% margin support the explicitly released learning allocation: two one-CPU/4-GiB/4-h workers. This replaces completed gates and adds no concurrency beyond this graph. All 24 configurations were frozen before the gates; all outcomes remain required before selection.

Ant seeds 6000/6001 run with their registered twenty-module trainer, native versions and streams. HC confirmation has four of ten valid completed runs; aggregate decisions remain withheld. The [resource snapshot](../experiments/live_resource_check_20261029_independent_studies.json) records actual live tasks; the table is the maximum, not a claim that all slots run at each instant. Scheduler-priority waits remain ordinary pending states. No task is restarted or replaced.


## Prey learning completion collector, 29 October 2026

The Prey branch now continues **21287360 → 21288041**. The collector replaces completed learning workers with one CPU/4 GiB for 45 minutes; the maximum remains fourteen workers/sixteen CPUs/120 GiB. It audits all 24 complete curves and performs all 24 full-protocol final replays before the frozen common-coefficient rule. It cannot release new training. [Plan](../experiments/prey_distillation_learning_audit_plan_20261029_v2.json); [submission](../experiments/prey_distillation_learning_audit_schedule_20261029.json). Every other allocation, data dependency and unsubmitted study reservation remains as above.
