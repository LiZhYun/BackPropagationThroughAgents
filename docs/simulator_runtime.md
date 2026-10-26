# Measuring simulator runtime

Before changing rollout execution, separate policy computation, environment
calls and PPO updates on the same worker. `scripts/profile_simulator.py` adds
wall-clock timers around the existing CPU trainer and environment adapters.
It forwards the original calls, arguments, results and exceptions unchanged.
It restores the trainer functions when it exits, including after a failure.

Use a flat simulator configuration and a fresh output directory:

```bash
python scripts/profile_simulator.py --config RUN_CONFIG.json --output NEW_PROFILE_DIRECTORY
```

Run with the corresponding native environment and library paths from the
[SMAC](smac.md), [MuJoCo](mujoco_setup.md) or [Football](football.md) guide. When
profiling a frozen experiment, select its source through `PYTHONPATH`. The
instrumentation is a separate script: `runtime_profile.json` records its hash
alongside the trainer's source hashes and metadata hash. Copy that script into
the experiment snapshot when freezing a new measurement campaign.

The regular training logs, checkpoint and summary are retained. The additional
timing report records call counts and accumulated seconds for collection,
optimization, evaluation, and environment construction/reset/step/close calls
within each phase. Environment times are **nested** inside collection or
evaluation; adding both totals would count that work twice. Setup, logging and
checkpoint overhead are not individually timed. CUDA is rejected because this
profiler does not synchronize asynchronous device work.

Six focused checks cover exact training-summary and checkpoint-state parity
for MAPPO, BPPO and recurrent AgentMixer on real Predator–Prey, exception
propagation, function restoration, output preservation and the CPU-only guard.
This establishes instrumentation behavior, not a native performance gain.

## Initial SMAC diagnostic

The [engine-free profile](../experiments/smac_cpu_profile_20261026.json) uses the
exact small-batch pilot source and its native tensor shapes: two environments,
five agents, 82 observation features, 120 state features and 11 actions. Each
method has one warmup update, three timed updates and a separate cProfile call
trace. Inputs, rewards, legal masks and episode lengths are synthetic.

| Method | Synthetic median seconds per 256-transition update | Recent native median |
| --- | ---: | ---: |
| MAPPO | 0.386 | 2.212 |
| AR-PPO | 0.631 | 2.442 |
| BPPO | 0.765 | 2.427 |
| AgentMixer | 0.853 | 2.456 |
| Identity modifier | 0.598 | Not started at this snapshot |

Synthetic timings were measured on an Intel Xeon Gold 6148 login host, while
the native runs execute on `milan1`. Native intervals include collection,
optimization, logging, checkpointing and any episode-boundary restart, but
exclude the preceding evaluation call. The different inputs and hardware mean
these differences are **not** an exact simulator-time fraction or a measured
speedup. They motivate direct native timing before selecting an optimization.

The adapter currently serializes native calls while installing private copies
of upstream global Python/NumPy random state. A thread pool around those calls
would therefore still serialize them. This observation does not justify
removing RNG isolation. Any future process-based execution must preserve native
trace parity, episode boundaries, seed streams and the resource ceiling.

## Frozen native measurement

`configs/smac_runtime_profile4096.json` declares MAPPO and BPPO at 4,096 decisions,
two environments × 128 rollout steps, and one initial/final evaluation episode.
The [study record](../experiments/smac_runtime_profile_study_20261026.json) freezes
the exact pilot source, full configurations, timing wrapper and launchers.
The frozen launcher invokes the profiling script instead of the ordinary CLI;
running the standard preparation script alone does not select instrumentation.

Each native profile must complete 16 rollout/update calls and 4,096 timed
training-environment steps. Its two evaluations must match the separately
counted evaluation interactions. Inspect nested timing fractions and Slurm peak
memory before changing execution. These are initial-policy, short-run timings;
trained policies, different batch sizes and native restart frequency can change
the result. No learning or paper-performance claim uses these profile rewards.

Job `21215880` contains two workers requesting 2CPUs/8GB for at most 30 minutes.
The [latest schedule](worker_schedule.md#short-diagnostics-first-2026-10-26)
runs worker 1 after SMAC worker 1, then worker 0, followed by the joint-matrix
control. Both profiles completed with scheduler exit 0. The four-worker ceiling
and all frozen measurement settings were preserved.

## Native measurements

The [measurement record](../experiments/smac_native_runtime_20261026.json)
verifies the source, configuration, instrumentation and launcher hashes, all
16 updates and 4,096 training steps per method, two evaluations, and separately
counted evaluation interactions. Both ran on an AMD EPYC 7713 worker.

| Method | Collect + update | Environment steps | Environment resets | Other collection | Updates | Environment fraction | Reported MaxRSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MAPPO | 55.28s | 22.00s | 28.48s | 4.10s | 0.70s | 91.3% | 2.47 GiB |
| BPPO | 60.25s | 22.07s | 28.06s | 7.49s | 2.63s | 83.2% | 2.95 GiB |

The environment fraction is `(step seconds + reset seconds) / (collection
seconds + update seconds)`. These four training components do not overlap.
Evaluation adds 14.17s/14.87s respectively and is excluded from that denominator.
The full instrumented wall times are 71.22s/79.59s; initialization, logging,
checkpointing and final cleanup account for the remainder.

These short runs include engine startup during first reset and frequent resets
from initial policies: 106 MAPPO and 103 BPPO training reset calls. They are not
steady-state timings, and larger batches or trained policies can change the
fractions. MaxRSS is Slurm's sampled batch-step accounting, not a guaranteed
memory ceiling for future process workers or larger rollouts.

The result supports testing separate environment processes. A proposed change
must first preserve sequential/parallel native traces, seed streams, episode
endings and cleanup, then demonstrate throughput and memory within the same
resource allocation. No process-based execution or measured speedup is claimed
by these profiles.

## Requirements for a process diagnostic

The current collector steps every environment, evaluates the resulting terminal
states for bootstrapping, then resets completed environments. Responses must
remain in environment-index order. Each environment needs its own process and
unchanged derived seed; policy sampling and optimization must stay in the parent.
The diagnostic should compare full native observations, masks, rewards, ending
flags and reset traces before measuring throughput under the same allocation.
Separate startup from subsequent step/reset time and include communication cost.

Observation normalization adds an ordering constraint. `NormalizedEnv` shares
training moments across environments and updates them on each reset and step.
Raw native calls could overlap, but the parent must apply those moment updates
in the same order as sequential execution. Merely forwarding asynchronous calls
through `NormalizedEnv.__getattr__` would bypass its observation transformation.
Any later rollout integration therefore needs exact normalized-batch, recurrent
carry, parameter-update and checkpoint parity, plus owned-worker cleanup checks.
The standalone diagnostic below implements the raw-call comparison. A process
backend has not been integrated into training or adopted by frozen experiments.


## Standalone process comparison

`scripts/profile_environment_processes.py` owns one spawned process per
environment and retains a sequential reference. It sends all step/reset calls
before collecting replies in environment-index order. Each variant creates
fresh environments; the same private action generators choose legal actions.
No learned policy, shared normalizer or PPO optimizer enters this measurement.

Use the corresponding native Python environment and SMAC setup:

```bash
python scripts/profile_environment_processes.py --config configs/smac_process_profile.json --output results/profile_output/new_smac_process_check
```

`--source-manifest` additionally requires the loaded package to match a selected
source snapshot. Full JSONL traces retain array dtype/shape, every observation,
action, reward, ending flag, info and explicit reset. Initial observations and
terminal observations remain distinct. Every pair and repeat must have identical
per-environment trace hashes before a timing ratio is reported.

Native step plus subsequent reset wall time includes IPC. Construction, initial
reset, close, and the loop including trace I/O are reported separately. The
sequential/process call-time ratio is not an end-to-end training speedup.
Worker errors, timeouts, changed traces or nongraceful cleanup fail the
diagnostic and remain in `report.json`. Cleanup acts only on owned children;
a forced cleanup invalidates the measurement and native descendants then rely
on allocation cleanup. It is not an accepted training-worker recovery mechanism.

### Frozen native study

The [study](../experiments/smac_process_profile_study_20261026.json) retains the
exact 19-module source of the earlier native timing profiles:

| Condition | Environment decisions per mode/trial | Trials | Restart threshold | Purpose |
| --- | ---: | ---: | ---: | --- |
| Normal | 2 environments × 512 | 3 | 480,000 native ticks | Compare native calls with IPC under the usual lifecycle |
| Frequent restart | 2 environments × 512 | 1 | 128 native ticks | Require trace parity through at least one planned restart per environment in both modes |

Each trial includes sequential and process variants, totaling 8,192 environment
decisions. Normal trials alternate their order. Seeds 2865252013 and 1680950762
are the earlier pilot's derived environment streams; these are runtime checks,
not independent training seeds. Both modes keep those streams unchanged.

The [preflight](../experiments/smac_process_profile_preflight_20261026.json)
passes 23 tests: real Prey trace/reset parity, parent RNG preservation,
pending-response draining, worker errors, stopped-child timeout/reaping,
output preservation, source checks and bounded configuration validation.
The copied script also passes exact raw-trace equality on 32 Prey decisions in
the native Python 3.11.15 / NumPy 1.23.5 prefix. That cheap-environment probe has
a sequential/process call-time ratio of 0.383; IPC costs exceed the saved native
work there. It supplies no SMAC speed claim.

Job **21218609** was queued with 2CPUs/8GB and a 30-minute cap; it has now
completed successfully. It follows
`21213860_0`, and entropy worker `21211006_0` follows it.
The [schedule proof](../experiments/smac_process_profile_schedule_20261026.json)
records both held workers' verified edges/resources before release. The four
serial chains and all active training jobs were preserved. Native trace,
restart, timing and sampled-memory gates passed as recorded below. Artifacts go into
`profile_output/` inside the frozen experiment directory.

### Instrumentation files

| File | Change | Purpose |
| --- | --- | --- |
| `scripts/profile_environment_processes.py` | Created | Standalone sequential/process calls, full traces, timers and owned-worker cleanup |
| `tests/test_profile_environment_processes.py` | Created | 23 parity, failure, cleanup and provenance checks |
| `configs/smac_process_profile.json` | Created | Normal native measurement |
| `configs/smac_process_restart_profile.json` | Created | Frequent-restart parity check |
| `experiments/smac_process_profile_study_20261026.json` | Created | Frozen source/configuration/instrumentation hashes and decision gates |
| `experiments/smac_process_profile_preflight_20261026.json` | Created | Test and copied-script verification |
| `experiments/smac_process_profile_schedule_20261026.json` | Created | Queue insertion and resource verification |

The frozen directory also contains copied instrumentation, source/configurations
and `submit.sbatch`. These profiling files can be removed after measurement
without changing the trainer, but retain their evidence for reported timings.

## Completed native process diagnostic

The [independent audit](../experiments/smac_process_profile_results_20261026.json)
passes all 8,192 recorded environment decisions. It recomputes each raw trace
hash, every private legal-action RNG sequence, reset/episode counts, planned
restart counters and paired timing ratios using the native NumPy version.
Whole trace files, including observations, actions, rewards, endings and info,
are byte-identical between sequential and process execution and across all
repeats within each condition. All children exit gracefully without forced
cleanup or close errors. The exact 19-module source and profiler hashes match
the frozen plan.

| Condition | Decisions across both modes | Sequential/process call-time ratios | Median ratio |
| --- | ---: | --- | ---: |
| Normal 480k native-loop restart threshold | 6,144 | 1.2753, 1.2914, 1.2548 | 1.2753 |
| Diagnostic 128 native-loop threshold | 2,048 | 0.9979 | 0.9979 |

Normal environment calls take about 21.6% less time with processes in these
paired measurements. Frequent-restart calls show no gain: reset time dominates
and the two environments usually reset at different decisions. Each mode
records 10 and 12 planned restarts for its two environments in that condition.
The job completes in 467s and peaks at 1.73GiB resident memory. This standalone
profiler has no policy or optimizer; its memory use does not establish a full
training bound.

These measurements use one host, two fixed environment seeds and uniformly
sampled legal actions. Call times exclude policy inference, optimization,
normalization and trace I/O. They do not establish a training speedup. No
process collector has been added to the training path; normalized-rollout,
recurrent-memory, RNG, optimizer and failure-cleanup parity would be required
first. The [released SMAC learning study](recurrent_actors.md#completed-native-gate-and-learning-release)
keeps its sequential backend and uses bounded single-run tasks.
