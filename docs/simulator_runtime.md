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
Each waits for the matching SMAC pilot worker (`21209234_i`), and the pending
joint-matrix worker (`21211813_i`) waits for that profile. Both arrays were
released after all four changed edges were verified, preserving four serial
chains. Submission and verified dependencies are recorded in
`experiments/jobs.jsonl`. Results are pending; no native timing fraction or
speedup has been established by this study yet.
