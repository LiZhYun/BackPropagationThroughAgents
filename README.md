# BPTA and AgentMixer, reproduced together

A readable, shared implementation of [Backpropagation Through Agents](https://arxiv.org/abs/2401.12574) and [AgentMixer](https://arxiv.org/abs/2401.08728).

**Status: implementation and experiments in progress.** Matrix games, Predator–Prey and a shared SMAC/MuJoCo/Football trainer are runnable. Real simulator checks pass. Full benchmark reproduction and performance beyond the papers are not yet established. This repository documents corrections to the old implementation instead of silently preserving its bugs.

## Quick start

Use Python 3.10 or later. The development environment currently pins CPU PyTorch 2.6.0; larger simulator experiments will have their own recorded dependency locks.

With uv, `uv sync --locked --extra dev` uses the checked-in CPU dependency lock. The manual installation below is also supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python -m bta_repro.train --config configs/matrix_projected_entropy01.json \
  --algorithm bppo --game climbing --steps 100000 --seed 0
```

The BPPO example uses the projected predecessor-action inputs and entropy
schedule that solved both matrix games in all three development seeds. Replace
`bppo` with `arppo` for the matched baseline; that baseline also succeeds.
This example demonstrates learning, not a peer-feedback advantage.

For the experimental correlated AgentMixer variant:

```bash
python -m bta_repro.train --config configs/matrix_joint_lr_fast.json \
  --algorithm agentmixer --game climbing --steps 200000 --seed 0
```

This setting solves Climbing in the three development seeds but fails Penalty.
The factorized reference remains the default when no joint configuration is
provided. See the [configuration guide](configs/README.md) for experiment-plan
formats, tested settings and their limitations.

Algorithms: `mappo`, `arppo`, `bppo`, `agentmixer`. Games: `climbing`, `penalty`. Every run creates a fresh result directory with configuration, source hashes, raw metrics, an exact matrix-game evaluation, and a checkpoint containing optimizer and random-number-generator states. `--output` must name a directory that does not already exist.

Matrix training uses one-step episodes and reports reward per joint environment step, matching the pinned legacy environment's termination convention. Legacy training logs multiply mean reward by 200, while evaluation reports reward per decision. The exact connection to historical plot files is unverified; see the [protocol audit](docs/other_curve_targets.md).

## Reading the code

1. `src/bta_repro/envs.py`: the accepted BPTA payoff tables and exact joint-action enumeration.
2. `src/bta_repro/train.py`: collection, advantage estimation, update, and evaluation in one place.
3. `src/bta_repro/core.py`: named batches, PPO, termination/truncation-aware GAE, and the independent actor baseline.
4. `src/bta_repro/bppo.py`: autoregressive decisions and backward peer feedback. See [the BPPO implementation notes](docs/bppo.md).
5. `src/bta_repro/agentmixer.py`: centralized policy modification and local execution. See [the AgentMixer implementation notes](docs/agentmixer.md).
6. `src/bta_repro/checkpoint.py`: model, optimizer, and RNG persistence.

The [shared simulator guide](docs/simulator_training.md) connects these policies to [temporal rollout collection](docs/runner.md). Continuous variants use explicit Gaussian vector events. Optional simulators have separate setup instructions: [SMACv2](docs/smac.md), [legacy MA-MuJoCo](docs/mujoco_setup.md) and [Football](docs/football.md). [Predator–Prey](docs/predator_prey_protocol.md) needs only the core dependencies. Optional engines are not needed for numerical tests or matrix games.

[Matrix development notes](docs/matrix_development.md) retain negative results and explain subsequent controls. The [two-agent joint modifier](docs/agentmixer_joint.md) is an experimental correlated redesign with exact mode-preservation checks; it is not silently substituted for the factorized AgentMixer reference.

[Simulator development results](docs/development_results.md) report the completed
optimization controls and their limits. Current jobs and remaining acceptance
requirements are recorded in [STATUS.md](STATUS.md).

Saved simulator policies can be evaluated with
`python scripts/evaluate_simulator.py RUN_DIRECTORY --output NEW_EVALUATION.json`.
The command restores observation preprocessing and checks source hashes; see
the [checkpoint replay instructions](docs/simulator_training.md#evaluate-a-saved-policy).

Three read-only reports need only Python's standard library:

```bash
python scripts/report_progress.py
python scripts/check_paper_targets.py
python scripts/report_benchmark_coverage.py
```

The first reports all declared runs in recent manifests, including unfinished
and failed runs. The second checks that all45 benchmark targets have either an
explicit source value or a linked curve estimate; it does not certify achieved
performance. The third maps every target to recorded experiment artifacts,
retaining missing runs and basic task mismatches; it does not compare scores
or certify reproduction. See the [coverage guide](docs/benchmark_coverage.md).

Full-budget learning curves, curve areas and training-seed uncertainty can be
exported with `scripts/summarize_learning_curves.py`. Every declared seed must
finish at matching settings before the report pools a group. See the
[curve-report guide](docs/learning_curve_reports.md) for commands, metric
denominators, bootstrap conventions and optional PDF/PNG/SVG figures.

The [matrix validation guide](docs/matrix_confirmation.md) records the frozen
unused-seed study and separate paper-budget development extension. Its reporter
checks provenance, recomputes scores and retains every declared seed before
making a configuration-validation decision.

BPPO executes autoregressively and requires predecessor actions. AgentMixer executes using each local policy alone. The evaluator preserves this distinction.

## What counts as reproduced

The accepted local AAAI sources define the protocol. [The source audit](docs/source_audit.md) and `configs/paper_targets.json` distinguish exact numeric targets, plot-only results, and unresolved inconsistencies. The benchmark plan is in [refine-logs/EXPERIMENT_PLAN.md](refine-logs/EXPERIMENT_PLAN.md).

Release requires method checks, reproducible installation and execution, full benchmark evidence under comparable task definitions and step budgets, and improvement beyond published performance where possible. Mathematical ceilings are recorded explicitly: improving an already optimal matrix-game result means greater reliability or sample efficiency, not a reward above the optimum.

Development seeds select configurations. Independent final seeds validate them. All final runs, including failures, are reported. Corrected implementations are labeled as such; passing tests does not establish the papers' convergence guarantees.

The original source archives and the other reproduction in the parent workspace are kept untouched. Reference source copies and local environments are excluded from version control.
