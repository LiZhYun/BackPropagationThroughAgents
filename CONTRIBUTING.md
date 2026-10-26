# Contributing

Start with the README quick start and the implementation note for the method
you want to change. Matrix games and Predator–Prey need only the core install;
native simulators are optional. Keep a single run runnable with a flat JSON
configuration before adding a larger experiment plan.

## Code changes

Keep shared numerical operations in `core.py`, environment behavior in its
adapter, and method-specific objectives in the corresponding policy module.
Use names that identify the probability event, reduction or gradient path.
Document any equation correction or changed execution information alongside
the code. A new architecture or estimator should have an explicit configuration
option and a matched control.

Run the checks relevant to the change. For changes to shared policy, rollout or
evaluation behavior, run the full core suite:

```bash
uv sync --locked --extra dev
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --locked pytest -q
```

Tests should check mathematical or behavioral properties: likelihood identity,
causal gradients, legal actions, episode boundaries, execution information and
checkpoint replay. Native integration and learning performance are separate
evidence. Include the commands and results with the change.

## Experiments

Use the [configuration guide](configs/README.md) to distinguish a flat training
configuration from a multi-run plan. Preparation freezes source and settings;
keep those snapshots unchanged once a run starts. Write new runs to new output
directories. Record simulator versions, actual environment decisions, evaluation
protocol, every declared seed and all failures. The [curve reports](docs/learning_curve_reports.md)
and [matrix validation report](docs/matrix_confirmation.md) check consistency
before aggregating results.

Use development seeds for selection. Freeze the configuration, evaluation rule
and unused confirmation seeds before launching validation. Publish the complete
seed table and uncertainty, including unsuccessful outcomes. A successful smoke
run or a selected best checkpoint does not establish a paper-level result.

For a bug report, include the commit, configuration, command, expected behavior,
observed behavior and a minimal log or failing check. For a result report,
include the frozen manifest and metric definition. Keep original papers and
reference captures unchanged; document findings in a separate note.
