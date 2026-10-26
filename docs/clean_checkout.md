# Clean installation verification

On 2026-10-26, two tracked source exports were installed in fresh private
environments. Neither export contained the local paper archives, experiment
results or an existing environment. The machine-readable
[verification record](../experiments/core_release_verification_20261026.json)
identifies the commits, commands, package hashes and local artifacts.

## What passed

The first export, commit `1fd8c61`, used `uv sync --locked --extra dev` with
Python 3.11.15, CPU PyTorch 2.6.0, NumPy 2.4.6 and pytest 8.4.2. The full suite
passed **693 tests**, with one CUDA-only skip, in 24.18 seconds. SMACv2,
mujoco-py and Football were absent. The package imported from that fresh export.
The initial offline installation lacked a cached PyTorch wheel; the online
locked installation succeeded.

The second export, commit `7021938`, built a wheel using the pinned build tools.
That exact wheel was installed into a separate environment. Its 23 Python
modules match the first export byte for byte. The wheel contains the README
and the original MIT notice, and excludes local results, references and
environments. Commands below ran from outside the source directory, with an
empty `PYTHONPATH`; the package imported from `site-packages`.

| Installed command | Training steps | Greedy return | Expected local execution return |
| --- | ---: | ---: | ---: |
| BPPO Climbing, `matrix_projected_entropy01.json`, seed 0 | 100,000 | 11 | 10.998430 |
| AgentMixer Climbing, `matrix_joint_lr_fast.json`, seed 0 | 200,000 | 11 | 8.310479 |

These are the README examples with explicit output directories. Both complete
their stated budgets. AgentMixer's central expected return is 11, but its local
expected return is lower: a greedy optimum does not establish reliable local
sampling or close the training/execution gap. These reused development seeds
are installation checks, not independent performance evidence.

The installed `bta-train-simulator` command also completed a 16-transition
recurrent AgentMixer Predator–Prey check. The saved evaluator reconstructed
the policy from its configuration and exactly replayed both episode and
fixed-window evaluations with matching source hashes. Its shortened episode
limit and tiny budget make this a CLI/checkpoint check only.

The first clean export exposed a progress-report crash when local manifests
were absent. Commit `7021938` fixes that case and rejects paths outside the
checkout. All five new progress-report tests pass in the second environment
(0.09 seconds). These are additional targeted tests; the full suite was not
repeated after the package-only changes. Raw pytest stdout was observed in the
session but was not saved to a log file. The recorded totals do not imply a
saved test report.

The standard-library-only target checker validates all 45 target references
(six numeric and 39 digitized); the progress command now returns unknown
evidence for unavailable manifests. Both commands exit successfully with
`python3 -S`, without importing the installed learning dependencies.

## Continuous integration

The [core workflow](../.github/workflows/core.yml) builds and installs the exact
wheel before running the full suite with pytest's source-path injection
disabled (`-o pythonpath=`). It also checks the installed console commands and
checkpoint replay outside the checkout, followed by all three public reports.
The workflow uses a CPU-only locked environment, one numerical thread, pinned
action commits and read-only repository permissions. Native engines and local
experiment artifacts are unnecessary.

A third clean export, commit `e47bc0d`, passed these commands locally:
**698 passed, one CUDA-only skip in 37.50 seconds**, both short matrix CLI runs,
the recurrent Prey CLI run and exact replay, and all three standard-library
reports. The [CI verification record](../experiments/core_ci_verification_20261026.json)
includes the JUnit/stdout hashes and exact installed module hashes. This full
installed-wheel run includes the five added progress-report tests. Workflow
YAML and shell syntax were checked locally. No hosted GitHub Actions run has
been triggered; its runner and action setup remain unverified here.

## Repeat on another machine

From a fresh checkout, run:

```bash
uv sync --locked --extra dev
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --no-sync python -m pytest -q
uv build --wheel
```

Use a separate environment to install the resulting wheel and run the README
commands from outside the source tree. Pass absolute paths to the JSON
configurations and fresh output directories. The wheel supplies `bta-train`
and `bta-train-simulator`; scripts, example configurations and research records
remain in the source checkout.

After installing the wheel, run its Python on
`scripts/check_installed_package.py --output NEW_DIRECTORY` from outside the
source checkout. The output must also be outside that checkout. This reusable
check compares installed modules, README and license with the source, exercises
both console entry points and verifies exact recurrent Prey checkpoint replay.
Its tiny budgets test integration only. Command logs and a verification JSON
are retained in the chosen directory.

This verifies the core CPU installation on the recorded platform. Separate
native-engine installations are covered by the [SMAC](smac.md),
[MuJoCo](mujoco_setup.md) and [Football](football.md) guides. Full benchmark
results, independent training seeds and qualified paper comparisons remain
release requirements in [STATUS.md](../STATUS.md).
