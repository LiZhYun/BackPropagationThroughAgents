# Clean installation verification

## Current wheel: 29 October 2026

The exact committed export of `f7f9de3` passes **1,044 installed-package tests
and one CUDA-only skip** in 54.93 seconds, with pytest source-path injection
disabled. The package imports from `site-packages`; all 25 installed module
files match the export byte for byte. The packaged README and MIT notice
match. This includes all 28 local-distillation tests and 41 continuous-memory tests; the continuous
checkpoint tests use synthetic environments.

The dependency lock is unchanged from the fresh check below. This run reuses
its inactive CPU validation prefix: Python 3.11.15, PyTorch 2.6.0+cpu, NumPy
2.4.6 and pytest 8.4.2. Only this project's wheel is temporarily replaced,
using an offline build and reinstall without resolving dependencies. The
dependency check passes, all installed package versions remain unchanged, and the original
wheel, module bytes and packaged README are restored afterward. No new
validation environment is built and no training environment changes.

Both matrix console commands and the original recurrent Predator–Prey replay
pass outside the export. An additional recurrent Prey CLI run with
`agentmixer_distillation_coef=0.1` completes with finite auxiliary metrics and
exact final episode/window replay. The matrix runs use 64 decisions each;
each Prey run uses 16. These budgets check integration. All three public
standard-library reports pass, keeping the 74 absent local manifests explicit
in the clean export. They do not certify achieved performance.

The [current verification record](../experiments/core_wheel_controls_verification_20261029.json)
retains the exact commit, archive, wheel, modules, JUnit report, fourteen
command logs, replay evidence and restoration checks. This is a current-wheel
check in reused locked dependencies. A fresh or uncached online installation
is not repeated; optional native engines are absent and hosted GitHub Actions
remains untested.

## Fresh installation: 29 October 2026

Commit `ddde9dc` now passes a fresh, non-editable wheel installation with
**975 tests passed and one CUDA-only skip**, in about 65 seconds. Pytest's
source-path injection was disabled. All 25 installed Python modules match the
committed export byte for byte, and the installed package imports from
`site-packages`. This includes the optional continuous local-scale control,
portable-study exporter tests and later reporting checks.

The export contains no local experiment results, paper archives or preexisting
environment. Locked dependencies, wheel construction and installation succeed
offline using the project cache. CPU PyTorch 2.6.0, NumPy 2.4.6 and Python
3.11.15 are recorded; SMACv2, mujoco-py and Football are absent. The dependency
check passes. The wheel includes the matching README and MIT notice.

Both matrix console checks and recurrent Predator–Prey training/replay pass
from outside the checkout. Saved-policy replay exactly matches every final
episode and fixed-window record. Their 64/16-step budgets are interface checks.
All three standard-library reports run successfully: absent local manifests
remain unknown or explicit missing-artifact errors, never completed evidence.

The [verification record](../experiments/core_wheel_verification_20261029.json)
contains the exact commit, archive, wheel, modules, dependency versions,
commands, JUnit/stdout hashes and installed-command replay evidence. This
validates the core CPU installation on the recorded platform. An uncached
online installation and hosted GitHub Actions were not repeated. Native
engines and full benchmark performance remain separate requirements.

## Earlier verification history

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

## Updated core and direct Prey example

A later clean export, commit `1e38c2d`, includes the source-snapshot preparation,
simulator confirmation reporter and CPU timing wrapper checks added since that
CI run. Its exact installed wheel passes **763 tests, one CUDA-only skip in
38.40 seconds**, with source-path injection disabled. All 24 installed modules,
README and license match the exported source. Both short matrix console checks,
recurrent Prey replay and the three standard-library reports also pass.

The new README `configs/prey_example.json` command then completes its unchanged
10,240-decision budget and 20 updates in 42.16 seconds on the recorded host.
Initial and final evaluations each contain 20 complete episodes and 100 windows
of 200 decisions; total evaluation work is 44,393 interactions, separately
counted from training. Both window reductions were recomputed from their raw
window totals. Saved-policy replay exactly matches every final episode and
window record. This remains a short interface check using a development seed.

The [updated verification record](../experiments/core_example_verification_20261026.json)
retains commands, the wheel and module hashes, JUnit/stdout hashes, example
configuration, logs, checkpoint and replay hashes. An initial offline install
failed because the locked CPU PyTorch wheel was absent from the cache; its logs
are retained. Online locked installation succeeded. Hosted CI remains untested.

## Repeat on another machine

The later [matrix identity-control verification](../experiments/matrix_identity_wheel_verification_20261026.json)
builds another fresh wheel and passes **899 tests, one CUDA-only skip in
55.48 seconds**, with source-path injection disabled. All installed module,
README, license, console-command and recurrent Prey replay checks pass. This
includes the optional matrix identity control and the paired-study reporter.
The existing native training environments remain unchanged.

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
