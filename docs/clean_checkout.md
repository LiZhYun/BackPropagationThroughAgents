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

This verifies the core CPU installation on the recorded platform. Separate
native-engine installations are covered by the [SMAC](smac.md),
[MuJoCo](mujoco_setup.md) and [Football](football.md) guides. Full benchmark
results, independent training seeds and qualified paper comparisons remain
release requirements in [STATUS.md](../STATUS.md).
