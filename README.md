# BPTA and AgentMixer

One readable implementation of [Backpropagation Through Agents](https://arxiv.org/abs/2401.12574) and [AgentMixer](https://arxiv.org/abs/2401.08728), sharing rollout collection, PPO utilities, critics, evaluation and checkpointing.

The goal is a clear, faithful implementation of the papers’ core mechanisms, with explicit corrections where the original equations or code are ambiguous. Full benchmark sweeps are optional. **This repository does not claim that every published score has been reproduced or exceeded.**

## Quick start

The core needs Python and CPU PyTorch; no simulator download or external logging account is required. The tested lock uses Python 3.11.15 and PyTorch 2.6.0. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --extra dev
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q
```

Run BPPO on the accepted paper’s Climbing payoff table:

```bash
python -m bta_repro.train --algorithm bppo --game climbing \
  --steps 4096 --seed 0 --output results/bppo_example
```

Run the corrected, factorized AgentMixer reference on the same game:

```bash
python -m bta_repro.train --algorithm agentmixer --game climbing \
  --steps 4096 --seed 0 --output results/agentmixer_example
```

These short commands demonstrate the training path; they are not convergence or performance tests. Available algorithms are `mappo`, `arppo`, `bppo` and `agentmixer`; games are `climbing` and `penalty`. Each output directory must be new. Runs save configuration, source hashes, metrics and a checkpoint.

To exercise temporal rollouts and decentralized AgentMixer execution without an optional engine:

```bash
python -m bta_repro.simulator_train --config configs/prey_example.json \
  --output results/prey_example
python scripts/evaluate_simulator.py results/prey_example \
  --output results/prey_example_replay.json
```

This example trains for 10,240 joint environment decisions. Predator–Prey reports both complete-episode diagnostics and the released 100 × 200-decision evaluation windows; the primary window score is `final_evaluation.fixed_window.reward_per_environment_step`. Evaluation decisions are counted separately from training.

## Read the implementation

| Start here | What it explains |
| --- | --- |
| [Method fidelity](docs/method_fidelity.md) | Paper mechanisms, explicit corrections and optional extensions |
| [envs.py](src/bta_repro/envs.py), [train.py](src/bta_repro/train.py) | Small matrix games and a complete training loop |
| [core.py](src/bta_repro/core.py) | Batches, clipped PPO, GAE and the independent actor baseline |
| [bppo.py](src/bta_repro/bppo.py), [continuous variant](src/bta_repro/bppo_continuous.py) | Autoregressive decisions, reverse updates and peer feedback |
| [agentmixer.py](src/bta_repro/agentmixer.py), [continuous variant](src/bta_repro/agentmixer_continuous.py) | Centralized policy modification and local execution |
| [runner.py](src/bta_repro/runner.py), [simulator_train.py](src/bta_repro/simulator_train.py) | Shared temporal collection, update and evaluation |
| [checkpoint.py](src/bta_repro/checkpoint.py) | Model, optimizer and random-state persistence |

BPPO and AR-PPO execute autoregressively and require predecessor actions. AgentMixer deploys each local policy without the centralized modifier or global state. The evaluator preserves this distinction.

The [configuration guide](configs/README.md) distinguishes a single runnable configuration from a multi-run research plan. Optional action projections, correlated two-agent modifiers, recurrent actors, Gaussian scale controls and local distillation are documented extensions; they are not silently substituted for the reference examples above.

## Optional simulators

Use a separate private environment for each engine: [MA-MuJoCo](docs/mujoco_setup.md), [SMACv2](docs/smac.md), or [Google Research Football](docs/football.md). The [shared training guide](docs/simulator_training.md) documents configurations, seeds, episode boundaries, metrics and saved-policy evaluation. [Runtime versions](docs/environment_versions.md) record the tested stacks. The core examples do not import these engines.

## Validation and results

The current numerical suite contains 1,093 passing tests and one CUDA-only skip on the recorded CPU stack. Checks cover likelihoods, causal gradients, masks, mode consistency, termination/truncation, local information isolation and checkpoint replay. [Installed-wheel verification](docs/clean_checkout.md) distinguishes local validation from hosted CI and native-engine checks.

Existing successful, unsuccessful and interrupted experiments are retained. See [STATUS.md](STATUS.md) for release readiness and [the experiment tracker](refine-logs/EXPERIMENT_TRACKER.md) for history. Tests and short examples establish implementation behavior; performance claims require matched protocols, complete seed groups and uncertainty. No benchmark sweep is required to use or contribute to the code.

For deeper details, read the [BPPO objective](docs/bppo.md), [AgentMixer correction](docs/agentmixer.md), and [accepted-source audit](docs/source_audit.md). Original paper archives, private environments and local training artifacts are excluded from the source checkout. The concurrent unrelated reproduction is untouched.

## License, citation and contributions

The code retains the [MIT license](LICENSE) from the [pinned original implementation](https://github.com/LiZhYun/BackPropagationThroughAgents/tree/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0). Optional engines retain their own licenses. Cite the accepted papers using [CITATION.bib](CITATION.bib), and identify the commit and configuration when reporting results. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.
