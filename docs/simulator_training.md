# Shared simulator training

`bta-train-simulator` (or `python -m bta_repro.simulator_train`) connects SMACv2, MA-MuJoCo, Football and Predator–Prey to the shared rollout and update implementation. It supports `mappo`, `arppo`, `bppo`, and `agentmixer` with discrete or continuous actions. Optional simulator packages are imported only when the corresponding environment is constructed.

`recurrent_actor` defaults to `false`. Discrete policies can enable private GRU
actor memory with `true`; factorized AgentMixer and its identity control are
supported. The shared critic remains feed-forward. See the [recurrent actor
guide](recurrent_actors.md) for buffer layout, episode resets, one-decision
gradient truncation, checkpoint replay and the explicit architecture differences.

For a runnable example without native engines or a scheduler, use the checked-in
flat configuration:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m bta_repro.simulator_train --config configs/prey_example.json \
  --output results/prey_example
```

It performs 10,240 training decisions and evaluates local AgentMixer actors
with both complete episodes and the released Prey fixed-window reduction.
This short interface example does not reproduce the paper's 1M-step budget.

Create an explicit JSON configuration, for example:

```json
{
  "environment": "smac",
  "env_config": {"scenario": "protoss_5v5", "game_version": "4.10"},
  "algorithm": "agentmixer",
  "seed": 0,
  "eval_seed": 1000000,
  "steps": 100000,
  "num_envs": 1,
  "rollout_steps": 128,
  "epochs": 4,
  "eval_episodes": 32,
  "eval_every_steps": 0
}
```

Then run:

```bash
PYTHONNOUSERSITE=1 .venv-smac/bin/python -m bta_repro.simulator_train \
  --config configs/my_smac_run.json --output results/my_smac_run
```

The example budget illustrates the interface; it is not the papers' full SMAC training budget. Existing output directories are rejected. `--algorithm`, `--seed`, and `--steps` can override the corresponding JSON fields. All other choices remain explicit in the JSON or recorded dataclass defaults.

For MA-MuJoCo, use `"environment": "mujoco"` and an adapter configuration such as:

```json
{"scenario": "HalfCheetah-v2", "agent_conf": "2x3", "agent_obsk": 0, "episode_limit": 1000}
```

See the adapter setup documents for simulator installation, supported partitions, versions, and deviations from the original releases. Unsupported tasks fail instead of silently substituting another benchmark.

Predator–Prey uses the core interpreter, `"environment":"prey"`, and
`"env_config":{"penalty":-1,"episode_limit":128}`. Football uses its private
interpreter, `"environment":"football"`, and for example
`"env_config":{"scenario":"academy_3_vs_1_with_keeper"}`. See their
[Prey protocol](predator_prey_protocol.md) and [Football setup](football.md).

Files such as `configs/prey_coordination300k.json` are experiment plans containing
multiple variants, rather than single-run configurations. Freeze a plan with
`python scripts/prepare_simulator.py configs/prey_coordination300k.json`.
The printed directory contains individual JSON configurations, copied source,
a manifest, and a Slurm submission script. Preparation does not submit jobs.
`wall_time_minutes` optionally changes its bounded runtime allocation; the
default is120 minutes, with at most two workers, each2CPUs/8GB. The measured
3M MuJoCo follow-up uses240 minutes. Keep total project concurrency within the
experiment plan's resource ceiling when submitting multiple manifests.

For continuous policies, `initial_std` sets the initial diagonal Gaussian standard deviation in policy action units. It defaults to `1.0` and must lie in `[exp(-5), exp(2)]`. For example, `0.3` reduces initial exploration while preserving the policy family and identity modifier. SMAC configurations reject nondefault values because their actions are categorical. The selected value is saved in the run configuration.

For discrete AgentMixer, `modifier_log_scale_bound` optionally selects the
existing smooth log-scale bound. For example, `2.302585092994046` constrains
gap scales to approximately `[0.1, 10]`. The default `null` retains the original
hard log-scale clamp. This option preserves local modes and shared initial
weights, but changes centralized exploration. It is rejected for continuous
policies, other algorithms and disabled modifiers. See the
[Prey control](prey_modifier_control.md) for its measured motivation, matched
configuration and limits.

## Training and evaluation

The run `seed` controls model initialization and sampling. Simulator seeds are derived with NumPy `SeedSequence([0x425441, 0x53494D, 1, seed]).spawn(num_envs)`, taking one uint32 value from each child. The fixed BTA/SIM/version namespace avoids the overlapping consecutive simulator streams that `seed + env_index` creates for adjacent run seeds. Increasing `num_envs` preserves the earlier environments' derived seeds. The scheme name and actual simulator seeds are recorded in `config.json`.

`env_config` cannot contain its own seed. Validation rejects collisions within the derived training seeds and any overlap with the explicit evaluation seed. The run seed may equal the evaluation seed if their actual simulator streams differ. Reducing child streams to 32-bit simulator seeds cannot guarantee collision-free seeds across every possible run; check the recorded lists when assembling a large multi-run campaign. Existing results retain their recorded seed scheme and are not changed retroactively. Training uses synchronous rollouts and configurable minibatch updates from [`runner.py`](../src/bta_repro/runner.py); the [runner documentation](runner.md) explains GAE, active-agent masks, raw Gaussian actions, clipping, and the algorithm-specific PPO objectives.

`steps` counts joint environment transitions, not individual agent actions or optimizer updates. It must be divisible by `num_envs`. The final rollout is shortened to meet the exact requested budget. Unfinished episodes and their observations carry across updates; they are not reset at rollout boundaries. Completed episode returns include the entire episode, and true terminations and time-limit truncations are counted separately. Partial episodes at the final budget are reported as unfinished and excluded from completed-episode return statistics.

`num_minibatches` defaults to `1`, preserving the original full-rollout update and RNG stream. Set it to `2` for two shuffled batches per epoch, for example. Every transition is retained, including uneven remainders. Validation rejects counts larger than the smallest rollout in the run. Advantage and value-target moments use the full rollout; BPPO keeps one behavior snapshot across every actor, epoch and minibatch. See the [update contract](runner.md#update-interface).

Initial and final evaluations use exactly `eval_episodes` episodes, each time constructing a fresh environment with `eval_seed`. Optional periodic evaluation runs after crossing `eval_every_steps`; zero disables it. Evaluation interactions are counted separately and excluded from the training budget. Python, NumPy, PyTorch, and already-initialized CUDA RNG states are saved and restored around evaluation, so adding evaluation does not change the training random stream.

Evaluation is deterministic: categorical actors use greedy actions and Gaussian actors use means. AgentMixer uses only its local actors, with no state or modifier. BPPO and autoregressive PPO retain ordered predecessor-action conditioning; their execution is explicitly labeled autoregressive. Continuous actions remain raw policy actions until the adapter applies actuator bounds. A failure to finish within `eval_max_steps` is an error rather than an invented truncation.

Prey can additionally report fixed-window mean reward with
`eval_window_steps=200, eval_windows=100`. This includes unfinished episodes
and resets completed ones within each window, matching the released logger's
reduction. This is the author-selected primary Prey comparison protocol
(2026-10-26). It appears separately under `fixed_window`; the normal episode
statistics retain their meaning. All additional evaluation steps are counted.

An evaluation stream reused during development is not automatically a held-out final benchmark. Select configurations using development runs, then evaluate the frozen selection on separately reserved seeds. Episode standard deviations within one run are not uncertainty across independently trained seeds.

## Observation normalization and value diagnostics

`normalize_observations` defaults to `false`, preserving the raw-input convention. When enabled, one running normalizer is shared by all training environments. It maintains separate moments for each agent's observation feature and for each global-state feature, updating once on every reset and step observation, including terminal observations. It centers and scales features using population variance plus a small epsilon, then clips them to `[-observation_clip, observation_clip]`; the default clip is `10`. This is running feature normalization, not the historical wrapper's independent normalization of each observation vector. Rewards, GAE targets, action masks, active-agent masks, and actions are unchanged.

Evaluation uses an independent frozen snapshot and never updates training moments. Initial evaluation starts with empty moments (zero mean, unit variance, with clipping still applied). The normalization convention is recorded in `config.json`. Each checkpoint includes the current moments at `extra_state["observation_normalizer"]`; deployment must restore them and apply the same transform before invoking the policy. For example, after loading the policy weights:

```python
from bta_repro.normalization import ObservationNormalizer, NormalizedEnv

normalizer = ObservationNormalizer(env.num_agents, env.obs_dim, env.state_dim)
normalizer.load_state_dict(checkpoint["extra_state"]["observation_normalizer"])
env = NormalizedEnv(env, normalizer.frozen_copy(), training=False)
```

Every update records `critic_diagnostics`: the mean and population standard deviation of the raw GAE return targets and rollout-time value predictions, plus explained variance before optimization. Explained variance is `null` when the targets are constant. Optional `normalize_value_targets`, `value_clip_coef`, and `value_huber_delta` control critic training while preserving raw values for GAE. Defaults retain raw MSE. See [critic target normalization](value_normalization.md) for the cumulative estimator, clipping reference, loss scaling, diagnostics, and checkpoint contract. PopArt and reward normalization are not implemented.

## Modifier ablation

For `algorithm="agentmixer"`, `modifier_enabled=false` freezes the modifier at its identity initialization and excludes its parameters from the optimizer. Local actor architecture, observation normalization, and the joint PPO objective are preserved. Metadata labels this `mode_consistent_no_modifier_control`. This is the direct modifier ablation; the MAPPO baseline uses a different Gaussian variance parameterization and a per-agent PPO objective. Other algorithms reject `modifier_enabled=false`.

## Artifacts and limits

Each output directory contains:

- `config.json`: all training parameters, resolved adapter configurations, environment dimensions, seeds, installed package versions, adapter version information, and SHA-256 hashes of the Python sources.
- `metrics.jsonl`: initial/periodic/final evaluation records and one loss/rollout summary with critic diagnostics per update.
- `episodes.jsonl`: raw completed training episode returns, lengths, ending flags, and environment info.
- `checkpoint.pt`: policy, critic (including value-target moments when enabled), optimizers, configuration, step count, shared checkpoint RNG fields, and observation-normalizer state when enabled. Writes replace a temporary file atomically; the final model is saved before final evaluation.
- `summary.json`: completion status, exact successful training budget, separate evaluation interaction count, episode statistics, and initial/final evaluation. Exceptions produce a failed summary and are re-raised.

`checkpoint_every_updates` controls intermediate checkpoint replacement; zero retains only the final checkpoint. Simulator states and unfinished rollouts are not serialized. The CLI therefore does not claim exact simulator continuation or offer a resume flag. The shared checkpoint currently stores CPU/Python/NumPy RNG state; complete CUDA continuation is also outside that checkpoint contract.

### Evaluate a saved policy

Use the appropriate simulator's Python environment and native-library setup:

```bash
python scripts/evaluate_simulator.py results/my_run --output results/my_run_replay.json
```

This restores the policy and its saved observation moments, then uses the
recorded evaluation seed, episode count, fixed-window settings and execution
rule. It creates no optimizer and does not modify the checkpoint. The JSON
includes checkpoint/source/evaluator hashes, package versions, training status,
checkpoint step and per-episode results. A checkpoint from a failed or unfinished
run remains labeled as such. New evaluation episodes do not supply independent
training seeds.

`--eval-seed 2000000 --episodes 100` requests a separately recorded evaluation;
`--device cpu` changes only the inference device. By default, the imported
package must match the training source hashes. For a plan prepared by this
repository, its frozen source can be selected directly:

```bash
PYTHONPATH=results/experiment/source python scripts/evaluate_simulator.py \
  results/experiment/agentmixer_seed0 --output results/replayed_evaluation.json
```

Use `--allow-source-change` only for an explicitly labeled compatibility check
with revised code. Source and package-version matches are separate report
fields. The evaluator requires saved normalization moments when training used
normalization; it never substitutes fresh statistics. Evaluation outputs must
use new paths, preserving earlier results.

All created environments are closed on success or failure. No jobs, simulator downloads, external logging accounts, or automatic retries are created by this CLI. Collection failures invalidate the run; failed-summary step counts include only successfully returned rollouts, since a backend failure can occur partway through collection.

The fake integration tests execute the actual rollout collector, PPO/BPPO updates, optimizer grouping, evaluation, and checkpoint code for every algorithm in both action spaces, with raw and normalized inputs. They check budgets, raw episode returns, episode endings, evaluation RNG/statistics isolation, restored normalization moments, the frozen modifier control, local AgentMixer evaluation, and cleanup. Passing these tests establishes integration behavior; comparable multi-seed simulator results are still required for a reproduction claim.
