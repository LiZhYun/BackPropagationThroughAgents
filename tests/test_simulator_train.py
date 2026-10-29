from dataclasses import replace
from types import SimpleNamespace
import json
import math
import random
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import bta_repro.simulator_train as training


class FakeEnvironment:
    """Two-step episodes exercise both genuine terminals and truncations."""

    num_agents, obs_dim, state_dim = 2, 3, 4
    num_actions, action_dim = 3, 2

    def __init__(self, config, seed):
        self.config = {"seed": seed, **config.env_config}
        self.versions = {"fake_simulator": "1"}
        self.is_discrete = config.environment == "smac"
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.episode = 0
        self.timestep = 0
        self.closed = False
        self.ready = False
        self.steps = 0
        self.fail = bool(config.env_config.get("fail_training", False)) and seed != config.eval_seed
        self.offset = 0.0
        self.episode_return = 0.0
        self.completed_returns = []

    def _observation(self):
        result = {
            "obs": np.full((2, 3), self.offset + self.timestep / 10, dtype=np.float32),
            "state": np.full(4, self.timestep / 10, dtype=np.float32),
            "active_agents": np.ones(2, dtype=bool),
        }
        if self.is_discrete:
            result["available_actions"] = np.ones((2, 3), dtype=bool)
        return result

    def reset(self, seed=None):
        assert not self.closed and seed is None
        self.episode += 1
        self.timestep = 0
        self.offset = self.rng.uniform(-1, 1)
        self.ready = True
        self.episode_return = 0.0
        return self._observation(), {}

    def step(self, actions):
        assert self.ready and not self.closed
        if self.fail:
            raise RuntimeError("injected simulator failure")
        if self.is_discrete:
            assert actions.shape == (2,) and np.issubdtype(actions.dtype, np.integer)
            reward = float(actions.sum())
        else:
            assert actions.shape == (2, 2)
            # Deliberately mutate adapter input to exercise the runner's raw
            # action copy; likelihood evaluation must retain the sampled values.
            np.clip(actions, -0.1, 0.1, out=actions)
            reward = float(actions.sum())
        # Global random calls make evaluation RNG isolation observable.
        reward += np.random.random() + random.random()
        self.episode_return += reward
        self.steps += 1
        self.timestep += 1
        done = self.timestep == 2
        truncated = done and self.episode % 2 == 0
        terminated = done and not truncated
        self.ready = not done
        if done:
            self.completed_returns.append(self.episode_return)
        info = {"battle_won": bool(reward > 1)} if self.is_discrete else {"cost": np.float32(0.1)}
        return self._observation(), reward, terminated, truncated, info

    def close(self):
        self.closed = True


@pytest.fixture
def environments(monkeypatch):
    instances = []

    def make_env(config, seed):
        env = FakeEnvironment(config, seed)
        instances.append(env)
        return env

    monkeypatch.setattr(training, "make_env", make_env)
    return instances


def _config(**overrides):
    arguments = dict(steps=8, num_envs=2, rollout_steps=3, hidden_dim=8, epochs=1,
                     eval_episodes=2, eval_seed=99, checkpoint_every_updates=1)
    arguments.update(overrides)
    return training.Config(**arguments)


@pytest.mark.parametrize("algorithm", ["mappo", "arppo", "bppo", "agentmixer"])
def test_continuous_policy_factory_passes_initial_standard_deviation(algorithm):
    config = _config(environment="mujoco", algorithm=algorithm, initial_std=.3)
    config.validate()
    env = FakeEnvironment(config, 7)
    policy = training.make_policy(config, env)
    obs, state = torch.zeros(4, 2, 3), torch.zeros(4, 4)
    arguments = (obs, state) if algorithm == "agentmixer" else (obs,)
    actions, log_probs = policy.sample(*arguments, deterministic=True)
    expected = -env.action_dim * math.log(config.initial_std * math.sqrt(2 * math.pi))
    torch.testing.assert_close(log_probs, torch.full((4, 2), expected), atol=1e-6, rtol=1e-6)
    assert actions.shape == (4, 2, 2)


def test_discrete_policy_factory_rejects_gaussian_scale_control():
    config = _config(initial_std=.3)
    with pytest.raises(ValueError, match="continuous environment"):
        training.make_policy(config, FakeEnvironment(config, 7))


def test_behavior_peer_noise_reaches_continuous_trainer(tmp_path, environments):
    config = _config(environment='mujoco', algorithm='bppo', continuous_peer_noise='behavior')
    summary = training.train(config, tmp_path / 'coupled')
    assert summary['status'] == 'completed'
    records = [json.loads(line) for line in (tmp_path / 'coupled/metrics.jsonl').read_text().splitlines()]
    updates = [record for record in records if record.get('event') == 'update']
    assert updates and all(record['losses']['peer_noise_behavior'] == 1 for record in updates)


@pytest.mark.parametrize('options', [
    {'continuous_peer_noise': 'invalid'},
    {'continuous_peer_noise': True},
    {'continuous_peer_noise': 'behavior', 'environment': 'smac', 'algorithm': 'bppo'},
    {'continuous_peer_noise': 'behavior', 'environment': 'mujoco', 'algorithm': 'arppo'},
])
def test_rejects_unsupported_continuous_peer_noise(options):
    with pytest.raises(ValueError, match='peer noise|peer_noise'):
        _config(**options).validate()


def test_bounded_modifier_reaches_trainer_and_checkpoint_evaluator(tmp_path, environments):
    config = _config(environment='mujoco', continuous_modifier_scale='bounded', initial_std=.2,
                     normalize_observations=True)
    config.validate()
    policy = training.make_policy(config, FakeEnvironment(config, 7))
    mean, std = policy.joint_parameters(torch.zeros(4, 2, 3), torch.zeros(4, 4))
    assert torch.equal(std, torch.full_like(std, .375))
    output = tmp_path / 'bounded'
    summary = training.train(config, output)
    assert summary['status'] == 'completed'
    metadata = json.loads((output / 'config.json').read_text())
    assert metadata['method_variant'] == 'mode_consistent_bounded_modifier'
    checkpoint = torch.load(output / 'checkpoint.pt', weights_only=True)
    assert checkpoint['config']['continuous_modifier_scale'] == 'bounded'
    replay = _checkpoint_evaluator()(output)
    assert replay['evaluation'] == summary['final_evaluation']


@pytest.mark.parametrize('options', [
    {'continuous_modifier_scale': 'invalid'},
    {'continuous_modifier_scale': True},
    {'continuous_modifier_scale': 'bounded', 'environment': 'smac'},
    {'continuous_modifier_scale': 'bounded', 'environment': 'mujoco', 'algorithm': 'bppo'},
    {'continuous_modifier_scale': 'bounded', 'environment': 'mujoco', 'modifier_enabled': False},
])
def test_rejects_unsupported_continuous_modifier_scale(options):
    with pytest.raises(ValueError, match='modifier.scale|modifier_scale'):
        _config(**options).validate()


@pytest.mark.parametrize('modifier_scale,modifier_enabled', [
    ('residual', True), ('bounded', True), ('residual', False),
])
@pytest.mark.parametrize('local_scale', ['state_independent', 'state_independent_sigmoid'])
def test_state_independent_local_scale_trains_and_replays_checkpoint(
    tmp_path, environments, modifier_scale, modifier_enabled, local_scale
):
    config = _config(environment='mujoco', continuous_local_scale=local_scale,
                     continuous_modifier_scale=modifier_scale, modifier_enabled=modifier_enabled,
                     initial_std=.3, normalize_observations=True, normalize_value_targets=True)
    config.validate()
    output = tmp_path / 'state_independent'
    summary = training.train(config, output)
    assert summary['status'] == 'completed'
    checkpoint = torch.load(output / 'checkpoint.pt', weights_only=True)
    metadata = json.loads((output / 'config.json').read_text())
    assert checkpoint['config']['continuous_local_scale'] == local_scale
    assert metadata['config']['continuous_local_scale'] == local_scale
    assert _checkpoint_evaluator()(output)['evaluation'] == summary['final_evaluation']
    if not modifier_enabled:
        policy = training.make_policy(config, FakeEnvironment(config, 7))
        policy.load_state_dict(checkpoint['policy'])
        obs, state = torch.randn(4, 2, 3), torch.randn(4, 4)
        for local, joint in zip(policy.local_parameters(obs), policy.joint_parameters(obs, state)):
            assert torch.equal(local, joint)


@pytest.mark.parametrize('options', [
    {'continuous_local_scale': 'invalid'},
    {'continuous_local_scale': True},
    {'continuous_local_scale': 'state_independent', 'environment': 'smac'},
    {'continuous_local_scale': 'state_independent', 'environment': 'prey'},
    {'continuous_local_scale': 'state_independent', 'environment': 'football'},
    {'continuous_local_scale': 'state_independent', 'environment': 'mujoco', 'algorithm': 'bppo'},
    {'continuous_local_scale': 'state_independent', 'environment': 'mujoco', 'algorithm': 'arppo'},
    {'continuous_local_scale': 'state_independent', 'environment': 'mujoco', 'algorithm': 'mappo'},
])
def test_rejects_unsupported_continuous_local_scale(options):
    with pytest.raises(ValueError, match='local.scale|local_scale'):
        _config(**options).validate()


def test_policy_factory_rejects_local_scale_with_discrete_environment():
    config = _config(environment='mujoco', continuous_local_scale='state_independent')
    config.validate()
    env = FakeEnvironment(_config(), 7)
    with pytest.raises(ValueError, match='local scale'):
        training.make_policy(config, env)


def _checkpoint_evaluator():
    path = Path(__file__).resolve().parents[1] / 'scripts/evaluate_simulator.py'
    spec = importlib.util.spec_from_file_location('evaluate_simulator_script', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate_saved


@pytest.mark.parametrize('environment', ['smac', 'mujoco'])
@pytest.mark.parametrize('algorithm', ['mappo', 'arppo', 'bppo', 'agentmixer'])
def test_checkpoint_evaluation_replays_final_result(tmp_path, environments, environment, algorithm):
    config = _config(environment=environment, algorithm=algorithm, normalize_observations=True,
                     normalize_value_targets=True)
    run = tmp_path / 'original'
    summary = training.train(config, run)
    checkpoint_bytes = (run / 'checkpoint.pt').read_bytes()
    rng = torch.get_rng_state().clone()
    report = _checkpoint_evaluator()(run)
    assert report['evaluation'] == summary['final_evaluation']
    assert report['source_matches_training'] and report['versions_match_training']
    assert report['checkpoint_step'] == config.steps
    assert report['training_run_status'] == 'completed' and report['checkpoint_is_final']
    assert (run / 'checkpoint.pt').read_bytes() == checkpoint_bytes
    assert torch.equal(rng, torch.get_rng_state())


def test_checkpoint_evaluation_reports_source_override_and_rejects_missing_moments(tmp_path, environments):
    run = tmp_path / 'original'
    training.train(_config(normalize_observations=True), run)
    evaluate_saved = _checkpoint_evaluator()
    path = run / 'config.json'
    metadata = json.loads(path.read_text())
    metadata['source_sha256']['core.py'] = 'different source'
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='Source differs'):
        evaluate_saved(run)
    report = evaluate_saved(run, allow_source_change=True, eval_seed=123, episodes=3)
    assert not report['source_matches_training']
    assert report['evaluation']['seed'] == 123 and report['evaluation']['count'] == 3
    checkpoint = torch.load(run / 'checkpoint.pt', weights_only=True)
    checkpoint.pop('extra_state')
    torch.save(checkpoint, run / 'checkpoint.pt')
    with pytest.raises(ValueError, match='missing observation moments'):
        evaluate_saved(run, allow_source_change=True)


@pytest.mark.parametrize("environment", ["smac", "mujoco"])
@pytest.mark.parametrize("algorithm", ["mappo", "arppo", "bppo", "agentmixer"])
@pytest.mark.parametrize("normalize_observations", [False, True])
@pytest.mark.parametrize("normalize_value_targets", [False, True])
def test_real_runner_updates_all_policy_families_and_preserves_exact_budget(tmp_path, environments, environment, algorithm, normalize_observations, normalize_value_targets):
    config = _config(environment=environment, algorithm=algorithm, normalize_observations=normalize_observations,
                     normalize_value_targets=normalize_value_targets,
                     value_clip_coef=.2 if normalize_value_targets else None,
                     value_huber_delta=10. if normalize_value_targets else None)
    output = tmp_path / f"{environment}_{algorithm}"
    summary = training.train(config, output)
    assert summary["status"] == "completed"
    assert summary["steps"] == summary["train_environment_steps"] == 8
    assert summary["updates"] == 2  # 3 steps/env followed by the remaining 1.
    assert summary["evaluation_environment_steps"] == 8
    assert summary["training_episodes"]["count"] == 4
    assert summary["training_episodes"]["terminated"] == 2
    assert summary["training_episodes"]["truncated"] == 2
    assert summary["unfinished_training_episodes"] == 0
    expected_seeds = training.training_environment_seeds(config.seed, config.num_envs)
    assert [env.seed for env in environments] == [*expected_seeds, 99, 99]
    assert [env.steps for env in environments[:2]] == [4, 4]
    assert all(env.closed for env in environments)
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["event"] for row in metrics] == ["evaluation", "update", "update", "evaluation"]
    assert [row["steps"] for row in metrics] == [0, 6, 8, 8]
    assert all(np.isfinite(value) for row in metrics if row["event"] == "update" for value in row["losses"].values())
    records = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert len(records) == 4
    for i, env in enumerate(environments[:2]):
        assert [record["return"] for record in records if record["env_index"] == i] == env.completed_returns
    metadata = json.loads((output / "config.json").read_text())
    assert metadata["training_environment_seeds"] == expected_seeds
    assert metadata["training_environment_seed_scheme"] == training.ENVIRONMENT_SEED_SCHEME
    assert metadata["evaluation_environment_seed"] == config.eval_seed
    assert metadata["observation_normalization"]["enabled"] is normalize_observations
    assert metadata["value_regression"]["normalize_targets"] is normalize_value_targets
    assert metadata["environment_versions"] == {"fake_simulator": "1"}
    assert "runner.py" in metadata["source_sha256"] and "simulator_train.py" in metadata["source_sha256"]
    checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
    assert checkpoint["step"] == 8 and not (output / "checkpoint.pt.tmp").exists()
    assert all(torch.isfinite(value).all() for value in checkpoint["policy"].values())
    if normalize_value_targets:
        restored_critic = training.make_critic(config, environments[0])
        restored_critic.load_state_dict(checkpoint["critic"])
        assert restored_critic.normalizer.count == 8 and restored_critic.normalizer.updates == 2
        for name, value in checkpoint["critic"].items():
            torch.testing.assert_close(restored_critic.state_dict()[name], value, atol=0, rtol=0)
        assert restored_critic.normalizer.mean.dtype == torch.float64
    if normalize_observations:
        from bta_repro.normalization import ObservationNormalizer
        state = checkpoint["extra_state"]["observation_normalizer"]
        restored = ObservationNormalizer(2, 3, 4)
        restored.load_state_dict(state)
        assert restored.state_dict() == state
        # Two initial resets, eight transitions and four completed-episode resets.
        assert state["obs"]["count"] == state["state"]["count"] == 14
    else:
        assert not checkpoint.get("extra_state")
    # Compare the checkpoint against the actual deterministic initialization.
    torch.manual_seed(config.seed)
    initial = training.make_policy(config, environments[0]).state_dict()
    assert any(not torch.equal(initial[key], value) for key, value in checkpoint["policy"].items())
    expected_execution = "decentralized" if algorithm == "agentmixer" else (
        "autoregressive" if algorithm in ("bppo", "arppo") else "independent")
    assert summary["final_evaluation"]["execution"] == expected_execution


@pytest.mark.parametrize("environment", ["smac", "mujoco"])
@pytest.mark.parametrize("algorithm", ["mappo", "arppo", "bppo", "agentmixer"])
def test_periodic_evaluation_does_not_change_training_weights_rng_or_normalization(tmp_path, environments, environment, algorithm):
    config = _config(algorithm=algorithm, environment=environment, steps=12, normalize_observations=True,
                     normalize_value_targets=True, value_clip_coef=.2, value_huber_delta=10.)
    first = tmp_path / "initial_final_only"
    second = tmp_path / "with_periodic_eval"
    training.train(config, first)
    training.train(replace(config, eval_every_steps=3), second)
    baseline = torch.load(first / "checkpoint.pt", weights_only=True)
    periodic = torch.load(second / "checkpoint.pt", weights_only=True)
    for component in ("policy", "critic"):
        for name, value in baseline[component].items():
            assert torch.equal(value, periodic[component][name]), name
    assert torch.equal(baseline["rng"]["torch"], periodic["rng"]["torch"])
    assert baseline["rng"]["python"] == periodic["rng"]["python"]
    assert baseline["rng"]["numpy"] == periodic["rng"]["numpy"]
    assert baseline["extra_state"]["observation_normalizer"] == periodic["extra_state"]["observation_normalizer"]
    assert baseline["critic"]["normalizer.count"] == periodic["critic"]["normalizer.count"] == 12
    assert baseline["critic"]["normalizer.updates"] == periodic["critic"]["normalizer.updates"] == 2
    baseline_summary = json.loads((first / "summary.json").read_text())
    periodic_summary = json.loads((second / "summary.json").read_text())
    assert periodic_summary["evaluation_environment_steps"] > baseline_summary["evaluation_environment_steps"]


@pytest.mark.parametrize("environment", ["smac", "mujoco"])
def test_no_modifier_control_keeps_identity_modifier_and_updates_local_actors(tmp_path, environments, environment):
    config = _config(environment=environment, modifier_enabled=False)
    output = tmp_path / "no_modifier"
    training.train(config, output)
    checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
    torch.manual_seed(config.seed)
    policy = training.make_policy(config, environments[0])
    assert all(not parameter.requires_grad for parameter in policy.modifier.parameters())
    initial = policy.state_dict()
    assert all(torch.equal(value, checkpoint["policy"][name])
               for name, value in initial.items() if name.startswith("modifier."))
    assert any(not torch.equal(value, checkpoint["policy"][name])
               for name, value in initial.items() if name.startswith("local_actors."))
    assert len(checkpoint["optimizers"]["policy"]["param_groups"]) == 1
    metadata = json.loads((output / "config.json").read_text())
    assert metadata["method_variant"] == "mode_consistent_no_modifier_control"


def test_critic_diagnostics_use_rollout_predictions_and_leave_targets_unchanged():
    returns = torch.tensor([1., 2., 3., 4.])
    predictions = returns.reshape(2, 2).clone()
    batch = SimpleNamespace(returns=returns, values=predictions)
    diagnostics = training.critic_diagnostics(batch)
    assert diagnostics["explained_variance"] == 1.0
    assert diagnostics["return_mean"] == 2.5
    assert diagnostics["return_std"] == pytest.approx(np.sqrt(1.25))
    assert torch.equal(returns, torch.tensor([1., 2., 3., 4.]))
    batch.values = torch.zeros(2, 2)
    assert training.critic_diagnostics(batch)["explained_variance"] == 0.0
    batch.returns = torch.ones(4)
    assert training.critic_diagnostics(batch)["explained_variance"] is None


def test_default_critic_factory_preserves_raw_network_initialization_and_rng():
    config = _config()
    env = FakeEnvironment(config, 0)
    assert config.normalize_value_targets is False
    assert config.value_clip_coef is config.value_huber_delta is None
    torch.manual_seed(87)
    expected = training.mlp(env.state_dim, 1, config.hidden_dim)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(87)
    actual = training.make_critic(config, env)
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value, atol=0, rtol=0)


def test_normalized_evaluation_requires_training_statistics_before_creating_env(environments):
    config = _config(normalize_observations=True)
    policy = training.make_policy(config, FakeEnvironment(config, 0))
    with pytest.raises(ValueError, match="training normalizer"):
        training.evaluate(policy, config)
    assert not environments


@pytest.mark.parametrize("environment", ["smac", "mujoco"])
def test_agentmixer_evaluation_never_calls_centralized_modifier(environments, environment):
    config = _config(environment=environment)
    env = FakeEnvironment(config, 0)
    policy = training.make_policy(config, env)

    class ForbiddenModifier(nn.Module):
        def forward(self, *args):
            raise AssertionError("evaluation used centralized information")

    policy.modifier = ForbiddenModifier()
    policy.train()
    evaluation = training.evaluate(policy, config, training.environment_spec(env))
    assert evaluation["count"] == 2 and policy.training
    assert environments[-1].closed


def test_existing_output_directory_is_rejected_before_creating_environments(tmp_path, environments):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        training.train(_config(), output)
    assert not environments and (output / "keep.txt").read_text() == "preserve"


def test_simulator_failure_closes_all_environments_and_records_failure(tmp_path, environments):
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected simulator failure"):
        training.train(_config(env_config={"fail_training": True}), output)
    assert all(env.closed for env in environments)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["error_type"] == "RuntimeError"
    assert not (output / "checkpoint.pt").exists()


@pytest.mark.parametrize("arguments", [
    {"steps": 7}, {"eval_seed": training.training_environment_seeds(0, 2)[0]}, {"env_config": {"seed": 5}},
    {"gamma": float("nan")}, {"learning_rate": float("inf")}, {"num_envs": 0},
    {"observation_clip": 0}, {"normalize_observations": 1},
    {"modifier_enabled": 0}, {"algorithm": "mappo", "modifier_enabled": False},
    {"normalize_value_targets": 1}, {"value_clip_coef": 0}, {"value_clip_coef": float("nan")},
    {"value_huber_delta": -1}, {"value_huber_delta": float("inf")}, {"value_huber_delta": True},
    {"initial_std": .3}, {"environment": "mujoco", "initial_std": 0},
    {"environment": "mujoco", "initial_std": float("nan")},
    {"environment": "mujoco", "initial_std": True},
    {"environment": "mujoco", "initial_std": math.exp(2) + .01},
    {"environment": "mujoco", "initial_std": math.exp(-5) / 2},
])
def test_invalid_configuration_is_rejected_before_output_creation(tmp_path, environments, arguments):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        training.train(_config(**arguments), output)
    assert not output.exists() and not environments


def test_adjacent_run_seeds_produce_disjoint_repeatable_environment_streams():
    first = training.training_environment_seeds(0, 40)
    second = training.training_environment_seeds(1, 40)
    assert first == training.training_environment_seeds(0, 40)
    assert second == training.training_environment_seeds(1, 40)
    assert len(set(first)) == len(set(second)) == 40
    assert set(first).isdisjoint(second)
    assert first[:2] == training.training_environment_seeds(0, 2)
    assert all(0 <= seed < 2**32 for seed in first + second)


def test_seed_validation_checks_derived_streams_and_accepts_full_uint32_run_seed():
    # Evaluation may use the run/model seed when it differs from simulator seeds.
    _config(eval_seed=0).validate()
    _config(seed=2**32 - 1, num_envs=40, steps=80).validate()


def test_training_seed_collision_is_rejected_before_output_creation(tmp_path, environments, monkeypatch):
    monkeypatch.setattr(training, "training_environment_seeds", lambda seed, count: [17] * count)
    output = tmp_path / "colliding_seeds"
    with pytest.raises(ValueError, match="seeds collide"):
        training.train(_config(), output)
    assert not output.exists() and not environments


def test_cli_reads_config_and_runs_shared_runner(tmp_path, environments, monkeypatch, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(training.asdict(_config(algorithm="mappo"))))
    output = tmp_path / "cli"
    monkeypatch.setattr("sys.argv", ["bta-train-simulator", "--config", str(config_path), "--output", str(output)])
    training.main()
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
