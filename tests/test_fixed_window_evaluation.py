from dataclasses import replace
import random

import numpy as np
import pytest
import torch

import bta_repro.simulator_train as training
from bta_repro.normalization import ObservationNormalizer


class WindowEnv:
    is_discrete = True
    num_agents, num_actions, obs_dim, state_dim = 2, 2, 1, 2

    def __init__(self, seed):
        self.seed, self.steps, self.resets, self.closed = seed, 0, 0, False

    def observation(self):
        return {'obs': np.ones((2, 1), dtype=np.float32), 'state': np.ones(2, dtype=np.float32),
                'available_actions': np.ones((2, 2), dtype=bool)}

    def reset(self):
        self.steps = 0
        self.resets += 1
        return self.observation(), {}

    def step(self, action):
        assert action.tolist() == [1, 1]
        # Exercise preservation of global RNG even for badly isolated backends.
        random.random(), np.random.random(), torch.rand(1)
        self.steps += 1
        done = self.steps == 2
        return self.observation(), 3. if done else 1., done, False, {'capture_success': done}

    def close(self):
        self.closed = True


def setup(monkeypatch, *, normalized=False):
    instances = []
    def make_env(config, seed):
        env = WindowEnv(seed)
        instances.append(env)
        return env
    monkeypatch.setattr(training, 'make_env', make_env)
    config = training.Config(environment='prey', algorithm='agentmixer', eval_episodes=1,
                             eval_windows=2, eval_window_steps=3, normalize_observations=normalized)
    policy = training.make_policy(config, WindowEnv(0))
    for actor in policy.local_actors:
        with torch.no_grad():
            actor[-1].weight.zero_()
            actor[-1].bias[:] = torch.tensor([0., 1.])
    # Central sampling is forbidden at execution.
    monkeypatch.setattr(policy, 'sample', lambda *args, **kwargs: pytest.fail('central execution'))
    return config, policy, instances


def test_fixed_window_includes_partial_episode_and_uses_all_step_denominator(monkeypatch):
    config, policy, instances = setup(monkeypatch)
    result = training.evaluate(policy, config)
    assert result['return_mean'] == 4.
    assert result['reward_per_environment_step'] == 2.
    window = result['fixed_window']
    assert window['reward_sum'] == 10.
    assert window['reward_per_environment_step'] == pytest.approx(5/3)
    assert window['environment_steps'] == 6
    assert result['environment_steps'] == 8
    for record in window['windows']:
        assert record['completed_episodes'] == record['captures'] == 1
        assert record['unfinished_episode_steps'] == 1
        assert record['unfinished_episode_return'] == 1.
    assert [env.resets for env in instances] == [1, 2, 2]
    assert all(env.closed for env in instances)


def test_window_replay_preserves_policy_mode_rng_and_training_normalizer(monkeypatch):
    config, policy, instances = setup(monkeypatch, normalized=True)
    normalizer = ObservationNormalizer(2, 1, 2)
    normalizer.update(WindowEnv(0).observation())
    before_stats = normalizer.state_dict()
    torch_before = torch.get_rng_state()
    np_before, python_before = np.random.get_state(), random.getstate()
    first = training.evaluate_fixed_windows(policy, config, normalizer=normalizer)
    second = training.evaluate_fixed_windows(policy, config, normalizer=normalizer)
    assert first == second
    assert policy.training
    assert normalizer.state_dict() == before_stats
    assert torch.equal(torch_before, torch.get_rng_state())
    assert python_before == random.getstate()
    np.testing.assert_equal(np_before, np.random.get_state())
    seeds = [r['seed'] for r in first['windows']]
    assert len(set(seeds)) == 2
    assert config.eval_seed not in seeds
    assert not set(seeds).intersection(training.training_environment_seeds(config.seed, config.num_envs))
    assert all(env.closed for env in instances)


def test_window_boundary_does_not_reset_a_finished_last_episode(monkeypatch):
    config, policy, instances = setup(monkeypatch)
    result = training.evaluate_fixed_windows(policy, replace(config, eval_window_steps=2))
    assert result['reward_per_environment_step'] == 2.
    assert all(record['unfinished_episode_steps'] == 0 for record in result['windows'])
    assert all(env.resets == 1 for env in instances)


@pytest.mark.parametrize('arguments', [
    {'eval_windows': 1}, {'eval_window_steps': 200},
    {'eval_windows': True}, {'eval_windows': -1},
    {'environment': 'smac', 'eval_windows': 1, 'eval_window_steps': 200},
])
def test_fixed_window_configuration_validation(arguments):
    with pytest.raises(ValueError):
        training.Config(**{'environment': 'prey', **arguments}).validate()


def test_real_prey_training_counts_both_evaluation_protocols(tmp_path):
    config = training.Config(environment='prey', algorithm='agentmixer', agentmixer_correlation='joint',
                             steps=8, num_envs=2, rollout_steps=4, epochs=1, hidden_dim=8,
                             env_config={'episode_limit': 4}, eval_episodes=1,
                             eval_windows=2, eval_window_steps=5)
    result = training.train(config, tmp_path / 'run')
    assert result['status'] == 'completed'
    assert result['evaluation_environment_steps'] == sum(
        result[key]['environment_steps'] for key in ('initial_evaluation', 'final_evaluation'))
    assert result['final_evaluation']['fixed_window']['environment_steps'] == 10
