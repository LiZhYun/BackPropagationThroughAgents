from types import SimpleNamespace
import random

import numpy as np
import pytest

from bta_repro.football_env import FootballEnv, SCENARIOS


class Backend:
    def __init__(self, **arguments):
        self.arguments = arguments
        self.count = arguments['number_of_left_players_agent_controls']
        self.action_space = SimpleNamespace(nvec=np.full(self.count, 19))
        self.unwrapped = self
        self._config = dict(arguments['other_config_options'])
        self.score = [0, 0]
        self.step_count = 0
        self.ends_at = 3
        self.closed = False
        random.random()
        np.random.random()

    def _obs(self):
        return np.full((self.count, 115), self.step_count + random.random() + np.random.random())

    def reset(self):
        self.step_count = 0
        self.score = [0, 0]
        self._config['episode_number'] = self._config.get('episode_number', 0) + 1
        return self._obs()

    def step(self, actions):
        self.step_count += 1
        done = self.step_count == self.ends_at
        if done:
            self.score = [2, 1]
        actions[:] = 0
        return self._obs(), np.arange(1, self.count + 1), done, {'score_reward': 0}

    def observation(self):
        return [{'score': list(self.score)} for _ in range(self.count)]

    def close(self):
        self.closed = True


@pytest.mark.parametrize('scenario', list(SCENARIOS))
def test_shapes_shared_rewards_and_complete_match_score(scenario):
    env = FootballEnv({'scenario': scenario}, backend_factory=Backend)
    assert env.num_agents == SCENARIOS[scenario][0]
    observation, metadata = env.reset()
    assert observation['obs'].shape == (env.num_agents, 115)
    np.testing.assert_array_equal(observation['state'], observation['obs'].flatten())
    assert observation['available_actions'].all() and observation['active_agents'].all()
    assert metadata['native_horizon'] == SCENARIOS[scenario][1]
    actions = np.full(env.num_agents, 7)
    for _ in range(3):
        final, reward, terminal, truncated, info = env.step(actions)
    assert reward == sum(range(1, env.num_agents + 1))
    assert terminal and not truncated and info['match_won']
    # The final per-step scoring reward is zero; the match score defines winning.
    assert info['native_score_reward'] == 0 and info['score_difference'] == 1
    np.testing.assert_array_equal(actions, 7)
    assert final['obs'][0, 0] >= 3
    with pytest.raises(RuntimeError):
        env.step(actions)
    env.close()
    env.close()
    assert env._backend.closed


def test_external_cap_and_native_terminal_precedence():
    env = FootballEnv({'episode_limit': 2}, backend_factory=Backend)
    env.reset()
    env.step(np.zeros(3, dtype=int))
    _, _, terminal, truncated, info = env.step(np.zeros(3, dtype=int))
    assert truncated and not terminal and 'match_won' not in info
    env.reset()
    env._backend.ends_at = 2
    env.step(np.zeros(3, dtype=int))
    assert env.step(np.zeros(3, dtype=int))[2:4] == (True, False)
    env.close()


def test_seeding_is_private_repeatable_and_reaches_native_engine():
    random.seed(71)
    np.random.seed(72)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    left = FootballEnv({'seed': 42}, backend_factory=Backend)
    right = FootballEnv({'seed': 42}, backend_factory=Backend)
    seeds = []
    episode_numbers = []
    for _ in range(3):
        a, ai = left.reset()
        b, bi = right.reset()
        np.testing.assert_array_equal(a['obs'], b['obs'])
        assert ai == bi
        seeds.append(ai['engine_seed'])
        episode_numbers.append(left._backend._config['episode_number'])
        assert left._backend._config['game_engine_random_seed'] == seeds[-1]
        assert left._backend._config['reverse_team_processing'] == bool(seeds[-1] % 2)
        aa = left.step(np.zeros(3, dtype=int))
        bb = right.step(np.zeros(3, dtype=int))
        np.testing.assert_array_equal(aa[0]['obs'], bb[0]['obs'])
    assert len(set(seeds)) == 3
    assert episode_numbers == [1, 2, 3]
    a, ai = left.reset(seed=42)
    b, bi = right.reset(seed=42)
    assert ai['engine_seed'] == bi['engine_seed'] == seeds[0]
    assert left._backend._config['episode_number'] == episode_numbers[0]
    np.testing.assert_array_equal(a['obs'], b['obs'])
    left.close(); right.close()
    assert random.getstate() == python_state
    actual = np.random.get_state()
    assert actual[0] == numpy_state[0] and actual[2:] == numpy_state[2:]
    np.testing.assert_array_equal(actual[1], numpy_state[1])


@pytest.mark.parametrize('arguments', [
    {'scenario': 'unknown'}, {'seed': -1}, {'seed': 2**32}, {'seed': True},
    {'episode_limit': 0}, {'episode_limit': 401}, {'episode_limit': True},
    {'rewards': 'checkpoints'}, {'render': True}, {'num_agents': 2},
])
def test_invalid_config_is_rejected_before_backend_creation(arguments):
    def forbidden(**kwargs):
        raise AssertionError('backend constructed for invalid configuration')
    with pytest.raises(ValueError):
        FootballEnv(arguments, backend_factory=forbidden)


@pytest.mark.parametrize('actions', [[0, 0], [0, 0, 19], [-1, 0, 0], [0., 0., 0.], [False] * 3])
def test_invalid_actions_do_not_advance_backend(actions):
    env = FootballEnv(backend_factory=Backend)
    env.reset()
    with pytest.raises(ValueError):
        env.step(actions)
    assert env._backend.step_count == 0
    env.close()


def test_invalid_backend_action_space_is_closed():
    created = []
    def factory(**kwargs):
        backend = Backend(**kwargs)
        backend.action_space.nvec[:] = 20
        created.append(backend)
        return backend
    with pytest.raises(ValueError, match='19 actions'):
        FootballEnv(backend_factory=factory)
    assert created[0].closed


@pytest.mark.parametrize('algorithm', ['mappo', 'arppo', 'bppo', 'agentmixer'])
def test_shared_training_uses_complete_match_metrics(tmp_path, monkeypatch, algorithm):
    from bta_repro import simulator_train as training
    created = []
    def make_env(config, seed):
        env = FootballEnv({**config.env_config, 'seed': seed}, backend_factory=Backend)
        created.append(env)
        return env
    monkeypatch.setattr(training, 'make_env', make_env)
    config = training.Config(environment='football', algorithm=algorithm,
                             steps=12, num_envs=2, rollout_steps=3, hidden_dim=8,
                             epochs=1, eval_episodes=2, normalize_value_targets=True)
    summary = training.train(config, tmp_path / algorithm)
    assert summary['status'] == 'completed'
    assert summary['final_evaluation']['count'] == 2
    assert summary['final_evaluation']['win_rate'] == 1
    assert summary['final_evaluation']['return_mean'] == 18
    assert all(env._closed for env in created)
