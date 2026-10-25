import random

import numpy as np
import pytest

from bta_repro.prey_env import LEGACY_COMMIT, PreyEnv


class Moves:
    def __init__(self, *moves):
        self.moves = iter(moves)
        self.calls = 0

    def choice(self, n, size, p):
        assert n == 5 and size == 1
        np.testing.assert_allclose(p, [.175, .175, .175, .175, .3])
        self.calls += 1
        return np.array([next(self.moves)])


def placed(predators, prey, **config):
    env = PreyEnv(config)
    env.reset()
    env._predators[:] = predators
    env._prey[:] = prey
    return env


def assert_same(left, right):
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def test_observation_is_exact_legacy_coordinates_visibility_and_time():
    env = placed([[0, 0], [4, 4]], [1, 2])
    obs = env._observation()
    assert (env.num_agents, env.num_actions, env.obs_dim, env.state_dim) == (2, 5, 28, 56)
    assert env.is_discrete and env.episode_limit == 128
    np.testing.assert_allclose(obs['obs'][:, :2], [[0, 0], [.8, 1]])
    assert obs['obs'][0, 2 + 3 * 5 + 4] == 1
    assert obs['obs'][0, 2:27].sum() == 1
    assert obs['obs'][1, 2:27].sum() == 0
    assert obs['obs'][:, -1].sum() == 0
    np.testing.assert_array_equal(obs['state'], obs['obs'].reshape(-1))
    assert obs['available_actions'].all() and obs['active_agents'].all()
    assert env.versions['legacy_commit'] == LEGACY_COMMIT


@pytest.mark.parametrize('seed', range(20))
def test_reset_places_distinct_entities_and_no_adjacent_prey(seed):
    env = PreyEnv({'seed': seed})
    env.reset()
    assert len(set(map(tuple, [*env._predators, env._prey]))) == 3
    assert ((env._predators >= 0) & (env._predators < 5)).all()
    assert ((env._prey >= 0) & (env._prey < 5)).all()
    assert env._neighbour_count(env._prey) == 0


@pytest.mark.parametrize('action, expected', [(0, [2, 1]), (1, [1, 0]), (2, [0, 1]), (3, [1, 2]), (4, [1, 1])])
def test_action_numbering(action, expected):
    env = placed([[1, 1], [4, 4]], [0, 4])
    env._numpy_rng = Moves(4)
    actions = np.array([action, 4])
    before = actions.copy()
    _, reward, term, trunc, _ = env.step(actions)
    np.testing.assert_array_equal(env._predators[0], expected)
    np.testing.assert_array_equal(actions, before)
    assert reward == -.01 and not term and not trunc


@pytest.mark.parametrize('predators, actions, expected', [
    ([[0, 0], [4, 4]], [2, 0], [[0, 0], [4, 4]]),
    ([[1, 1], [1, 2]], [3, 1], [[1, 1], [1, 2]]),
    ([[1, 1], [1, 2]], [1, 1], [[1, 0], [1, 1]]),
    ([[1, 0], [1, 2]], [3, 1], [[1, 1], [1, 2]]),
])
def test_boundaries_collisions_and_index_order(predators, actions, expected):
    env = placed(predators, [4, 0])
    env._numpy_rng = Moves(4)
    env.step(np.array(actions))
    np.testing.assert_array_equal(env._predators, expected)


@pytest.mark.parametrize('penalty', [0, -1])
def test_single_neighbour_penalty_is_shared_and_repeats(penalty):
    env = placed([[2, 1], [0, 0]], [2, 2], penalty=penalty)
    env._numpy_rng = Moves(*([4] * 10))
    for _ in range(2):
        _, reward, term, trunc, info = env.step(np.array([4, 4]))
        assert reward == pytest.approx(-.01 + penalty)
        assert not term and not trunc and info['prey_alive'] == [True]
    assert env._numpy_rng.calls == 10


def test_two_cardinal_neighbours_capture_before_prey_movement():
    env = placed([[2, 1], [2, 3]], [2, 2], penalty=-1)
    env._numpy_rng = Moves()  # Capture must consume no prey movement draw.
    obs, reward, term, trunc, info = env.step(np.array([4, 4]))
    assert reward == 9.99 and term and not trunc
    assert info['capture_success'] and info['prey_alive'] == [False]
    assert obs['obs'][:, 2:27].sum() == 0
    assert obs['available_actions'].all()
    with pytest.raises(RuntimeError, match='reset'):
        env.step(np.array([4, 4]))


def test_diagonal_neighbours_do_not_count_as_capture():
    env = placed([[1, 1], [3, 3]], [2, 2])
    env._numpy_rng = Moves(4)
    _, reward, term, _, _ = env.step(np.array([4, 4]))
    assert reward == -.01 and not term


@pytest.mark.parametrize('moves, expected, draws', [
    ([4, 4, 4, 4, 4], [2, 2], 5),  # Five dangerous proposals: stay.
    ([4, 3], [2, 3], 2),            # Reject danger, accept escape.
    ([1, 3], [2, 2], 1),            # Occupied destination consumes acceptance.
])
def test_prey_rejection_and_occupied_cell_order(moves, expected, draws):
    env = placed([[2, 1], [0, 0]], [2, 2], penalty=-1)
    env._numpy_rng = Moves(*moves)
    _, reward, _, _, _ = env.step(np.array([4, 4]))
    np.testing.assert_array_equal(env._prey, expected)
    assert env._numpy_rng.calls == draws
    assert reward == -1.01  # Penalty precedes any escape.


def test_out_of_grid_proposal_consumes_one_accepted_draw():
    env = placed([[4, 4], [4, 3]], [0, 0])
    env._numpy_rng = Moves(2, 3)
    env.step(np.array([4, 4]))
    np.testing.assert_array_equal(env._prey, [0, 0])
    assert env._numpy_rng.calls == 1


@pytest.mark.parametrize('capture', [False, True])
def test_timeout_and_capture_precedence(capture):
    positions = [[2, 1], [2, 3]] if capture else [[0, 0], [4, 4]]
    env = placed(positions, [2, 2], episode_limit=1)
    env._numpy_rng = Moves(4)
    obs, _, term, trunc, info = env.step(np.array([4, 4]))
    assert term == capture and trunc != capture
    assert info['episode_limit'] == trunc
    np.testing.assert_array_equal(obs['obs'][:, -1], [1, 1])
    with pytest.raises(RuntimeError, match='reset'):
        env.step(np.array([4, 4]))


def test_arrays_are_copied_and_state_does_not_alias_local_observations():
    env = PreyEnv()
    obs, _ = env.reset()
    expected = env._observation()
    for values in obs.values():
        values[:] = 0
    assert_same(env._observation(), expected)
    fresh = env._observation()
    state = fresh['state'].copy()
    fresh['obs'][:] = 0
    np.testing.assert_array_equal(fresh['state'], state)


def test_seed_replay_interleaving_and_caller_rng_isolation():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    first = PreyEnv({'seed': 91})
    second = PreyEnv({'seed': 91})
    unrelated = PreyEnv({'seed': 13})
    initial, _ = first.reset()
    other_initial, _ = second.reset()
    assert_same(initial, other_initial)
    unrelated.reset()
    actions = np.random.RandomState(123).randint(5, size=(40, 2))
    for action in actions:
        a = first.step(action)
        unrelated.reset()
        b = second.step(action)
        assert_same(a[0], b[0])
        assert a[1:] == b[1:]
        if a[2] or a[3]:
            assert_same(first.reset()[0], second.reset()[0])
    assert_same(first.reset(seed=91)[0], initial)
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert numpy_state[0] == after[0] and numpy_state[2:] == after[2:]
    np.testing.assert_array_equal(numpy_state[1], after[1])
    replay = PreyEnv({'seed': 91})
    replay.reset()
    assert_same(first.reset()[0], replay.reset()[0])


@pytest.mark.parametrize('config', [
    {'seed': -1}, {'seed': 2**32}, {'seed': True}, {'seed': 1.5},
    {'penalty': -1.5}, {'penalty': True}, {'penalty': float('nan')}, {'penalty': 0j},
    {'episode_limit': 0}, {'episode_limit': 129}, {'episode_limit': False},
    {'grid_shape': [7, 7]}, {'num_agents': 4}, {'full_observable': True},
])
def test_unsupported_or_invalid_config_rejected(config):
    with pytest.raises(ValueError):
        PreyEnv(config)


@pytest.mark.parametrize('actions', [[0], [0, 1, 2], [0., 1.], [-1, 0], [0, 5], [True, False]])
def test_invalid_actions_leave_transition_state_unchanged(actions):
    env = PreyEnv()
    original, _ = env.reset()
    with pytest.raises(ValueError):
        env.step(np.array(actions))
    assert env._steps == 0
    assert_same(env._observation(), original)


def test_lifecycle_and_reseed_validation():
    env = PreyEnv()
    with pytest.raises(RuntimeError, match='reset'):
        env.step(np.array([4, 4]))
    env.reset()
    with pytest.raises(ValueError):
        env.reset(seed=True)
    env.close()
    env.close()
    with pytest.raises(RuntimeError, match='closed'):
        env.reset()
    with pytest.raises(RuntimeError, match='closed'):
        env.step(np.array([4, 4]))


def test_runner_receives_timeout_observation_before_reset():
    import torch
    from bta_repro.runner import collect_rollout

    class StayPolicy(torch.nn.Module):
        def sample(self, obs, available_actions):
            shape = obs.shape[:2]
            return torch.full(shape, 4, dtype=torch.long), torch.zeros(shape)

    class ClockCritic(torch.nn.Module):
        def forward(self, state):
            return state[:, -1]

    envs = [PreyEnv({'seed': seed, 'episode_limit': 1}) for seed in (0, 1)]
    batch, carry, records = collect_rollout(envs, StayPolicy(), ClockCritic(), 'mappo', 3, gamma=.9)
    assert batch.rollout_shape == (3, 2)
    assert len(records) == 6 and batch.truncated.all() and not batch.terminated.any()
    torch.testing.assert_close(batch.rewards, torch.full((3, 2), -.01))
    torch.testing.assert_close(batch.next_values, torch.ones(3, 2))
    torch.testing.assert_close(batch.returns, torch.full((6,), .89))
    assert all(record['length'] == 1 and record['return'] == -.01 for record in records)
    assert all(entry['obs'][:, -1].sum() == 0 for entry in carry)
