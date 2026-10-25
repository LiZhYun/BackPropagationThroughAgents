"""Optional Google Research Football 2.10.2 adapter for the four BPTA tasks."""

from contextlib import contextmanager
from copy import deepcopy
from importlib import metadata
import random
import threading

import numpy as np


SCENARIOS = {
    'academy_3_vs_1_with_keeper': (3, 400),
    'academy_counterattack_hard': (4, 400),
    'academy_corner': (10, 400),
    '5_vs_5': (4, 3000),
}
_RNG_LOCK = threading.RLock()


def _seed(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    return int(value)


class FootballEnv:
    """Simple115v2 observations and summed player rewards, without rendering.

    Native match endings terminate. An optional shorter external cap truncates.
    The shared critic receives all local observations concatenated; each actor
    receives only its own115 features. Engine seeding repairs the legacy wrapper.
    """

    is_discrete = True
    obs_dim = 115
    num_actions = 19

    def __init__(self, config=None, *, backend_factory=None):
        supplied = deepcopy(config) if config is not None else {}
        if not isinstance(supplied, dict) or supplied.keys() - {'scenario', 'seed', 'rewards', 'episode_limit'}:
            raise ValueError('Football config accepts scenario, seed, rewards, and episode_limit')
        scenario = supplied.get('scenario', 'academy_3_vs_1_with_keeper')
        if scenario not in SCENARIOS:
            raise ValueError('unsupported BPTA Football scenario')
        self.num_agents, self.native_horizon = SCENARIOS[scenario]
        self.state_dim = self.num_agents * self.obs_dim
        self.seed = _seed(supplied.get('seed', 0))
        rewards = supplied.get('rewards', 'scoring,checkpoints')
        if rewards not in ('scoring', 'scoring,checkpoints'):
            raise ValueError('rewards must be scoring or scoring,checkpoints')
        self.episode_limit = supplied.get('episode_limit')
        if self.episode_limit is not None and (
            isinstance(self.episode_limit, bool) or not isinstance(self.episode_limit, int)
            or not 1 <= self.episode_limit <= self.native_horizon
        ):
            raise ValueError('episode_limit must be None or an integer within the native horizon')
        self.config = {'scenario': scenario, 'seed': self.seed, 'rewards': rewards,
                       'episode_limit': self.episode_limit, 'num_agents': self.num_agents,
                       'representation': 'simple115v2', 'shared_reward': 'sum_of_player_rewards',
                       'critic_state': 'concatenated_local_observations'}
        self._seed_streams(self.seed)
        self._needs_reset, self._closed, self._steps = True, False, 0
        self._backend = None
        if backend_factory is None:
            from gfootball.env import create_environment
            backend_factory = create_environment
            version = metadata.version('gfootball')
            if version != '2.10.2':
                raise ValueError('the canonical Football profile requires gfootball2.10.2')
        else:
            version = 'injected_backend'
        self.versions = {'gfootball': version, 'upstream_commit': 'ba9952130898bde2879b01ff2b49d3d48d26ce81',
                         'seeding': 'per_reset_native_seed_private_rng_episode_counter_v2'}
        try:
            with self._random_context():
                self._backend = backend_factory(
                    env_name=scenario, stacked=False, representation='simple115v2', rewards=rewards,
                    number_of_left_players_agent_controls=self.num_agents,
                    number_of_right_players_agent_controls=0, render=False, write_video=False,
                    write_goal_dumps=False, write_full_episode_dumps=False,
                    other_config_options={'game_engine_random_seed': 0, 'reverse_team_processing': False})
            sizes = np.asarray(self._backend.action_space.nvec)
            if sizes.shape != (self.num_agents,) or not (sizes == self.num_actions).all():
                raise ValueError('Football backend must expose19 actions per controlled player')
            # Some scenarios alternate kickoff using this counter. Preserve the
            # constructor's starting phase, then restart it on explicit reseed.
            native_config = self._backend.unwrapped._config
            self._initial_episode_number = native_config['episode_number'] if 'episode_number' in native_config else 0
            # Discard constructor-only random draws before the first real reset.
            self._seed_streams(self.seed)
        except Exception:
            if self._backend is not None:
                self._backend.close()
            raise

    def _seed_streams(self, seed):
        self._python_state = random.Random(seed).getstate()
        self._numpy_state = np.random.RandomState(seed).get_state()
        self._engine_seeds = np.random.default_rng(np.random.SeedSequence([0x475246, 1, seed]))
        self._episode_index = 0

    @contextmanager
    def _random_context(self):
        # Calls are synchronous; this lock cannot protect unrelated RNG users.
        with _RNG_LOCK:
            python_state, numpy_state = random.getstate(), np.random.get_state()
            random.setstate(self._python_state)
            np.random.set_state(self._numpy_state)
            try:
                yield
            finally:
                self._python_state, self._numpy_state = random.getstate(), np.random.get_state()
                random.setstate(python_state)
                np.random.set_state(numpy_state)

    def _observation(self, raw):
        obs = np.asarray(raw, dtype=np.float32)
        if obs.shape != (self.num_agents, 115) or not np.isfinite(obs).all():
            raise ValueError('Football simple115v2 observations have invalid shape or values')
        return {'obs': obs.copy(), 'state': obs.flatten().copy(),
                'available_actions': np.ones((self.num_agents, 19), dtype=bool),
                'active_agents': np.ones(self.num_agents, dtype=bool)}

    def reset(self, seed=None):
        if self._closed:
            raise RuntimeError('Football environment is closed')
        if seed is not None:
            self.seed = _seed(seed)
            self.config['seed'] = self.seed
            self._seed_streams(self.seed)
        self._needs_reset = True
        engine_seed = int(self._engine_seeds.integers(0, 2_000_000_001))
        with self._random_context():
            native_config = self._backend.unwrapped._config
            native_config['game_engine_random_seed'] = engine_seed
            native_config['reverse_team_processing'] = bool(engine_seed % 2)
            native_config['episode_number'] = self._initial_episode_number + self._episode_index
            raw = self._backend.reset()
        self._episode_index += 1
        observation = self._observation(raw)
        self._steps, self._needs_reset = 0, False
        return observation, {'seed': self.seed, 'engine_seed': engine_seed, 'native_horizon': self.native_horizon,
                             'episode_index': self._episode_index - 1}

    def step(self, actions):
        if self._closed or self._needs_reset:
            raise RuntimeError('reset an open Football environment before stepping')
        actions = np.asarray(actions)
        if (actions.shape != (self.num_agents,) or not np.issubdtype(actions.dtype, np.integer)
                or ((actions < 0) | (actions >= self.num_actions)).any()):
            raise ValueError('Football actions must be one legal integer per agent')
        with self._random_context():
            raw, rewards, done, native_info = self._backend.step(actions.copy())
            state = self._backend.unwrapped.observation()
        rewards = np.asarray(rewards, dtype=np.float64)
        if rewards.shape != (self.num_agents,) or not np.isfinite(rewards).all():
            raise ValueError('Football must return finite rewards for every controlled player')
        if not isinstance(done, (bool, np.bool_)):
            raise ValueError('Football backend done flag must be boolean')
        self._steps += 1
        terminated = bool(done)
        truncated = bool(self.episode_limit is not None and self._steps >= self.episode_limit and not terminated)
        self._needs_reset = terminated or truncated
        score = np.asarray(state[0]['score'])
        if score.shape != (2,) or not np.isfinite(score).all():
            raise ValueError('Football score must contain both teams')
        info = {'score': score.tolist(), 'score_difference': float(score[0] - score[1]),
                'player_rewards': rewards.tolist(), 'episode_limit': truncated, 'elapsed_steps': self._steps,
                'native_score_reward': native_info.get('score_reward')}
        if terminated:
            info['match_won'] = bool(score[0] > score[1])
        return self._observation(raw), float(rewards.sum()), terminated, truncated, info

    def close(self):
        if not self._closed:
            with self._random_context():
                self._backend.close()
            self._closed = True
