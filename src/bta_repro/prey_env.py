"""The released AgentMixer Predator–Prey task, without optional simulators.

Mechanics are independently implemented from the MIT-licensed author release
at LEGACY_COMMIT. See docs/predator_prey_protocol.md for source locators,
compatibility details, seeding repairs, and unresolved published metric units.
"""

from copy import deepcopy
import random

import numpy as np


LEGACY_COMMIT = "13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0"
PROTOCOL_VERSION = "legacy_predator_prey_5x5_2predators_1prey_v1"
# The action numbers matter: DOWN, LEFT, UP, RIGHT, NOOP.
_MOVES = np.array(((1, 0), (0, -1), (-1, 0), (0, 1), (0, 0)))
_PREY_PROBABILITIES = (0.175, 0.175, 0.175, 0.175, 0.3)


def _integer(name, value, minimum, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return int(value)


class PreyEnv:
    """Two predators and one prey on a bounded 5×5 grid.

    Reset advances private Python/NumPy streams, or restarts both when given a
    seed. The only benchmark variants are penalties 0 and -1. Shorter episode
    limits are explicit diagnostics; the released task uses 128 steps.
    """

    is_discrete = True
    num_agents = 2
    num_actions = 5
    obs_dim = 28
    state_dim = 56

    def __init__(self, config: dict | None = None):
        supplied = deepcopy(config) if config is not None else {}
        if not isinstance(supplied, dict) or supplied.keys() - {"seed", "penalty", "episode_limit"}:
            raise ValueError("PreyEnv config accepts only seed, penalty, and episode_limit")
        self.seed = _integer("seed", supplied.get("seed", 0), 0, 2**32 - 1)
        penalty = supplied.get("penalty", 0)
        if isinstance(penalty, (bool, np.bool_)) or not isinstance(penalty, (int, float, np.integer, np.floating)) or penalty not in (0, -1):
            raise ValueError("paper-target penalty must be 0 or -1")
        self.penalty = float(penalty)
        self.episode_limit = _integer("episode_limit", supplied.get("episode_limit", 128), 1, 128)
        self.config = {"seed": self.seed, "penalty": self.penalty, "episode_limit": self.episode_limit}
        self.versions = {"protocol": PROTOCOL_VERSION, "legacy_commit": LEGACY_COMMIT,
                         "numpy": np.__version__, "seeding": "private_random_and_numpy_randomstate_v1"}
        self._python_rng = random.Random(self.seed)
        self._numpy_rng = np.random.RandomState(self.seed)
        self._predators = np.empty((2, 2), dtype=np.int64)
        self._prey = np.empty(2, dtype=np.int64)
        self._prey_alive = True
        self._steps = 0
        self._needs_reset = True
        self._closed = False

    def _assert_open(self):
        if self._closed:
            raise RuntimeError("Predator–Prey environment is closed")

    @staticmethod
    def _in_grid(position):
        return bool(((0 <= position) & (position < 5)).all())

    def _neighbour_count(self, position):
        # Also defined for out-of-grid proposals, as in the legacy rejection
        # test. Such proposals may be accepted but cannot subsequently move.
        return int((np.abs(self._predators - position).sum(axis=1) == 1).sum())

    def _vacant(self, position):
        return (self._in_grid(position)
                and not (self._predators == position).all(axis=1).any()
                and not (self._prey_alive and np.array_equal(self._prey, position)))

    def _observation(self):
        local = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
        for agent, position in enumerate(self._predators):
            # Preserve the released coordinate asymmetry: row/5, column/4.
            local[agent, :2] = position / np.array((5, 4))
            if self._prey_alive:
                offset = self._prey - position + 2
                if ((0 <= offset) & (offset < 5)).all():
                    local[agent, 2 + int(offset[0]) * 5 + int(offset[1])] = 1
            local[agent, -1] = self._steps / self.episode_limit
        return {"obs": local, "state": local.reshape(-1).copy(),
                "available_actions": np.ones((2, 5), dtype=bool),
                "active_agents": np.ones(2, dtype=bool)}

    def reset(self, seed: int | None = None):
        self._assert_open()
        if seed is not None:
            self.seed = _integer("seed", seed, 0, 2**32 - 1)
            self.config["seed"] = self.seed
            self._python_rng = random.Random(self.seed)
            self._numpy_rng = np.random.RandomState(self.seed)
        self._needs_reset = True
        self._predators.fill(-10)
        self._prey_alive = False
        self._prey.fill(-10)
        for agent in range(self.num_agents):
            while True:
                candidate = np.array([self._python_rng.randint(0, 4), self._python_rng.randint(0, 4)])
                if self._vacant(candidate):
                    self._predators[agent] = candidate
                    break
        while True:
            candidate = np.array([self._python_rng.randint(0, 4), self._python_rng.randint(0, 4)])
            if self._vacant(candidate) and self._neighbour_count(candidate) == 0:
                self._prey = candidate
                break
        self._prey_alive = True
        self._steps = 0
        self._needs_reset = False
        return self._observation(), {"seed": self.seed, "episode_limit": self.episode_limit,
                                    "protocol": PROTOCOL_VERSION}

    def step(self, actions):
        self._assert_open()
        if self._needs_reset:
            raise RuntimeError("call reset before stepping or after the episode ends")
        actions = np.asarray(actions)
        if (actions.shape != (2,) or not np.issubdtype(actions.dtype, np.integer)
                or ((actions < 0) | (actions >= 5)).any()):
            raise ValueError("actions must be integer indices in [0, 5) with shape [2]")
        self._steps += 1
        # This is intentionally sequential by index. A later agent can enter
        # an earlier agent's vacated cell; swaps and occupied-cell moves fail.
        for agent, action in enumerate(actions):
            candidate = self._predators[agent] + _MOVES[action]
            if self._vacant(candidate):
                self._predators[agent] = candidate
        neighbours = self._neighbour_count(self._prey)
        reward = -0.01
        if neighbours == 1:
            reward += self.penalty
        elif neighbours >= 2:
            reward += 10.0
            self._prey_alive = False
        if self._prey_alive:
            for _ in range(5):
                move = int(self._numpy_rng.choice(5, 1, p=_PREY_PROBABILITIES)[0])
                candidate = self._prey + _MOVES[move]
                if self._neighbour_count(candidate) == 0:
                    # Legacy rejection checks danger first, not vacancy.
                    # Walls/occupied proposals consume the accepted draw.
                    if self._vacant(candidate):
                        self._prey = candidate
                    break
            # If all five proposals are dangerous, the prey stays put.
        terminated = not self._prey_alive
        truncated = self._steps >= self.episode_limit and not terminated
        self._needs_reset = terminated or truncated
        info = {"prey_alive": [self._prey_alive], "capture_success": terminated,
                "episode_limit": truncated, "step_count": self._steps}
        return self._observation(), float(reward), terminated, truncated, info

    def close(self):
        self._closed = True
