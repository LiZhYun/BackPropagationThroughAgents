"""Optional SMACv2 adapter with explicit episode boundaries and seeded capabilities."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import random
import re
import threading

import numpy as np


SMACV2_COMMIT = "577ab5a2cff2391f8df582da5731ea9cd6adf3c6"
_RNG_LOCK = threading.RLock()
_TEAMS = {
    "protoss": (["stalker", "zealot", "colossus"], [0.45, 0.45, 0.1], []),
    "terran": (["marine", "marauder", "medivac"], [0.45, 0.45, 0.1], ["medivac"]),
    "zerg": (["zergling", "baneling", "hydralisk"], [0.45, 0.1, 0.45], ["baneling"]),
}


def smac_config(scenario: str, seed: int = 0, **overrides) -> dict:
    """Build one of the nine source-listed race/team-size scenarios.

    Capability details follow pinned official SMACv2 examples; the accepted
    papers do not completely specify those details. Overrides are explicit
    backend keyword arguments, except ``sc2_path`` which the adapter consumes.
    """
    normalized = scenario.lower().strip().replace(" ", "_")
    match = re.fullmatch(r"(protoss|terran|zerg)_(5v5|10v10|10v11)", normalized)
    if match is None:
        raise ValueError("scenario must be a race plus 5v5, 10v10, or 10v11, e.g. protoss_5v5")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    race, counts = match.groups()
    n_agents, n_enemies = map(int, counts.split("v"))
    units, weights, exceptions = _TEAMS[race]
    config = {
        "scenario": normalized,
        "seed": int(seed),
        "map_name": f"10gen_{race}",
        "game_version": "4.10",
        "difficulty": "7",
        "step_mul": 8,
        "move_amount": 2,
        "continuing_episode": True,
        "conic_fov": False,
        "obs_own_pos": True,
        "use_unit_ranges": True,
        "min_attack_range": 2,
        "obs_all_health": True,
        "obs_own_health": True,
        "obs_last_action": False,
        "state_last_action": True,
        "obs_instead_of_state": False,
        "obs_pathing_grid": False,
        "obs_terrain_height": False,
        "obs_timestep_number": False,
        "state_timestep_number": False,
        "prob_obs_enemy": 1.0,
        "action_mask": True,
        "reward_only_positive": True,
        "reward_scale": True,
        "reward_scale_rate": 20,
        "reward_sparse": False,
        "reward_death_value": 10,
        "reward_win": 200,
        "reward_defeat": 0,
        "reward_negative_scale": 0.5,
        "capability_config": {
            "n_units": n_agents,
            "n_enemies": n_enemies,
            "team_gen": {
                "dist_type": "weighted_teams", "observe": True,
                "unit_types": list(units), "weights": list(weights),
                "exception_unit_types": list(exceptions),
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect", "p": 0.5,
                "map_x": 32, "map_y": 32,
            },
        },
    }
    config.update(deepcopy(overrides))
    if config["continuing_episode"] is not True:
        raise ValueError("continuing_episode=True is required to expose SMAC time-limit truncations")
    if config["action_mask"] is not True:
        raise ValueError("action_mask=True is required for legal actions and active-agent masks")
    if config["prob_obs_enemy"] != 1.0:
        raise ValueError("this seeded adapter currently supports prob_obs_enemy=1.0 only")
    if {"attack", "health", "enemy_mask"} & config["capability_config"].keys():
        raise ValueError("this adapter currently supports team and start-position capabilities only")
    return config


def _make_backend(config: dict):
    """Import the optional simulator only when an environment is constructed."""
    try:
        from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
    except ImportError as exc:
        raise ImportError(
            "SMACv2 is optional and unavailable or incompatible. Install the pinned "
            f"SMACv2 dependency at commit {SMACV2_COMMIT}; see docs/smac.md."
        ) from exc
    sc2_path = os.environ.get("SC2PATH")
    if not sc2_path or not (Path(sc2_path) / "Versions").is_dir() or not (Path(sc2_path) / "Maps").is_dir():
        raise FileNotFoundError("Set SC2PATH to an existing StarCraft II installation with Versions and Maps")
    return StarCraftCapabilityEnvWrapper(**deepcopy(config))


def _seed_capability_generators(backend, seed: int) -> int:
    """Replace upstream unseeded default_rng instances, including nested ones."""
    sequence = np.random.SeedSequence(seed)
    visited = set()
    count = 0

    def visit(distribution):
        nonlocal count
        if id(distribution) in visited:
            return
        visited.add(id(distribution))
        if isinstance(getattr(distribution, "rng", None), np.random.Generator):
            distribution.rng = np.random.default_rng(sequence.spawn(1)[0])
            count += 1
        for name, child in sorted(vars(distribution).items()):
            if name != "rng" and hasattr(child, "generate") and hasattr(child, "__dict__"):
                visit(child)

    for _, distribution in sorted(getattr(backend, "env_key_to_distribution_map", {}).items()):
        visit(distribution)
    return count


class SmacSimulatorError(RuntimeError):
    """A simulator failure invalidated the transition; it must not enter PPO."""


class SmacEnv:
    """SMACv2's list API adapted to explicit observations and episode endings.

    Seed is fixed at construction. ``reset()`` advances its stream; passing a
    seed is accepted only for the first reset and only if it matches that seed.
    Create another environment for a fresh independent or replayed stream.
    """

    is_discrete = True

    def __init__(self, config: dict):
        supplied = deepcopy(config)
        scenario = supplied.pop("scenario", "protoss_5v5")
        seed = supplied.pop("seed", 0)
        self.config = smac_config(scenario, seed, **supplied)
        self.seed = self.config["seed"]
        self.scenario = self.config["scenario"]
        self._sc2_path = self.config.get("sc2_path")
        self._numpy_state = np.random.RandomState(self.seed).get_state()
        self._python_state = random.Random(self.seed).getstate()
        self._closed = False
        self._needs_reset = True
        self._has_reset = False
        self._last_available_actions = None
        backend_config = {key: value for key, value in self.config.items() if key not in ("scenario", "sc2_path")}
        with self._backend_scope():
            self._env = _make_backend(backend_config)
            self.capability_rng_streams = _seed_capability_generators(self._env, self.seed)
            metadata = self._env.get_env_info()
        self.num_agents = int(metadata["n_agents"])
        self.obs_dim = int(np.prod(metadata["obs_shape"]))
        self.state_dim = int(np.prod(metadata["state_shape"]))
        self.num_actions = int(metadata["n_actions"])
        self.episode_limit = int(metadata["episode_limit"])
        if min(self.num_agents, self.obs_dim, self.state_dim, self.num_actions, self.episode_limit) < 1:
            self.close()
            raise ValueError("SMAC reported invalid environment dimensions")
        if self.num_agents != self.config["capability_config"]["n_units"]:
            self.close()
            raise ValueError("SMAC agent count does not match capability_config")

    @contextmanager
    def _backend_scope(self):
        # Upstream also calls Python random and legacy np.random. Preserve the
        # caller's streams and serialize adapters while installing private ones.
        with _RNG_LOCK:
            numpy_state, python_state = np.random.get_state(), random.getstate()
            previous_path = os.environ.get("SC2PATH")
            np.random.set_state(self._numpy_state)
            random.setstate(self._python_state)
            if self._sc2_path is not None:
                os.environ["SC2PATH"] = str(self._sc2_path)
            try:
                yield
            finally:
                self._numpy_state, self._python_state = np.random.get_state(), random.getstate()
                np.random.set_state(numpy_state)
                random.setstate(python_state)
                if previous_path is None:
                    os.environ.pop("SC2PATH", None)
                else:
                    os.environ["SC2PATH"] = previous_path

    def _assert_open(self):
        if self._closed:
            raise RuntimeError("SMAC environment is closed")

    def _observation(self) -> dict:
        obs = np.asarray(self._env.get_obs(), dtype=np.float32)
        state = np.asarray(self._env.get_state(), dtype=np.float32)
        raw_mask = np.asarray(self._env.get_avail_actions())
        if obs.shape != (self.num_agents, self.obs_dim) or state.shape != (self.state_dim,):
            raise ValueError("SMAC observation/state shapes do not match get_env_info")
        if not np.isfinite(obs).all() or not np.isfinite(state).all():
            raise ValueError("SMAC observation and state must be finite")
        if raw_mask.shape != (self.num_agents, self.num_actions) or not np.isin(raw_mask, (0, 1)).all():
            raise ValueError("SMAC returned invalid available_actions")
        available = raw_mask.astype(bool)
        if not available.any(axis=-1).all():
            raise ValueError("every SMAC agent needs an available action, including noop for dead agents")
        self._last_available_actions = available.copy()
        return {
            "obs": obs.copy(), "state": state.copy(), "available_actions": available,
            "active_agents": available[:, 1:].any(axis=-1),
        }

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        self._assert_open()
        if seed is not None and (seed != self.seed or self._has_reset):
            raise ValueError("SMAC cannot reseed in place; construct a new SmacEnv with the desired seed")
        self._needs_reset = True
        with self._backend_scope():
            restarts = getattr(self._env, "force_restarts", 0)
            self._env.reset()
            if getattr(self._env, "force_restarts", 0) != restarts:
                raise SmacSimulatorError("SMAC restarted the simulator during reset; discard this episode")
            # Query explicitly: upstream's reset retry path can return None.
            observation = self._observation()
        self._needs_reset = False
        self._has_reset = True
        return observation, {
            "seed": self.seed, "scenario": self.scenario,
            "episode_limit": self.episode_limit, "capability_rng_streams": self.capability_rng_streams,
        }

    def step(self, actions: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        self._assert_open()
        if self._needs_reset:
            raise RuntimeError("call reset before stepping or after the episode ends")
        actions = np.asarray(actions)
        if actions.shape != (self.num_agents,) or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("actions must be integer indices with shape [num_agents]")
        if ((actions < 0) | (actions >= self.num_actions)).any():
            raise ValueError("action index out of range")
        if not self._last_available_actions[np.arange(self.num_agents), actions].all():
            raise ValueError("actions contain an unavailable action")
        try:
            with self._backend_scope():
                restarts = getattr(self._env, "force_restarts", 0)
                reward, done, backend_info = self._env.step(actions.tolist())
                if getattr(self._env, "force_restarts", 0) != restarts:
                    raise SmacSimulatorError("SMAC restarted during step; discard the invalid transition")
                info = dict(backend_info)
                if done and not info:
                    raise SmacSimulatorError("SMAC returned an unexplained empty terminal result")
                observation = self._observation()
        except Exception:
            self._needs_reset = True
            raise
        truncated = bool(info.get("episode_limit", False))
        if truncated and not done:
            self._needs_reset = True
            raise SmacSimulatorError("SMAC signaled an episode limit without ending the episode")
        if not np.isfinite(reward):
            self._needs_reset = True
            raise SmacSimulatorError("SMAC returned a nonfinite reward")
        terminated = bool(done) and not truncated
        self._needs_reset = terminated or truncated
        return observation, float(reward), terminated, truncated, info

    def close(self):
        if not self._closed:
            try:
                with self._backend_scope():
                    self._env.close()
            finally:
                self._closed = True
