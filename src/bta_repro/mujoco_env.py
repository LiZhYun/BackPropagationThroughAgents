"""Legacy MA-MuJoCo observations with audited controls and explicit timeouts.

The optional simulator is imported only at construction. See
``docs/mujoco_adapter.md`` for the action coordinates and supported partitions.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from importlib import metadata
import json
import platform
import random
import threading

import numpy as np

from .mujoco_partitions import (
    ACTUATOR_MAPPINGS, AUTHOR_COMMIT, PARTITION_EXTENSION_VERSION,
    bootstrap_partition, extend_backend_partition,
)

MAMUJOCO_COMMIT = "b212ddd74b258e7cea006ff1d642b5ffada4b99d"
_RNG_LOCK = threading.RLock()

def _integer(value, name: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is out of range")
    return int(value)


def mujoco_config(config: dict) -> dict:
    """Resolve a legacy task without silently selecting a different partition."""
    result = {
        "scenario": "HalfCheetah-v2", "agent_conf": "2x3",
        "agent_obsk": 0, "episode_limit": 1000, "seed": 0,
        "k_categories": None, "global_categories": None, "action_units": "normalized",
    }
    unknown = set(config) - set(result)
    if unknown:
        raise ValueError(f"unknown MuJoCo config keys: {sorted(unknown)}")
    result.update(deepcopy(config))
    if not isinstance(result["scenario"], str) or not isinstance(result["agent_conf"], str):
        raise ValueError("scenario and agent_conf must be strings")
    if (result["scenario"], result["agent_conf"]) not in ACTUATOR_MAPPINGS:
        raise ValueError(
            f"unsupported audited partition {result['scenario']} / {result['agent_conf']}; "
            "see docs/mujoco_adapter.md for supported tasks and extension requirements"
        )
    result["seed"] = _integer(result["seed"], "seed", 0, 2**32 - 1)
    result["episode_limit"] = _integer(result["episode_limit"], "episode_limit", 1)
    if result["action_units"] not in ("normalized", "physical"):
        raise ValueError("action_units must be normalized or physical")
    if result["agent_obsk"] is not None:
        result["agent_obsk"] = _integer(result["agent_obsk"], "agent_obsk", 0)
    for key in ("k_categories", "global_categories"):
        if result[key] is not None and not isinstance(result[key], str):
            raise ValueError(f"{key} must be a string or None")
    return result


def _make_backend(config: dict):
    """Require the pinned legacy stack, rather than importing a modern v5 task."""
    try:
        versions = {name: metadata.version(name) for name in ("gym", "mujoco-py", "MAMujoco")}
        provenance = json.loads(metadata.distribution("MAMujoco").read_text("direct_url.json") or "{}")
    except metadata.PackageNotFoundError as exc:
        raise ImportError("Legacy MA-MuJoCo is optional; run scripts/setup_mujoco.sh and use its environment") from exc
    commit = provenance.get("vcs_info", {}).get("commit_id")
    if versions["gym"] != "0.10.8" or versions["mujoco-py"] != "2.1.2.14" or commit != MAMUJOCO_COMMIT:
        raise RuntimeError("MuJoCo adapter requires Gym 0.10.8, mujoco-py 2.1.2.14 and the pinned MA-MuJoCo commit")
    try:
        import mujoco_py
        from multiagent_mujoco.mujoco_multi import MujocoMulti
    except ImportError as exc:
        raise ImportError("Legacy MuJoCo could not load; use the environment exports in docs/mujoco_setup.md") from exc
    engine_version = int(mujoco_py.cymj._mj_version())
    if engine_version != 210:
        raise RuntimeError(f"expected MuJoCo engine 2.1.0 (210), found {engine_version}")
    backend_config = {key: value for key, value in config.items() if key != "seed"}
    backend_config["agent_conf"] = bootstrap_partition(config["scenario"], config["agent_conf"])
    backend = MujocoMulti(env_args=backend_config)
    try:
        extended = extend_backend_partition(backend, config["agent_conf"])
    except Exception:
        backend.env.close()
        raise
    backend._bta_versions = {
        **versions, "mujoco_engine": engine_version, "ma_mujoco_commit": commit,
        "python": platform.python_version(), "numpy": np.__version__,
    }
    if extended:
        backend._bta_versions.update({"partition_extension": PARTITION_EXTENSION_VERSION,
                                      "partition_author_commit": AUTHOR_COMMIT,
                                      "partition_actuator_corrections": backend._bta_partition_corrections})
    return backend


class MujocoEnv:
    """Shared runner interface for finite, equally sized local action vectors.

    Actions are raw Gaussian samples, normalized to actuator ranges by default.
    They remain unchanged for PPO likelihoods; a private copy is clipped/scaled.
    ``reset()`` advances the seed stream; ``reset(seed=...)`` restarts it.
    """

    is_discrete = False

    def __init__(self, config: dict):
        self.config = mujoco_config(config)
        self.seed = self.config["seed"]
        self.scenario = self.config["scenario"]
        self.episode_limit = self.config["episode_limit"]
        self._numpy_state = np.random.RandomState(self.seed).get_state()
        self._python_state = random.Random(self.seed).getstate()
        self._closed = False
        self._needs_reset = True
        self._steps = 0
        self._env = None
        try:
            with self._backend_scope():
                self._env = _make_backend(self.config)
                # MujocoMulti.seed() is a no-op. Seed the actual Gym physics env.
                self._env.env.seed(self.seed)
                info = self._env.get_env_info()
            self.num_agents = _integer(info["n_agents"], "n_agents", 1)
            self.obs_dim = _integer(info["obs_shape"], "obs_shape", 1)
            self.state_dim = _integer(info["state_shape"], "state_shape", 1)
            self.action_dim = _integer(info["n_actions"], "n_actions", 1)
            if info["episode_limit"] != self.episode_limit:
                raise ValueError("backend episode_limit differs from config")
            self._audit_controls()
            self.versions = deepcopy(getattr(self._env, "_bta_versions", {}))
            # Discard constructor-only global draws. First reset and explicit
            # reseeding must start at the same point of every random stream.
            self._numpy_state = np.random.RandomState(self.seed).get_state()
            self._python_state = random.Random(self.seed).getstate()
        except Exception:
            self.close()
            raise

    def _audit_controls(self):
        partitions = self._env.agent_partitions
        if len(partitions) != self.num_agents or any(len(part) != self.action_dim for part in partitions):
            raise ValueError("only equally sized, complete local actuator partitions are supported")
        mapping = tuple(tuple(_integer(node.act_ids, "act_ids", 0) for node in part) for part in partitions)
        expected = ACTUATOR_MAPPINGS[(self.scenario, self.config["agent_conf"])]
        if mapping != expected:
            raise ValueError(f"backend actuator mapping {mapping} differs from audited mapping {expected}")
        space = self._env.env.action_space
        space_low, space_high = np.asarray(space.low, dtype=np.float64), np.asarray(space.high, dtype=np.float64)
        total = self.num_agents * self.action_dim
        if space_low.shape != (total,) or space_high.shape != (total,) or sorted(sum(mapping, ())) != list(range(total)):
            raise ValueError("partition must control every base actuator exactly once")
        if not np.isfinite(space_low).all() or not np.isfinite(space_high).all() or not (space_low < space_high).all():
            raise ValueError("base actuator bounds must be finite and strictly ordered")
        model = self._env.env.model
        control_range = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
        if control_range.shape != (total, 2) or not np.asarray(model.actuator_ctrllimited).all():
            raise ValueError("model must provide finite bounded controls for every actuator")
        low, high = control_range[:, 0], control_range[:, 1]
        if (not np.isfinite(control_range).all() or not (low < high).all()
                or not np.allclose(space_low, low, rtol=1e-6, atol=0)
                or not np.allclose(space_high, high, rtol=1e-6, atol=0)):
            raise ValueError("Gym action bounds differ from the physical actuator bounds")
        # Gym casts bounds to float32. Clip against the model's exact doubles:
        # Humanoid's 0.4 would otherwise become 0.4000000059604645.
        # Check the model, not just a permutation: the observation's joint and
        # the commanded actuator must refer to the same physical degree of freedom.
        for part in partitions:
            for node in part:
                actuator = int(node.act_ids)
                if model.actuator_trntype[actuator] != 0:
                    raise ValueError("only audited joint-transmission actuators are supported")
                joint = int(model.actuator_trnid[actuator, 0])
                qpos = _integer(node.qpos_ids, "qpos_ids", -int(model.nq), int(model.nq) - 1) % model.nq
                qvel = _integer(node.qvel_ids, "qvel_ids", -int(model.nv), int(model.nv) - 1) % model.nv
                if model.jnt_qposadr[joint] != qpos or model.jnt_dofadr[joint] != qvel:
                    raise ValueError("partition observation joint does not match its physical actuator")
        self.actuator_indices = np.asarray(mapping, dtype=np.int64)
        self.actuator_low = low[self.actuator_indices].copy()
        self.actuator_high = high[self.actuator_indices].copy()
        if self.config["action_units"] == "normalized":
            self.action_low = np.full_like(self.actuator_low, -1.0)
            self.action_high = np.full_like(self.actuator_high, 1.0)
        else:
            self.action_low = self.actuator_low.copy()
            self.action_high = self.actuator_high.copy()
        for value in (self.actuator_indices, self.action_low, self.action_high, self.actuator_low, self.actuator_high):
            value.flags.writeable = False

    @contextmanager
    def _backend_scope(self):
        # Gym uses its own seeded RNG. Isolate legacy/global draws too, including
        # construction and close. Synchronous runner calls do not leak RNG state.
        with _RNG_LOCK:
            numpy_state, python_state = np.random.get_state(), random.getstate()
            np.random.set_state(self._numpy_state)
            random.setstate(self._python_state)
            try:
                yield
            finally:
                self._numpy_state, self._python_state = np.random.get_state(), random.getstate()
                np.random.set_state(numpy_state)
                random.setstate(python_state)

    def _assert_open(self):
        if self._closed:
            raise RuntimeError("MuJoCo environment is closed")

    def _observation(self) -> dict:
        obs = np.asarray(self._env.get_obs(), dtype=np.float32)
        state = np.asarray(self._env.get_state(), dtype=np.float32)
        if obs.shape != (self.num_agents, self.obs_dim) or state.shape != (self.state_dim,):
            raise ValueError("MuJoCo observation/state shapes do not match get_env_info")
        if not np.isfinite(obs).all() or not np.isfinite(state).all():
            raise ValueError("MuJoCo observation and state must be finite")
        return {"obs": obs.copy(), "state": state.copy(), "active_agents": np.ones(self.num_agents, dtype=bool)}

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        self._assert_open()
        if seed is not None:
            self.seed = _integer(seed, "seed", 0, 2**32 - 1)
            self.config["seed"] = self.seed
            self._numpy_state = np.random.RandomState(self.seed).get_state()
            self._python_state = random.Random(self.seed).getstate()
        self._needs_reset = True
        with self._backend_scope():
            if seed is not None:
                self._env.env.seed(self.seed)
            self._env.env.reset()
            self._env.steps = 0
            observation = self._observation()
        self._steps = 0
        self._needs_reset = False
        return observation, {
            "seed": self.seed, "scenario": self.scenario,
            "agent_conf": self.config["agent_conf"], "episode_limit": self.episode_limit,
        }

    def step(self, actions: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        self._assert_open()
        if self._needs_reset:
            raise RuntimeError("call reset before stepping or after the episode ends")
        raw = np.asarray(actions)
        if raw.shape != (self.num_agents, self.action_dim) or not np.issubdtype(raw.dtype, np.floating):
            raise ValueError("actions must be floating point with shape [num_agents, action_dim]")
        if not np.isfinite(raw).all():
            raise ValueError("actions must be finite")
        local = np.clip(np.array(raw, dtype=np.float64, copy=True), self.action_low, self.action_high)
        if self.config["action_units"] == "normalized":
            midpoint = (self.actuator_low + self.actuator_high) * 0.5
            half_range = (self.actuator_high - self.actuator_low) * 0.5
            physical = np.clip(midpoint + local * half_range, self.actuator_low, self.actuator_high)
        else:
            physical = local
        controls = np.empty(self.num_agents * self.action_dim, dtype=np.float64)
        controls[self.actuator_indices] = physical
        try:
            with self._backend_scope():
                # Legacy TimeLimit erases physical termination at the horizon.
                # Step the SAME base Gym task and own the horizon here instead.
                _, reward, terminated, backend_info = self._env.env.step(controls)
                if not isinstance(terminated, (bool, np.bool_)) or not isinstance(backend_info, dict):
                    raise ValueError("base Gym step must return a boolean terminal flag and dict info")
                reward = float(reward)
                if not np.isfinite(reward):
                    raise ValueError("MuJoCo reward must be finite")
                self._steps += 1
                self._env.steps = self._steps
                observation = self._observation()
        except Exception:
            self._needs_reset = True
            raise
        terminated = bool(terminated)
        truncated = self._steps >= self.episode_limit and not terminated
        self._needs_reset = terminated or truncated
        info = deepcopy(backend_info)
        info.update({"episode_limit": truncated, "elapsed_steps": self._steps,
                     "clipped_action_components": int(np.count_nonzero(local != raw))})
        return observation, reward, terminated, truncated, info

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._needs_reset = True
        if self._env is not None:
            with self._backend_scope():
                # Upstream MujocoMulti.close() raises NotImplementedError.
                self._env.env.close()
