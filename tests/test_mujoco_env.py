import random
from types import SimpleNamespace

import numpy as np
import pytest

import bta_repro.mujoco_env as mujoco


class FakePhysics:
    def __init__(self, count):
        self.action_space = SimpleNamespace(low=-np.arange(1, count + 1, dtype=float),
                                            high=np.arange(2, count + 2, dtype=float))
        self.model = SimpleNamespace(nq=count, nv=count, actuator_trntype=np.zeros(count, dtype=int),
                                     actuator_trnid=np.column_stack([np.arange(count), np.zeros(count, dtype=int)]),
                                     jnt_qposadr=np.arange(count), jnt_dofadr=np.arange(count),
                                     actuator_ctrlrange=np.column_stack([self.action_space.low, self.action_space.high]),
                                     actuator_ctrllimited=np.ones(count, dtype=bool))
        self.closed = 0
        self.terminal = False
        self.fail = False
        self.state = np.zeros(4)
        self.seed(0)

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def reset(self):
        self.state = self.rng.random(4) + np.random.random() + random.random()
        self.terminal = False
        return self.state.copy()

    def step(self, action):
        if self.fail:
            raise RuntimeError("physics failed")
        self.last_action = action.copy()
        self.state += self.rng.random() + np.random.random() + random.random()
        return self.state.copy(), float(action.sum()), self.terminal, {"custom": 9}

    def close(self):
        self.closed += 1


class FakeMulti:
    def __init__(self, config):
        self.config = config
        mapping = mujoco.ACTUATOR_MAPPINGS[(config["scenario"], config["agent_conf"])]
        self.agent_partitions = [[SimpleNamespace(act_ids=index, qpos_ids=index, qvel_ids=index)
                                  for index in part] for part in mapping]
        self.env = FakePhysics(sum(map(len, mapping)))
        self.bad_observation = False
        self._bta_versions = {"test_backend": True}
        # Exercise constructor leakage as well as reset/step leakage.
        np.random.random()
        random.random()

    def get_env_info(self):
        return {"n_agents": len(self.agent_partitions), "n_actions": len(self.agent_partitions[0]),
                "obs_shape": 3, "state_shape": 4, "episode_limit": self.config["episode_limit"]}

    def get_obs(self):
        obs = np.repeat(self.env.state[None, :3], len(self.agent_partitions), axis=0)
        if self.bad_observation:
            obs[0, 0] = np.nan
        return obs

    def get_state(self):
        return self.env.state

    def step(self, actions):
        raise AssertionError("legacy TimeLimit must not merge timeout and physical failure")

    def seed(self, seed):
        raise AssertionError("upstream seed is a no-op")

    def close(self):
        raise AssertionError("upstream close is unimplemented")


@pytest.fixture
def factory(monkeypatch):
    backends = []

    def build(config):
        backend = FakeMulti(config)
        backends.append(backend)
        return backend

    monkeypatch.setattr(mujoco, "_make_backend", build)
    return backends


@pytest.mark.parametrize("scenario,partition", mujoco.ACTUATOR_MAPPINGS)
def test_supported_partitions_and_raw_observations(factory, scenario, partition):
    env = mujoco.MujocoEnv({"scenario": scenario, "agent_conf": partition, "seed": 7})
    observation, info = env.reset()
    assert env.is_discrete is False and env.versions == {"test_backend": True}
    assert info["seed"] == 7 and info["episode_limit"] == 1000
    assert observation["obs"].shape == (env.num_agents, 3)
    assert observation["obs"].dtype == np.float32
    np.testing.assert_allclose(observation["state"], factory[-1].env.state)
    np.testing.assert_allclose(observation["obs"][0], observation["state"][:3])
    assert observation["active_agents"].all() and "available_actions" not in observation
    np.testing.assert_array_equal(env.actuator_indices, mujoco.ACTUATOR_MAPPINGS[(scenario, partition)])
    env.close()
    env.close()
    assert factory[-1].env.closed == 1
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


def test_physical_action_clipping_respects_anatomical_bounds_and_keeps_raw_samples(factory):
    env = mujoco.MujocoEnv({"scenario": "Ant-v2", "agent_conf": "2x4d", "action_units": "physical"})
    env.reset()
    raw = np.array([[30, -30, 0.25, 0.5], [0.75, 1, 2.5, -3]], dtype=np.float32)
    original = raw.copy()
    _, reward, terminal, timeout, info = env.step(raw)
    # Global indices are [2,3,6,7] and [4,5,0,1]; ranges differ per actuator.
    expected = np.array([2, -2, 4, -4, 0.75, 1, 0.25, 0.5])
    np.testing.assert_array_equal(factory[-1].env.last_action, expected)
    np.testing.assert_array_equal(raw, original)
    assert reward == expected.sum() and not terminal and not timeout
    assert info["clipped_action_components"] == 4 and info["custom"] == 9
    assert not env.actuator_indices.flags.writeable and not env.action_low.flags.writeable


@pytest.mark.parametrize("action_units", ["normalized", "physical"])
def test_clipping_uses_exact_model_bounds_instead_of_rounded_gym_box(monkeypatch, action_units):
    backend = FakeMulti(mujoco.mujoco_config({"scenario": "Humanoid-v2", "agent_conf": "17x1"}))
    backend.env.action_space.low = np.full(17, -0.4, dtype=np.float32)
    backend.env.action_space.high = np.full(17, 0.4, dtype=np.float32)
    backend.env.model.actuator_ctrlrange = np.tile([-0.4, 0.4], (17, 1))
    monkeypatch.setattr(mujoco, "_make_backend", lambda config: backend)
    env = mujoco.MujocoEnv({"scenario": "Humanoid-v2", "agent_conf": "17x1", "action_units": action_units})
    env.reset()
    env.step(np.ones((17, 1)))
    np.testing.assert_array_equal(backend.env.last_action, np.full(17, 0.4, dtype=np.float64))
    assert np.all(env.actuator_high == 0.4)
    assert np.all(env.action_high == (1.0 if action_units == "normalized" else 0.4))


def test_default_normalized_actions_scale_asymmetric_physical_bounds(factory):
    env = mujoco.MujocoEnv({})
    env.reset()
    raw = np.array([[-2, -0.5, 0], [0.25, 0.5, 2]], dtype=float)
    original = raw.copy()
    _, _, _, _, info = env.step(raw)
    # Local actuator order [2,1,0],[5,4,3]. Physical midpoints are all 0.5;
    # half ranges in global order are 1.5,2.5,3.5,4.5,5.5,6.5.
    np.testing.assert_array_equal(factory[-1].env.last_action, [0.5, -0.75, -3, 5, 3.25, 2.125])
    np.testing.assert_array_equal(raw, original)
    assert info["clipped_action_components"] == 2
    assert env.config["action_units"] == "normalized"
    assert np.all(env.action_low == -1) and np.all(env.action_high == 1)


@pytest.mark.parametrize("terminal,limit,expected", [(False, 1, (False, True)),
                                                      (True, 1, (True, False)),
                                                      (True, 5, (True, False))])
def test_terminal_and_timeout_including_failure_on_last_step(factory, terminal, limit, expected):
    env = mujoco.MujocoEnv({"episode_limit": limit})
    initial, _ = env.reset()
    factory[-1].env.terminal = terminal
    final, _, terminated, truncated, info = env.step(np.zeros((2, 3)))
    assert (terminated, truncated) == expected
    assert info["episode_limit"] is truncated and info["elapsed_steps"] == 1
    assert not np.array_equal(initial["state"], final["state"])
    np.testing.assert_allclose(final["state"], factory[-1].env.state)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros((2, 3)))
    env.reset()


def test_seeded_interleaving_reseed_and_caller_rng_are_independent(factory):
    numpy_state, python_state = np.random.get_state(), random.getstate()
    expected_numpy = np.random.RandomState()
    expected_numpy.set_state(numpy_state)
    expected_python = random.Random()
    expected_python.setstate(python_state)
    first = mujoco.MujocoEnv({"seed": 91})
    other = mujoco.MujocoEnv({"seed": 54})
    replay = mujoco.MujocoEnv({"seed": 91})
    initial, _ = first.reset()
    other.reset()
    replay_initial, _ = replay.reset()
    np.testing.assert_array_equal(initial["state"], replay_initial["state"])
    other.step(np.ones((2, 3)))
    transition = first.step(np.zeros((2, 3)))
    replay_transition = replay.step(np.zeros((2, 3)))
    np.testing.assert_array_equal(transition[0]["state"], replay_transition[0]["state"])
    assert not np.array_equal(first.reset()[0]["state"], initial["state"])
    np.testing.assert_array_equal(first.reset(seed=91)[0]["state"], initial["state"])
    first.reset(seed=54)
    assert first.config["seed"] == 54
    first.close()
    other.close()
    replay.close()
    assert np.random.random() == expected_numpy.random()
    assert random.random() == expected_python.random()


@pytest.mark.parametrize("bad_actions", [np.zeros(6), np.zeros((2, 3), dtype=int),
                                         np.full((2, 3), np.nan), np.full((2, 3), np.inf)])
def test_invalid_actions_do_not_advance_simulator(factory, bad_actions):
    env = mujoco.MujocoEnv({})
    env.reset()
    with pytest.raises(ValueError, match="actions"):
        env.step(bad_actions)
    assert not hasattr(factory[-1].env, "last_action")
    env.step(np.zeros((2, 3)))


@pytest.mark.parametrize("config", [{"scenario": "Ant-v5"}, {"agent_conf": "4x2"},
                                    {"seed": -1}, {"seed": True}, {"seed": 2**32},
                                    {"episode_limit": 0}, {"agent_obsk": -1},
                                    {"agent_obsk": False}, {"action_units": "tanh"}, {"foo": 1}])
def test_invalid_or_unaudited_configs_fail_before_simulator_load(factory, config):
    with pytest.raises(ValueError):
        mujoco.MujocoEnv(config)
    assert factory == []


@pytest.mark.parametrize("mutation", ["duplicate", "different_order", "wrong_joint", "unequal", "invalid_bounds"])
def test_mapping_and_model_mismatches_close_backend(monkeypatch, mutation):
    config = mujoco.mujoco_config({})
    backend = FakeMulti(config)
    if mutation == "duplicate":
        backend.agent_partitions[0][0].act_ids = 1
    elif mutation == "different_order":
        backend.agent_partitions[0].reverse()
    elif mutation == "wrong_joint":
        backend.env.model.jnt_qposadr[2] = 1
    elif mutation == "unequal":
        backend.agent_partitions[1].pop()
    else:
        backend.env.action_space.high[0] = backend.env.action_space.low[0]
    monkeypatch.setattr(mujoco, "_make_backend", lambda config: backend)
    with pytest.raises(ValueError):
        mujoco.MujocoEnv(config)
    assert backend.env.closed == 1


@pytest.mark.parametrize("failure", ["physics", "observation"])
def test_invalid_transition_requires_reset(factory, failure):
    env = mujoco.MujocoEnv({})
    env.reset()
    if failure == "physics":
        factory[-1].env.fail = True
    else:
        factory[-1].bad_observation = True
    with pytest.raises((RuntimeError, ValueError)):
        env.step(np.zeros((2, 3)))
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros((2, 3)))
    factory[-1].env.fail = False
    factory[-1].bad_observation = False
    env.reset()
    env.step(np.zeros((2, 3)))
