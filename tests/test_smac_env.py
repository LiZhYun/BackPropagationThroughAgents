import random
from types import SimpleNamespace

import numpy as np
import pytest

import bta_repro.smac_env as smac


class FakeDistribution:
    def __init__(self, nested=False):
        self.rng = np.random.default_rng()
        if nested:
            self.pos_generator = FakeDistribution()

    def generate(self):
        value = self.rng.random()
        if hasattr(self, "pos_generator"):
            value += self.pos_generator.generate()
        return value


class FakeSmac:
    def __init__(self, config):
        self.config = config
        self.n_agents = config["capability_config"]["n_units"]
        self.n_actions = config["capability_config"]["n_enemies"] + 6
        self.env_key_to_distribution_map = {
            "team_gen": FakeDistribution(), "start_positions": FakeDistribution(nested=True),
        }
        self.force_restarts = 0
        self._obs = SimpleNamespace(observation=SimpleNamespace(game_loop=0))
        self.reset_calls = 0
        self.step_calls = 0
        self.closed = False
        self.restart_on_step = False
        self.restart_on_reset = False
        self.next_result = (1.5, False, {"battle_won": False})
        self.obs = np.zeros((self.n_agents, 3), dtype=np.float64)
        self.state = np.zeros(4, dtype=np.float64)
        self.mask = np.zeros((self.n_agents, self.n_actions), dtype=np.int64)
        self.mask[:, 1] = 1
        self.mask[1, 1] = 0
        self.mask[1, 0] = 1

    def get_env_info(self):
        return {"n_agents": self.n_agents, "n_actions": self.n_actions,
                "obs_shape": 3, "state_shape": 4, "episode_limit": 200}

    def _draw(self):
        return sum(d.generate() for d in self.env_key_to_distribution_map.values()) + random.random() + np.random.random()

    def reset(self):
        self.reset_calls += 1
        self._obs.observation.game_loop += 1
        self.force_restarts += int(self.restart_on_reset)
        self.obs.fill(self._draw())
        self.state.fill(self.reset_calls)
        # Intentionally no return: the adapter must query observations itself.

    def get_obs(self):
        return self.obs

    def get_state(self):
        return self.state

    def get_avail_actions(self):
        return self.mask

    def step(self, actions):
        assert isinstance(actions, list)
        self.step_calls += 1
        self._obs.observation.game_loop += self.config['step_mul']
        self.force_restarts += int(self.restart_on_step)
        self.obs.fill(self._draw())
        self.state.fill(10 + self.step_calls)
        return self.next_result

    def full_restart(self):
        self.force_restarts += 1
        self._obs.observation.game_loop = 0

    def close(self):
        self.closed = True


@pytest.fixture
def factory(monkeypatch):
    instances = []

    def build(config):
        backend = FakeSmac(config)
        instances.append(backend)
        return backend

    monkeypatch.setattr(smac, "_make_backend", build)
    return instances


def _legal_actions(env):
    actions = np.ones(env.num_agents, dtype=np.int64)
    actions[1] = 0
    return actions


@pytest.mark.parametrize("race", ["protoss", "terran", "zerg"])
@pytest.mark.parametrize("counts", ["5v5", "10v10", "10v11"])
def test_factory_builds_source_scenarios_without_simulator(race, counts):
    config = smac.smac_config(f"{race}_{counts}", seed=12)
    agents, enemies = map(int, counts.split("v"))
    assert config["map_name"] == f"10gen_{race}"
    assert config["capability_config"]["n_units"] == agents
    assert config["capability_config"]["n_enemies"] == enemies
    assert config["continuing_episode"] is True
    assert sum(config["capability_config"]["team_gen"]["weights"]) == pytest.approx(1)
    config["capability_config"]["team_gen"]["unit_types"][0] = "mutated"
    assert smac.smac_config(f"{race}_{counts}")["capability_config"]["team_gen"]["unit_types"][0] != "mutated"


def test_reset_contract_masks_and_no_implicit_reset(factory):
    env = smac.SmacEnv({"scenario": "terran_5v5", "seed": 3})
    assert env.is_discrete and env.num_agents == 5 and env.num_actions == 11
    with pytest.raises(RuntimeError, match="reset"):
        env.step(_legal_actions(env))
    observation, info = env.reset(seed=3)
    assert observation["obs"].shape == (5, 3) and observation["obs"].dtype == np.float32
    assert observation["state"].shape == (4,) and observation["state"].dtype == np.float32
    assert observation["available_actions"].shape == (5, 11)
    assert observation["available_actions"].dtype == bool
    assert observation["active_agents"].tolist() == [True, False, True, True, True]
    assert info["capability_rng_streams"] == 3
    assert factory[0].reset_calls == 1
    # Public arrays neither alias the backend nor corrupt the internal mask.
    observation["available_actions"].fill(False)
    observation["obs"].fill(-99)
    assert not (factory[0].obs == -99).any()
    _, reward, terminated, truncated, _ = env.step(_legal_actions(env))
    assert reward == 1.5 and not terminated and not truncated
    assert factory[0].reset_calls == 1


@pytest.mark.parametrize("timeout", [False, True])
def test_terminal_or_truncation_preserves_final_observation_and_requires_reset(factory, timeout):
    env = smac.SmacEnv({"scenario": "zerg_10v11"})
    env.reset()
    factory[0].next_result = (2, True, {"battle_won": not timeout, "episode_limit": timeout})
    final, reward, terminated, truncated, info = env.step(_legal_actions(env))
    assert terminated is (not timeout) and truncated is timeout
    assert reward == 2.0 and info["battle_won"] is (not timeout)
    assert (final["state"] == 11).all()
    assert factory[0].reset_calls == 1
    with pytest.raises(RuntimeError, match="reset"):
        env.step(_legal_actions(env))
    env.reset()
    assert (final["state"] == 11).all()


@pytest.mark.parametrize("bad_actions", [np.ones(4, dtype=int), np.ones(5), np.full(5, -1), np.full(5, 99), np.zeros(5, dtype=int)])
def test_rejects_bad_or_unavailable_actions_without_stepping(factory, bad_actions):
    env = smac.SmacEnv({})
    env.reset()
    with pytest.raises(ValueError, match="action"):
        env.step(bad_actions)
    assert factory[0].step_calls == 0


def test_fixed_seed_isolates_nested_capability_and_global_random_streams(factory):
    random.seed(56)
    np.random.seed(78)
    expected_python = random.getstate()
    expected_numpy = np.random.get_state()
    first = smac.SmacEnv({"seed": 99})
    second = smac.SmacEnv({"seed": 99})
    third = smac.SmacEnv({"seed": 100})
    obs1, _ = first.reset()
    obs2, _ = second.reset()
    obs3, _ = third.reset()
    np.testing.assert_array_equal(obs1["obs"], obs2["obs"])
    assert not np.array_equal(obs1["obs"], obs3["obs"])
    assert random.getstate() == expected_python
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == expected_numpy[0]
    np.testing.assert_array_equal(actual_numpy[1], expected_numpy[1])
    assert actual_numpy[2:] == expected_numpy[2:]
    step1 = first.step(_legal_actions(first))[0]
    random.random()
    np.random.random(50)
    step2 = second.step(_legal_actions(second))[0]
    np.testing.assert_array_equal(step1["obs"], step2["obs"])
    for env in (first, second, third):
        env.close()


def test_reseed_semantics_are_explicit(factory):
    env = smac.SmacEnv({"seed": 3})
    with pytest.raises(ValueError, match="reseed"):
        env.reset(seed=4)
    env.reset(seed=3)
    with pytest.raises(ValueError, match="reseed"):
        env.reset(seed=3)
    env.reset()
    assert factory[0].reset_calls == 2


def test_restart_invalidates_step_and_reset(factory):
    env = smac.SmacEnv({})
    env.reset()
    factory[0].restart_on_step = True
    with pytest.raises(smac.SmacSimulatorError, match="restarted"):
        env.step(_legal_actions(env))
    with pytest.raises(RuntimeError, match="reset"):
        env.step(_legal_actions(env))
    factory[0].restart_on_reset = True
    with pytest.raises(smac.SmacSimulatorError, match="restarted"):
        env.reset()


def test_native_clock_restart_only_at_reset_preserves_capability_streams(factory):
    env = smac.SmacEnv({"restart_game_loop": 10, "seed": 9})
    control = smac.SmacEnv({"seed": 9})
    assert "restart_game_loop" not in factory[0].config
    env.reset()
    control.reset()
    for index in range(2):
        factory[0].next_result = factory[1].next_result = (1.5, index == 1, {"battle_won": index == 1})
        result = env.step(_legal_actions(env))
        other = control.step(_legal_actions(control))
        np.testing.assert_array_equal(result[0]["obs"], other[0]["obs"])
        assert result[1:4] == other[1:4]
        assert result[4]["planned_restarts"] == 0
    assert factory[0].force_restarts == 0
    observed, info = env.reset()
    expected, _ = control.reset()
    assert info["planned_restart"] and info["planned_restarts"] == 1
    assert info["native_game_loop"] == 1
    np.testing.assert_array_equal(observed["obs"], expected["obs"])
    # A second, unexpected restart during the subsequent reset is still fatal.
    factory[0]._obs.observation.game_loop = 10
    factory[0].restart_on_reset = True
    with pytest.raises(smac.SmacSimulatorError, match="during reset"):
        env.reset()


@pytest.mark.parametrize("threshold", [True, 0, -1, 1.5, 524288])
def test_rejects_invalid_restart_clock(threshold):
    with pytest.raises(ValueError, match="restart_game_loop"):
        smac.smac_config("terran_5v5", restart_game_loop=threshold)


def test_restart_clock_leaves_room_for_episode(factory):
    with pytest.raises(ValueError, match="complete SMAC episode"):
        smac.SmacEnv({"restart_game_loop": 524000})
    assert factory[0].closed


def test_rejects_invalid_backend_observations(factory):
    env = smac.SmacEnv({})
    factory[0].mask[0] = 0
    with pytest.raises(ValueError, match="available action"):
        env.reset()
    factory[0].mask[0, 1] = 1
    factory[0].state = np.zeros(8)
    with pytest.raises(ValueError, match="shapes"):
        env.reset()


def test_close_is_idempotent_and_rejects_future_use(factory, monkeypatch):
    monkeypatch.setenv("SC2PATH", "/external/original")
    env = smac.SmacEnv({"sc2_path": "/explicit/read-only/installation"})
    assert smac.os.environ["SC2PATH"] == "/external/original"
    env.reset()
    assert smac.os.environ["SC2PATH"] == "/external/original"
    env.close()
    env.close()
    assert factory[0].closed
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


@pytest.mark.parametrize("config", [
    {"scenario": "unknown"}, {"seed": -1}, {"continuing_episode": False},
    {"action_mask": False}, {"prob_obs_enemy": 0.5},
])
def test_rejects_unsupported_config_before_simulator_construction(factory, config):
    with pytest.raises(ValueError):
        smac.SmacEnv(config)
    assert not factory
