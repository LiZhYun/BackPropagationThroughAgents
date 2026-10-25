"""Tiny physics check, not a learning benchmark. Use docs/mujoco_setup.md exports."""

import json
import random

import numpy as np

from bta_repro.mujoco_env import ACTUATOR_MAPPINGS, MujocoEnv


def main():
    numpy_state, python_state = np.random.get_state(), random.getstate()
    expected_numpy = np.random.RandomState()
    expected_numpy.set_state(numpy_state)
    expected_python = random.Random()
    expected_python.setstate(python_state)
    records = []
    for scenario, partition in ACTUATOR_MAPPINGS:
        config = {"scenario": scenario, "agent_conf": partition, "agent_obsk": 0,
                  "episode_limit": 2, "seed": 123}
        env = MujocoEnv(config)
        try:
            initial, _ = env.reset()
            actions = np.zeros((env.num_agents, env.action_dim), dtype=np.float32)
            first = env.step(actions)
            second = env.step(actions)
            assert not first[2] and not first[3]
            assert not second[2] and second[3]
            replay_initial, _ = env.reset(seed=123)
            replay_first = env.step(actions)
            np.testing.assert_array_equal(initial["state"], replay_initial["state"])
            np.testing.assert_array_equal(first[0]["state"], replay_first[0]["state"])
            assert first[1] == replay_first[1]
            for transition in (first, second, replay_first):
                assert np.isfinite(transition[0]["obs"]).all()
                assert np.isfinite(transition[0]["state"]).all()
                assert np.isfinite(transition[1])
            records.append({"config": env.config, "versions": env.versions,
                            "obs_dim": env.obs_dim, "state_dim": env.state_dim,
                            "num_agents": env.num_agents, "action_dim": env.action_dim,
                            "actuator_indices": env.actuator_indices.tolist(),
                            "zero_action_rewards": [first[1], second[1]],
                            "seed_replay_exact": True, "timeout_step": 2})
        finally:
            env.close()
    assert np.random.random() == expected_numpy.random()
    assert random.random() == expected_python.random()
    print(json.dumps({"check": "physics only; no training", "caller_rng_preserved": True,
                      "environments": records}, indent=2))


if __name__ == "__main__":
    main()
