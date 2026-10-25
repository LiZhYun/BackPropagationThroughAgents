from copy import deepcopy
import io
import json

import numpy as np
import pytest
import torch

from bta_repro.normalization import NormalizedEnv, ObservationNormalizer, RunningMoments


def observation(value=0.0):
    return {"obs": np.array([[value, value+2], [value+100, value+300]], dtype=np.float64),
            "state": np.array([value-7, value+30, value+50], dtype=np.float64),
            "active_agents": np.array([True, False]),
            "available_actions": np.array([[True, False], [False, True]]),
            "extra": {"raw": np.array([value])}}


def test_pooled_batches_and_merged_moments_match_direct_population_statistics():
    samples = np.random.default_rng(7).normal(size=(73, 3, 5)) * np.arange(1, 6) + 1000
    online = RunningMoments((3, 5))
    for block in np.array_split(samples, 9):
        online.update(block)
    first, second = RunningMoments((3, 5)), RunningMoments((3, 5))
    first.update(samples[:21])
    second.update(samples[21:])
    first.merge(second)
    for moments in (online, first):
        assert moments.count == len(samples)
        assert moments.mean.dtype == moments.M2.dtype == np.float64
        np.testing.assert_allclose(moments.mean, samples.mean(0), rtol=1e-14, atol=1e-12)
        np.testing.assert_allclose(moments.variance, samples.var(0), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(moments.M2, samples.var(0)*len(samples), rtol=1e-12)


def test_large_offset_small_variance_is_stable():
    data = 1e9 + np.arange(-20, 21, dtype=np.float64).reshape(-1, 1) / 8
    moments = RunningMoments(1)
    for row in data:
        moments.update(row)
    np.testing.assert_allclose(moments.mean, data.mean(0), rtol=0, atol=1e-7)
    np.testing.assert_allclose(moments.variance, data.var(0), rtol=0, atol=1e-7)


def test_empty_single_and_constant_samples_have_defined_finite_behavior():
    moments = RunningMoments((2,), epsilon=1e-6)
    moments.update(np.empty((0, 2)))
    assert moments.count == 0
    np.testing.assert_array_equal(moments.variance, [1, 1])
    moments.update([4, -9])
    assert moments.count == 1
    np.testing.assert_array_equal(moments.normalize([4, -9]), [0, 0])
    moments.update(np.tile([4, -9], (11, 1)))
    np.testing.assert_array_equal(moments.variance, [0, 0])
    np.testing.assert_allclose(moments.normalize([5, -8]), [1000, 1000])


def test_local_statistics_are_per_agent_and_features_are_not_mixed():
    normalizer = ObservationNormalizer(2, 2, 3)
    values = [observation(-2), observation(2)]
    batch = {key: np.stack([entry[key] for entry in values]) for key in ("obs", "state")}
    normalizer.update(batch)
    assert normalizer.obs_moments.count == normalizer.state_moments.count == 2
    np.testing.assert_array_equal(normalizer.obs_moments.mean, [[0, 2], [100, 300]])
    np.testing.assert_array_equal(normalizer.obs_moments.variance, np.full((2, 2), 4))
    transformed = normalizer.transform(observation(2))
    np.testing.assert_allclose(transformed["obs"], np.ones((2, 2)), rtol=1e-6)
    np.testing.assert_allclose(transformed["state"], np.ones(3), rtol=1e-6)


def test_transform_clips_copies_every_array_and_never_changes_statistics():
    normalizer = ObservationNormalizer(2, 2, 3, clip=3)
    normalizer.update(observation())
    before = normalizer.state_dict()
    raw = observation(10)
    original = deepcopy(raw)
    result = normalizer.transform(raw)
    assert result["obs"].dtype == result["state"].dtype == np.float32
    np.testing.assert_array_equal(result["obs"], np.full((2, 2), 3))
    assert normalizer.state_dict() == before
    for key in ("obs", "state", "active_agents", "available_actions"):
        np.testing.assert_array_equal(raw[key], original[key])
        assert not np.shares_memory(raw[key], result[key])
    np.testing.assert_array_equal(result["available_actions"], raw["available_actions"])
    np.testing.assert_array_equal(result["active_agents"], raw["active_agents"])
    result["extra"]["raw"][0] = -999
    assert raw["extra"]["raw"][0] == 10


def test_frozen_snapshot_is_independent_and_rejects_updates():
    normalizer = ObservationNormalizer(2, 2, 3)
    normalizer.update(observation(-1))
    normalizer.update(observation(1))
    frozen = normalizer.frozen_copy()
    expected = frozen.transform(observation(2))["obs"]
    saved = frozen.state_dict()
    normalizer.update(observation(100))
    np.testing.assert_array_equal(frozen.transform(observation(2))["obs"], expected)
    assert frozen.state_dict() == saved
    with pytest.raises(RuntimeError, match="frozen"):
        frozen.update(observation(3))
    assert not np.shares_memory(frozen.obs_moments.mean, normalizer.obs_moments.mean)


def test_json_and_weights_only_state_roundtrip_and_continuation():
    normalizer = ObservationNormalizer(2, 2, 3, epsilon=1e-5, clip=4)
    normalizer.update(observation(-2))
    normalizer.update(observation(3))
    payload = json.loads(json.dumps(normalizer.state_dict()))
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)
    resumed = ObservationNormalizer(2, 2, 3)
    resumed.load_state_dict(loaded)
    assert resumed.state_dict() == normalizer.state_dict()
    for entry in (observation(7), observation(-4)):
        normalizer.update(entry)
        resumed.update(entry)
    assert resumed.state_dict() == normalizer.state_dict()
    frozen = resumed.frozen_copy()
    resumed.load_state_dict(frozen.state_dict())
    assert resumed.frozen


@pytest.mark.parametrize("failure", ["shape", "nan", "negative_m2", "count_mismatch", "epsilon_mismatch"])
def test_bad_serialized_state_is_rejected_without_partial_restore(failure):
    normalizer = ObservationNormalizer(2, 2, 3)
    normalizer.update(observation(1))
    original = normalizer.state_dict()
    damaged = deepcopy(original)
    if failure == "shape":
        damaged["obs"]["shape"] = [2, 3]
    elif failure == "nan":
        damaged["state"]["mean"][0] = float("nan")
    elif failure == "negative_m2":
        damaged["obs"]["M2"][0][0] = -1
    elif failure == "count_mismatch":
        damaged["state"]["count"] += 1
    else:
        damaged["obs"]["epsilon"] = 1e-4
    with pytest.raises(ValueError):
        normalizer.load_state_dict(damaged)
    assert normalizer.state_dict() == original


def test_invalid_input_or_overflow_does_not_partially_update_statistics():
    normalizer = ObservationNormalizer(2, 2, 3)
    original = normalizer.state_dict()
    bad = observation(1)
    bad["state"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        normalizer.update(bad)
    assert normalizer.state_dict() == original
    huge = {"obs": np.zeros((2, 2, 2)), "state": np.array([[1e300]*3, [-1e300]*3])}
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="overflow"):
            normalizer.update(huge)
    assert normalizer.state_dict() == original


class FakeEnv:
    is_discrete = False
    num_agents, obs_dim, state_dim, action_dim = 2, 2, 3, 1
    config = {"scenario": "fake"}
    versions = {"test": True}

    def __init__(self, initial=0):
        self.initial = initial
        self.raw = observation(initial)
        self.closed = False

    def reset(self, seed=None):
        self.last_seed = seed
        self.raw = observation(self.initial)
        return self.raw, {"seed": seed}

    def step(self, actions):
        self.last_actions = actions
        self.raw["obs"] += 1
        self.raw["state"] += 1
        return self.raw, 123.5, False, True, {"episode_limit": True}

    def close(self):
        self.closed = True


def test_wrapper_delegates_updates_terminal_once_and_preserves_raw_observations():
    base = FakeEnv()
    normalizer = ObservationNormalizer(2, 2, 3)
    env = NormalizedEnv(base, normalizer)
    first, info = env.reset(seed=7)
    assert info == {"seed": 7} and base.last_seed == 7
    assert env.config is base.config and env.versions is base.versions
    assert (env.num_agents, env.obs_dim, env.state_dim, env.action_dim, env.is_discrete) == (2, 2, 3, 1, False)
    actions = np.zeros((2, 1))
    final, reward, terminated, truncated, info = env.step(actions)
    assert base.last_actions is actions
    assert reward == 123.5 and not terminated and truncated and info == {"episode_limit": True}
    assert normalizer.obs_moments.count == normalizer.state_moments.count == 2
    np.testing.assert_array_equal(base.raw["obs"], observation(1)["obs"])
    np.testing.assert_array_equal(first["obs"], np.zeros((2, 2)))
    np.testing.assert_allclose(final["obs"], np.ones((2, 2)), rtol=1e-6)
    env.reset()
    assert normalizer.obs_moments.count == 3
    env.close()
    assert base.closed


def test_shared_train_statistics_and_frozen_evaluation_never_leak():
    shared = ObservationNormalizer(2, 2, 3)
    first, second = NormalizedEnv(FakeEnv(-1), shared), NormalizedEnv(FakeEnv(1), shared)
    first.reset()
    second.reset()
    assert shared.obs_moments.count == 2
    snapshot = shared.frozen_copy()
    evaluation = NormalizedEnv(FakeEnv(100), snapshot, training=False)
    assert evaluation.normalizer is snapshot
    before = snapshot.state_dict()
    evaluation.reset()
    evaluation.step(np.zeros((2, 1)))
    assert snapshot.state_dict() == before and shared.obs_moments.count == 2
    first.step(np.zeros((2, 1)))
    assert shared.obs_moments.count == 3 and snapshot.state_dict() == before
    implicit_snapshot = NormalizedEnv(FakeEnv(), shared, training=False)
    assert implicit_snapshot.normalizer.frozen and implicit_snapshot.normalizer is not shared


def test_wrapper_rejects_frozen_training_or_wrong_dimensions():
    normalizer = ObservationNormalizer(2, 2, 3)
    with pytest.raises(ValueError, match="unfrozen"):
        NormalizedEnv(FakeEnv(), normalizer.frozen_copy())
    with pytest.raises(ValueError, match="dimensions"):
        NormalizedEnv(FakeEnv(), ObservationNormalizer(3, 2, 3))
