"""Training observation moments and immutable evaluation snapshots.

Local features have independent statistics for each agent. Rewards, action
masks and episode boundaries pass through unchanged. No simulator is imported.
"""

from copy import deepcopy
import math

import numpy as np


def _shape(shape) -> tuple[int, ...]:
    if isinstance(shape, (int, np.integer)) and not isinstance(shape, bool):
        shape = (shape,)
    shape = tuple(shape)
    if not shape or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 1 for x in shape):
        raise ValueError("feature shape must contain positive integers")
    return tuple(int(x) for x in shape)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


class RunningMoments:
    """Float64 per-feature mean and summed squared deviations (M2).

    Leading dimensions are samples; trailing dimensions must equal ``shape``.
    ``count`` is an exact integer sample count, with no artificial prior count.
    Variance is the population variance M2/count, or one before any samples.
    """

    def __init__(self, shape, epsilon: float = 1e-8):
        self.shape = _shape(shape)
        self.epsilon = _positive(epsilon, "epsilon")
        self.count = 0
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.M2 = np.zeros(self.shape, dtype=np.float64)

    def _values(self, values):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim < len(self.shape) or array.shape[-len(self.shape):] != self.shape:
            raise ValueError(f"values must have trailing feature shape {self.shape}")
        if not np.isfinite(array).all():
            raise ValueError("moment inputs must be finite")
        return array

    @property
    def variance(self):
        return self.M2 / self.count if self.count else np.ones(self.shape, dtype=np.float64)

    def update(self, values):
        samples = self._values(values).reshape((-1, *self.shape))
        count = len(samples)
        if count == 0:
            return
        mean = samples.mean(axis=0, dtype=np.float64)
        centered = samples - mean
        m2 = np.sum(centered * centered, axis=0, dtype=np.float64)
        self._merge(count, mean, m2)

    def _merge(self, count, mean, m2):
        if count == 0:
            return
        total = self.count + count
        if self.count:
            delta = mean - self.mean
            new_mean = self.mean + delta * (count / total)
            new_m2 = self.M2 + m2 + delta * delta * (self.count * count / total)
        else:
            new_mean, new_m2 = mean.copy(), m2.copy()
        if not np.isfinite(new_mean).all() or not np.isfinite(new_m2).all():
            raise ValueError("moment update overflowed float64; statistics were not changed")
        self.count, self.mean, self.M2 = total, new_mean, np.maximum(new_m2, 0)

    def merge(self, other):
        """Combine independent sample sets without retaining their raw values."""
        if not isinstance(other, RunningMoments) or other.shape != self.shape:
            raise ValueError("merged moments must have the same feature shape")
        self._merge(other.count, other.mean, other.M2)

    def normalize(self, values):
        result = (self._values(values) - self.mean) / np.sqrt(self.variance + self.epsilon)
        if not np.isfinite(result).all():
            raise ValueError("normalization overflowed float64")
        return result

    def state_dict(self):
        # Plain lists keep this JSON serializable and torch weights_only safe.
        return {"format_version": 1, "shape": list(self.shape), "epsilon": self.epsilon,
                "count": self.count, "mean": self.mean.tolist(), "M2": self.M2.tolist()}

    def load_state_dict(self, state):
        if state.get("format_version") != 1 or _shape(state["shape"]) != self.shape:
            raise ValueError("moment state version or feature shape does not match")
        count = state["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("moment count must be a nonnegative integer")
        epsilon = _positive(state["epsilon"], "epsilon")
        mean, m2 = np.asarray(state["mean"], dtype=np.float64), np.asarray(state["M2"], dtype=np.float64)
        if mean.shape != self.shape or m2.shape != self.shape:
            raise ValueError("stored moment arrays have incorrect shapes")
        if not np.isfinite(mean).all() or not np.isfinite(m2).all() or (m2 < 0).any():
            raise ValueError("stored moments must be finite with nonnegative M2")
        if count == 0 and (mean.any() or m2.any()):
            raise ValueError("empty moments must have zero mean and M2")
        self.count, self.epsilon = count, epsilon
        self.mean, self.M2 = mean.copy(), m2.copy()


class ObservationNormalizer:
    """Separate local [agent, feature] and centralized [state feature] moments.

    Updates accept individual observation dictionaries or batched dictionaries
    with matching leading sample dimensions. Transformation never updates stats.
    """

    def __init__(self, num_agents: int, obs_dim: int, state_dim: int, *, clip: float = 10.0, epsilon: float = 1e-8):
        self.num_agents, self.obs_dim, self.state_dim = _shape((num_agents, obs_dim, state_dim))
        self.clip = _positive(clip, "clip")
        self.epsilon = _positive(epsilon, "epsilon")
        self.obs_moments = RunningMoments((self.num_agents, self.obs_dim), epsilon)
        self.state_moments = RunningMoments((self.state_dim,), epsilon)
        self._frozen = False

    @property
    def frozen(self):
        return self._frozen

    def _values(self, observation):
        obs = self.obs_moments._values(observation["obs"])
        state = self.state_moments._values(observation["state"])
        if obs.shape[:-2] != state.shape[:-1]:
            raise ValueError("obs and state must have matching leading sample dimensions")
        return obs, state

    def update(self, observation):
        if self.frozen:
            raise RuntimeError("evaluation normalization is frozen")
        obs, state = self._values(observation)
        # Commit both together, including when finite inputs overflow arithmetic.
        local, central = deepcopy(self.obs_moments), deepcopy(self.state_moments)
        local.update(obs)
        central.update(state)
        self.obs_moments, self.state_moments = local, central

    def transform(self, observation):
        obs, state = self._values(observation)
        result = deepcopy(observation)
        result["obs"] = np.clip(self.obs_moments.normalize(obs), -self.clip, self.clip).astype(np.float32)
        result["state"] = np.clip(self.state_moments.normalize(state), -self.clip, self.clip).astype(np.float32)
        if not np.isfinite(result["obs"]).all() or not np.isfinite(result["state"]).all():
            raise ValueError("normalized observations must fit finite float32")
        return result

    def frozen_copy(self):
        snapshot = deepcopy(self)
        snapshot._frozen = True
        return snapshot

    def state_dict(self):
        return {"format_version": 1, "num_agents": self.num_agents, "obs_dim": self.obs_dim,
                "state_dim": self.state_dim, "clip": self.clip, "epsilon": self.epsilon,
                "frozen": self.frozen, "obs": self.obs_moments.state_dict(),
                "state": self.state_moments.state_dict()}

    def load_state_dict(self, state):
        dimensions = (state["num_agents"], state["obs_dim"], state["state_dim"])
        if state.get("format_version") != 1 or dimensions != (self.num_agents, self.obs_dim, self.state_dim):
            raise ValueError("normalizer state version or dimensions do not match")
        restored = ObservationNormalizer(*dimensions, clip=state["clip"], epsilon=state["epsilon"])
        restored.obs_moments.load_state_dict(state["obs"])
        restored.state_moments.load_state_dict(state["state"])
        if (restored.obs_moments.count != restored.state_moments.count
                or restored.obs_moments.epsilon != restored.epsilon
                or restored.state_moments.epsilon != restored.epsilon):
            raise ValueError("local and state moments must have equal counts and epsilon")
        if not isinstance(state["frozen"], bool):
            raise ValueError("frozen must be boolean")
        restored._frozen = state["frozen"]
        self.__dict__.update(restored.__dict__)


class NormalizedEnv:
    """Wrap an adapter with shared training statistics or a frozen eval snapshot."""

    def __init__(self, env, normalizer: ObservationNormalizer, *, training: bool = True):
        if not isinstance(training, bool):
            raise ValueError("training must be boolean")
        if (env.num_agents, env.obs_dim, env.state_dim) != (normalizer.num_agents, normalizer.obs_dim, normalizer.state_dim):
            raise ValueError("normalizer dimensions differ from the environment")
        if training and normalizer.frozen:
            raise ValueError("training requires an unfrozen normalizer")
        self.env = env
        self.training = training
        self.normalizer = normalizer if training or normalizer.frozen else normalizer.frozen_copy()

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _observation(self, raw):
        if self.training:
            self.normalizer.update(raw)
        return self.normalizer.transform(raw)

    def reset(self, seed=None):
        raw, info = self.env.reset() if seed is None else self.env.reset(seed=seed)
        return self._observation(raw), info

    def step(self, actions):
        raw, reward, terminated, truncated, info = self.env.step(actions)
        return self._observation(raw), reward, terminated, truncated, info

    def close(self):
        return self.env.close()
