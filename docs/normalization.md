# Observation normalization

`ObservationNormalizer` maintains running feature statistics from training observations. Local observations use separate statistics for each `(agent, feature)` pair; centralized state features have their own statistics. Rewards, actions, available-action masks and active-agent masks are unchanged. This is running per-feature normalization, distinct from the historical wrapper's normalization of each individual observation vector.

```python
from bta_repro.normalization import ObservationNormalizer, NormalizedEnv

statistics = ObservationNormalizer(env.num_agents, env.obs_dim, env.state_dim)
training_env = NormalizedEnv(env, statistics, training=True)

# Wrap every training environment with the same statistics object.
# Evaluation uses one independent snapshot shared by its evaluation environments.
evaluation_statistics = statistics.frozen_copy()
evaluation_env = NormalizedEnv(eval_env, evaluation_statistics, training=False)
```

`RunningMoments(shape, epsilon=1e-8)` stores a Python integer count and float64 mean/M2 arrays. M2 is the sum of squared deviations. Batch updates and merging independent statistics use the Chan/Welford formulas, avoiding the unstable subtraction of squared means. Leading array dimensions are sample dimensions; trailing dimensions must equal `shape`. Thus `shape=(agents, features)` with input `[batch, agents, features]` pools samples without pooling agents or features.

The variance is the population variance `M2/count`. Normalization computes `(x - mean) / sqrt(variance + epsilon)`, clips to `[-10,10]` by default, and returns float32 arrays. `clip` and `epsilon` are configurable finite positive values. Before any samples, mean is zero and variance is one; there is no artificial initial sample count. After a constant feature is observed, it normalizes to zero, and epsilon stabilizes later deviations. The count includes all observations, including inactive agents' observations; masks are preserved rather than used to filter the statistics.

Training wrappers update once before transforming each `reset()` or `step()` result, including final terminal/truncated observations. They do not reset the underlying environment automatically. The next explicit reset contributes a separate initial observation. Shared training statistics therefore count the observations emitted by every wrapped environment. Transformations use the statistics available at that point in the synchronous environment traversal; already collected rollout observations are not retrospectively transformed when statistics change.

`transform()` never updates statistics and returns a copied dictionary. Observation/state arrays, masks and other nested arrays do not alias the original simulator buffers. Training statistics are committed together only after both local and state updates succeed, preventing a failed input from partially updating their counts. Nonfinite inputs, invalid shapes, inconsistent serialized statistics and arithmetic overflow raise errors.

`frozen_copy()` makes an independent evaluation snapshot. Its public `update()` raises an error. `NormalizedEnv(..., training=False)` uses an already frozen snapshot directly, or takes one automatically if given mutable training statistics. Evaluation never changes training statistics or its own counts. Training wrappers reject a frozen normalizer. Adapter dimensions, config, version metadata and other attributes are delegated; rewards and episode-ending flags pass through unchanged.

## Saving and restoring

`state_dict()` returns plain dictionaries/lists/numbers compatible with JSON and `torch.load(..., weights_only=True)`. Format version 1 stores dimensions, clip, epsilon, frozen status, and two moment dictionaries under `obs` and `state`; each records shape, count, mean, M2 and epsilon. `load_state_dict()` requires matching dimensions, validates both moment sets, and restores the serialized parameters and frozen status atomically.

```python
save_checkpoint(..., extra_state={
    "observation_normalizer": statistics.state_dict(),
})
payload = load_checkpoint(...)
statistics.load_state_dict(payload["extra_state"]["observation_normalizer"])
```

The optional checkpoint `extra_state` field preserves checkpoint format version 1. Older checkpoints load with an empty `extra_state` dictionary. Statistics are sufficient to reproduce preprocessing; exact simulator continuation still requires simulator state, which these checkpoints do not contain.

Tests cover pooled/merged moments, large offsets with small variance, constant features, separate agent statistics, clipping, copied raw buffers and masks, frozen evaluation, restricted checkpoint loading, shared wrapper counts, unchanged rewards and episode boundaries, and rejected malformed state/input.
