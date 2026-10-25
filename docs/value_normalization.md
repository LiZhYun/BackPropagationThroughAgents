# Critic target normalization and loss controls

The simulator trainer supports optional scalar ValueNorm, clipped value regression, and Huber loss for every algorithm and action space. Defaults preserve the existing raw-value MSE objective and network initialization:

```json
{
  "normalize_value_targets": false,
  "value_clip_coef": null,
  "value_huber_delta": null,
  "value_coef": 0.5
}
```

Set `normalize_value_targets=true` to train the critic in standardized return units. Set a finite positive `value_clip_coef` or `value_huber_delta` to enable those independent controls. A clipping radius is measured in standardized units when normalization is enabled and raw reward units otherwise. These are optimization controls, not evidence of improved benchmark performance.

## Raw values and training statistics

[`ValueNormalizedCritic`](../src/bta_repro/value_normalization.py) wraps the existing feedforward critic. Its public `forward(state)` returns raw values, `mean + scale * network(state)`. Rollout collection, termination/truncation bootstrapping, GAE, and explained-variance diagnostics therefore use raw reward units. Only critic regression calls `normalized_forward(state)`. Rewards and stored actor advantages are never standardized by ValueNorm; the existing optional advantage normalization remains separate.

Before critic epochs, the wrapper updates its scalar moments once using all detached raw return targets in that training rollout. Each target contributes once, independently of the number of optimization epochs. Statistics cover the shared team critic, across training environments and rollouts. Evaluation and inference never update them. Updates still occur when `value_coef=0`, although the critic optimizer is skipped; disabling the loss does not freeze an enabled normalizer.

The estimator accumulates the sample count, mean, and centered squared deviations using float64 Chan/Welford merging. Its scale is `sqrt(max(M2 / count, 0.01))`, with population variance and an explicit variance floor. Before the first update, the default transform has mean zero and scale one. Statistics remain float64 when the network is converted to a different floating dtype, and follow device moves. The count and update counter are integer buffers.

This is simple ValueNorm without PopArt output rescaling. Updating moments changes the critic's raw predictions even before its network weights change. Old rollout values remain the values that actually produced the rollout, rather than being recomputed after the statistics update.

## Clipping and Huber loss

After updating moments, transform both raw return targets and detached old raw rollout predictions with the **current** mean and scale. Let these be `target` and `old`, and let `prediction` be the current network output in the same units. With clipping radius `c`, compute:

```text
clipped = old + clamp(prediction - old, -c, c)
loss = mean(max(error_loss(prediction - target),
                error_loss(clipped - target)))
```

When normalization is disabled, all three quantities are already raw values. Without clipping, only the first error loss is used. Targets and old predictions are detached; gradients flow through the current prediction.

The default error loss is the full squared error, preserving the previous `value_coef * MSE` scaling. With Huber threshold `d`, it is `0.5 * error²` for `abs(error) <= d` and `d * (abs(error) - 0.5 * d)` otherwise. Consequently, enabling Huber also changes the quadratic-region coefficient by a factor of two relative to full MSE. The final mean is multiplied by `value_coef`.

## Checkpoints and diagnostics

The critic state dictionary contains both the network and normalizer buffers. Restore a matching critic before loading it:

```python
from bta_repro.simulator_train import Config, make_critic

config = Config(**checkpoint["config"])
critic = make_critic(config, env)
critic.load_state_dict(checkpoint["critic"])
critic.eval()
```

Observation-normalizer state remains separate in the checkpoint's `extra_state`. Policy-only deployment does not need the critic or its target moments.

Update metrics distinguish `value_regression_loss` in the selected loss units, its weighted `value_loss`, and `raw_value_mse` (`value_mse` is the same raw metric for compatibility). These are averaged over critic epochs before each optimizer step. `value_target_mean`, `value_target_scale`, `value_target_count`, and `value_normalizer_updates` describe the current moments; raw critics report the identity transform and zero counts. The existing `critic_diagnostics` remain based on raw rollout predictions before optimization.

## Difference from the legacy implementation

The audited legacy `bta/utils/valuenorm.py` at commit `13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0` uses a debiased exponential moving average of first and second moments with `beta=0.99999`, `epsilon=1e-5`, and a variance floor of `0.01`. Its training loop updates moments in each minibatch of each epoch, with a separate normalizer per agent critic. This redesign uses stable cumulative population moments, one update per rollout, and one shared team critic. It does not claim identical legacy optimization trajectories. PopArt, reward normalization, minibatch updates, and recurrent critics are separate changes and are not provided by these controls.
