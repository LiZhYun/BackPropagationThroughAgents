"""Scalar cumulative ValueNorm and explicit clipped value-regression losses.

This is cumulative population-moment normalization, not the legacy exponential
moving average and not PopArt: updating statistics does not rescale the head.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


DEFAULT_VARIANCE_FLOOR = 1e-2


def validate_value_loss_options(clip_coef=None, huber_delta=None):
    for name, value in (("value_clip_coef", clip_coef), ("value_huber_delta", huber_delta)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be None or finite and positive")


class RunningValueStats(nn.Module):
    """Float64 Chan/Welford moments of scalar training return targets."""

    def __init__(self, variance_floor: float = DEFAULT_VARIANCE_FLOOR):
        super().__init__()
        if isinstance(variance_floor, bool) or not math.isfinite(variance_floor) or variance_floor <= 0:
            raise ValueError("variance_floor must be finite and positive")
        self.register_buffer("count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("updates", torch.zeros((), dtype=torch.int64))
        self.register_buffer("mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("m2", torch.zeros((), dtype=torch.float64))
        self.register_buffer("variance_floor", torch.tensor(variance_floor, dtype=torch.float64))

    def _apply(self, fn, recurse=True):
        # Module.float()/half() may change the network dtype, but must not lose
        # already accumulated moment precision. Device moves still apply.
        original = {name: self._buffers[name] for name in ("mean", "m2", "variance_floor")}
        result = super()._apply(fn, recurse=recurse)
        for name, value in original.items():
            self._buffers[name] = value.to(device=self._buffers[name].device, dtype=torch.float64)
        return result

    @torch.no_grad()
    def update(self, returns: Tensor):
        if not returns.is_floating_point() or returns.numel() == 0 or not torch.isfinite(returns).all():
            raise ValueError("value targets must be nonempty finite floating-point values")
        values = returns.detach().to(device=self.mean.device, dtype=torch.float64).flatten()
        batch_count = values.numel()
        batch_mean = values.mean()
        batch_m2 = (values - batch_mean).square().sum()
        total = self.count + batch_count
        if self.count == 0:
            new_mean, new_m2 = batch_mean, batch_m2
        else:
            delta = batch_mean - self.mean
            total_float = total.double()
            new_mean = self.mean + delta * (batch_count / total_float)
            new_m2 = self.m2 + batch_m2 + delta.square() * (self.count.double() * batch_count / total_float)
        if not torch.isfinite(new_mean) or not torch.isfinite(new_m2):
            raise FloatingPointError("value moments overflowed")
        self.mean.copy_(new_mean)
        self.m2.copy_(new_m2)
        self.count.copy_(total)
        self.updates.add_(1)

    @property
    def scale(self) -> Tensor:
        variance = torch.where(self.count > 0, self.m2 / self.count.clamp_min(1), torch.ones_like(self.m2))
        return variance.clamp_min(self.variance_floor).sqrt()

    def normalize(self, values: Tensor) -> Tensor:
        mean, scale = self.mean.to(values.device), self.scale.to(values.device)
        return ((values.double() - mean) / scale).to(values.dtype)

    def denormalize(self, values: Tensor) -> Tensor:
        mean, scale = self.mean.to(values.device), self.scale.to(values.device)
        return (values.double() * scale + mean).to(values.dtype)


class ValueNormalizedCritic(nn.Module):
    """The public forward is raw-valued; only critic regression uses z-scores."""

    def __init__(self, base: nn.Module, variance_floor: float = DEFAULT_VARIANCE_FLOOR):
        super().__init__()
        self.base = base
        self.normalizer = RunningValueStats(variance_floor)

    def forward(self, state: Tensor) -> Tensor:
        return self.normalizer.denormalize(self.normalized_forward(state))

    def normalized_forward(self, state: Tensor) -> Tensor:
        return self.base(state)

    def update_statistics(self, returns: Tensor):
        if not self.training:
            raise RuntimeError("value statistics cannot update in evaluation mode")
        self.normalizer.update(returns)


def value_regression_loss(
    prediction: Tensor, targets: Tensor, *, old_prediction: Tensor | None = None,
    clip_coef: float | None = None, huber_delta: float | None = None,
) -> Tensor:
    """Mean max(unclipped, clipped) error loss in one consistent value space.

    MSE is the full squared error, preserving the existing runner's coefficient.
    Huber uses 0.5*error² near zero and delta*(abs(error)-0.5*delta) outside.
    """
    validate_value_loss_options(clip_coef, huber_delta)
    if prediction.shape != targets.shape or prediction.numel() == 0:
        raise ValueError("value predictions and targets must have matching nonempty shapes")
    targets = targets.detach()

    def error_loss(value):
        error = value - targets
        if huber_delta is None:
            return error.square()
        absolute = error.abs()
        quadratic = absolute.clamp(max=huber_delta)
        return .5 * quadratic.square() + huber_delta * (absolute - quadratic)

    losses = error_loss(prediction)
    if clip_coef is not None:
        if old_prediction is None or old_prediction.shape != prediction.shape:
            raise ValueError("value clipping requires old predictions matching current predictions")
        old = old_prediction.detach()
        clipped = old + (prediction - old).clamp(-clip_coef, clip_coef)
        losses = torch.maximum(losses, error_loss(clipped))
    return losses.mean()
