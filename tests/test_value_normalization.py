from io import BytesIO

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from bta_repro.value_normalization import RunningValueStats, ValueNormalizedCritic, value_regression_loss


def test_cumulative_moments_are_stable_and_do_not_reweight_old_batches():
    stats = RunningValueStats()
    values = 1e9 + torch.tensor([-3., -1., 1., 3.], dtype=torch.float64)
    stats.update(values[:1])
    stats.update(values[1:])
    assert stats.count == 4 and stats.updates == 2
    torch.testing.assert_close(stats.mean, values.mean(), rtol=0, atol=0)
    torch.testing.assert_close(stats.scale.square(), values.var(unbiased=False))
    torch.testing.assert_close(stats.normalize(values), (values - values.mean()) / values.std(unbiased=False))
    torch.testing.assert_close(stats.denormalize(stats.normalize(values)), values)


def test_empty_statistics_are_identity_and_constant_targets_use_explicit_floor():
    stats = RunningValueStats(variance_floor=.04)
    values = torch.tensor([3., 3., 3.])
    torch.testing.assert_close(stats.normalize(values), values, atol=0, rtol=0)
    stats.update(values)
    assert float(stats.scale) == pytest.approx(.2)
    torch.testing.assert_close(stats.normalize(values), torch.zeros_like(values), atol=0, rtol=0)


def test_module_dtype_conversion_preserves_accumulated_float64_moments():
    stats = RunningValueStats()
    stats.update(torch.tensor([1e9 + .125, 1e9 + .375], dtype=torch.float64))
    before = {key: value.clone() for key, value in stats.state_dict().items()}
    stats.float()
    assert stats.mean.dtype == stats.m2.dtype == stats.variance_floor.dtype == torch.float64
    for key, value in before.items():
        torch.testing.assert_close(stats.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_wrapper_returns_raw_values_and_evaluation_does_not_update_statistics(device):
    base = nn.Linear(1, 1)
    with torch.no_grad():
        base.weight.fill_(2.)
        base.bias.fill_(1.)
    critic = ValueNormalizedCritic(base).to(device)
    critic.update_statistics(torch.tensor([10., 14.], device=device))
    state = torch.tensor([[3.]], device=device)
    torch.testing.assert_close(critic.normalized_forward(state), torch.tensor([[7.]], device=device))
    torch.testing.assert_close(critic(state), torch.tensor([[26.]], device=device))
    before = {key: value.clone() for key, value in critic.normalizer.state_dict().items()}
    critic.eval()
    critic(state)
    critic(state)
    for key, value in before.items():
        torch.testing.assert_close(critic.normalizer.state_dict()[key], value, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="evaluation mode"):
        critic.update_statistics(torch.tensor([1.], device=device))


def test_state_dict_roundtrip_preserves_raw_predictions_and_future_moment_updates():
    critic = ValueNormalizedCritic(nn.Linear(2, 1), variance_floor=.03)
    critic.update_statistics(torch.tensor([10., 20., 40.]))
    state = torch.randn(4, 2)
    archive = BytesIO()
    torch.save(critic.state_dict(), archive)
    archive.seek(0)
    restored = ValueNormalizedCritic(nn.Linear(2, 1))
    restored.load_state_dict(torch.load(archive, weights_only=True))
    torch.testing.assert_close(restored(state), critic(state), atol=0, rtol=0)
    for model in (critic, restored):
        model.update_statistics(torch.tensor([100., 200.]))
    for key, value in critic.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)


def test_value_norm_does_not_apply_popart_output_rescaling():
    critic = ValueNormalizedCritic(nn.Linear(1, 1))
    state = torch.ones(2, 1)
    normalized = critic.normalized_forward(state).detach().clone()
    raw = critic(state).detach().clone()
    critic.update_statistics(torch.tensor([100., 104.]))
    torch.testing.assert_close(critic.normalized_forward(state), normalized, atol=0, rtol=0)
    assert not torch.equal(critic(state), raw)


def test_default_loss_preserves_full_mse_and_detaches_targets():
    prediction = torch.tensor([2., -2.], requires_grad=True)
    targets = torch.tensor([3., -1.], requires_grad=True)
    loss = value_regression_loss(prediction, targets)
    torch.testing.assert_close(loss, (prediction - targets).square().mean(), rtol=0, atol=0)
    loss.backward()
    assert targets.grad is None
    torch.testing.assert_close(prediction.grad, torch.tensor([-1., -1.]))


def test_clipping_uses_maximum_error_and_detaches_old_predictions():
    prediction = torch.tensor([2., -2.], requires_grad=True)
    targets = torch.tensor([3., -1.], requires_grad=True)
    old = torch.zeros(2, requires_grad=True)
    loss = value_regression_loss(prediction, targets, old_prediction=old, clip_coef=.5)
    assert float(loss) == pytest.approx((6.25 + 1) / 2)
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([0., -1.]))
    assert old.grad is None and targets.grad is None


@pytest.mark.parametrize("clip", [None, .2])
def test_huber_and_clipped_huber_match_pytorch_reference(clip):
    prediction = torch.tensor([0., .5, 2., -3.], requires_grad=True)
    target = torch.tensor([0., 0., 0., 0.])
    old = torch.tensor([0., 1., 0., -1.])
    expected = F.huber_loss(prediction, target, reduction="none", delta=1.5)
    if clip is not None:
        clipped = old + (prediction - old).clamp(-clip, clip)
        expected = torch.maximum(expected, F.huber_loss(clipped, target, reduction="none", delta=1.5))
    result = value_regression_loss(prediction, target, old_prediction=old, clip_coef=clip, huber_delta=1.5)
    torch.testing.assert_close(result, expected.mean())


@pytest.mark.parametrize("returns", [torch.tensor([]), torch.tensor([float("nan")]), torch.tensor([1])])
def test_invalid_targets_do_not_mutate_statistics(returns):
    stats = RunningValueStats()
    with pytest.raises(ValueError):
        stats.update(returns)
    assert stats.count == stats.updates == 0


@pytest.mark.parametrize("kwargs", [{"clip_coef": 0}, {"clip_coef": float("nan")},
                                    {"huber_delta": -1}, {"huber_delta": float("inf")},
                                    {"huber_delta": True}])
def test_invalid_loss_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        value_regression_loss(torch.ones(2), torch.ones(2), **kwargs)
