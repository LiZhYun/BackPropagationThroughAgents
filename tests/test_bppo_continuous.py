"""Density, causality, and peer-gradient checks for Gaussian BPPO."""

from copy import deepcopy
import math

import pytest
import torch
from torch import nn
from torch.distributions import Independent, Normal

from bta_repro.bppo import _actor_objective, _successor_feedback
from bta_repro.bppo_continuous import ContinuousAutoregressivePolicy, continuous_bppo_update


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(211)


def make_batch(agents=2, dimensions=3, batch=9):
    policy = ContinuousAutoregressivePolicy(2, agents, dimensions, hidden_dim=8)
    obs = torch.randn(batch, agents, 2)
    actions, log_probs = policy.sample(obs)
    return policy, obs, actions, log_probs


@pytest.mark.parametrize("deterministic", [False, True])
def test_sample_and_evaluation_likelihoods_agree(deterministic):
    policy, obs, _, _ = make_batch()
    actions, sampled_log_probs = policy.sample(obs, deterministic)
    log_probs, entropy = policy.evaluate_actions(obs, actions)
    assert actions.shape == (9, 2, 3)
    assert log_probs.shape == entropy.shape == (9, 2)
    torch.testing.assert_close(sampled_log_probs, log_probs)
    assert torch.isfinite(entropy).all()


def test_gaussian_event_log_probability_and_entropy_sum_coordinates():
    policy, obs, actions, _ = make_batch(dimensions=4)
    log_probs, entropy = policy.evaluate_actions(obs, actions)
    for i in range(2):
        dist = policy._distribution(obs, actions, i)
        base = dist.base_dist
        coordinate_log_probs = (
            -0.5 * ((actions[:, i] - base.loc) / base.scale).square()
            - base.scale.log()
            - 0.5 * math.log(2 * math.pi)
        )
        torch.testing.assert_close(log_probs[:, i], coordinate_log_probs.sum(-1))
        torch.testing.assert_close(entropy[:, i], base.entropy().sum(-1))
    old_log_probs = log_probs.detach()
    with torch.no_grad():
        policy.actors[0].mean[-1].bias.add_(0.2)
    changed = policy._distribution(obs, actions, 0)
    # Reconstruct the original actor density before its constant mean shift.
    original_mean = changed.base_dist.loc - 0.2
    original = Normal(original_mean, changed.base_dist.scale)
    per_coordinate_delta = changed.base_dist.log_prob(actions[:, 0]) - original.log_prob(actions[:, 0])
    joint_ratio = (changed.log_prob(actions[:, 0]) - old_log_probs[:, 0]).exp()
    torch.testing.assert_close(joint_ratio, per_coordinate_delta.exp().prod(-1))


def test_raw_actions_are_neither_clipped_nor_squashed():
    policy, obs, _, _ = make_batch(dimensions=1)
    with torch.no_grad():
        for actor in policy.actors:
            actor.mean[-1].weight.zero_()
            actor.mean[-1].bias.fill_(3.0)
    actions, old_log_probs = policy.sample(obs, deterministic=True)
    torch.testing.assert_close(actions, torch.full_like(actions, 3.0))
    raw_log_probs, _ = policy.evaluate_actions(obs, actions)
    clipped_log_probs, _ = policy.evaluate_actions(obs, actions.clamp(-1, 1))
    torch.testing.assert_close(raw_log_probs, old_log_probs)
    assert not torch.allclose(clipped_log_probs, old_log_probs)


def test_standard_deviation_remains_positive_and_bounded():
    policy, obs, actions, _ = make_batch(dimensions=2)
    with torch.no_grad():
        policy.actors[0].log_std.copy_(torch.tensor([-100.0, 100.0]))
    std = policy._distribution(obs, actions, 0).base_dist.scale
    expected = torch.tensor([-5.0, 2.0]).exp().expand_as(std)
    torch.testing.assert_close(std, expected)
    assert (std > 0).all() and torch.isfinite(std).all()


def test_future_actions_cannot_change_previous_conditionals():
    policy, obs, actions, _ = make_batch(agents=3)
    log_probs, _ = policy.evaluate_actions(obs, actions)
    changed = actions.clone()
    changed[:, 2] += 2.0
    changed_obs = obs.clone()
    changed_obs[:, 2] += 10.0
    updated_log_probs, _ = policy.evaluate_actions(changed_obs, changed)
    torch.testing.assert_close(log_probs[:, :2], updated_log_probs[:, :2])
    log_probs[:, 2].sum().backward()
    for actor in policy.actors[:2]:
        assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in actor.parameters())


class AffineGaussian(nn.Module):
    """One-dimensional conditional normal for an analytic derivative oracle."""

    def __init__(self, slope, bias, std):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(float(slope), dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor(float(bias), dtype=torch.float64))
        self.log_std = nn.Parameter(torch.tensor(math.log(std), dtype=torch.float64))

    def forward(self, inputs):
        mean = self.slope * inputs[:, -1:] + self.bias
        return Independent(Normal(mean, self.log_std.exp().expand_as(mean)), 1)


def test_successor_ratio_derivative_matches_analytic_gaussian_formula():
    policy = ContinuousAutoregressivePolicy(2, 2, 1, hidden_dim=8).double()
    behavior = deepcopy(policy).requires_grad_(False)
    policy.actors[1] = AffineGaussian(0.7, 0.2, 1.1)
    behavior.actors[1] = AffineGaussian(-0.3, -0.1, 0.9).requires_grad_(False)
    obs = torch.zeros(3, 2, 2, dtype=torch.float64)
    actions = torch.tensor([[[-0.3], [0.4]], [[0.1], [-0.2]], [[0.5], [0.3]]], dtype=torch.float64)
    multiplier, coefficients = _successor_feedback(policy, behavior, obs, actions, actions, 0)
    x, y = actions[:, 0, 0], actions[:, 1, 0]
    mu, old_mu = 0.7 * x + 0.2, -0.3 * x - 0.1
    expected_ratio = (Normal(mu, 1.1).log_prob(y) - Normal(old_mu, 0.9).log_prob(y)).exp()
    expected_derivative = expected_ratio * (
        (y - mu) * 0.7 / 1.1**2 - (y - old_mu) * (-0.3) / 0.9**2
    )
    torch.testing.assert_close(multiplier, expected_ratio)
    torch.testing.assert_close(coefficients[:, 0], expected_derivative)
    assert not multiplier.requires_grad and not coefficients.requires_grad
    assert all(p.grad is None for p in behavior.parameters())


def test_multisuccessor_ratio_derivative_matches_finite_differences():
    policy, obs, actions, _ = make_batch(agents=3, dimensions=2, batch=4)
    policy, obs, actions = policy.double(), obs.double(), actions.double()
    behavior = deepcopy(policy).requires_grad_(False)
    before_ratio, before_derivative = _successor_feedback(policy, behavior, obs, actions, actions, 0)
    torch.testing.assert_close(before_ratio, torch.ones_like(before_ratio))
    torch.testing.assert_close(before_derivative, torch.zeros_like(before_derivative), atol=1e-12, rtol=0)
    with torch.no_grad():
        for actor in policy.actors[1:]:
            actor.mean[-1].weight.add_(torch.randn_like(actor.mean[-1].weight) * 0.2)
    _, coefficients = _successor_feedback(policy, behavior, obs, actions, actions, 0)

    def ratio(probed):
        log_ratio = torch.zeros(len(obs), dtype=obs.dtype)
        for j in (1, 2):
            log_ratio += policy._distribution(obs, probed, j).log_prob(actions[:, j])
            log_ratio -= behavior._distribution(obs, probed, j).log_prob(actions[:, j])
        return log_ratio.exp()

    numerical = torch.zeros_like(coefficients)
    eps = 1e-5
    with torch.no_grad():
        for d in range(2):
            plus, minus = actions.clone(), actions.clone()
            plus[:, 0, d] += eps
            minus[:, 0, d] -= eps
            numerical[:, d] = (ratio(plus) - ratio(minus)) / (2 * eps)
    torch.testing.assert_close(coefficients, numerical, atol=1e-8, rtol=1e-5)


def test_gaussian_peer_path_reaches_only_current_actor():
    policy, obs, actions, _ = make_batch(dimensions=2)
    behavior = deepcopy(policy).requires_grad_(False)
    with torch.no_grad():
        policy.actors[1].mean[-1].weight.add_(torch.randn_like(policy.actors[1].mean[-1].weight))
    multiplier, coefficients = _successor_feedback(policy, behavior, obs, actions, actions, 0)
    relaxed = policy._distribution(obs, actions, 0).rsample()
    loss, _ = _actor_objective(
        torch.zeros(len(obs), requires_grad=True), torch.zeros(len(obs)), torch.zeros(len(obs)),
        torch.ones(len(obs)), multiplier, coefficients, relaxed,
        clip_coef=0.2, entropy_coef=0,
    )
    loss.backward()
    assert sum(p.grad.abs().sum() for p in policy.actors[0].parameters()) > 0
    assert all(p.grad is None for p in policy.actors[1].parameters())
    assert all(p.grad is None for p in behavior.parameters())


def test_single_agent_matches_gaussian_ppo():
    policy, obs, actions, old_log_probs = make_batch(agents=1, dimensions=3)
    reference = deepcopy(policy)
    advantages = torch.linspace(-1, 1, len(obs))
    optimizer = torch.optim.SGD(policy.actors[0].parameters(), lr=0.03)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
    metrics = continuous_bppo_update(
        policy, [optimizer], obs, actions, old_log_probs, advantages,
        epochs=3, entropy_coef=0.02, max_grad_norm=1e9,
    )
    for _ in range(3):
        log_probs, entropy = reference.evaluate_actions(obs, actions)
        ratio = (log_probs[:, 0] - old_log_probs[:, 0]).exp()
        loss = -torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages).mean()
        loss -= 0.02 * entropy.mean()
        reference_optimizer.zero_grad()
        loss.backward()
        reference_optimizer.step()
    for actual, expected in zip(policy.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)
    assert metrics["peer_loss"] == metrics["peer_coefficient_norm"] == 0


def test_peer_dot_sums_coordinates_and_own_loss_is_scalar():
    def objective(dimensions, batch=2):
        return _actor_objective(
            torch.zeros(batch, requires_grad=True), torch.ones(batch), torch.zeros(batch),
            torch.ones(batch), torch.ones(batch), torch.ones(batch, dimensions),
            torch.ones(batch, dimensions, requires_grad=True), clip_coef=0.2, entropy_coef=0,
        )

    _, scalar = objective(1)
    _, vector = objective(4)
    _, repeated = objective(4, batch=8)
    torch.testing.assert_close(scalar["own_loss"], vector["own_loss"])
    torch.testing.assert_close(vector["peer_loss"], 4 * scalar["peer_loss"])
    for key in vector:
        torch.testing.assert_close(vector[key], repeated[key])


def test_two_agent_update_is_finite_and_reverse_ordered():
    policy, obs, actions, old_log_probs = make_batch()
    calls = []

    class RecordingSGD(torch.optim.SGD):
        def __init__(self, actor, index):
            super().__init__(actor.parameters(), lr=0.01)
            self.index = index

        def step(self, closure=None):
            calls.append(self.index)
            return super().step(closure)

    optimizers = [RecordingSGD(actor, i) for i, actor in enumerate(policy.actors)]
    metrics = continuous_bppo_update(
        policy, optimizers, obs, actions, old_log_probs, torch.linspace(-1, 1, len(obs)), epochs=2
    )
    assert calls == [1, 1, 0, 0]
    assert metrics["peer_coefficient_norm"] > 0
    assert metrics["num_updates"] == 4
    assert all(math.isfinite(value) for value in metrics.values())
    assert all(torch.isfinite(p).all() for p in policy.parameters())


def test_stale_behavior_and_clipped_rollout_actions_are_rejected():
    policy, obs, actions, old_log_probs = make_batch()
    optimizers = [torch.optim.SGD(actor.parameters(), lr=0.1) for actor in policy.actors]
    assert (actions.abs() > 1).any()
    with pytest.raises(ValueError, match="Policy changed since rollout"):
        continuous_bppo_update(
            policy, optimizers, obs, actions.clamp(-1, 1), old_log_probs, torch.ones(len(obs))
        )


def test_empty_or_nonfinite_raw_action_inputs_fail_explicitly():
    policy, obs, actions, _ = make_batch()
    with pytest.raises(ValueError, match="batch must not be empty"):
        policy.sample(obs[:0])
    actions[0, 0, 0] = torch.inf
    with pytest.raises(ValueError, match="finite floating-point raw actions"):
        policy.evaluate_actions(obs, actions)
