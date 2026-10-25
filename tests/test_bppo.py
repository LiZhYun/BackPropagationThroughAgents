"""Behavioral and gradient checks for the discrete BPPO implementation."""

from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from bta_repro.bppo import (
    AutoregressivePolicy,
    _actor_objective,
    _successor_feedback,
    bppo_update,
)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(173)


def make_batch(agents=3, actions=4, batch=11, action_embed_dim=0):
    policy = AutoregressivePolicy(2, agents, actions, hidden_dim=8, action_embed_dim=action_embed_dim)
    obs = torch.randn(batch, agents, 2)
    chosen, log_probs = policy.sample(obs)
    return policy, obs, chosen, log_probs


@pytest.mark.parametrize("deterministic", [False, True])
def test_sampling_and_teacher_forced_likelihood_agree(deterministic):
    policy, obs, _, _ = make_batch()
    available = torch.ones(11, 3, 4, dtype=torch.bool)
    available[:, 1, 0] = False
    available[0, 2] = torch.tensor([False, False, True, False])
    actions, sampled_log_probs = policy.sample(obs, available, deterministic)
    evaluated_log_probs, entropy = policy.evaluate_actions(obs, actions, available)
    torch.testing.assert_close(sampled_log_probs, evaluated_log_probs)
    assert available.gather(-1, actions.unsqueeze(-1)).all()
    assert torch.isfinite(entropy).all()
    assert entropy[0, 2] == 0


def test_future_actions_and_observations_cannot_change_previous_conditionals():
    policy, obs, actions, _ = make_batch()
    log_probs, _ = policy.evaluate_actions(obs, actions)
    changed_actions = actions.clone()
    changed_actions[:, 2] = (changed_actions[:, 2] + 1) % 4
    changed_obs = obs.clone()
    changed_obs[:, 2] += 100
    changed_log_probs, _ = policy.evaluate_actions(changed_obs, changed_actions)
    torch.testing.assert_close(log_probs[:, :2], changed_log_probs[:, :2])
    policy.zero_grad(set_to_none=True)
    log_probs[:, 2].sum().backward()
    # Stacking all conditionals can materialize zero gradients on unused actors.
    for actor in policy.actors[:2]:
        assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in actor.parameters())


def test_behavior_denominator_is_differentiated_with_respect_to_action():
    policy, obs, actions, _ = make_batch(agents=2)
    policy = policy.double()
    obs = obs.double()
    behavior = deepcopy(policy).requires_grad_(False)
    features = F.one_hot(actions, 4).to(obs.dtype)
    multiplier, coefficients = _successor_feedback(policy, behavior, obs, actions, features, 0)
    torch.testing.assert_close(multiplier, torch.ones_like(multiplier))
    torch.testing.assert_close(coefficients, torch.zeros_like(coefficients), atol=1e-12, rtol=0)

    with torch.no_grad():
        policy.actors[1][-1].weight.add_(torch.randn_like(policy.actors[1][-1].weight) * 0.8)
    multiplier, coefficients = _successor_feedback(policy, behavior, obs, actions, features, 0)
    assert coefficients.abs().sum() > 0
    assert not multiplier.requires_grad
    assert not coefficients.requires_grad

    def ratio_at(probed_features):
        current = policy._distribution(obs, probed_features, 1, None).log_prob(actions[:, 1])
        old = behavior._distribution(obs, probed_features, 1, None).log_prob(actions[:, 1])
        return (current - old).exp()

    eps = 1e-5
    numerical = torch.zeros_like(coefficients)
    with torch.no_grad():
        for coordinate in range(4):
            plus, minus = features.clone(), features.clone()
            plus[:, 0, coordinate] += eps
            minus[:, 0, coordinate] -= eps
            numerical[:, coordinate] = (ratio_at(plus) - ratio_at(minus)) / (2 * eps)
    torch.testing.assert_close(coefficients, numerical, atol=1e-8, rtol=1e-6)


def test_peer_gradient_reaches_only_current_actor():
    policy, obs, actions, _ = make_batch(agents=2)
    behavior = deepcopy(policy).requires_grad_(False)
    features = F.one_hot(actions, 4).to(obs.dtype)
    with torch.no_grad():
        policy.actors[1][-1].weight.add_(torch.randn_like(policy.actors[1][-1].weight))
    multiplier, coefficients = _successor_feedback(policy, behavior, obs, actions, features, 0)
    dist = policy._distribution(obs, features, 0, None)
    relaxed = F.gumbel_softmax(dist.logits, tau=1.0, hard=True)
    # A standalone scalar log-probability leaf removes the own-policy pathway.
    log_probs = torch.zeros(len(obs), requires_grad=True)
    loss, _ = _actor_objective(
        log_probs,
        torch.zeros(len(obs)),
        torch.zeros(len(obs)),
        torch.ones(len(obs)),
        multiplier,
        coefficients,
        relaxed,
        clip_coef=0.2,
        entropy_coef=0,
    )
    loss.backward()
    assert sum(p.grad.abs().sum() for p in policy.actors[0].parameters()) > 0
    assert all(p.grad is None for p in policy.actors[1].parameters())
    assert all(p.grad is None for p in behavior.parameters())


def test_single_agent_reduces_to_standard_ppo():
    policy, obs, actions, old_log_probs = make_batch(agents=1)
    reference = deepcopy(policy)
    advantages = torch.linspace(-1, 1, len(obs))
    optimizer = torch.optim.SGD(policy.actors[0].parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    result = bppo_update(
        policy, [optimizer], obs, actions, old_log_probs, advantages,
        epochs=3, entropy_coef=0.03, max_grad_norm=1e9,
    )
    for _ in range(3):
        log_probs, entropy = reference.evaluate_actions(obs, actions)
        ratio = (log_probs[:, 0] - old_log_probs[:, 0]).exp()
        loss = -torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages).mean()
        loss = loss - 0.03 * entropy.mean()
        reference_optimizer.zero_grad()
        loss.backward()
        reference_optimizer.step()
    for actual, expected in zip(policy.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)
    assert result["peer_loss"] == 0
    assert result["peer_coefficient_norm"] == 0
    assert result["num_updates"] == 3


def objective_and_gradients(extra_coordinates=0, repeats=1):
    log_probs = torch.tensor([0.1, -0.2], requires_grad=True)
    relaxed_action = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]], requires_grad=True)
    coefficients = torch.tensor([[2.0, -1.0, 3.0], [4.0, 0.5, -2.0]])
    padded_relaxed = F.pad(relaxed_action, (0, extra_coordinates))
    padded_coefficients = F.pad(coefficients, (0, extra_coordinates))
    loss, metrics = _actor_objective(
        log_probs.repeat(repeats),
        torch.tensor([0.8, 0.7]).repeat(repeats),
        torch.tensor([-0.3, 0.2]).repeat(repeats),
        torch.tensor([-0.4, 0.9]).repeat(repeats),
        torch.tensor([1.2, 0.8]).repeat(repeats),
        padded_coefficients.repeat(repeats, 1),
        padded_relaxed.repeat(repeats, 1),
        clip_coef=0.2,
        entropy_coef=0.01,
    )
    gradients = torch.autograd.grad(loss, (log_probs, relaxed_action))
    return loss, metrics, gradients


@pytest.mark.parametrize("extra_coordinates,repeats", [(7, 1), (0, 5), (7, 5)])
def test_peer_dot_is_summed_over_actions_and_averaged_only_over_batch(extra_coordinates, repeats):
    reference_loss, reference_metrics, reference_gradients = objective_and_gradients()
    loss, metrics, gradients = objective_and_gradients(extra_coordinates, repeats)
    torch.testing.assert_close(loss, reference_loss)
    for key in metrics:
        torch.testing.assert_close(metrics[key], reference_metrics[key])
    for actual, expected in zip(gradients, reference_gradients):
        torch.testing.assert_close(actual, expected)


def test_updates_complete_actor_epochs_in_reverse_order():
    policy, obs, actions, old_log_probs = make_batch()
    calls = []

    class RecordingSGD(torch.optim.SGD):
        def __init__(self, actor, index):
            super().__init__(actor.parameters(), lr=0.03)
            self.index = index

        def step(self, closure=None):
            calls.append(self.index)
            return super().step(closure)

    optimizers = [RecordingSGD(actor, i) for i, actor in enumerate(policy.actors)]
    metrics = bppo_update(
        policy, optimizers, obs, actions, old_log_probs, torch.linspace(-1, 1, len(obs)), epochs=2
    )
    assert calls == [2, 2, 1, 1, 0, 0]
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["peer_coefficient_norm"] > 0


@pytest.mark.parametrize("operation", ["sample", "evaluate", "update"])
def test_empty_action_masks_fail_explicitly(operation):
    policy, obs, actions, old_log_probs = make_batch()
    available = torch.ones(11, 3, 4, dtype=torch.bool)
    available[4, 1] = False
    with pytest.raises(ValueError, match="at least one available action"):
        if operation == "sample":
            policy.sample(obs, available)
        elif operation == "evaluate":
            policy.evaluate_actions(obs, actions, available)
        else:
            optimizers = [torch.optim.SGD(actor.parameters(), lr=0.1) for actor in policy.actors]
            bppo_update(
                policy, optimizers, obs, actions, old_log_probs, torch.ones(len(obs)),
                available_actions=available,
            )


def test_unavailable_recorded_actions_are_rejected():
    policy, obs, actions, _ = make_batch()
    available = torch.ones(11, 3, 4, dtype=torch.bool)
    available[0, 0, actions[0, 0]] = False
    with pytest.raises(ValueError, match="unavailable action"):
        policy.evaluate_actions(obs, actions, available)


def test_update_with_single_available_actions_has_finite_gradients():
    policy, obs, _, _ = make_batch(agents=2)
    available = torch.ones(11, 2, 4, dtype=torch.bool)
    available[::2, 0] = torch.tensor([False, True, False, False])
    available[1::2, 1] = torch.tensor([True, False, False, False])
    actions, old_log_probs = policy.sample(obs, available)
    optimizers = [torch.optim.SGD(actor.parameters(), lr=0.1) for actor in policy.actors]
    metrics = bppo_update(
        policy, optimizers, obs, actions, old_log_probs, torch.linspace(-1, 1, len(obs)),
        available_actions=available, epochs=2,
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_empty_batch_is_rejected():
    policy = AutoregressivePolicy(2, 3, 4, hidden_dim=8)
    with pytest.raises(ValueError, match="batch must not be empty"):
        policy.sample(torch.zeros(0, 3, 2))


def test_stale_rollout_snapshot_is_rejected_before_updating():
    policy, obs, actions, old_log_probs = make_batch()
    optimizers = [torch.optim.SGD(actor.parameters(), lr=0.1) for actor in policy.actors]
    with torch.no_grad():
        policy.actors[0][-1].bias[0] += 1
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    with pytest.raises(ValueError, match="Policy changed since rollout"):
        bppo_update(policy, optimizers, obs, actions, old_log_probs, torch.ones(len(obs)))
    for current, expected in zip(policy.parameters(), before):
        torch.testing.assert_close(current, expected)


def test_shared_or_wrong_optimizer_ownership_is_rejected():
    policy, obs, actions, old_log_probs = make_batch()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="exactly its corresponding actor"):
        bppo_update(policy, [optimizer] * 3, obs, actions, old_log_probs, torch.ones(len(obs)))


def test_inactive_samples_have_zero_own_peer_and_entropy_gradient():
    log_probs = torch.tensor([0.1, -0.2, 0.2], requires_grad=True)
    entropy = torch.tensor([0.7, 0.8, 0.5], requires_grad=True)
    relaxed = torch.ones(3, 4, requires_grad=True)
    loss, _ = _actor_objective(
        log_probs, entropy, torch.zeros(3), torch.tensor([1.0, -2.0, 3.0]),
        torch.ones(3), torch.ones(3, 4), relaxed,
        clip_coef=0.2, entropy_coef=0.1, active=torch.tensor([True, False, True]),
    )
    loss.backward()
    assert log_probs.grad[1] == entropy.grad[1] == 0
    assert torch.count_nonzero(relaxed.grad[1]) == 0
    assert torch.count_nonzero(relaxed.grad[[0, 2]]) > 0


def test_masked_single_agent_update_matches_active_subset():
    policy, obs, actions, old_log_probs = make_batch(agents=1)
    reference = deepcopy(policy)
    active = torch.arange(len(obs)) % 2 == 0
    advantages = torch.linspace(-1, 1, len(obs))
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.05)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    bppo_update(
        policy, [optimizer], obs, actions, old_log_probs, advantages,
        active_agents=active[:, None], epochs=2,
    )
    bppo_update(
        reference, [reference_optimizer], obs[active], actions[active], old_log_probs[active],
        advantages[active], epochs=2,
    )
    for actual, expected in zip(policy.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)


def test_all_inactive_actors_skip_optimizers_and_remain_finite():
    policy, obs, actions, old_log_probs = make_batch()
    before = deepcopy(policy.state_dict())
    optimizers = [torch.optim.Adam(actor.parameters(), lr=0.1) for actor in policy.actors]
    metrics = bppo_update(
        policy, optimizers, obs, actions, old_log_probs, torch.ones(len(obs)),
        active_agents=torch.zeros_like(actions, dtype=torch.bool), epochs=2,
    )
    assert all(value == 0 for value in metrics.values())
    assert all(not optimizer.state for optimizer in optimizers)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_inactive_single_noop_successor_has_unit_ratio_and_zero_feedback():
    policy, obs, _, _ = make_batch(agents=2)
    available = torch.ones(11, 2, 4, dtype=torch.bool)
    available[:, 1] = torch.tensor([True, False, False, False])
    actions, old_log_probs = policy.sample(obs, available)
    active = torch.ones(11, 2, dtype=torch.bool)
    active[:, 1] = False
    before = deepcopy(policy.actors[1].state_dict())
    optimizers = [torch.optim.Adam(actor.parameters(), lr=0.03) for actor in policy.actors]
    metrics = bppo_update(
        policy, optimizers, obs, actions, old_log_probs, torch.linspace(-1, 1, len(obs)),
        available_actions=available, active_agents=active, epochs=2,
    )
    assert metrics["num_updates"] == 2
    assert metrics["peer_coefficient_norm"] == 0
    assert metrics["successor_ratio"] == 1
    assert not optimizers[1].state
    for key, value in policy.actors[1].state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_zero_embedding_preserves_original_actor_structure_initialization_and_likelihood():
    torch.manual_seed(821)
    default = AutoregressivePolicy(2, 3, 4, hidden_dim=8)
    torch.manual_seed(821)
    explicit_zero = AutoregressivePolicy(2, 3, 4, hidden_dim=8, action_embed_dim=0)
    assert all(isinstance(actor, torch.nn.Sequential) for actor in default.actors)
    assert default.state_dict().keys() == explicit_zero.state_dict().keys()
    for key, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit_zero.state_dict()[key], atol=0, rtol=0)
    obs = torch.randn(5, 3, 2)
    actions, old_log_probs = default.sample(obs)
    log_probs, _ = explicit_zero.evaluate_actions(obs, actions)
    torch.testing.assert_close(old_log_probs, log_probs, atol=0, rtol=0)


def test_projected_actor_owns_embedding_and_retains_action_input_gradients():
    policy, obs, actions, old_log_probs = make_batch(action_embed_dim=7)
    assert isinstance(policy.actors[0], torch.nn.Sequential)
    assert policy.actors[1].action_projection is not policy.actors[2].action_projection
    assert policy.actors[2].network[0].in_features == 2 + 2 * 7
    log_probs, _ = policy.evaluate_actions(obs, actions)
    torch.testing.assert_close(log_probs, old_log_probs)
    features = F.one_hot(actions, 4).to(obs.dtype).requires_grad_(True)
    distribution = policy._distribution(obs, features, 2, None)
    distribution.log_prob(actions[:, 2]).sum().backward()
    assert torch.count_nonzero(features.grad[:, :2]) > 0
    assert torch.count_nonzero(features.grad[:, 2]) == 0
    assert torch.count_nonzero(policy.actors[2].action_projection.weight.grad) > 0
    assert all(p.grad is None for p in policy.actors[0].parameters())
    assert all(p.grad is None for p in policy.actors[1].parameters())


def test_projected_peer_derivative_matches_finite_difference():
    policy, obs, actions, _ = make_batch(agents=2, action_embed_dim=7)
    policy, obs = policy.double(), obs.double()
    behavior = deepcopy(policy).requires_grad_(False)
    with torch.no_grad():
        policy.actors[1].network[-1].weight.add_(torch.randn_like(policy.actors[1].network[-1].weight) * 0.3)
    features = F.one_hot(actions, 4).to(obs.dtype)
    _, coefficients = _successor_feedback(policy, behavior, obs, actions, features, 0)
    numerical = torch.zeros_like(coefficients)
    eps = 1e-5

    def ratio_at(probed):
        current = policy._distribution(obs, probed, 1, None).log_prob(actions[:, 1])
        old = behavior._distribution(obs, probed, 1, None).log_prob(actions[:, 1])
        return (current - old).exp()

    with torch.no_grad():
        for coordinate in range(4):
            plus, minus = features.clone(), features.clone()
            plus[:, 0, coordinate] += eps
            minus[:, 0, coordinate] -= eps
            numerical[:, coordinate] = (ratio_at(plus) - ratio_at(minus)) / (2 * eps)
    torch.testing.assert_close(coefficients, numerical, atol=1e-8, rtol=1e-6)


def test_single_agent_gradient_diagnostics_match_unclipped_update():
    policy, obs, actions, old_log_probs = make_batch(agents=1)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.03)
    metrics = bppo_update(
        policy, [optimizer], obs, actions, old_log_probs, torch.linspace(-1, 1, len(obs)),
        epochs=1, entropy_coef=0, max_grad_norm=1e9,
    )
    assert metrics["own_grad_norm"] == pytest.approx(metrics["grad_norm"], rel=1e-6)
    assert metrics["peer_grad_norm"] == metrics["own_peer_grad_cosine"] == 0
    assert metrics["successor_ratio_abs_change"] == 0


def test_projected_update_reports_separate_gradient_scales_and_owns_all_parameters():
    policy, obs, actions, old_log_probs = make_batch(agents=2, action_embed_dim=7)
    optimizers = [torch.optim.SGD(actor.parameters(), lr=0.03) for actor in policy.actors]
    before = policy.actors[1].action_projection.weight.detach().clone()
    metrics = bppo_update(
        policy, optimizers, obs, actions, old_log_probs, torch.linspace(-1, 1, len(obs)), epochs=2,
    )
    assert metrics["own_grad_norm"] > 0
    assert metrics["peer_grad_norm"] > 0
    assert -1 <= metrics["own_peer_grad_cosine"] <= 1
    assert metrics["successor_ratio_abs_change"] > 0
    assert not torch.equal(before, policy.actors[1].action_projection.weight)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
