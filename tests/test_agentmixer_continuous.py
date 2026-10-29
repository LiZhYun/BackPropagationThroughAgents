import math

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from bta_repro.agentmixer_continuous import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    ContinuousAgentMixerPolicy,
)


def _example(seed=12, initial_std=1.0, modifier_scale="residual", local_scale="observation"):
    torch.manual_seed(seed)
    policy = ContinuousAgentMixerPolicy(4, 7, 3, 2, hidden_dim=16, initial_std=initial_std,
                                       modifier_scale=modifier_scale, local_scale=local_scale)
    return policy, torch.randn(11, 3, 4), torch.randn(11, 7)


@pytest.mark.parametrize("initial_std", [1.0, 0.36552928931500245])
@pytest.mark.parametrize("local_scale", ["observation", "state_independent"])
def test_initial_identity_and_exact_mean_consistency_after_modifier_changes(initial_std, local_scale):
    policy, obs, state = _example(initial_std=initial_std, local_scale=local_scale)
    local_mean, local_std = policy.local_parameters(obs)
    mean, std = policy.joint_parameters(obs, state)
    assert torch.equal(mean, local_mean)
    assert torch.equal(std, local_std)
    nn.init.normal_(policy.modifier.log_std_head.weight, std=0.2)
    mixed_mean, mixed_std = policy.joint_parameters(obs, state)
    changed_mean, changed_std = policy.joint_parameters(obs, state + 13)
    assert torch.equal(mixed_mean, local_mean)
    assert torch.equal(changed_mean, local_mean)
    assert not torch.allclose(mixed_std, local_std)
    assert not torch.allclose(changed_std, mixed_std)


@pytest.mark.parametrize("modifier_scale", ["residual", "bounded"])
@pytest.mark.parametrize("local_scale", ["observation", "state_independent"])
def test_local_execution_is_independent_of_other_agents_and_modifier(modifier_scale, local_scale):
    policy, obs, _ = _example(2, modifier_scale=modifier_scale, local_scale=local_scale)
    obs.requires_grad_()
    mean, _ = policy.local_parameters(obs)
    changed = obs.detach().clone()
    changed[:, 1:] += 100
    assert torch.equal(mean[:, 0], policy.local_actions(changed)[:, 0])
    mean[:, 0].sum().backward()
    assert torch.count_nonzero(obs.grad[:, 1:]) == 0
    assert obs.grad[:, 0].abs().sum() > 0
    assert len({actor.mean_head.weight.data_ptr() for actor in policy.local_actors}) == 3

    class ForbiddenModifier(nn.Module):
        def forward(self, *args):
            raise AssertionError("local execution used the policy modifier")

    policy.modifier = ForbiddenModifier()
    actions, logp = policy.sample(obs.detach(), decentralized=True, deterministic=True)
    assert torch.equal(actions, policy.local_actions(obs.detach()))
    assert torch.isfinite(logp).all()


@pytest.mark.parametrize("bias", [-1e30, 1e30])
@pytest.mark.parametrize("modifier_scale", ["residual", "bounded"])
@pytest.mark.parametrize("local_scale", ["observation", "state_independent"])
def test_local_and_joint_standard_deviations_are_finite_positive_and_bounded(bias, modifier_scale, local_scale):
    policy, obs, state = _example(3, modifier_scale=modifier_scale, local_scale=local_scale)
    with torch.no_grad():
        for actor in policy.local_actors:
            (actor.log_std if actor.log_std_head is None else actor.log_std_head.bias).fill_(bias)
        policy.modifier.log_std_head.bias.fill_(-bias)
    for _, std in (policy.local_parameters(obs), policy.joint_parameters(obs, state)):
        assert torch.isfinite(std).all()
        assert (std > 0).all()
        assert (std >= math.exp(LOG_STD_MIN) * (1 - 1e-6)).all()
        assert (std <= math.exp(LOG_STD_MAX) * (1 + 1e-6)).all()
    actions, logp = policy.sample(obs, state)
    _, entropy = policy.evaluate_actions(obs, state, actions)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(logp).all() and torch.isfinite(entropy).all()


@pytest.mark.parametrize("initial_std", [1.0, 0.36552928931500245])
@pytest.mark.parametrize("modifier_scale", ["residual", "bounded"])
@pytest.mark.parametrize("local_scale", ["observation", "state_independent"])
def test_log_probabilities_and_entropies_sum_action_coordinates(initial_std, modifier_scale, local_scale):
    policy, obs, state = _example(5, initial_std=initial_std, modifier_scale=modifier_scale, local_scale=local_scale)
    nn.init.normal_(policy.modifier.log_std_head.weight, std=0.1)
    mean, std = policy.joint_parameters(obs, state)
    raw_actions = mean.detach() + torch.randn_like(mean) * std.detach()
    logp, entropy = policy.evaluate_actions(obs, state, raw_actions)
    distribution = Normal(mean, std)
    assert logp.shape == entropy.shape == (11, 3)
    torch.testing.assert_close(logp, distribution.log_prob(raw_actions).sum(-1))
    torch.testing.assert_close(entropy, distribution.entropy().sum(-1))
    sampled_actions, sampled_logp = policy.sample(obs, state)
    evaluated_logp, _ = policy.evaluate_actions(obs, state, sampled_actions)
    torch.testing.assert_close(sampled_logp, evaluated_logp)


def test_raw_actions_are_not_squashed_or_clipped_and_deterministic_actions_are_means():
    policy, obs, state = _example(6)
    with torch.no_grad():
        for actor in policy.local_actors:
            actor.mean_head.weight.zero_()
            actor.mean_head.bias.fill_(4.0)
    mean, _ = policy.local_parameters(obs)
    for decentralized in (True, False):
        actions, _ = policy.sample(obs, state, deterministic=True, decentralized=decentralized)
        assert torch.equal(actions, mean)
        assert (actions == 4).all()
    raw_actions, raw_logp = policy.sample(obs, state)
    assert (raw_actions > 1).any()
    clipped_logp, _ = policy.evaluate_actions(obs, state, raw_actions.clamp(-1, 1))
    assert not torch.allclose(raw_logp, clipped_logp)


@pytest.mark.parametrize("initial_std", [1.0, 0.36552928931500245])
@pytest.mark.parametrize("modifier_scale", ["residual", "bounded"])
@pytest.mark.parametrize("local_scale", ["observation", "state_independent"])
def test_gradients_reach_local_means_local_uncertainty_and_modifier(initial_std, modifier_scale, local_scale):
    policy, obs, state = _example(7, initial_std=initial_std, modifier_scale=modifier_scale, local_scale=local_scale)
    nn.init.normal_(policy.modifier.log_std_head.weight, std=0.1)
    actions, _ = policy.sample(obs, state)
    logp, entropy = policy.evaluate_actions(obs, state, actions)
    loss = -(logp * torch.randn_like(logp)).mean() - 0.01 * entropy.mean()
    loss.backward()
    for component in [*policy.local_actors, policy.modifier]:
        grads = [parameter.grad for parameter in component.parameters() if parameter.grad is not None]
        assert grads and all(torch.isfinite(gradient).all() for gradient in grads)
        assert sum(gradient.abs().sum() for gradient in grads) > 0
    for actor in policy.local_actors:
        assert actor.mean_head.weight.grad.abs().sum() > 0
        scale_parameter = actor.log_std if actor.log_std_head is None else actor.log_std_head.weight
        assert scale_parameter.grad.abs().sum() > 0
    assert policy.modifier.policy_embedding.weight.grad.abs().sum() > 0
    assert policy.modifier.state_embedding.weight.grad.abs().sum() > 0
    assert policy.modifier.blocks[0].agent_mlp[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("modifier_scale", ["residual", "bounded"])
@pytest.mark.parametrize("local_scale", ["observation", "state_independent"])
def test_checkpoint_roundtrip_restores_distribution_and_seeded_samples(tmp_path, modifier_scale, local_scale):
    policy, obs, state = _example(9, modifier_scale=modifier_scale, local_scale=local_scale)
    nn.init.normal_(policy.modifier.log_std_head.weight, std=0.1)
    path = tmp_path / "continuous_policy.pt"
    torch.save(policy.state_dict(), path)
    restored = ContinuousAgentMixerPolicy(4, 7, 3, 2, hidden_dim=16, modifier_scale=modifier_scale, local_scale=local_scale)
    restored.load_state_dict(torch.load(path, weights_only=True))
    for expected, actual in zip(policy.joint_parameters(obs, state), restored.joint_parameters(obs, state)):
        assert torch.equal(expected, actual)
    torch.manual_seed(18)
    actions, logp = policy.sample(obs, state)
    torch.manual_seed(18)
    restored_actions, restored_logp = restored.sample(obs, state)
    assert torch.equal(actions, restored_actions)
    assert torch.equal(logp, restored_logp)


def test_reject_malformed_inputs():
    policy, obs, state = _example(10)
    with pytest.raises(ValueError, match="state"):
        policy.sample(obs)
    with pytest.raises(ValueError, match="obs"):
        policy.local_parameters(obs[:, :2])
    with pytest.raises(ValueError, match="state"):
        policy.joint_parameters(obs, torch.full_like(state, torch.nan))
    with pytest.raises(ValueError, match="raw_actions"):
        policy.evaluate_actions(obs, state, torch.zeros(11, 3))
    with pytest.raises(ValueError, match="finite"):
        policy.evaluate_actions(obs, state, torch.full((11, 3, 2), torch.inf))


@pytest.mark.parametrize("initial_std", [math.exp(-5), 0.36552928931500245, 1.0, math.exp(2)])
def test_initial_std_preserves_mean_initialization_and_modifier_identity(initial_std):
    torch.manual_seed(23)
    reference = ContinuousAgentMixerPolicy(4, 7, 3, 2, 8)
    torch.manual_seed(23)
    policy = ContinuousAgentMixerPolicy(4, 7, 3, 2, 8, initial_std=initial_std)
    obs, state = torch.randn(7, 3, 4), torch.randn(7, 7)
    local_mean, local_std = policy.local_parameters(obs)
    mean, std = policy.joint_parameters(obs, state)
    torch.testing.assert_close(local_std, torch.full_like(local_std, initial_std))
    assert torch.equal(local_std, std)
    assert torch.equal(local_mean, mean)
    assert torch.equal(mean, reference.local_actions(obs))
    actions, sampled_logp = policy.sample(obs, state)
    logp, _ = policy.evaluate_actions(obs, state, actions)
    assert torch.equal(sampled_logp, logp)
    for decentralized in (False, True):
        modes, _ = policy.sample(obs, state, deterministic=True, decentralized=decentralized)
        assert torch.equal(modes, mean)
    for name, value in reference.state_dict().items():
        if not (name.startswith("local_actors.") and name.endswith("log_std_head.bias")) or initial_std == 1.0:
            assert torch.equal(value, policy.state_dict()[name])


@pytest.mark.parametrize("initial_std", [0, -1, math.nan, math.inf, -math.inf,
                                         math.exp(-5) * 0.99, math.exp(2) * 1.01, True, None])
def test_invalid_initial_std_is_rejected(initial_std):
    with pytest.raises(ValueError, match="initial_std"):
        ContinuousAgentMixerPolicy(4, 7, 3, 2, 8, initial_std=initial_std)


def test_bounded_scale_preserves_initial_weights_rng_and_exact_local_modes():
    residual, obs, state = _example(19, initial_std=.2)
    bounded, bounded_obs, bounded_state = _example(19, initial_std=.2, modifier_scale="bounded")
    assert torch.equal(obs, bounded_obs) and torch.equal(state, bounded_state)
    for name, value in residual.state_dict().items():
        assert torch.equal(value, bounded.state_dict()[name])
    local_mean, local_std = bounded.local_parameters(obs)
    mean, std = bounded.joint_parameters(obs, state)
    assert torch.equal(mean, local_mean)
    assert torch.equal(mean, residual.local_actions(obs))
    assert torch.equal(std, torch.full_like(std, .375))
    assert not torch.equal(std, local_std)
    nn.init.normal_(bounded.modifier.log_std_head.weight, std=.2)
    mean, std = bounded.joint_parameters(obs, state)
    changed_mean, changed_std = bounded.joint_parameters(obs, state + 13)
    assert torch.equal(mean, local_mean) and torch.equal(changed_mean, local_mean)
    assert not torch.allclose(std, changed_std)
    assert (std <= .75).all() and (changed_std <= .75).all()
    actions, _ = bounded.sample(obs, state, deterministic=True)
    assert torch.equal(actions, local_mean)


@pytest.mark.parametrize("score", [-1e30, -2., 0., 2., 1e30])
def test_bounded_scale_replaces_local_std_and_has_analytic_output_gradient(score):
    policy, obs, state = _example(20, initial_std=1.0, modifier_scale="bounded")
    with torch.no_grad():
        policy.modifier.log_std_head.bias.fill_(score)
    mean, std = policy.joint_parameters(obs, state)
    sigmoid = torch.sigmoid(torch.tensor(.3 * score)).item()
    expected = max(.75 * sigmoid, math.exp(LOG_STD_MIN))
    torch.testing.assert_close(std, torch.full_like(std, expected))
    std.sum().backward()
    slope = .75 * .3 * sigmoid * (1 - sigmoid) if .75 * sigmoid > math.exp(LOG_STD_MIN) else 0.
    expected_grad = torch.full_like(policy.modifier.log_std_head.bias, 11 * 3 * slope)
    torch.testing.assert_close(policy.modifier.log_std_head.bias.grad, expected_grad)
    with torch.no_grad():
        for actor in policy.local_actors:
            actor.log_std_head.bias.fill_(math.log(.1))
    new_mean, new_std = policy.joint_parameters(obs, state)
    assert torch.equal(mean, new_mean) and torch.equal(std, new_std)


@pytest.mark.parametrize("modifier_scale", ["invalid", True, None])
def test_invalid_modifier_scale_is_rejected(modifier_scale):
    with pytest.raises(ValueError, match="modifier_scale"):
        ContinuousAgentMixerPolicy(4, 7, 3, 2, 8, modifier_scale=modifier_scale)


@pytest.mark.parametrize("local_scale", ["invalid", True, None])
def test_invalid_local_scale_is_rejected(local_scale):
    with pytest.raises(ValueError, match="local_scale"):
        ContinuousAgentMixerPolicy(4, 7, 3, 2, 8, local_scale=local_scale)


@pytest.mark.parametrize("modifier_scale", ["residual", "bounded"])
def test_local_scale_control_preserves_initial_distribution_weights_and_rng(modifier_scale):
    conditioned, obs, state = _example(31, initial_std=.36552928931500245,
                                      modifier_scale=modifier_scale)
    independent, other_obs, other_state = _example(31, initial_std=.36552928931500245,
                                                  modifier_scale=modifier_scale,
                                                  local_scale="state_independent")
    assert torch.equal(obs, other_obs) and torch.equal(state, other_state)
    reference = conditioned.state_dict()
    for name, value in independent.state_dict().items():
        original_name = name + "_head.bias" if name.endswith(".log_std") else name
        assert torch.equal(value, reference[original_name])
    for left, right in zip(conditioned.joint_parameters(obs, state),
                           independent.joint_parameters(obs, state)):
        assert torch.equal(left, right)
    torch.manual_seed(90)
    actions, logp = conditioned.sample(obs, state)
    torch.manual_seed(90)
    other_actions, other_logp = independent.sample(obs, state)
    assert torch.equal(actions, other_actions) and torch.equal(logp, other_logp)


def test_state_independent_local_scales_learn_per_agent_and_remain_constant_across_observations():
    policy, obs, _ = _example(37, initial_std=.3, local_scale="state_independent")
    optimizer = torch.optim.Adam(policy.local_actors.parameters(), lr=.01)
    initial_mean, initial_std = policy.local_parameters(obs)
    # Opposite entropy objectives make private agent scales move oppositely.
    directions = torch.tensor([1., -1., 2.]).view(1, 3, 1)
    (initial_std.log() * directions).sum().backward()
    optimizer.step()
    mean, std = policy.local_parameters(obs)
    changed_mean, changed_std = policy.local_parameters(obs + 10)
    assert torch.equal(mean, initial_mean)
    assert not torch.equal(mean, changed_mean)
    assert torch.equal(std, changed_std)
    assert (std[:, 0] < initial_std[:, 0]).all()
    assert (std[:, 1] > initial_std[:, 1]).all()
    assert all(actor.log_std.grad.abs().sum() > 0 for actor in policy.local_actors)
    assert len({actor.log_std.data_ptr() for actor in policy.local_actors}) == 3
    assert not any("log_std_head" in name for name, _ in policy.local_actors.named_parameters())
