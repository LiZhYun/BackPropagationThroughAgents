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


def _example(seed=12, initial_std=1.0):
    torch.manual_seed(seed)
    policy = ContinuousAgentMixerPolicy(4, 7, 3, 2, hidden_dim=16, initial_std=initial_std)
    return policy, torch.randn(11, 3, 4), torch.randn(11, 7)


@pytest.mark.parametrize("initial_std", [1.0, 0.36552928931500245])
def test_initial_identity_and_exact_mean_consistency_after_modifier_changes(initial_std):
    policy, obs, state = _example(initial_std=initial_std)
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


def test_local_execution_is_independent_of_other_agents_and_modifier():
    policy, obs, _ = _example(2)
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
def test_local_and_joint_standard_deviations_are_finite_positive_and_bounded(bias):
    policy, obs, state = _example(3)
    with torch.no_grad():
        for actor in policy.local_actors:
            actor.log_std_head.bias.fill_(bias)
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
def test_log_probabilities_and_entropies_sum_action_coordinates(initial_std):
    policy, obs, state = _example(5, initial_std=initial_std)
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
def test_gradients_reach_local_means_local_uncertainty_and_modifier(initial_std):
    policy, obs, state = _example(7, initial_std=initial_std)
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
        assert actor.log_std_head.weight.grad.abs().sum() > 0
    assert policy.modifier.policy_embedding.weight.grad.abs().sum() > 0
    assert policy.modifier.state_embedding.weight.grad.abs().sum() > 0
    assert policy.modifier.blocks[0].agent_mlp[0].weight.grad.abs().sum() > 0


def test_checkpoint_roundtrip_restores_distribution_and_seeded_samples(tmp_path):
    policy, obs, state = _example(9)
    nn.init.normal_(policy.modifier.log_std_head.weight, std=0.1)
    path = tmp_path / "continuous_policy.pt"
    torch.save(policy.state_dict(), path)
    restored = ContinuousAgentMixerPolicy(4, 7, 3, 2, hidden_dim=16)
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
