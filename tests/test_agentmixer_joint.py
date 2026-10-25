import math

import pytest
import torch
from torch import nn
from torch.distributions import Categorical

from bta_repro.agentmixer_joint import JointAgentMixerPolicy, joint_mode_preserving_logits
from bta_repro.core import ppo_objective


def _fixed_local_logits(policy, logits):
    with torch.no_grad():
        for actor, row in zip(policy.local_actors, logits):
            actor[-1].weight.zero_()
            actor[-1].bias.copy_(torch.as_tensor(row, dtype=actor[-1].bias.dtype))


@pytest.mark.parametrize("seed", range(5))
def test_joint_mode_set_is_exact_product_of_masked_local_mode_sets(seed):
    generator = torch.Generator().manual_seed(seed)
    local = torch.randn(17, 2, 5, generator=generator)
    scales = 20 * torch.randn(17, 5, 5, generator=generator)
    mask = torch.rand(17, 2, 5, generator=generator) > 0.3
    mask[..., 0] = True
    joint = joint_mode_preserving_logits(local, scales, mask)
    local = local.masked_fill(~mask, -torch.inf)
    modes = local == local.amax(-1, keepdim=True)
    expected = modes[:, 0, :, None] & modes[:, 1, None, :]
    actual = joint == joint.flatten(1).amax(-1)[:, None, None]
    assert torch.equal(actual, expected)
    probabilities = joint.flatten(1).softmax(-1).reshape_as(joint)
    torch.testing.assert_close(probabilities.sum((1, 2)), torch.ones(17))
    joint_mask = mask[:, 0, :, None] & mask[:, 1, None, :]
    assert (probabilities[~joint_mask] == 0).all()


def test_ties_masked_maximum_dead_agent_and_uniform_policies():
    local = torch.tensor([[[100., 2., 2.], [3., 1., 3.]], [[1., 2., 3.], [0., 0., 0.]]])
    mask = torch.tensor([[[False, True, True], [True, True, True]],
                         [[True, False, False], [True, True, True]]])
    scales = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3) - 9
    joint = joint_mode_preserving_logits(local, scales, mask)
    assert (joint[0] == 0).sum() == 4
    assert (joint[1] == 0).sum() == 3
    assert torch.isneginf(joint[1, 1:]).all()
    uniform = joint_mode_preserving_logits(torch.zeros(2, 2, 3), scales)
    assert torch.equal(uniform, torch.zeros_like(uniform))


def test_identity_initialization_equals_product_distribution_with_masks():
    torch.manual_seed(7)
    policy = JointAgentMixerPolicy(3, 4, 2, 5, hidden_dim=8)
    obs, state = torch.randn(11, 2, 3), torch.randn(11, 4)
    mask = torch.ones(11, 2, 5, dtype=torch.bool)
    mask[:, 0, 1] = False
    local = policy.local_logits(obs).masked_fill(~mask, -torch.inf).softmax(-1)
    expected = local[:, 0, :, None] * local[:, 1, None, :]
    actual = policy.joint_logits(obs, state, mask).flatten(1).softmax(-1).reshape(11, 5, 5)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    assert torch.equal(policy.modifier_log_scales(policy.local_logits(obs), state), torch.zeros(11, 5, 5))


def test_joint_probabilities_can_be_correlated_while_global_mode_stays_local():
    local = torch.tensor([[[0., -1.], [0., -1.]]])
    scales = torch.tensor([[[1., 10.], [10., .05]]]).log()
    logits = joint_mode_preserving_logits(local, scales)
    probability = logits.flatten(1).softmax(-1).reshape(2, 2)
    assert probability[1, 1] > .47
    assert probability[0, 0] > probability[1, 1]
    determinant = probability[0, 0] * probability[1, 1] - probability[0, 1] * probability[1, 0]
    assert determinant > .24  # A product distribution would have determinant zero.


def test_sample_and_evaluate_report_one_exact_joint_event():
    torch.manual_seed(11)
    policy = JointAgentMixerPolicy(3, 4, 2, 3, hidden_dim=8)
    nn.init.normal_(policy.modifier.scale_head.weight, std=.1)
    obs, state = torch.randn(13, 2, 3), torch.randn(13, 4)
    actions, sampled = policy.sample(obs, state)
    evaluated, entropy = policy.evaluate_actions(obs, state, actions)
    assert actions.shape == (13, 2)
    assert sampled.shape == evaluated.shape == entropy.shape == (13, 1)
    torch.testing.assert_close(sampled, evaluated)
    logits = policy.joint_logits(obs, state).flatten(1)
    indices = actions[:, 0] * 3 + actions[:, 1]
    torch.testing.assert_close(evaluated[:, 0], logits.log_softmax(-1).gather(1, indices[:, None])[:, 0])
    expected_entropy = Categorical(logits=logits).entropy()
    torch.testing.assert_close(entropy[:, 0], expected_entropy)


def test_deterministic_actions_use_joint_mode_not_root_marginal_mode():
    policy = JointAgentMixerPolicy(1, 1, 2, 3, hidden_dim=4)
    _fixed_local_logits(policy, [[0., -1., -1.], [0., -1., -1.]])
    target_energy = torch.tensor([[0., 10., 10.], [.1, .1, .1], [10., 10., 10.]])
    gaps = torch.tensor([0., 1., 1.])
    pair_gaps = gaps[:, None] + gaps[None, :]
    scale = torch.where(pair_gaps > 0, target_energy / pair_gaps.clamp_min(1), torch.ones_like(pair_gaps))
    with torch.no_grad():
        policy.modifier.scale_head.bias.copy_(scale.log().flatten())
    obs, state = torch.ones(1, 2, 1), torch.ones(1, 1)
    probability = policy.joint_logits(obs, state).flatten(1).softmax(-1).reshape(3, 3)
    assert probability.sum(1).argmax() == 1
    actions, _ = policy.sample(obs, state, deterministic=True)
    assert actions.tolist() == [[0, 0]]
    assert torch.equal(actions, policy.local_actions(obs))


def test_exact_joint_score_gradient_matches_expected_return_gradient():
    local = torch.tensor([[[.7, -.3, .1], [.4, -.2, .8]]], dtype=torch.float64, requires_grad=True)
    scales = torch.linspace(-.3, .4, 9, dtype=torch.float64).reshape(1, 3, 3).requires_grad_()
    logits = joint_mode_preserving_logits(local, scales).flatten()
    logp = logits.log_softmax(0)
    probabilities = logp.exp()
    rewards = torch.tensor([1., -2., .5, -1., 3., .7, 4., -.2, 2.], dtype=torch.float64)
    expected_return = (probabilities * rewards).sum()
    # Weight exhaustive actions by the behavior probabilities; the helper
    # averages its batch, hence the factor nine in these detached advantages.
    advantages = (9 * probabilities * (rewards - expected_return)).detach()
    loss, _ = ppo_objective(logp[:, None], logp.detach()[:, None], advantages,
                            torch.zeros(9, 1, dtype=torch.float64), joint=True, entropy_coef=0.)
    surrogate = torch.autograd.grad(loss, (local, scales), retain_graph=True)
    exact = torch.autograd.grad(-expected_return, (local, scales))
    for actual, expected in zip(surrogate, exact):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)


def test_energy_gradcheck_and_exact_entropy_gradient_away_from_ties():
    local = torch.tensor([[[1.4, -.2], [.3, 1.1]]], dtype=torch.float64, requires_grad=True)
    scales = torch.tensor([[[.1, -.3], [.2, .4]]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(joint_mode_preserving_logits, (local, scales))
    entropy = Categorical(logits=joint_mode_preserving_logits(local, scales).flatten(1)).entropy().sum()
    derivative = torch.autograd.grad(entropy, scales)[0][0, 0, 0]
    plus, minus = scales.detach().clone(), scales.detach().clone()
    plus[0, 0, 0] += 1e-6
    minus[0, 0, 0] -= 1e-6
    h_plus = Categorical(logits=joint_mode_preserving_logits(local.detach(), plus).flatten(1)).entropy()
    h_minus = Categorical(logits=joint_mode_preserving_logits(local.detach(), minus).flatten(1)).entropy()
    torch.testing.assert_close(derivative, ((h_plus - h_minus) / 2e-6)[0], atol=1e-9, rtol=1e-6)


def test_locality_and_gradients_to_local_actors_and_joint_modifier():
    torch.manual_seed(13)
    policy = JointAgentMixerPolicy(3, 4, 2, 3, hidden_dim=8)
    nn.init.normal_(policy.modifier.scale_head.weight, std=.1)
    obs, state = torch.randn(9, 2, 3), torch.randn(9, 4)
    changed = obs.clone()
    changed[:, 1] += 10
    torch.testing.assert_close(policy.local_logits(obs)[:, 0], policy.local_logits(changed)[:, 0])
    assert not torch.allclose(policy.joint_logits(obs, state), policy.joint_logits(obs, state + 3))
    actions, _ = policy.sample(obs, state)
    logp, entropy = policy.evaluate_actions(obs, state, actions)
    (-logp.mean() - .01 * entropy.mean()).backward()
    for component in [*policy.local_actors, policy.modifier]:
        gradients = [p.grad for p in component.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    assert policy.modifier.policy_embedding.weight.grad.abs().sum() > 0
    assert policy.modifier.state_embedding.weight.grad.abs().sum() > 0

    class ForbiddenModifier(nn.Module):
        def forward(self, *args):
            raise AssertionError("local execution accessed the modifier")

    policy.modifier = ForbiddenModifier()
    local_actions, local_logp = policy.sample(obs, decentralized=True, deterministic=True)
    assert local_logp.shape == (9, 2)
    assert torch.equal(local_actions, policy.local_actions(obs))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_extreme_finite_values_subnormal_gaps_and_single_legal_pair(dtype):
    largest = torch.finfo(dtype).max
    smallest = torch.nextafter(torch.tensor(0., dtype=dtype), torch.tensor(1., dtype=dtype))
    local = torch.tensor([[[largest, -largest], [0., -smallest]]], dtype=dtype, requires_grad=True)
    scales = torch.tensor([[[largest, -largest], [largest, -largest]]], dtype=dtype, requires_grad=True)
    logits = joint_mode_preserving_logits(local, scales)
    assert torch.isfinite(logits).all()
    assert logits[0, 0, 0] == 0 and (logits.flatten()[1:] < 0).all()
    logits.sum().backward()
    assert torch.isfinite(local.grad).all() and torch.isfinite(scales.grad).all()
    mask = torch.tensor([[[True, False], [False, True]]])
    single = joint_mode_preserving_logits(local.detach(), scales.detach(), mask)
    distribution = Categorical(logits=single.flatten(1))
    assert distribution.probs[0, 1] == 1 and distribution.entropy().item() == 0


def test_optional_smooth_log_scale_bound():
    policy = JointAgentMixerPolicy(1, 1, 2, 3, hidden_dim=4, log_scale_bound=math.log(2))
    with torch.no_grad():
        policy.modifier.scale_head.bias.copy_(torch.linspace(-100, 100, 9))
    scales = policy.modifier_log_scales(policy.local_logits(torch.ones(1, 2, 1)), torch.ones(1, 1)).exp()
    assert (scales >= .5).all() and (scales <= 2).all()


def test_float64_distribution_retains_small_gap_gradients_and_strict_probability_modes():
    local = torch.tensor([[[.001, 0., -.5], [.002, 0., -.5]]], requires_grad=True)
    scales = torch.full((1, 3, 3), -8.)
    low = joint_mode_preserving_logits(local, scales)
    low_gradient, = torch.autograd.grad(low[0, 1, 0], local, retain_graph=True)
    assert low_gradient.abs().sum() == 0  # Float32 mode guard is active.
    precise = joint_mode_preserving_logits(local.double(), scales.double())
    gradient, = torch.autograd.grad(precise[0, 1, 0], local)
    expected = torch.zeros_like(local)
    expected[0, 0, 0], expected[0, 0, 1] = -math.exp(-8), math.exp(-8)
    torch.testing.assert_close(gradient, expected)
    probabilities = Categorical(logits=precise.flatten(1)).probs
    assert probabilities[0, 0] > probabilities[0, 1:].max()


def test_precise_joint_policy_keeps_network_dtype_and_likelihood_identity():
    torch.manual_seed(93)
    policy = JointAgentMixerPolicy(1, 1, 2, 3, hidden_dim=8, distribution_float64=True)
    assert all(p.dtype == torch.float32 for p in policy.parameters())
    obs, state = torch.ones(32, 2, 1), torch.ones(32, 1)
    actions, behavior_logp = policy.sample(obs, state)
    logp, entropy = policy.evaluate_actions(obs, state, actions)
    assert behavior_logp.dtype == logp.dtype == entropy.dtype == torch.float64
    torch.testing.assert_close(behavior_logp, logp, rtol=0, atol=0)
    (-logp.mean()).backward()
    assert all(p.grad.dtype == torch.float32 for p in policy.parameters() if p.grad is not None)
    _, local_logp = policy.sample(obs, decentralized=True)
    assert local_logp.dtype == torch.float32


def test_invalid_agent_count_masks_and_unavailable_scored_actions():
    with pytest.raises(ValueError, match="two agents"):
        JointAgentMixerPolicy(1, 1, 3, 3)
    policy = JointAgentMixerPolicy(1, 1, 2, 3, hidden_dim=4)
    obs, state = torch.ones(1, 2, 1), torch.ones(1, 1)
    with pytest.raises(ValueError, match="available action"):
        policy.sample(obs, state, torch.zeros(1, 2, 3, dtype=torch.bool))
    mask = torch.zeros(1, 2, 3, dtype=torch.bool)
    mask[..., 1] = True
    with pytest.raises(ValueError, match="unavailable"):
        policy.evaluate_actions(obs, state, torch.zeros(1, 2, dtype=torch.long), mask)
