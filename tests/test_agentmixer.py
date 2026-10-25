import pytest
import torch
from torch import nn

from bta_repro.agentmixer import AgentMixerPolicy, mode_preserving_logits


@pytest.mark.parametrize("seed", range(6))
def test_randomized_igc_preserves_entire_masked_mode_set(seed):
    generator = torch.Generator().manual_seed(seed)
    local = torch.randn(23, 4, 7, generator=generator)
    scales = 30 * torch.randn(local.shape, generator=generator)
    mask = torch.rand(local.shape, generator=generator) > 0.3
    mask[..., 0] = True
    joint = mode_preserving_logits(local, scales, mask)
    masked_local = local.masked_fill(~mask, -torch.inf)
    local_modes = masked_local == masked_local.amax(-1, keepdim=True)
    joint_modes = joint == joint.amax(-1, keepdim=True)
    assert torch.equal(local_modes, joint_modes)
    assert torch.isneginf(joint[~mask]).all()
    assert torch.isfinite(joint[mask]).all()


def test_masks_ties_and_single_available_action():
    local = torch.tensor([[[100.0, 2.0, 2.0, -1.0], [99.0, -3.0, 11.0, 0.0]]])
    mask = torch.tensor([[[False, True, True, True], [False, True, False, False]]])
    scales = torch.tensor([[[100.0, -100.0, 100.0, -100.0], [0.0, 100.0, 0.0, 0.0]]])
    joint = mode_preserving_logits(local, scales, mask)
    assert joint[0, 0, 1] == joint[0, 0, 2] == 0
    assert joint[0, 0, 3] < 0
    assert joint[0, 1, 1] == 0
    distribution = torch.distributions.Categorical(logits=joint)
    assert distribution.probs[0, 1, 1] == 1
    assert distribution.entropy()[0, 1] == 0
    assert torch.isfinite(distribution.entropy()).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_extreme_finite_inputs_and_subnormal_gaps_preserve_modes(dtype):
    largest = torch.finfo(dtype).max
    smallest = torch.nextafter(torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype))
    local = torch.tensor([[[largest, -largest, 0.0], [0.0, -smallest, -1.0]]], dtype=dtype)
    local.requires_grad_()
    log_scales = torch.tensor([[[largest, largest, -largest], [-largest, -largest, largest]]], dtype=dtype)
    log_scales.requires_grad_()
    joint = mode_preserving_logits(local, log_scales)
    assert torch.isfinite(joint).all()
    assert torch.equal(joint == 0, local == local.amax(-1, keepdim=True))
    distribution = torch.distributions.Categorical(logits=joint)
    assert (distribution.probs[..., 0] > distribution.probs[..., 1]).all()
    joint.sum().backward()
    assert torch.isfinite(local.grad).all()
    assert torch.isfinite(log_scales.grad).all()


def test_gap_maximum_is_not_detached():
    local = torch.tensor([[3.0, 1.0, 0.0]], requires_grad=True)
    log_scales = torch.tensor([[1.0, 2.0, 3.0]]).log()
    mode_preserving_logits(local, log_scales)[0, 1].backward()
    torch.testing.assert_close(local.grad, torch.tensor([[-2.0, 2.0, 0.0]]))


def test_gap_scaling_gradcheck_away_from_ties():
    local = torch.tensor([[[2.0, 0.2, -1.0], [0.3, 1.5, -0.9]]], dtype=torch.float64, requires_grad=True)
    scales = torch.tensor([[[0.2, -0.1, 0.3], [0.1, 0.4, -0.3]]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(mode_preserving_logits, (local, scales))


@pytest.mark.parametrize("bad_mask", [torch.zeros(2, 3, 4), torch.ones(2, 3, 5), torch.full((2, 3, 4), 2)])
def test_reject_invalid_masks(bad_mask):
    with pytest.raises(ValueError, match="available|action"):
        mode_preserving_logits(torch.zeros(2, 3, 4), torch.zeros(2, 3, 4), bad_mask)


def test_identity_initialization_and_sampling_evaluation_agree():
    torch.manual_seed(41)
    policy = AgentMixerPolicy(5, 7, 3, 4, hidden_dim=16)
    obs, state = torch.randn(11, 3, 5), torch.randn(11, 7)
    local = policy.local_logits(obs)
    joint = policy.joint_logits(obs, state)
    torch.testing.assert_close(joint.softmax(-1), local.softmax(-1), atol=1e-7, rtol=1e-6)
    actions, old_logp = policy.sample(obs, state)
    logp, entropy = policy.evaluate_actions(obs, state, actions)
    assert actions.shape == logp.shape == entropy.shape == (11, 3)
    torch.testing.assert_close(logp, old_logp)
    greedy, _ = policy.sample(obs, state, deterministic=True)
    assert torch.equal(greedy, policy.local_actions(obs))


def test_local_execution_cannot_use_state_other_observations_or_modifier():
    torch.manual_seed(5)
    policy = AgentMixerPolicy(4, 6, 3, 5, hidden_dim=16)
    obs = torch.randn(9, 3, 4, requires_grad=True)
    local = policy.local_logits(obs)
    changed = obs.detach().clone()
    changed[:, 1:] += 100
    torch.testing.assert_close(local[:, 0], policy.local_logits(changed)[:, 0])
    local[:, 0].sum().backward()
    assert torch.count_nonzero(obs.grad[:, 1:]) == 0
    assert torch.count_nonzero(obs.grad[:, 0]) > 0
    assert len({actor[0].weight.data_ptr() for actor in policy.local_actors}) == 3

    class ForbiddenModifier(nn.Module):
        def forward(self, *args):
            raise AssertionError("decentralized execution invoked the modifier")

    policy.modifier = ForbiddenModifier()
    actions, logp = policy.sample(obs.detach(), None, decentralized=True, deterministic=True)
    assert torch.equal(actions, policy.local_actions(obs.detach()))
    assert torch.isfinite(logp).all()


def test_modifier_uses_state_and_other_agents_and_gradients_reach_both_components():
    torch.manual_seed(19)
    policy = AgentMixerPolicy(4, 6, 3, 5, hidden_dim=16)
    # Zero initialization intentionally postpones hidden modifier gradients until
    # its output head learns. Here exercise a nonzero modifier after that step.
    nn.init.normal_(policy.modifier.scale_head.weight, std=0.1)
    obs, state = torch.randn(9, 3, 4), torch.randn(9, 6)
    logits = policy.joint_logits(obs, state)
    changed_obs = obs.clone()
    changed_obs[:, 1:] += 10
    assert not torch.allclose(logits[:, 0], policy.joint_logits(changed_obs, state)[:, 0])
    assert not torch.allclose(logits, policy.joint_logits(obs, state + 10))
    actions, _ = policy.sample(obs, state)
    logp, entropy = policy.evaluate_actions(obs, state, actions)
    loss = -(logp * torch.randn_like(logp)).mean() - 0.01 * entropy.mean()
    loss.backward()
    for component in [*policy.local_actors, policy.modifier]:
        grads = [p.grad for p in component.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)
        assert sum(grad.abs().sum() for grad in grads) > 0
    assert policy.modifier.policy_embedding.weight.grad.abs().sum() > 0
    assert policy.modifier.state_embedding.weight.grad.abs().sum() > 0
    assert policy.modifier.blocks[0].agent_mlp[0].weight.grad.abs().sum() > 0


def test_masked_policy_samples_only_available_actions():
    torch.manual_seed(17)
    policy = AgentMixerPolicy(2, 3, 2, 4, hidden_dim=8)
    obs, state = torch.randn(19, 2, 2), torch.randn(19, 3)
    mask = torch.zeros(19, 2, 4, dtype=torch.bool)
    mask[:, 0, 2] = True
    mask[:, 1, 1] = True
    actions, logp = policy.sample(obs, state, mask)
    assert (actions[:, 0] == 2).all() and (actions[:, 1] == 1).all()
    evaluated, entropy = policy.evaluate_actions(obs, state, actions, mask)
    torch.testing.assert_close(logp, evaluated)
    assert (logp == 0).all() and (entropy == 0).all()
    with pytest.raises(ValueError, match="unavailable"):
        policy.evaluate_actions(obs, state, torch.zeros_like(actions), mask)


def test_bounded_modifier_preserves_identity_modes_and_scale_limits():
    bound = torch.tensor(2.).log().item()
    policy = AgentMixerPolicy(2, 3, 2, 4, hidden_dim=8, log_scale_bound=bound)
    obs, state = torch.randn(19, 2, 2), torch.randn(19, 3)
    local = policy.local_logits(obs)
    torch.testing.assert_close(policy.joint_logits(obs, state).softmax(-1), local.softmax(-1))
    with torch.no_grad():
        policy.modifier.scale_head.weight.normal_(std=100.)
    scales = policy.modifier_log_scales(local, state).exp()
    assert scales.min() >= .5 and scales.max() <= 2.
    assert torch.equal(policy.joint_logits(obs, state).argmax(-1), policy.local_actions(obs))
