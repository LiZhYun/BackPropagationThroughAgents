"""The continuous baseline scores vector events and uses only local inputs."""
import torch
from bta_repro.core import ContinuousIndependentPolicy


def test_rollout_likelihood_is_event_sum_and_matches_evaluation():
    torch.manual_seed(7)
    policy = ContinuousIndependentPolicy(4, 3, 2, 8)
    obs = torch.randn(11, 3, 4)
    actions, old_log_probs = policy.sample(obs)
    log_probs, entropy = policy.evaluate_actions(obs, actions)
    assert actions.shape == (11, 3, 2)
    assert log_probs.shape == entropy.shape == (11, 3)
    torch.testing.assert_close(log_probs, old_log_probs)
    base = policy.distribution(obs).base_dist
    torch.testing.assert_close(log_probs, base.log_prob(actions).sum(-1))
    # Raw Gaussian actions must not be clipped or tanh-transformed by the actor.
    assert actions.abs().max() > 1


def test_local_actions_ignore_other_agents_and_gradients_reach_scale():
    torch.manual_seed(11)
    policy = ContinuousIndependentPolicy(4, 3, 2, 8)
    obs = torch.randn(11, 3, 4)
    changed = obs.clone()
    changed[:, 1:] += 100
    torch.testing.assert_close(policy.local_actions(obs)[:, 0], policy.local_actions(changed)[:, 0])
    actions, _ = policy.sample(obs)
    log_probs, entropy = policy.evaluate_actions(obs, actions)
    (-log_probs.mean() - .01 * entropy.mean()).backward()
    assert policy.log_std.grad is not None and policy.log_std.grad.norm() > 0
    assert all(any(p.grad is not None and p.grad.norm() > 0 for p in actor.parameters())
               for actor in policy.actors)


def test_deterministic_action_is_gaussian_mode():
    policy = ContinuousIndependentPolicy(4, 3, 2, 8)
    obs = torch.randn(5, 3, 4)
    actions, _ = policy.sample(obs, deterministic=True)
    torch.testing.assert_close(actions, policy.local_actions(obs))
