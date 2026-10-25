import pytest
import torch
from bta_repro.core import generalized_advantage, ppo_objective, IndependentPolicy
from bta_repro.envs import MatrixGame
from bta_repro.core import normalize_advantages


@pytest.mark.parametrize('value', [0.1, -0.1, 1.1234567, -19.987654, 10000.123])
def test_constant_advantages_do_not_create_a_learning_signal(value):
    advantages = torch.full((1024,), value, dtype=torch.float32)
    assert torch.equal(normalize_advantages(advantages), torch.zeros_like(advantages))


def test_advantage_normalization_uses_only_valid_samples():
    result = normalize_advantages(torch.tensor([1., 3., 100.]), torch.tensor([True, True, False]))
    torch.testing.assert_close(result, torch.tensor([-1., 1., 0.]))
    assert normalize_advantages(torch.ones(3), torch.zeros(3, dtype=torch.bool)).sum() == 0


def test_truncation_bootstraps_terminal_observation_and_breaks_recursion():
    rewards = torch.tensor([[1.], [100.]])
    values = torch.tensor([[2.], [7.]])
    next_values = torch.tensor([[3.], [0.]])
    term = torch.tensor([[False], [True]])
    trunc = torch.tensor([[True], [False]])
    advantage, returns = generalized_advantage(rewards, values, next_values, term, trunc, gamma=.9)
    assert advantage[0].item() == pytest.approx(1 + .9 * 3 - 2)
    assert returns[1].item() == 100
    term[0], trunc[0] = True, False
    _, returns = generalized_advantage(rewards, values, next_values, term, trunc, gamma=.9)
    assert returns[0].item() == 1


def test_ppo_clipping_for_both_advantage_signs():
    lp = torch.tensor([[1.4], [.6]]).log().requires_grad_()
    loss, _ = ppo_objective(lp, torch.zeros_like(lp), torch.tensor([2., -3.]), torch.zeros_like(lp), entropy_coef=0)
    assert loss.item() == pytest.approx((-2.4 + 2.4) / 2, abs=1e-6)
    loss.backward()
    assert torch.equal(lp.grad, torch.zeros_like(lp))


def test_empty_active_samples_are_finite_with_zero_gradient():
    lp = torch.zeros(3, 2, requires_grad=True)
    loss, _ = ppo_objective(lp, lp.detach(), torch.ones(3), torch.ones_like(lp), active=torch.zeros_like(lp))
    loss.backward()
    assert loss.item() == 0 and lp.grad.abs().sum() == 0


def test_policy_collection_likelihood_and_mask():
    torch.manual_seed(1)
    policy = IndependentPolicy(2, 3, 4, 16)
    obs = torch.randn(7, 3, 2)
    mask = torch.ones(7, 3, 4, dtype=torch.bool)
    mask[..., 2] = False
    action, lp = policy.sample(obs, mask)
    assert (action != 2).all()
    torch.testing.assert_close(lp, policy.evaluate_actions(obs, action, mask)[0])
    with pytest.raises(ValueError):
        policy.sample(obs, torch.zeros_like(mask))


def test_matrix_payoffs_and_exact_optimum():
    climbing, penalty = MatrixGame('climbing'), MatrixGame('penalty')
    assert climbing.reward(torch.tensor([[0, 0], [1, 1], [2, 1]])).tolist() == [11, 7, 6]
    assert penalty.reward(torch.tensor([[0, 2], [2, 0], [0, 0]])).tolist() == [10, 10, -100]
    assert climbing.reward(climbing.joint_actions()).max() == climbing.optimum == 11
    assert penalty.optimum == 10
