"""Temporal runner checks with deterministic, simulator-free environments."""

from copy import deepcopy
import math

import numpy as np
import pytest
import torch
from torch import nn

from bta_repro.agentmixer import AgentMixerPolicy
from bta_repro.agentmixer_continuous import ContinuousAgentMixerPolicy
from bta_repro.bppo import AutoregressivePolicy
from bta_repro.bppo_continuous import ContinuousAutoregressivePolicy
from bta_repro.core import ContinuousIndependentPolicy, IndependentPolicy, mlp
from bta_repro.runner import ObservationCarry, collect_rollout, update_rollout
from bta_repro.value_normalization import ValueNormalizedCritic


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(307)


class CounterEnv:
    num_agents = 2
    obs_dim = state_dim = 1
    num_actions = 3
    action_dim = 2

    def __init__(self, index=0, horizon=2, truncates=False, is_discrete=True, masks=False):
        self.index, self.horizon, self.truncates = index, horizon, truncates
        self.is_discrete, self.masks = is_discrete, masks
        self.episode = -1
        self.t = 0
        self.raw_actions = []
        self.closed = False

    def observation(self):
        value = 100 * self.index + 10 * self.episode + self.t
        result = {"obs": np.full((2, 1), value, dtype=np.float32),
                  "state": np.array([value], dtype=np.float32)}
        if self.masks:
            available = np.ones((2, 3), dtype=bool)
            active = np.ones(2, dtype=bool)
            if self.t % 2 == 1:
                active[1] = False
                available[1] = [True, False, False]
            if self.t == self.horizon:
                # Final states need a critic bootstrap, but no next action.
                available[:] = False
            result.update(available_actions=available, active_agents=active)
        return result

    def reset(self, seed=None):
        self.episode += 1
        self.t = 0
        return self.observation(), {"seed": seed}

    def step(self, raw_actions):
        self.raw_actions.append(raw_actions.copy())
        if not self.is_discrete:
            np.clip(raw_actions, -1, 1, out=raw_actions)
        self.t += 1
        ended = self.t == self.horizon
        return (self.observation(), float(self.index + 1), ended and not self.truncates,
                ended and self.truncates, {"episode": self.episode})

    def close(self):
        self.closed = True


class IdentityCritic(nn.Module):
    def forward(self, state):
        return state[:, 0]


def make_policy(algorithm, discrete):
    if algorithm == "agentmixer":
        return AgentMixerPolicy(1, 1, 2, 3, 8) if discrete else ContinuousAgentMixerPolicy(1, 1, 2, 2, 8)
    if algorithm == "mappo":
        return IndependentPolicy(1, 2, 3, 8) if discrete else ContinuousIndependentPolicy(1, 2, 2, 8)
    return AutoregressivePolicy(1, 2, 3, 8) if discrete else ContinuousAutoregressivePolicy(1, 2, 2, 8)


def optimizers_for(policy, critic, algorithm):
    result = {"critic": torch.optim.SGD(critic.parameters(), lr=0.01)}
    if algorithm == "bppo":
        result.update({f"actor_{i}": torch.optim.SGD(actor.parameters(), lr=0.01)
                       for i, actor in enumerate(policy.actors)})
    else:
        result["policy"] = torch.optim.SGD(policy.parameters(), lr=0.01)
    return result


def test_unequal_episode_lengths_terminal_bootstrap_and_flatten_identity():
    envs = [CounterEnv(0, 2), CounterEnv(1, 3, truncates=True)]
    policy = make_policy("arppo", True)
    batch, carry, records = collect_rollout(
        envs, policy, IdentityCritic(), "arppo", 4, gamma=1, gae_lambda=1
    )
    assert batch.rollout_shape == (4, 2)
    expected_states = torch.tensor([0., 100., 1., 101., 10., 102., 11., 110.])
    torch.testing.assert_close(batch.state[:, 0], expected_states)
    torch.testing.assert_close(batch.obs[:, 0, 0], expected_states)
    torch.testing.assert_close(batch.values, expected_states.reshape(4, 2))
    torch.testing.assert_close(batch.next_values, torch.tensor([[1., 101.], [2., 102.], [11., 103.], [12., 111.]]))
    torch.testing.assert_close(batch.advantages.reshape(4, 2), torch.tensor([[2., 9.], [0., 6.], [-8., 3.], [-10., 3.]]))
    torch.testing.assert_close(batch.returns.reshape(4, 2), torch.tensor([[2., 109.], [1., 107.], [2., 105.], [1., 113.]]))
    assert [(r["env_index"], r["length"], r["return"]) for r in records] == [(0, 2, 2.), (1, 3, 6.), (0, 2, 2.)]
    assert records[0]["terminated"] and records[1]["truncated"]
    assert isinstance(carry, ObservationCarry)
    assert carry[0]["state"][0] == 20 and carry[1]["state"][0] == 111
    assert carry.episode_lengths == [0, 1]
    assert not any(env.closed for env in envs)


def test_partial_episode_accounting_continues_across_rollout_calls():
    envs = [CounterEnv(0, 2), CounterEnv(1, 3, truncates=True)]
    policy = make_policy("mappo", True)
    _, carry, _ = collect_rollout(envs, policy, IdentityCritic(), "mappo", 4)
    _, next_carry, records = collect_rollout(envs, policy, IdentityCritic(), "mappo", 2, observations=carry)
    assert [(r["env_index"], r["length"], r["return"]) for r in records] == [(0, 2, 2.), (1, 3, 6.)]
    assert next_carry.episode_lengths == [0, 0]
    assert next_carry.episode_returns == [0., 0.]


@pytest.mark.parametrize("algorithm,discrete", [
    ("mappo", True), ("arppo", True), ("bppo", True), ("agentmixer", True),
    ("mappo", False), ("arppo", False), ("bppo", False), ("agentmixer", False),
])
@pytest.mark.parametrize("normalize_values", [False, True])
def test_collected_likelihoods_match_and_small_updates_are_finite(algorithm, discrete, normalize_values):
    envs = [CounterEnv(is_discrete=discrete), CounterEnv(horizon=3, truncates=True, is_discrete=discrete)]
    policy, critic = make_policy(algorithm, discrete), mlp(1, 1, 8)
    if normalize_values:
        critic = ValueNormalizedCritic(critic)
    batch, _, _ = collect_rollout(envs, policy, critic, algorithm, 3)
    if algorithm == "agentmixer":
        log_probs, _ = (policy.evaluate_actions(batch.obs, batch.state, batch.actions, batch.available_actions)
                        if discrete else policy.evaluate_actions(batch.obs, batch.state, batch.actions))
    else:
        log_probs, _ = (policy.evaluate_actions(batch.obs, batch.actions, batch.available_actions)
                        if discrete else policy.evaluate_actions(batch.obs, batch.actions))
    torch.testing.assert_close(log_probs, batch.old_log_probs)
    assert not batch.old_log_probs.requires_grad and not batch.advantages.requires_grad
    assert batch.available_actions is not None if discrete else batch.available_actions is None
    before = deepcopy(critic.state_dict())
    raw_returns, raw_advantages = batch.returns.clone(), batch.advantages.clone()
    metrics = update_rollout(
        policy, critic, optimizers_for(policy, critic, algorithm), batch, algorithm, discrete, epochs=2,
        value_clip_coef=.2 if normalize_values else None, value_huber_delta=5. if normalize_values else None,
    )
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["num_updates"] > 0 and metrics["critic_updates"] == 2
    assert any(not torch.equal(value, before[key]) for key, value in critic.state_dict().items())
    torch.testing.assert_close(batch.returns, raw_returns, atol=0, rtol=0)
    torch.testing.assert_close(batch.advantages, raw_advantages, atol=0, rtol=0)
    if normalize_values:
        assert critic.normalizer.count == 6 and critic.normalizer.updates == 1
        assert metrics["value_targets_normalized"] == 1
        assert metrics["raw_value_mse"] == metrics["value_mse"]


def test_raw_gaussian_actions_survive_in_place_environment_clipping():
    env = CounterEnv(is_discrete=False)
    policy = make_policy("bppo", False)
    with torch.no_grad():
        for actor in policy.actors:
            actor.mean[-1].weight.zero_()
            actor.mean[-1].bias.fill_(3.)
            actor.log_std.fill_(-5.)
    batch, _, _ = collect_rollout([env], policy, IdentityCritic(), "bppo", 3)
    assert (batch.actions > 1).all()
    torch.testing.assert_close(batch.actions, torch.tensor(np.stack(env.raw_actions)))
    log_probs, _ = policy.evaluate_actions(batch.obs, batch.actions)
    torch.testing.assert_close(log_probs, batch.old_log_probs)


@pytest.mark.parametrize("algorithm", ["mappo", "arppo", "bppo", "agentmixer"])
def test_availability_and_active_masks_survive_terminal_states(algorithm):
    env = CounterEnv(horizon=2, masks=True)
    policy, critic = make_policy(algorithm, True), mlp(1, 1, 8)
    batch, _, _ = collect_rollout([env], policy, critic, algorithm, 4)
    assert batch.active_agents.tolist() == [[True, True], [True, False], [True, True], [True, False]]
    assert (batch.actions[1::2, 1] == 0).all()
    assert batch.available_actions.gather(-1, batch.actions.unsqueeze(-1)).all()
    metrics = update_rollout(
        policy, critic, optimizers_for(policy, critic, algorithm), batch, algorithm, True, epochs=2
    )
    assert metrics["active_fraction"] == .75
    assert all(math.isfinite(value) for value in metrics.values())


def test_value_coefficient_is_explicit_and_zero_skips_optimizer():
    policy, critic = make_policy("mappo", True), mlp(1, 1, 8)
    batch, _, _ = collect_rollout([CounterEnv()], policy, critic, "mappo", 2)
    expected_mse = (critic(batch.state).squeeze(-1) - batch.returns).square().mean().item()
    optimizers = optimizers_for(policy, critic, "mappo")
    metrics = update_rollout(policy, critic, optimizers, batch, "mappo", True, epochs=1, value_coef=.25)
    assert metrics["value_mse"] == pytest.approx(expected_mse)
    assert metrics["value_loss"] == pytest.approx(.25 * expected_mse)
    batch, _, _ = collect_rollout([CounterEnv()], policy, critic, "mappo", 2)
    before = deepcopy(critic.state_dict())
    expected_disabled_mse = (critic(batch.state).squeeze(-1) - batch.returns).square().mean().item()
    metrics = update_rollout(policy, critic, optimizers, batch, "mappo", True, epochs=1, value_coef=0)
    assert metrics["critic_updates"] == 0
    assert metrics["value_loss"] == 0
    assert metrics["value_mse"] == pytest.approx(expected_disabled_mse)
    for key, value in critic.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_observation_carry_rejects_different_environment_order():
    envs = [CounterEnv(), CounterEnv()]
    policy = make_policy("mappo", True)
    _, carry, _ = collect_rollout(envs, policy, IdentityCritic(), "mappo", 1)
    with pytest.raises(ValueError, match="different environment list or order"):
        collect_rollout(list(reversed(envs)), policy, IdentityCritic(), "mappo", 1, observations=carry)


def test_seeded_initial_observations_are_not_reset_again():
    env = CounterEnv()
    observations = [env.reset(seed=123)[0]]
    policy = make_policy("mappo", True)
    batch, _, _ = collect_rollout([env], policy, IdentityCritic(), "mappo", 1, observations=observations)
    assert env.episode == 0
    assert batch.state[0, 0] == 0


def test_identical_returns_do_not_create_spurious_policy_score_gradients():
    class ConstantRewardEnv(CounterEnv):
        def step(self, raw_actions):
            observation, _, terminated, truncated, info = super().step(raw_actions)
            return observation, 12345.678, terminated, truncated, info

    policy, critic = make_policy("mappo", True), mlp(1, 1, 8)
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
    batch, _, _ = collect_rollout([ConstantRewardEnv(horizon=1)], policy, critic, "mappo", 7)
    assert torch.unique(batch.returns).numel() == torch.unique(batch.advantages).numel() == 1
    before = deepcopy(policy.state_dict())
    metrics = update_rollout(
        policy, critic, optimizers_for(policy, critic, "mappo"), batch, "mappo", True,
        epochs=2, entropy_coef=0, value_coef=0,
    )
    assert metrics["policy_loss"] == metrics["grad_norm"] == 0
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


@pytest.mark.parametrize("truncates,expected_returns", [(False, [2., 1.]), (True, [18., 17.])])
def test_normalized_critic_supplies_raw_terminal_and_truncation_values_to_gae(truncates, expected_returns):
    critic = ValueNormalizedCritic(IdentityCritic())
    critic.update_statistics(torch.tensor([10., 14.]))  # raw value = 12 + 2*state.
    before = deepcopy(critic.normalizer.state_dict())
    batch, _, _ = collect_rollout([CounterEnv(truncates=truncates)], make_policy("mappo", True),
                                  critic, "mappo", 2, gamma=1., gae_lambda=1.)
    torch.testing.assert_close(batch.values.flatten(), torch.tensor([12., 14.]))
    torch.testing.assert_close(batch.next_values.flatten(), torch.tensor([14., 16.]))
    torch.testing.assert_close(batch.returns, torch.tensor(expected_returns))
    for key, value in before.items():
        torch.testing.assert_close(critic.normalizer.state_dict()[key], value, atol=0, rtol=0)


def test_value_clipping_transforms_old_raw_values_with_current_statistics_once_per_rollout():
    base = nn.Linear(1, 1)
    with torch.no_grad():
        base.weight.zero_()
        base.bias.zero_()
    critic = ValueNormalizedCritic(base)
    critic.update_statistics(torch.tensor([0., 2.]))
    policy = make_policy("mappo", True)
    batch, _, _ = collect_rollout([CounterEnv()], policy, critic, "mappo", 2)
    torch.testing.assert_close(batch.values.flatten(), torch.ones(2))
    batch.returns = torch.tensor([10., 14.])
    optimizers = optimizers_for(policy, critic, "mappo")
    for optimizer in optimizers.values():
        for group in optimizer.param_groups:
            group["lr"] = 0.
    metrics = update_rollout(policy, critic, optimizers, batch, "mappo", True,
                             epochs=3, value_coef=.25, value_clip_coef=.2)
    history = torch.tensor([0., 2., 10., 14.], dtype=torch.float64)
    mean, scale = history.mean(), history.std(unbiased=False)
    targets = ((batch.returns.double() - mean) / scale).float()
    old = ((torch.ones(2, dtype=torch.float64) - mean) / scale).float()
    prediction = torch.zeros(2)
    clipped = old + (prediction - old).clamp(-.2, .2)
    expected = torch.maximum((prediction - targets).square(), (clipped - targets).square()).mean()
    assert metrics["value_regression_loss"] == pytest.approx(float(expected))
    assert metrics["value_loss"] == pytest.approx(.25 * float(expected))
    assert metrics["raw_value_mse"] == pytest.approx(34.25)
    assert metrics["value_target_mean"] == pytest.approx(float(mean))
    assert metrics["value_target_scale"] == pytest.approx(float(scale))
    assert critic.normalizer.count == 4 and critic.normalizer.updates == 2


def test_raw_value_clipping_and_huber_are_supported_without_normalization():
    critic = nn.Linear(1, 1)
    with torch.no_grad():
        critic.weight.zero_()
        critic.bias.zero_()
    policy = make_policy("mappo", True)
    batch, _, _ = collect_rollout([CounterEnv()], policy, critic, "mappo", 2)
    batch.returns = torch.tensor([10., 14.])
    with torch.no_grad():
        critic.bias.fill_(3.)
    metrics = update_rollout(policy, critic, optimizers_for(policy, critic, "mappo"), batch,
                             "mappo", True, epochs=1, value_clip_coef=.2, value_huber_delta=2.)
    expected = torch.nn.functional.huber_loss(torch.full((2,), .2), batch.returns, delta=2.)
    assert metrics["value_regression_loss"] == pytest.approx(float(expected))
    assert metrics["raw_value_mse"] == pytest.approx((7**2 + 11**2) / 2)
    assert metrics["value_targets_normalized"] == 0
