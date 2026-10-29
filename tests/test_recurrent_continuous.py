"""Private continuous history, IGC, raw likelihood and rollout boundary checks."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from bta_repro import simulator_train as training
from bta_repro.core import mlp
from bta_repro.recurrent import RecurrentGaussianActor
from bta_repro.runner import collect_rollout, update_rollout
from test_runner import CounterEnv, IdentityCritic
from test_simulator_train import FakeEnvironment

VARIANTS = [(modifier, local, enabled) for local in ("observation", "state_independent")
            for modifier, enabled in (("residual", True), ("bounded", True), ("residual", False))]


def config(variant, **kwargs):
    modifier, local, enabled = variant
    return training.Config(environment="mujoco", algorithm="agentmixer", recurrent_actor=True,
                           continuous_modifier_scale=modifier, continuous_local_scale=local,
                           modifier_enabled=enabled, initial_std=.3, hidden_dim=8, **kwargs)


@pytest.mark.parametrize("variant", VARIANTS)
def test_rollout_incoming_memory_raw_likelihood_and_per_environment_resets(variant):
    cfg = config(variant)
    cfg.validate()
    envs = [CounterEnv(horizon=2, is_discrete=False),
            CounterEnv(index=1, horizon=3, truncates=True, is_discrete=False)]
    policy = training.make_policy(cfg, envs[0])
    batch, carry, records = collect_rollout(envs, policy, IdentityCritic(), cfg.algorithm, 7)
    packed = batch.obs.reshape(7, 2, 2, 9)
    memory = None
    for step in range(7):
        expected = policy.pack_observations(packed[step, :, :, :1], memory)
        torch.testing.assert_close(packed[step], expected, rtol=0, atol=0)
        memory = policy.advance_memory(expected)
        memory[batch.terminated[step] | batch.truncated[step]] = 0
    torch.testing.assert_close(carry.policy_memory, memory, rtol=0, atol=0)
    logs, entropy = policy.evaluate_actions(batch.obs, batch.state, batch.actions)
    mean, std = policy.joint_parameters(batch.obs, batch.state)
    torch.testing.assert_close(logs, batch.old_log_probs, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(logs, Normal(mean, std).log_prob(batch.actions).sum(-1))
    torch.testing.assert_close(entropy, Normal(mean, std).entropy().sum(-1))
    assert batch.actions.shape == (14, 2, 2) and batch.actions.is_floating_point()
    assert not batch.obs.requires_grad and not carry.policy_memory.requires_grad
    assert len(records) == 5


@pytest.mark.parametrize("variant", VARIANTS)
def test_split_rollout_preserves_history_and_sampling_rng(variant):
    cfg = config(variant)
    policy = training.make_policy(cfg, CounterEnv(is_discrete=False))
    make = lambda: [CounterEnv(horizon=3, is_discrete=False),
                    CounterEnv(index=1, horizon=5, truncates=True, is_discrete=False)]
    torch.manual_seed(38)
    whole, whole_carry, _ = collect_rollout(make(), policy, IdentityCritic(), cfg.algorithm, 7)
    rng = torch.get_rng_state().clone()
    envs = make()
    torch.manual_seed(38)
    first, carry, _ = collect_rollout(envs, policy, IdentityCritic(), cfg.algorithm, 2)
    second, split_carry, _ = collect_rollout(envs, policy, IdentityCritic(), cfg.algorithm, 5, carry)
    for name in ("obs", "actions", "old_log_probs"):
        torch.testing.assert_close(getattr(whole, name), torch.cat((getattr(first, name), getattr(second, name))), rtol=0, atol=0)
    assert torch.equal(whole_carry.policy_memory, split_carry.policy_memory)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_modes_stay_equal_and_only_local_history_controls_local_actions(variant, dtype):
    torch.manual_seed(93)
    cfg = config(variant)
    policy = training.make_policy(cfg, CounterEnv(is_discrete=False)).to(dtype=dtype)
    if cfg.modifier_enabled:
        nn.init.normal_(policy.modifier.log_std_head.weight, std=.2)
    previous = torch.randn(4, 2, 1, dtype=dtype)
    changed = previous.clone()
    changed[:, 0] += 5
    memory = policy.advance_memory(policy.pack_observations(previous))
    other_memory = policy.advance_memory(policy.pack_observations(changed))
    obs = torch.randn(4, 2, 1, dtype=dtype)
    packed = policy.pack_observations(obs, memory)
    different = policy.pack_observations(obs, other_memory)
    local = policy.local_actions(packed)
    assert torch.equal(local[:, 1], policy.local_actions(different)[:, 1])
    assert not torch.equal(local[:, 0], policy.local_actions(different)[:, 0])
    for state in (torch.zeros(4, 1, dtype=dtype), torch.full((4, 1), 20., dtype=dtype)):
        central, _ = policy.sample(packed, state, deterministic=True)
        assert torch.equal(local, central)
    snapshot = deepcopy(policy.state_dict())
    policy.advance_memory(packed)
    assert all(torch.equal(value, policy.state_dict()[key]) for key, value in snapshot.items())

    class ForbiddenModifier(nn.Module):
        def forward(self, *args):
            raise AssertionError("decentralized execution consulted the modifier")

    policy.modifier = ForbiddenModifier()
    assert torch.equal(policy.local_actions(packed), local)
    sampled, _ = policy.sample(packed, decentralized=True)
    assert sampled.shape == local.shape
    independent = packed.detach().clone().requires_grad_()
    policy.local_actions(independent)[:, 0].sum().backward()
    assert torch.count_nonzero(independent.grad[:, 1]) == 0
    assert torch.count_nonzero(independent.grad[:, :, 1:]) == 0
    assert torch.count_nonzero(independent.grad[:, 0, :1]) > 0


@pytest.mark.parametrize("local_scale", ["observation", "state_independent"])
def test_gaussian_gru_gradient_matches_finite_difference_and_stops_at_incoming_memory(local_scale):
    torch.manual_seed(117)
    actor = RecurrentGaussianActor(3 + 8, 2, 8, raw_obs_dim=3, initial_std=.3, local_scale=local_scale).double()
    inputs = torch.randn(4, 11, dtype=torch.float64, requires_grad=True)
    actions = torch.randn(4, 2, dtype=torch.float64)
    def score():
        mean, std = actor(inputs)
        return Normal(mean, std).log_prob(actions).sum()
    score().backward()
    analytic = actor.cell.weight_hh.grad[0, 0].clone()
    assert torch.count_nonzero(inputs.grad[:, :3]) > 0
    assert torch.count_nonzero(inputs.grad[:, 3:]) == 0
    for parameter in (actor.features[0].weight, actor.cell.weight_ih, actor.cell.weight_hh, actor.mean_head.weight):
        assert torch.count_nonzero(parameter.grad) > 0
    with torch.no_grad():
        original = actor.cell.weight_hh[0, 0].clone()
        actor.cell.weight_hh[0, 0] = original + 1e-5
        plus = score().item()
        actor.cell.weight_hh[0, 0] = original - 1e-5
        minus = score().item()
        actor.cell.weight_hh[0, 0] = original
    torch.testing.assert_close(analytic, analytic.new_tensor((plus - minus) / 2e-5), rtol=2e-5, atol=1e-8)


@pytest.mark.parametrize("variant", VARIANTS)
def test_ppo_updates_encoders_gru_and_heads_without_changing_recorded_history(variant):
    torch.manual_seed(72)
    cfg = config(variant)
    envs = [CounterEnv(horizon=5, is_discrete=False), CounterEnv(index=1, horizon=4, is_discrete=False)]
    policy = training.make_policy(cfg, envs[0])
    critic = mlp(1, 1, 8)
    optimizers = training.make_optimizers(cfg, policy, critic)
    batch, carry, _ = collect_rollout(envs, policy, critic, cfg.algorithm, 8)
    before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
    packed, memory = batch.obs.clone(), carry.policy_memory.clone()
    losses = update_rollout(policy, critic, optimizers, batch, cfg.algorithm, False, epochs=2, num_minibatches=3)
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    for term in ("features.0.weight", "cell.weight_ih", "cell.weight_hh", "mean_head.weight"):
        assert any(not torch.equal(before[name], parameter) for name, parameter in policy.named_parameters() if term in name)
    assert torch.equal(packed, batch.obs) and torch.equal(memory, carry.policy_memory)
    owned = {id(parameter) for group in optimizers['policy'].param_groups for parameter in group['params']}
    assert owned == {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
    assert torch.equal(policy.local_actions(batch.obs), policy.sample(batch.obs, batch.state, deterministic=True)[0])
    if not cfg.modifier_enabled:
        assert all(torch.equal(before[name], parameter) for name, parameter in policy.named_parameters() if name.startswith('modifier.'))


@pytest.mark.parametrize("variant", VARIANTS)
def test_training_and_saved_policy_replay_with_normalization_and_episode_resets(tmp_path, monkeypatch, variant):
    cfg = config(variant, steps=16, num_envs=2, rollout_steps=3, epochs=2, num_minibatches=2,
                 eval_episodes=3, eval_seed=99, normalize_observations=True, normalize_value_targets=True)
    monkeypatch.setattr(training, 'make_env', lambda config, seed: FakeEnvironment(config, seed))
    run = tmp_path / 'continuous_recurrent'
    summary = training.train(cfg, run)
    metadata = json.loads((run / 'config.json').read_text())
    assert metadata['actor_memory']['enabled'] and metadata['actor_memory']['gradient_chunk_length'] == 1
    checkpoint = torch.load(run / 'checkpoint.pt', weights_only=True)
    assert checkpoint['config']['recurrent_actor'] and any('cell.weight_hh' in key for key in checkpoint['policy'])
    path = Path(__file__).resolve().parents[1] / 'scripts/evaluate_simulator.py'
    spec = importlib.util.spec_from_file_location('evaluate_continuous_recurrent', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    replay = module.evaluate_saved(run)
    assert replay['source_matches_training'] and replay['checkpoint_is_final'] and replay['versions_match_training']
    assert replay['evaluation'] == summary['final_evaluation']


@pytest.mark.parametrize('algorithm', ['mappo', 'arppo', 'bppo'])
def test_continuous_recurrence_outside_agentmixer_is_rejected_by_config_and_factory(algorithm):
    cfg = training.Config(environment='mujoco', algorithm=algorithm, recurrent_actor=True)
    with pytest.raises(ValueError, match='recurrent'):
        cfg.validate()
    with pytest.raises(ValueError, match='recurrent'):
        training.make_policy(cfg, CounterEnv(is_discrete=False))
