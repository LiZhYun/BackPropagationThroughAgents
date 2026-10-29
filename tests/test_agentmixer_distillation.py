"""An opt-in actor-only KL, checked against an independent categorical reference."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from bta_repro.agentmixer import AgentMixerPolicy
from bta_repro.agentmixer_joint import JointAgentMixerPolicy
from bta_repro.agentmixer_continuous import ContinuousAgentMixerPolicy
from bta_repro.core import mlp
from bta_repro.recurrent import RecurrentAgentMixerPolicy
from bta_repro.runner import collect_rollout, update_rollout
from bta_repro.simulator_train import Config, train
from test_runner import CounterEnv, optimizers_for


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('all_dead', [False, True])
def test_kl_value_and_analytic_gradient_with_masks_and_inactive_agents(monkeypatch, dtype, all_dead):
    policy = AgentMixerPolicy(1, 1, 2, 3, 8).to(dtype)
    local = torch.tensor([[[1., 2., 3.], [4., 1., 2.]],
                          [[3., 1., 2.], [2., 3., 1.]],
                          [[1., 3., 2.], [3., 2., 1.]]], dtype=dtype, requires_grad=True)
    teacher = (local.detach() * 2).requires_grad_()
    mask = torch.tensor([[[1, 0, 1], [1, 1, 1]], [[0, 1, 0], [1, 0, 1]], [[1, 1, 1], [1, 1, 1]]]).bool()
    active = torch.tensor([[1, 0], [1, 1], [0, 0]]).bool()
    if all_dead:
        active.zero_()
    monkeypatch.setattr(policy, 'local_logits', lambda obs: local)
    monkeypatch.setattr(policy, 'joint_logits', lambda obs, state, available: teacher.masked_fill(~available, -torch.inf))
    loss = policy.distillation_loss(torch.empty(3, 2, 1), torch.empty(3, 1), mask, active)
    expected = 0.
    denominator = max(1, int(active.any(-1).sum()))
    gradient = torch.zeros_like(local)
    # Enumerate each factor's legal support; no masked -inf arithmetic.
    for b, i in active.nonzero().tolist():
        legal = mask[b, i]
        p = torch.softmax(local.detach()[b, i, legal], -1)
        q = torch.softmax(teacher.detach()[b, i, legal], -1)
        expected += float((q * (q.log() - p.log())).sum()) / denominator
        gradient[b, i, legal] = (p - q) / denominator
    assert float(loss.detach()) == pytest.approx(expected, abs=2e-7)
    loss.backward()
    torch.testing.assert_close(local.grad, gradient, atol=1e-7, rtol=1e-5)
    assert teacher.grad is None
    assert all(p.grad is None for p in policy.modifier.parameters())


@pytest.mark.parametrize('recurrent', [False, True])
@pytest.mark.parametrize('bounded', [False, True])
def test_actual_policy_routes_only_local_gradient_and_preserves_modes(recurrent, bounded):
    torch.manual_seed(18)
    cls = RecurrentAgentMixerPolicy if recurrent else AgentMixerPolicy
    policy = cls(2, 3, 2, 4, 8, log_scale_bound=1. if bounded else None).double()
    with torch.no_grad():
        policy.modifier.scale_head.bias.fill_(.8)
    obs, state = torch.randn(5, 2, 2, dtype=torch.float64), torch.randn(5, 3, dtype=torch.float64)
    if recurrent:
        obs = policy.pack_observations(obs, torch.randn(5, 2, 8, dtype=torch.float64))
    mask = torch.ones(5, 2, 4, dtype=torch.bool)
    mask[:, :, 1] = False
    before = deepcopy(policy.state_dict())
    loss = policy.distillation_loss(obs, state, mask)
    assert loss > 0
    loss.backward()
    assert all(p.grad is None for p in policy.modifier.parameters())
    for actor in policy.local_actors:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())
        assert sum(p.grad.norm() for p in actor.parameters()) > 0
    optimizer = torch.optim.SGD(policy.parameters(), lr=.05)
    optimizer.step()
    assert all(torch.equal(value, policy.state_dict()[key]) for key, value in before.items() if key.startswith('modifier.'))
    assert torch.equal(policy.local_actions(obs, mask), policy.joint_logits(obs, state, mask).argmax(-1))


def test_runner_applies_coefficient_and_joint_reduction_once():
    torch.manual_seed(49)
    env = CounterEnv(masks=True, horizon=5)
    policy = AgentMixerPolicy(1, 1, 2, 3, 8)
    with torch.no_grad():
        policy.modifier.scale_head.bias.fill_(1.)
    critic = mlp(1, 1, 8)
    batch, _, _ = collect_rollout([env], policy, critic, 'agentmixer', 4)
    batch.advantages.zero_()  # Isolate the auxiliary update from PPO and entropy.
    reference = deepcopy(policy)
    expected = reference.distillation_loss(batch.obs, batch.state, batch.available_actions, batch.active_agents)
    (.3 * expected).backward()
    torch.optim.SGD(reference.parameters(), lr=.01).step()
    result = update_rollout(policy, critic, optimizers_for(policy, critic, 'agentmixer'), batch,
                            'agentmixer', True, epochs=1, entropy_coef=0, value_coef=0,
                            normalize_advantages=False, max_grad_norm=1e6, agentmixer_distillation_coef=.3)
    assert result['distillation_kl'] == pytest.approx(float(expected.detach()), abs=1e-7)
    assert result['actor_total_loss'] == pytest.approx(.3 * result['distillation_kl'])
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, policy.state_dict()[name], rtol=0, atol=1e-7)
    # An all-dead rollout must not create an actor update or change its optimizer.
    before = deepcopy(policy.state_dict())
    batch.active_agents.zero_()
    result = update_rollout(policy, critic, optimizers_for(policy, critic, 'agentmixer'), batch,
                            'agentmixer', True, agentmixer_distillation_coef=.3, value_coef=0)
    assert result['num_updates'] == result['distillation_loss'] == 0
    assert all(torch.equal(value, policy.state_dict()[key]) for key, value in before.items())


@pytest.mark.parametrize('options', [
    {'agentmixer_distillation_coef': value} for value in [-1, True, '1', None, math.inf, math.nan]
] + [
    {'agentmixer_distillation_coef': .1, **option} for option in [
        {'algorithm': 'mappo'}, {'algorithm': 'arppo'}, {'algorithm': 'bppo'},
        {'environment': 'mujoco'}, {'agentmixer_correlation': 'joint'}, {'modifier_enabled': False},
    ]
])
def test_configuration_rejects_unsupported_objective(options):
    with pytest.raises(ValueError, match='agentmixer_distillation_coef'):
        Config(**{'environment': 'prey', **options}).validate()


@pytest.mark.parametrize('kind', ['continuous', 'joint', 'identity', 'other_algorithm'])
def test_runner_rejects_unsupported_policy_before_any_update(kind):
    policy = (ContinuousAgentMixerPolicy(1, 1, 2, 2, 8) if kind == 'continuous' else
              JointAgentMixerPolicy(1, 1, 2, 3, 8) if kind == 'joint' else
              AgentMixerPolicy(1, 1, 2, 3, 8))
    if kind == 'identity':
        policy.modifier.requires_grad_(False)
    with pytest.raises(ValueError, match='agentmixer_distillation_coef'):
        update_rollout(policy, None, {}, None, 'mappo' if kind == 'other_algorithm' else 'agentmixer',
                       kind != 'continuous', agentmixer_distillation_coef=.1)


def test_zero_coefficient_does_not_compute_auxiliary_loss(monkeypatch):
    policy, critic = AgentMixerPolicy(1, 1, 2, 3, 8), mlp(1, 1, 8)
    batch, _, _ = collect_rollout([CounterEnv()], policy, critic, 'agentmixer', 2)
    def forbidden(*args, **kwargs):
        raise AssertionError('Disabled auxiliary loss must not be evaluated')
    monkeypatch.setattr(policy, 'distillation_loss', forbidden)
    stats = update_rollout(policy, critic, optimizers_for(policy, critic, 'agentmixer'), batch,
                            'agentmixer', True, agentmixer_distillation_coef=0.)
    assert 'distillation_kl' not in stats and 'actor_total_loss' not in stats


@pytest.mark.parametrize('recurrent', [False, True])
def test_real_prey_training_checkpoint_replay_and_explicit_label(tmp_path, recurrent):
    config = Config(environment='prey', recurrent_actor=recurrent, agentmixer_distillation_coef=.1,
                    env_config={'penalty': -1, 'episode_limit': 4}, steps=16, num_envs=2,
                    rollout_steps=4, hidden_dim=8, epochs=2, eval_episodes=2,
                    eval_windows=2, eval_window_steps=4, normalize_value_targets=True)
    run = tmp_path / 'prey'
    summary = train(config, run)
    metadata = json.loads((run / 'config.json').read_text())
    assert metadata['method_variant'] == 'mode_consistent_modifier_experimental_local_distillation'
    assert metadata['auxiliary_objective']['paper_objective'] is False
    checkpoint = torch.load(run / 'checkpoint.pt', weights_only=True)
    assert checkpoint['config']['agentmixer_distillation_coef'] == .1
    updates = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    updates = [entry for entry in updates if entry['event'] == 'update']
    assert len(updates) == 2
    assert all(math.isfinite(entry['losses']['distillation_loss']) for entry in updates)
    path = Path(__file__).resolve().parents[1] / 'scripts/evaluate_simulator.py'
    spec = importlib.util.spec_from_file_location('evaluate_distillation_checkpoint', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    replay = module.evaluate_saved(run)
    assert replay['source_matches_training']
    assert replay['evaluation'] == summary['final_evaluation']
