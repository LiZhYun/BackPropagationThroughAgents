"""Memory boundary, likelihood, gradient and execution checks for recurrent actors."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from bta_repro import simulator_train as training
from bta_repro.core import mlp
from bta_repro.prey_env import PreyEnv
from bta_repro.recurrent import RecurrentActor
from bta_repro.runner import collect_rollout, update_rollout
from test_runner import CounterEnv, IdentityCritic


METHODS = ["mappo", "arppo", "bppo", "agentmixer", "agentmixer_identity"]


def config(method, **kwargs):
    return training.Config(algorithm="agentmixer" if method.endswith("identity") else method,
                           modifier_enabled=not method.endswith("identity"), recurrent_actor=True,
                           hidden_dim=8, **kwargs)


def likelihood(policy, cfg, batch):
    arguments = (batch.obs, batch.state, batch.actions) if cfg.algorithm == "agentmixer" else (batch.obs, batch.actions)
    return policy.evaluate_actions(*arguments, available_actions=batch.available_actions)[0]


@pytest.mark.parametrize("method", METHODS)
def test_rollout_stores_incoming_memory_replays_likelihood_and_resets_each_environment(method):
    cfg = config(method)
    envs = [CounterEnv(horizon=2), CounterEnv(index=1, horizon=3, truncates=True)]
    policy = training.make_policy(cfg, envs[0])
    batch, carry, records = collect_rollout(envs, policy, IdentityCritic(), cfg.algorithm, 7)
    packed = batch.obs.reshape(7, 2, 2, 9)
    memory = None
    for step in range(7):
        expected = policy.pack_observations(packed[step, :, :, :1], memory)
        torch.testing.assert_close(packed[step], expected)
        memory = policy.advance_memory(expected)
        memory[batch.terminated[step] | batch.truncated[step]] = 0
    torch.testing.assert_close(carry.policy_memory, memory)
    torch.testing.assert_close(likelihood(policy, cfg, batch), batch.old_log_probs, atol=2e-6, rtol=2e-6)
    assert not batch.obs.requires_grad and not carry.policy_memory.requires_grad
    assert len(records) == 5
    assert carry.policy_memory.shape == (2, 2, 8)


@pytest.mark.parametrize("method", METHODS)
def test_split_rollout_preserves_history_and_rng(method):
    cfg = config(method)
    policy = training.make_policy(cfg, CounterEnv())
    whole_envs = [CounterEnv(horizon=3), CounterEnv(index=1, horizon=5, truncates=True)]
    split_envs = [CounterEnv(horizon=3), CounterEnv(index=1, horizon=5, truncates=True)]
    torch.manual_seed(38)
    whole, whole_carry, _ = collect_rollout(whole_envs, policy, IdentityCritic(), cfg.algorithm, 7)
    torch.manual_seed(38)
    first, carry, _ = collect_rollout(split_envs, policy, IdentityCritic(), cfg.algorithm, 2)
    second, split_carry, _ = collect_rollout(split_envs, policy, IdentityCritic(), cfg.algorithm, 5, carry)
    for name in ("obs", "actions", "old_log_probs"):
        torch.testing.assert_close(getattr(whole, name), torch.cat((getattr(first, name), getattr(second, name))))
    torch.testing.assert_close(whole_carry.policy_memory, split_carry.policy_memory)


@pytest.mark.parametrize("method", METHODS)
def test_encoder_and_gru_receive_updates_with_correct_optimizer_ownership(method):
    torch.manual_seed(72)
    cfg = config(method, action_embed_dim=4 if method in ("arppo", "bppo") else 0)
    envs = [CounterEnv(horizon=5), CounterEnv(index=1, horizon=4, masks=True)]
    policy = training.make_policy(cfg, envs[0])
    critic = mlp(1, 1, 8)
    optimizers = training.make_optimizers(cfg, policy, critic)
    batch, _, _ = collect_rollout(envs, policy, critic, cfg.algorithm, 8)
    torch.testing.assert_close(likelihood(policy, cfg, batch), batch.old_log_probs, atol=2e-6, rtol=2e-6)
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    losses = update_rollout(policy, critic, optimizers, batch, cfg.algorithm, True, epochs=2)
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    for term in ("encoder.weight", "cell.weight_ih", "cell.weight_hh"):
        assert any(not torch.equal(before[name], p) for name, p in policy.named_parameters() if term in name)
    actors = policy.local_actors if cfg.algorithm == "agentmixer" else policy.actors
    if cfg.algorithm == "bppo":
        for i, actor in enumerate(actors):
            actual = {id(p) for g in optimizers[f"actor_{i}"].param_groups for p in g["params"]}
            assert actual == {id(p) for p in actor.parameters()}
    if not cfg.modifier_enabled:
        assert all(torch.equal(before[name], p) for name, p in policy.named_parameters() if name.startswith("modifier."))


@pytest.mark.parametrize("method", METHODS)
def test_memory_is_local_and_has_no_internal_mutable_state(method):
    cfg = config(method)
    policy = training.make_policy(cfg, CounterEnv())
    raw = torch.randn(3, 2, 1)
    incoming = torch.randn(3, 2, 8)
    packed = policy.pack_observations(raw, incoming)
    changed = packed.clone()
    changed[:, 0] += 7
    before = deepcopy(policy.state_dict())
    next_memory = policy.advance_memory(packed)
    changed_memory = policy.advance_memory(changed)
    torch.testing.assert_close(next_memory[:, 1], changed_memory[:, 1])
    assert not torch.equal(next_memory[:, 0], changed_memory[:, 0])
    assert all(torch.equal(value, policy.state_dict()[name]) for name, value in before.items())


def test_recurrent_actor_recomputes_gradients_but_stops_at_incoming_hidden_state():
    actor = RecurrentActor(3 + 8 + 6, 3, 8, raw_obs_dim=3).double()
    inputs = torch.randn(4, 17, dtype=torch.float64, requires_grad=True)
    actor(inputs).square().sum().backward()
    assert torch.count_nonzero(inputs.grad[:, :3])
    assert torch.count_nonzero(inputs.grad[:, 3:11]) == 0
    assert torch.count_nonzero(inputs.grad[:, 11:])
    assert torch.count_nonzero(actor.cell.weight_hh.grad)


def test_recurrent_successor_action_derivative_matches_finite_difference():
    policy = training.make_policy(config("bppo", action_embed_dim=4), CounterEnv()).double()
    packed = policy.pack_observations(torch.randn(2, 2, 1, dtype=torch.float64), torch.randn(2, 2, 8, dtype=torch.float64))
    features = F.one_hot(torch.tensor([[0, 1], [1, 2]]), 3).double().requires_grad_()
    score = policy._distribution(packed, features, 1, None).logits[:, 0].sum()
    analytical = torch.autograd.grad(score, features)[0][0, 0, 1]
    plus, minus = features.detach().clone(), features.detach().clone()
    plus[0, 0, 1] += 1e-5
    minus[0, 0, 1] -= 1e-5
    difference = (policy._distribution(packed, plus, 1, None).logits[:, 0].sum()
                  - policy._distribution(packed, minus, 1, None).logits[:, 0].sum()) / 2e-5
    torch.testing.assert_close(analytical, difference, atol=1e-8, rtol=1e-5)


def test_recurrent_agentmixer_preserves_local_modes_under_state_changes():
    policy = training.make_policy(config("agentmixer"), CounterEnv())
    with torch.no_grad():
        policy.modifier.scale_head.weight.normal_(0, 1)
    packed = policy.pack_observations(torch.randn(4, 2, 1), torch.randn(4, 2, 8))
    masks = torch.ones(4, 2, 3, dtype=torch.bool)
    masks[:, 0, 0] = False
    local = policy.local_actions(packed, masks)
    for state in (torch.zeros(4, 1), torch.full((4, 1), 100.)):
        central, _ = policy.sample(packed, state, available_actions=masks, deterministic=True)
        assert torch.equal(local, central)


@pytest.mark.parametrize("windows", [False, True])
def test_evaluation_starts_fresh_memory_at_each_episode_boundary(monkeypatch, windows):
    cfg = config("agentmixer", environment="prey", eval_episodes=3,
                 eval_windows=2 if windows else 0, eval_window_steps=5 if windows else 0)
    policy = training.make_policy(cfg, CounterEnv())
    monkeypatch.setattr(training, "make_env", lambda *args: CounterEnv(horizon=2, truncates=True))
    incoming_was_empty = []
    original = policy.pack_observations

    def record(observation, memory=None):
        incoming_was_empty.append(memory is None)
        return original(observation, memory)

    monkeypatch.setattr(policy, "pack_observations", record)
    if windows:
        training.evaluate_fixed_windows(policy, cfg)
        assert incoming_was_empty == [True, False, True, False, True] * 2
    else:
        training.evaluate(policy, cfg)
        assert incoming_was_empty == [True, False] * 3


@pytest.mark.parametrize("options", [{"recurrent_actor": 1}, {"environment": "mujoco", "algorithm": "bppo"},
                                     {"agentmixer_correlation": "joint"}])
def test_unsupported_recurrence_is_rejected(options):
    with pytest.raises(ValueError, match="recurrent"):
        training.Config(**{"recurrent_actor": True, **options}).validate()


@pytest.mark.parametrize("method", METHODS)
def test_real_prey_training_checkpoint_and_both_evaluation_modes_replay(tmp_path, method):
    cfg = config(method, environment="prey", env_config={"penalty": -1, "episode_limit": 4},
                 steps=16, num_envs=2, rollout_steps=3, epochs=1, eval_episodes=2,
                 eval_windows=2, eval_window_steps=8, normalize_value_targets=True)
    run = tmp_path / method
    summary = training.train(cfg, run)
    metadata = json.loads((run / "config.json").read_text())
    assert metadata["actor_memory"]["gradient_chunk_length"] == 1
    checkpoint = torch.load(run / "checkpoint.pt", weights_only=True)
    assert checkpoint["config"]["recurrent_actor"] is True
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_simulator.py"
    spec = importlib.util.spec_from_file_location("evaluate_recurrent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    replay = module.evaluate_saved(run)
    assert replay['source_matches_training']
    assert replay["evaluation"] == summary["final_evaluation"]
