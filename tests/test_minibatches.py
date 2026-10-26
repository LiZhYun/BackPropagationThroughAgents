"""Minibatch coverage, rollout references, masks, and native-free replay."""

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from bta_repro import bppo, bppo_continuous, runner, simulator_train as training
from bta_repro.core import minibatch_indices, mlp, normalize_advantages
from bta_repro.value_normalization import ValueNormalizedCritic
from test_runner import CounterEnv, make_policy, optimizers_for


def test_partition_visits_every_sample_once_retains_remainder_and_reshuffles():
    torch.manual_seed(16)
    orders = []
    for _ in range(3):
        batches = minibatch_indices(11, 3, torch.device("cpu"))
        assert [len(batch) for batch in batches] == [4, 4, 3]
        order = torch.cat(batches)
        assert torch.equal(order.sort().values, torch.arange(11))
        orders.append(order)
    assert not torch.equal(orders[0], orders[1])
    assert not torch.equal(orders[1], orders[2])
    state = torch.get_rng_state().clone()
    assert minibatch_indices(11, 1, "cpu") == (slice(None),)
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, 6])
def test_invalid_count_rejected_before_policy_or_optimizer_changes(value):
    policy, critic = make_policy("bppo", True), mlp(1, 1, 8)
    batch, _, _ = runner.collect_rollout([CounterEnv(horizon=10)], policy, critic, "bppo", 5)
    optimizers = optimizers_for(policy, critic, "bppo")
    before = deepcopy(policy.state_dict())
    with pytest.raises(ValueError, match="num_minibatches"):
        runner.update_rollout(policy, critic, optimizers, batch, "bppo", True, num_minibatches=value)
    assert all(torch.equal(before[key], value) for key, value in policy.state_dict().items())
    assert all(not optimizer.state for optimizer in optimizers.values())


@pytest.mark.parametrize("steps,num_envs,rollout_steps,num_minibatches", [
    (10, 2, 4, 3),  # Final rollout has only two transitions.
    (4, 2, 4, 5),   # Entire run is shorter than one nominal rollout.
    (16, 2, 4, 9),  # No partial rollout, but full rollouts are too small.
])
def test_config_rejects_empty_minibatches_including_final_partial(steps, num_envs, rollout_steps, num_minibatches):
    with pytest.raises(ValueError, match="smallest rollout"):
        training.Config(steps=steps, num_envs=num_envs, rollout_steps=rollout_steps,
                        num_minibatches=num_minibatches).validate()


@pytest.mark.parametrize("algorithm", ["mappo", "arppo", "bppo", "agentmixer"])
@pytest.mark.parametrize("discrete", [True, False])
def test_all_methods_update_unequal_minibatches_with_one_value_statistics_update(algorithm, discrete):
    torch.manual_seed(14)
    policy = make_policy(algorithm, discrete)
    critic = ValueNormalizedCritic(mlp(1, 1, 8))
    batch, _, _ = runner.collect_rollout([CounterEnv(horizon=10, is_discrete=discrete)],
                                        policy, critic, algorithm, 5)
    before = deepcopy(batch)
    metrics = runner.update_rollout(
        policy, critic, optimizers_for(policy, critic, algorithm), batch, algorithm, discrete,
        epochs=2, num_minibatches=3, value_clip_coef=.2, value_huber_delta=5,
    )
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["num_updates"] == (12 if algorithm == "bppo" else 6)
    assert metrics["critic_updates"] == 6
    assert critic.normalizer.count == 5 and critic.normalizer.updates == 1
    for field in ("obs", "actions", "old_log_probs", "advantages", "returns", "values"):
        assert torch.equal(getattr(before, field), getattr(batch, field))


def test_advantages_and_clipped_value_targets_use_full_rollout_moments(monkeypatch):
    policy = make_policy("mappo", True)
    critic = ValueNormalizedCritic(mlp(1, 1, 8))
    batch, _, _ = runner.collect_rollout([CounterEnv(horizon=10)], policy, critic, "mappo", 5)
    batch.active_agents[2] = False
    batch.advantages = torch.tensor([1., 2., 10000., 4., 8.])
    batch.returns = torch.tensor([3., 6., 9., 15., 21.])
    expected = normalize_advantages(batch.advantages, batch.active_agents.any(-1))
    order = (torch.tensor([0, 2]), torch.tensor([1, 4]), torch.tensor([3]))
    monkeypatch.setattr(runner, "minibatch_indices", lambda *args: order)
    actual_advantages, actual_values = [], []
    original_ppo, original_value = runner.ppo_objective, runner.value_regression_loss

    def record_policy(log_probs, old_log_probs, advantages, *args, **kwargs):
        actual_advantages.append(advantages.clone())
        return original_ppo(log_probs, old_log_probs, advantages, *args, **kwargs)

    def record_value(prediction, target, *, old_prediction, **kwargs):
        actual_values.append((target.clone(), old_prediction.clone()))
        return original_value(prediction, target, old_prediction=old_prediction, **kwargs)

    monkeypatch.setattr(runner, "ppo_objective", record_policy)
    monkeypatch.setattr(runner, "value_regression_loss", record_value)
    runner.update_rollout(policy, critic, optimizers_for(policy, critic, "mappo"), batch, "mappo", True,
                          epochs=2, num_minibatches=3, value_clip_coef=.2)
    assert len(actual_advantages) == len(actual_values) == 6
    for index, (advantages, (target, old)) in enumerate(zip(actual_advantages, actual_values)):
        indices = order[index % 3]
        torch.testing.assert_close(advantages, expected[indices])
        torch.testing.assert_close(target, critic.normalizer.normalize(batch.returns)[indices])
        torch.testing.assert_close(old, critic.normalizer.normalize(batch.values.flatten())[indices])
    assert critic.normalizer.count == 5 and critic.normalizer.updates == 1


def test_joint_event_minibatches_and_disabled_critic_keep_their_update_contract():
    env = CounterEnv(horizon=10)
    cfg = training.Config(algorithm="agentmixer", agentmixer_correlation="joint", hidden_dim=8)
    policy = training.make_policy(cfg, env)
    critic = ValueNormalizedCritic(mlp(1, 1, 8))
    optimizers = training.make_optimizers(cfg, policy, critic)
    batch, _, _ = runner.collect_rollout([env], policy, critic, "agentmixer", 5)
    before = [p.detach().clone() for p in critic.parameters()]
    metrics = runner.update_rollout(policy, critic, optimizers, batch, "agentmixer", True,
                                    epochs=2, num_minibatches=3, value_coef=0)
    assert batch.old_log_probs.shape == (5, 1)
    assert metrics["num_updates"] == 6 and metrics["critic_updates"] == 0
    assert all(math.isfinite(value) for value in metrics.values())
    assert critic.normalizer.count == 5 and critic.normalizer.updates == 1
    assert all(torch.equal(a, b) for a, b in zip(before, critic.parameters()))
    assert not optimizers["critic"].state


@pytest.mark.parametrize("variant", ["discrete", "recurrent", "fresh", "behavior"])
def test_bppo_retains_one_behavior_anchor_and_completes_successor_minibatches_first(monkeypatch, variant):
    discrete = variant in ("discrete", "recurrent")
    torch.manual_seed(26)
    env = CounterEnv(horizon=10, is_discrete=discrete)
    policy = (training.make_policy(training.Config(algorithm="bppo", hidden_dim=8, recurrent_actor=True), env)
              if variant == "recurrent" else make_policy("bppo", discrete))
    critic = mlp(1, 1, 8)
    batch, _, _ = runner.collect_rollout([env], policy, critic, "bppo", 5)
    behavior_weights = deepcopy(policy.state_dict())
    optimizers = optimizers_for(policy, critic, "bppo")
    steps, calls = [], []
    for i in range(2):
        original_step = optimizers[f"actor_{i}"].step

        def record_step(*args, actor=i, step=original_step, **kwargs):
            steps.append(actor)
            return step(*args, **kwargs)

        monkeypatch.setattr(optimizers[f"actor_{i}"], "step", record_step)
    module = bppo if discrete else bppo_continuous
    feedback = module._successor_feedback

    def record_feedback(current, behavior, obs, actions, features, agent, *args):
        calls.append((behavior, agent, tuple(steps)))
        assert all(torch.equal(behavior_weights[key], value) for key, value in behavior.state_dict().items())
        assert all(not parameter.requires_grad for parameter in behavior.parameters())
        assert torch.equal(obs, batch.obs)
        return feedback(current, behavior, obs, actions, features, agent, *args)

    monkeypatch.setattr(module, "_successor_feedback", record_feedback)
    runner.update_rollout(policy, critic, optimizers, batch, "bppo", discrete, epochs=2,
                          num_minibatches=3, continuous_peer_noise="behavior" if variant == "behavior" else "fresh")
    assert len(calls) == 2 and calls[0][0] is calls[1][0]
    assert [call[1:] for call in calls] == [(1, ()), (0, (1,) * 6)]
    assert steps == [1] * 6 + [0] * 6


@pytest.mark.parametrize("algorithm", ["mappo", "arppo", "bppo", "agentmixer"])
def test_inactive_minibatches_do_not_advance_actor_optimizer_state(monkeypatch, algorithm):
    policy, critic = make_policy(algorithm, True), mlp(1, 1, 8)
    batch, _, _ = runner.collect_rollout([CounterEnv(horizon=10)], policy, critic, algorithm, 4)
    batch.active_agents[2:] = False
    order = (torch.tensor([0, 1]), torch.tensor([2, 3]))
    monkeypatch.setattr(runner, "minibatch_indices", lambda *args: order)
    monkeypatch.setattr(bppo, "minibatch_indices", lambda *args: order)
    metrics = runner.update_rollout(policy, critic, optimizers_for(policy, critic, algorithm),
                                    batch, algorithm, True, epochs=2, num_minibatches=2)
    assert metrics["num_updates"] == (4 if algorithm == "bppo" else 2)
    assert metrics["critic_updates"] == 4  # State values include all transitions.


@pytest.mark.parametrize("method", ["mappo", "arppo", "bppo", "agentmixer", "identity"])
def test_real_recurrent_prey_minibatch_training_checkpoint_and_evaluation_replay(tmp_path, method):
    cfg = training.Config(
        environment="prey", env_config={"penalty": -1, "episode_limit": 4},
        algorithm="agentmixer" if method == "identity" else method, modifier_enabled=method != "identity",
        hidden_dim=8, recurrent_actor=True, num_minibatches=3, epochs=2,
        steps=16, num_envs=2, rollout_steps=3, eval_episodes=2, eval_windows=2, eval_window_steps=8,
        normalize_value_targets=True, value_clip_coef=.2,
    )
    run = tmp_path / method
    summary = training.train(cfg, run)
    checkpoint = torch.load(run / "checkpoint.pt", weights_only=True)
    assert checkpoint["config"]["num_minibatches"] == 3
    updates = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
               if json.loads(line)["event"] == "update"]
    assert [entry["rollout_shape"] for entry in updates] == [[3, 2], [3, 2], [2, 2]]
    assert all(entry["losses"]["critic_updates"] == 6 for entry in updates)
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_simulator.py"
    spec = importlib.util.spec_from_file_location("evaluate_minibatch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    replay = module.evaluate_saved(run)
    assert replay["source_matches_training"]
    assert replay["evaluation"] == summary["final_evaluation"]
