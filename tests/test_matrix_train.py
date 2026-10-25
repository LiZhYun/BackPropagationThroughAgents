import json
from dataclasses import replace

import pytest
import torch

from bta_repro.train import (
    Config, actor_learning_rate, make_optimizers, make_policy, set_actor_learning_rate, train,
)


@pytest.mark.parametrize('precise_distribution', [False, True])
def test_joint_modifier_matrix_training_integrates_one_joint_event(tmp_path, precise_distribution):
    output = tmp_path / 'joint_smoke'
    summary = train(Config(algorithm='agentmixer', agentmixer_correlation='joint',
                           joint_distribution_float64=precise_distribution,
                           steps=128, batch_size=64, hidden_dim=8, epochs=2), output)
    assert summary['steps'] == 128
    assert abs(sum(summary['training_joint_probabilities']) - 1) < 1e-5
    assert len(summary['training_joint_probabilities']) == 9
    assert (output / 'checkpoint.pt').exists()
    metadata = json.loads((output / 'config.json').read_text())
    assert metadata['method_variant'] == 'experimental_joint_gap_scaling'


@pytest.mark.parametrize('arguments', [
    {'joint_distribution_float64': 1},
    {'joint_distribution_float64': True},
    {'algorithm': 'agentmixer', 'joint_distribution_float64': True},
])
def test_joint_precision_control_requires_the_joint_method(arguments):
    with pytest.raises(ValueError):
        Config(**arguments).validate()


def test_default_actor_learning_rate_is_constant_and_late_change_uses_environment_steps():
    config = Config(steps=200_000)
    assert config.late_actor_lr_multiplier == 1
    assert config.actor_lr_change_fraction == .5
    for step in (0, 99_999, 100_000, 100_352, 200_000):
        assert actor_learning_rate(config, step) == config.learning_rate
    late = replace(config, late_actor_lr_multiplier=5)
    assert actor_learning_rate(late, 99_999) == 5e-4
    assert actor_learning_rate(late, 100_000) == 2.5e-3
    first_changed_rollout = next(step for step in range(0, config.steps, config.batch_size)
                                 if actor_learning_rate(late, step) != late.learning_rate)
    assert first_changed_rollout == 100_352
    assert actor_learning_rate(replace(late, actor_lr_change_fraction=0), 0) == 2.5e-3


@pytest.mark.parametrize("algorithm,correlation", [
    ("mappo", "factorized"), ("arppo", "factorized"), ("bppo", "factorized"),
    ("agentmixer", "factorized"), ("agentmixer", "joint"),
])
def test_schedule_changes_only_actual_actor_optimizer_groups(algorithm, correlation):
    config = Config(algorithm=algorithm, agentmixer_correlation=correlation, hidden_dim=8,
                    steps=200_000, late_actor_lr_multiplier=5,
                    modifier_learning_rate=7e-5, critic_learning_rate=3e-4)
    policy, critic = make_policy(config), torch.nn.Linear(1, 1)
    optimizers = make_optimizers(config, policy, critic)
    actor_module = policy.local_actors if algorithm == "agentmixer" else policy.actors
    actor_ids = {id(parameter) for parameter in actor_module.parameters()}
    modifier_ids = {id(parameter) for parameter in policy.modifier.parameters()} if algorithm == "agentmixer" else set()
    critic_ids = {id(parameter) for parameter in critic.parameters()}
    for step, actor_rate in ((0, 5e-4), (99_999, 5e-4), (100_000, 2.5e-3), (150_000, 2.5e-3)):
        assert set_actor_learning_rate(config, optimizers, step) == actor_rate
        for optimizer in optimizers.values():
            for group in optimizer.param_groups:
                owned = {id(parameter) for parameter in group["params"]}
                if owned <= actor_ids:
                    assert group["lr"] == actor_rate
                elif owned <= modifier_ids:
                    assert group["lr"] == 7e-5
                else:
                    assert owned <= critic_ids
                    assert group["lr"] == 3e-4


@pytest.mark.parametrize("correlation", ["factorized", "joint"])
def test_late_schedule_preserves_first_half_updates_and_logs_active_rate(tmp_path, monkeypatch, correlation):
    original_step = torch.optim.Adam.step
    captured, critic_rates = [], []

    def record_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        if len(optimizer.param_groups) == 2:
            captured.append({
                "rates": [group["lr"] for group in optimizer.param_groups],
                "parameters": [parameter.detach().clone() for group in optimizer.param_groups
                               for parameter in group["params"]],
            })
        else:
            critic_rates.append(optimizer.param_groups[0]["lr"])
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", record_step)
    config = Config(algorithm="agentmixer", agentmixer_correlation=correlation,
                    steps=128, batch_size=32, hidden_dim=8, epochs=1, eval_every=1,
                    modifier_learning_rate=7e-5, critic_learning_rate=3e-4)
    train(config, tmp_path / "constant")
    baseline, captured = captured, []
    train(replace(config, late_actor_lr_multiplier=5), tmp_path / "late")
    assert len(baseline) == len(captured) == 4
    for before, after in zip(baseline[:2], captured[:2]):
        assert all(torch.equal(a, b) for a, b in zip(before["parameters"], after["parameters"]))
    assert any(not torch.equal(a, b) for a, b in zip(baseline[-1]["parameters"], captured[-1]["parameters"]))
    assert [entry["rates"] for entry in captured] == [[5e-4, 7e-5]] * 2 + [[2.5e-3, 7e-5]] * 2
    assert critic_rates == [3e-4] * 8
    records = [json.loads(line) for line in (tmp_path / "late" / "metrics.jsonl").read_text().splitlines()]
    updates = records[1:]
    assert [record["actor_lr"] for record in updates] == [5e-4, 5e-4, 2.5e-3, 2.5e-3]
    assert [record["actor_lr_schedule_steps"] for record in updates] == [0, 32, 64, 96]
    assert [record["actor_lr_changed"] for record in updates] == [False, False, True, False]
    assert [record["steps"] for record in updates] == [32, 64, 96, 128]
    checkpoint = torch.load(tmp_path / "late" / "checkpoint.pt", weights_only=True)
    assert [group["lr"] for group in checkpoint["optimizers"]["policy"]["param_groups"]] == [2.5e-3, 7e-5]
    assert checkpoint["optimizers"]["critic"]["param_groups"][0]["lr"] == 3e-4


def test_first_changed_rollout_is_logged_between_periodic_evaluations(tmp_path):
    config = Config(algorithm="agentmixer", agentmixer_correlation="joint",
                    steps=100, batch_size=32, hidden_dim=8, epochs=1, eval_every=100,
                    late_actor_lr_multiplier=5)
    train(config, tmp_path / "crossing")
    records = [json.loads(line) for line in (tmp_path / "crossing" / "metrics.jsonl").read_text().splitlines()]
    changed = [record for record in records if record.get("actor_lr_changed")]
    assert len(changed) == 1
    assert changed[0]["actor_lr_schedule_steps"] == 64
    assert changed[0]["steps"] == 96
    assert changed[0]["actor_lr"] == 2.5e-3


@pytest.mark.parametrize("arguments", [
    {"late_actor_lr_multiplier": 0}, {"late_actor_lr_multiplier": -1},
    {"late_actor_lr_multiplier": float("nan")}, {"late_actor_lr_multiplier": float("inf")},
    {"late_actor_lr_multiplier": True}, {"late_actor_lr_multiplier": "5"},
    {"actor_lr_change_fraction": -.1}, {"actor_lr_change_fraction": 1},
    {"actor_lr_change_fraction": float("nan")}, {"actor_lr_change_fraction": float("inf")},
    {"actor_lr_change_fraction": False}, {"actor_lr_change_fraction": "0.5"},
    {"learning_rate": 1e300, "late_actor_lr_multiplier": 1e300},
])
def test_invalid_actor_schedule_is_rejected(arguments):
    with pytest.raises(ValueError):
        Config(**arguments).validate()
