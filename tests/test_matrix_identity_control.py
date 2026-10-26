"""The matrix ablation freezes only the modifier, retaining the actor path."""
from dataclasses import asdict, replace
import json

import pytest
import torch

from bta_repro.checkpoint import load_checkpoint
from bta_repro.core import mlp
from bta_repro.envs import MatrixGame
from bta_repro.train import Config, evaluate, make_optimizers, make_policy, train


FAMILIES = [("factorized", False), ("joint", False), ("joint", True)]


@pytest.mark.parametrize("correlation,precise", FAMILIES)
def test_identity_control_preserves_initialization_and_sampling_rng(correlation, precise):
    config = Config(algorithm="agentmixer", agentmixer_correlation=correlation,
                    joint_distribution_float64=precise, hidden_dim=8)
    torch.manual_seed(17)
    learned = make_policy(config)
    learned_rng = torch.get_rng_state()
    torch.manual_seed(17)
    identity = make_policy(replace(config, modifier_enabled=False))
    assert torch.equal(torch.get_rng_state(), learned_rng)
    for name, value in learned.state_dict().items():
        assert torch.equal(value, identity.state_dict()[name])
    obs, state = MatrixGame("climbing").observations(64)
    actions, logp = learned.sample(obs, state)
    sampled_rng = torch.get_rng_state()
    torch.set_rng_state(learned_rng)
    identity_actions, identity_logp = identity.sample(obs, state)
    assert torch.equal(actions, identity_actions)
    assert torch.equal(logp, identity_logp)
    assert torch.equal(torch.get_rng_state(), sampled_rng)


@pytest.mark.parametrize("correlation,precise", FAMILIES)
def test_identity_modifier_stays_fixed_while_actors_learn_and_checkpoint_replays(
        tmp_path, correlation, precise):
    config = Config(algorithm="agentmixer", agentmixer_correlation=correlation,
                    joint_distribution_float64=precise, modifier_enabled=False,
                    steps=192, batch_size=64, epochs=2, hidden_dim=8, eval_every=1)
    torch.manual_seed(config.seed)
    initial = make_policy(config)
    output = tmp_path / "identity"
    summary = train(config, output)
    saved = torch.load(output / "checkpoint.pt", weights_only=True)
    for name, parameter in initial.state_dict().items():
        if name.startswith("modifier."):
            assert torch.equal(parameter, saved["policy"][name])
    assert any(not torch.equal(parameter, saved["policy"][name])
               for name, parameter in initial.state_dict().items()
               if name.startswith("local_actors."))
    assert saved["config"] == asdict(config)
    rebuilt = make_policy(Config(**saved["config"]))
    critic = mlp(1, 1, config.hidden_dim)
    optimizers = make_optimizers(config, rebuilt, critic)
    actor_ids = {id(p) for p in rebuilt.local_actors.parameters()}
    optimizer_ids = {id(p) for group in optimizers["policy"].param_groups for p in group["params"]}
    assert optimizer_ids == actor_ids
    assert all(not p.requires_grad for p in rebuilt.modifier.parameters())
    assert all(p.requires_grad for p in rebuilt.local_actors.parameters())
    load_checkpoint(output / "checkpoint.pt", policy=rebuilt, critic=critic, optimizers=optimizers)
    replay = evaluate(rebuilt, config, MatrixGame(config.game))
    assert all(summary[key] == value for key, value in replay.items())
    assert replay["modifier_scale_min"] == replay["modifier_scale_max"] == 1
    # The guards and float64 central distribution permit tiny roundoff differences.
    torch.testing.assert_close(torch.tensor(replay["training_joint_probabilities"]),
                               torch.tensor(replay["execution_joint_probabilities"]),
                               rtol=1e-5, atol=1e-6)
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert all(row["modifier_scale_min"] == row["modifier_scale_max"] == 1 for row in rows)
    metadata = json.loads((output / "config.json").read_text())
    expected_label = "experimental_joint_identity_control" if correlation == "joint" else "identity_modifier_control"
    assert metadata["method_variant"] == expected_label


@pytest.mark.parametrize("arguments", [
    {"modifier_enabled": 0}, {"modifier_enabled": "false"},
    {"modifier_enabled": False},
    {"algorithm": "mappo", "modifier_enabled": False},
    {"algorithm": "arppo", "modifier_enabled": False},
    {"algorithm": "agentmixer", "modifier_enabled": False, "modifier_log_scale_bound": 2.},
])
def test_identity_control_rejects_ambiguous_or_ineffective_options(arguments):
    with pytest.raises(ValueError):
        Config(**arguments).validate()
