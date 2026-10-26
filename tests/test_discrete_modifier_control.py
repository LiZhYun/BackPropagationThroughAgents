"""The simulator's discrete exploration control preserves mode and replay contracts."""
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from bta_repro.prey_env import PreyEnv
from bta_repro.simulator_train import Config, make_policy, train


@pytest.mark.parametrize('correlation', ['factorized', 'joint'])
def test_bounded_discrete_factory_preserves_initialization_and_local_modes(correlation):
    env = PreyEnv({'seed': 13})
    config = Config(environment='prey', agentmixer_correlation=correlation)
    torch.manual_seed(7)
    original = make_policy(config, env)
    config.modifier_log_scale_bound = math.log(10)
    config.validate()
    torch.manual_seed(7)
    bounded = make_policy(config, env)
    assert all(torch.equal(value, bounded.state_dict()[name])
               for name, value in original.state_dict().items())
    observation, _ = env.reset()
    obs = torch.as_tensor(observation['obs']).unsqueeze(0)
    state = torch.as_tensor(observation['state']).unsqueeze(0)
    torch.testing.assert_close(original.joint_logits(obs, state), bounded.joint_logits(obs, state))
    with torch.no_grad():
        bounded.modifier.scale_head.bias.copy_(torch.linspace(-20, 20, bounded.modifier.scale_head.bias.numel()))
    scales = bounded.modifier_log_scales(bounded.local_logits(obs), state).exp()
    assert scales.min() >= .1 - 1e-7 and scales.max() <= 10 + 1e-6
    central, _ = bounded.sample(obs, state, deterministic=True)
    assert torch.equal(central, bounded.local_actions(obs))
    assert not torch.equal(original.joint_logits(obs, state), bounded.joint_logits(obs, state))
    env.close()


@pytest.mark.parametrize('options', [
    {'modifier_log_scale_bound': x} for x in [0, -1, 8.1, True, '2', float('inf'), float('nan')]
] + [
    {'modifier_log_scale_bound': 2, 'environment': 'mujoco'},
    {'modifier_log_scale_bound': 2, 'algorithm': 'mappo'},
    {'modifier_log_scale_bound': 2, 'modifier_enabled': False},
])
def test_rejects_unsupported_discrete_modifier_bound(options):
    with pytest.raises(ValueError, match='modifier_log_scale_bound'):
        Config(**{'environment': 'prey', **options}).validate()


@pytest.mark.parametrize('correlation', ['factorized', 'joint'])
def test_real_prey_bound_survives_training_checkpoint_and_replay(tmp_path, correlation):
    config = Config(environment='prey', agentmixer_correlation=correlation,
                    modifier_log_scale_bound=math.log(10),
                    env_config={'penalty': -1, 'episode_limit': 4},
                    steps=16, num_envs=2, rollout_steps=4, hidden_dim=8, epochs=1,
                    eval_episodes=2, eval_windows=2, eval_window_steps=4,
                    normalize_value_targets=True)
    run = tmp_path / correlation
    summary = train(config, run)
    metadata = json.loads((run / 'config.json').read_text())
    expected = ('experimental_joint_bounded_gap_scaling' if correlation == 'joint'
                else 'mode_consistent_bounded_gap_modifier')
    assert metadata['method_variant'] == expected
    checkpoint = torch.load(run / 'checkpoint.pt', weights_only=True)
    assert checkpoint['config']['modifier_log_scale_bound'] == math.log(10)
    path = Path(__file__).resolve().parents[1] / 'scripts/evaluate_simulator.py'
    spec = importlib.util.spec_from_file_location('evaluate_bounded_checkpoint', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    replay = module.evaluate_saved(run)
    assert replay['source_matches_training']
    assert replay['config']['modifier_log_scale_bound'] == math.log(10)
    assert replay['evaluation'] == summary['final_evaluation']
