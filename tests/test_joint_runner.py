import json

import pytest
import torch

from bta_repro.core import ppo_objective
from bta_repro.prey_env import PreyEnv
from bta_repro.runner import collect_rollout, update_rollout
from bta_repro.simulator_train import Config, make_critic, make_optimizers, make_policy, train


@pytest.mark.parametrize('precise_distribution', [False, True])
def test_joint_ppo_scores_the_single_event_exactly_once(precise_distribution):
    torch.manual_seed(3)
    config = Config(environment='prey', algorithm='agentmixer', agentmixer_correlation='joint',
                    hidden_dim=8, epochs=1, normalize_advantages=False,
                    joint_distribution_float64=precise_distribution)
    envs = [PreyEnv({'seed': i, 'episode_limit': 8}) for i in range(2)]
    policy, critic = make_policy(config, envs[0]), make_critic(config, envs[0])
    optimizers = make_optimizers(config, policy, critic)
    batch, _, _ = collect_rollout(envs, policy, critic, 'agentmixer', 4)
    assert batch.old_log_probs.shape == (8, 1)
    assert batch.old_log_probs.dtype == (torch.float64 if precise_distribution else torch.float32)
    with torch.no_grad():
        policy.local_actors[0][-1].bias[0] += .4
    log_probs, entropy = policy.evaluate_actions(batch.obs, batch.state, batch.actions, batch.available_actions)
    _, expected = ppo_objective(log_probs, batch.old_log_probs, batch.advantages, entropy,
                                clip_coef=.2, entropy_coef=.01, joint=True)
    actual = update_rollout(policy, critic, optimizers, batch, 'agentmixer', True,
                            epochs=1, value_coef=0, normalize_advantages=False)
    for key in ('policy_loss', 'entropy', 'approx_kl', 'clip_fraction'):
        assert actual[key] == pytest.approx(expected[key], abs=1e-6)
    batch.active_agents[0, 0] = False
    with pytest.raises(ValueError, match='every agent'):
        update_rollout(policy, critic, optimizers, batch, 'agentmixer', True)
    for env in envs:
        env.close()


@pytest.mark.parametrize('modifier_enabled', [True, False])
@pytest.mark.parametrize('precise_distribution', [True, False])
def test_joint_prey_training_and_local_evaluation(tmp_path, modifier_enabled, precise_distribution):
    config = Config(environment='prey', algorithm='agentmixer', agentmixer_correlation='joint',
                    modifier_enabled=modifier_enabled, joint_distribution_float64=precise_distribution,
                    env_config={'penalty': -1, 'episode_limit': 4},
                    steps=16, num_envs=2, rollout_steps=4, hidden_dim=8, epochs=1,
                    eval_episodes=2, normalize_value_targets=True)
    result = train(config, tmp_path / 'run')
    assert result['status'] == 'completed'
    assert result['final_evaluation']['execution'] == 'decentralized'
    metadata = json.loads((tmp_path / 'run/config.json').read_text())
    assert metadata['method_variant'].startswith('experimental_joint_')


@pytest.mark.parametrize('algorithm', ['arppo', 'bppo'])
def test_projected_autoregressive_policy_reaches_simulator_trainer(tmp_path, algorithm):
    config = Config(environment='prey', algorithm=algorithm, action_embed_dim=8,
                    env_config={'episode_limit': 4}, steps=8, num_envs=2, rollout_steps=4,
                    hidden_dim=8, epochs=1, eval_episodes=1)
    result = train(config, tmp_path / algorithm)
    assert result['status'] == 'completed'
    saved = torch.load(tmp_path / algorithm / 'checkpoint.pt', weights_only=True)
    assert saved['config']['action_embed_dim'] == 8


@pytest.mark.parametrize('arguments', [
    {'agentmixer_correlation': 'other'},
    {'environment': 'smac', 'agentmixer_correlation': 'joint'},
    {'algorithm': 'bppo', 'agentmixer_correlation': 'joint'},
    {'action_embed_dim': -1}, {'action_embed_dim': True},
    {'action_embed_dim': 8},
    {'algorithm': 'bppo', 'environment': 'mujoco', 'action_embed_dim': 8},
])
def test_unsupported_controls_rejected(arguments):
    with pytest.raises(ValueError):
        Config(**{'environment': 'prey', **arguments}).validate()
