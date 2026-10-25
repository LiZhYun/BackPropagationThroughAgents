import json

import pytest
import torch

from bta_repro.simulator_train import Config, EpisodeStatistics, train


@pytest.mark.parametrize('algorithm', ['mappo', 'arppo', 'bppo', 'agentmixer'])
def test_real_prey_uses_shared_trainer_and_saves_normalized_critic(tmp_path, algorithm):
    config = Config(environment='prey', algorithm=algorithm, env_config={'penalty': -1, 'episode_limit': 4},
                    steps=16, num_envs=2, rollout_steps=4, hidden_dim=8, epochs=1,
                    eval_episodes=2, normalize_value_targets=True, value_clip_coef=.2,
                    value_huber_delta=10)
    result = train(config, tmp_path / algorithm)
    assert result['status'] == 'completed' and result['train_environment_steps'] == 16
    evaluation = result['final_evaluation']
    assert evaluation['capture_records'] == evaluation['count'] == 2
    total_return = sum(episode['return'] for episode in evaluation['episodes'])
    total_steps = sum(episode['length'] for episode in evaluation['episodes'])
    assert evaluation['reward_per_environment_step'] == pytest.approx(total_return / total_steps)
    assert evaluation['reward_per_step_scope'] == 'completed_episodes_only'
    checkpoint = torch.load(tmp_path / algorithm / 'checkpoint.pt', weights_only=True)
    assert checkpoint['critic']['normalizer.count'].item() == 16
    assert checkpoint['critic']['normalizer.updates'].item() == 2
    metadata = json.loads((tmp_path / algorithm / 'config.json').read_text())
    assert metadata['environment_spec']['obs_dim'] == 28
    assert metadata['environment_versions']['legacy_commit'].startswith('13607d0')


def test_episode_statistics_distinguishes_capture_and_step_weighting():
    statistics = EpisodeStatistics()
    statistics.add([
        {'return': 9.98, 'length': 2, 'terminated': True, 'truncated': False,
         'info': {'capture_success': True}},
        {'return': -1.28, 'length': 128, 'terminated': False, 'truncated': True,
         'info': {'capture_success': False}},
    ])
    result = statistics.summary()
    assert result['capture_rate'] == .5
    assert result['reward_per_environment_step'] == pytest.approx((9.98 - 1.28) / 130)
    assert result['return_mean'] == pytest.approx((9.98 - 1.28) / 2)
    assert result['win_rate'] is None


def test_prey_rejects_continuous_exploration_setting():
    with pytest.raises(ValueError, match='continuous'):
        Config(environment='prey', initial_std=.3).validate()
