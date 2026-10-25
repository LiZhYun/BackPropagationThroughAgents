import json
from bta_repro.train import Config, train


def test_joint_modifier_matrix_training_integrates_one_joint_event(tmp_path):
    output = tmp_path / 'joint_smoke'
    summary = train(Config(algorithm='agentmixer', agentmixer_correlation='joint',
                           steps=128, batch_size=64, hidden_dim=8, epochs=2), output)
    assert summary['steps'] == 128
    assert abs(sum(summary['training_joint_probabilities']) - 1) < 1e-5
    assert len(summary['training_joint_probabilities']) == 9
    assert (output / 'checkpoint.pt').exists()
    metadata = json.loads((output / 'config.json').read_text())
    assert metadata['method_variant'] == 'experimental_joint_gap_scaling'
