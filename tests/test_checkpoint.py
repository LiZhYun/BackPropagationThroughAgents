import random
import numpy as np
import torch
from bta_repro.checkpoint import save_checkpoint, load_checkpoint
from bta_repro.core import IndependentPolicy, mlp


def test_checkpoint_restores_models_optimizer_and_all_rngs(tmp_path):
    torch.manual_seed(4); np.random.seed(4); random.seed(4)
    policy = IndependentPolicy(1, 2, 3, 8); critic = mlp(1, 1, 8)
    optimizer = torch.optim.Adam(policy.parameters())
    loss = policy.logits(torch.ones(4,2,1)).square().mean()
    loss.backward(); optimizer.step()
    expected = {k:v.clone() for k,v in policy.state_dict().items()}
    checkpoint = tmp_path/'checkpoint.pt'
    save_checkpoint(checkpoint, policy=policy, critic=critic, optimizers={'actor':optimizer}, config={'seed':4},step=100)
    expected_random = (random.random(), np.random.rand(), torch.rand(3))
    with torch.no_grad():
        for p in policy.parameters(): p.add_(10)
    result = load_checkpoint(checkpoint, policy=policy, critic=critic, optimizers={'actor':optimizer})
    assert result['step'] == 100
    for k,v in expected.items(): torch.testing.assert_close(policy.state_dict()[k],v)
    assert random.random() == expected_random[0]
    assert np.random.rand() == expected_random[1]
    torch.testing.assert_close(torch.rand(3),expected_random[2])
    assert optimizer.state_dict()['state']
    assert result['extra_state'] == {}


def test_checkpoint_preserves_normalization_with_weights_only_loading(tmp_path):
    from bta_repro.normalization import ObservationNormalizer

    policy, critic = torch.nn.Linear(2, 2), torch.nn.Linear(3, 1)
    normalizer = ObservationNormalizer(1, 2, 3)
    normalizer.update({'obs': np.array([[1., 4.]]), 'state': np.array([3., 2., 1.])})
    normalizer.update({'obs': np.array([[3., 6.]]), 'state': np.array([9., 4., 7.])})
    path = tmp_path / 'normalized.pt'
    save_checkpoint(path, policy=policy, critic=critic, optimizers={}, config={}, step=2,
                    extra_state={'observation_normalizer': normalizer.state_dict()})
    restricted = torch.load(path, weights_only=True)
    assert restricted['extra_state']['observation_normalizer'] == normalizer.state_dict()
    payload = load_checkpoint(path, policy=policy, critic=critic, optimizers={})
    restored = ObservationNormalizer(1, 2, 3)
    restored.load_state_dict(payload['extra_state']['observation_normalizer'])
    assert restored.state_dict() == normalizer.state_dict()
