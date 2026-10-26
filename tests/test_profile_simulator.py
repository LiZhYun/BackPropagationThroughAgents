"""Instrumentation must preserve learning, episode boundaries and failures."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from bta_repro import simulator_train as training


path = Path(__file__).resolve().parents[1] / 'scripts/profile_simulator.py'
spec = importlib.util.spec_from_file_location('profile_simulator', path)
profiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiler)


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('algorithm,recurrent', [('mappo', False), ('bppo', False), ('agentmixer', True)])
def test_timing_preserves_complete_training_and_checkpoint_state(tmp_path, algorithm, recurrent):
    config = training.Config(environment='prey', env_config={'episode_limit': 4}, algorithm=algorithm,
                             recurrent_actor=recurrent, steps=16, num_envs=2, rollout_steps=4,
                             hidden_dim=8, epochs=1, eval_episodes=2, normalize_value_targets=True,
                             eval_windows=2, eval_window_steps=8, seed=7)
    original = {name: getattr(training, name) for name in ('make_env', 'collect_rollout', 'update_rollout', 'evaluate')}
    plain = training.train(config, tmp_path / 'plain')
    measured = profiler.profile(config, tmp_path / 'measured')
    assert {k: v for k, v in plain.items() if k != 'elapsed_seconds'} == {
        k: v for k, v in measured.items() if k != 'elapsed_seconds'}
    for name, function in original.items():
        assert getattr(training, name) is function
    assert_equal(torch.load(tmp_path / 'plain/checkpoint.pt', weights_only=True),
                 torch.load(tmp_path / 'measured/checkpoint.pt', weights_only=True))
    report = json.loads((tmp_path / 'measured/runtime_profile.json').read_text())
    assert report['status'] == 'completed' and report['error'] is None
    assert report['timings']['collect_rollout']['calls'] == 2
    assert report['timings']['collect_rollout.env_step']['calls'] == 16
    assert report['timings']['update_rollout']['calls'] == 2
    assert report['timings']['evaluate']['calls'] == 2
    assert all(value['seconds'] >= 0 for value in report['timings'].values())
    assert report['timings']['collect_rollout']['seconds'] >= report['timings']['collect_rollout.env_step']['seconds']


def test_failure_is_propagated_and_instrumentation_is_restored(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError('injected rollout failure')

    monkeypatch.setattr(training, 'collect_rollout', broken)
    config = training.Config(environment='prey', env_config={'episode_limit': 2}, steps=4,
                             num_envs=2, rollout_steps=2, hidden_dim=8, epochs=1, eval_episodes=1)
    with pytest.raises(RuntimeError, match='injected rollout failure'):
        profiler.profile(config, tmp_path / 'failed')
    assert training.collect_rollout is broken
    report = json.loads((tmp_path / 'failed/runtime_profile.json').read_text())
    assert report['status'] == 'failed' and report['error']['type'] == 'RuntimeError'
    assert report['timings']['collect_rollout']['calls'] == 1
    assert json.loads((tmp_path / 'failed/summary.json').read_text())['status'] == 'failed'


def test_existing_results_are_not_modified(tmp_path):
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'marker'
    marker.write_text('preserved')
    original = training.make_env
    with pytest.raises(FileExistsError):
        profiler.profile(training.Config(environment='prey'), output)
    assert marker.read_text() == 'preserved' and list(output.iterdir()) == [marker]
    assert training.make_env is original


def test_async_device_timing_is_rejected_before_mutating_the_trainer(tmp_path):
    original = training.make_env
    with pytest.raises(ValueError, match='CPU execution only'):
        profiler.profile(training.Config(environment='prey', device='cuda'), tmp_path / 'gpu')
    assert training.make_env is original and not (tmp_path / 'gpu').exists()
