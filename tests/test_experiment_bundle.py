"""Relocation, immutable evidence and whole-study requirements for bundles."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

exporter = module('export_simulator_study')
runner = module('bundle_runner')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def study(tmp_path):
    root = tmp_path / 'project'
    scripts = root / 'scripts'
    scripts.mkdir(parents=True)
    for name in ('bundle_runner.py', 'evaluate_simulator.py'):
        shutil.copyfile(ROOT / 'scripts' / name, scripts / name)
    (root / 'LICENSE').write_text('Fixture license\n')
    source = root / 'results/study/source/bta_repro'
    source.mkdir(parents=True)
    (source / '__init__.py').write_text('')
    (source / 'simulator_train.py').write_text('''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--output');a=p.parse_args()
out=Path(a.output);out.mkdir();(out/'invocation.json').write_text(json.dumps({'source':__file__,'config':json.loads(Path(a.config).read_text())}))
''')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}
    manifest = {'scope': 'Synthetic integrity fixture, not learning evidence', 'source_sha256': hashes, 'runs': []}
    for seed, value in [(0, 8.), (1, -2.)]:
        out = root / f'results/study/seed{seed}'
        config = {'algorithm': 'mappo', 'seed': seed, 'steps': 8, 'num_envs': 2,
                  'rollout_steps': 2, 'eval_every_steps': 4, 'eval_seed': 1000000}
        cp = root / f'results/study/seed{seed}.json'
        write(cp, config)
        write(out / 'config.json', {'config': config, 'source_sha256': hashes, 'environment_versions': {'fixture': 1}})
        evaluation = {'return_mean': value, 'count': 2, 'seed': 1000000, 'execution': 'independent', 'deterministic': True}
        rows = [{'event': 'evaluation', 'steps': 0, 'stage': 'initial', 'evaluation': evaluation}]
        for i in (1, 2):
            rows += [{'event': 'update', 'steps': i*4, 'update': i, 'rollout_shape': [2, 2]},
                     {'event': 'evaluation', 'steps': i*4, 'stage': 'periodic' if i == 1 else 'final', 'evaluation': evaluation}]
        (out / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        write(out / 'summary.json', {'status': 'completed', 'steps': 8, 'train_environment_steps': 8,
                                    'updates': 2, 'initial_evaluation': evaluation, 'final_evaluation': evaluation})
        (out / 'checkpoint.pt').write_bytes(b'fixture bytes: no model replay claim')
        manifest['runs'].append({'variant': 'control', 'algorithm': 'mappo', 'seed': seed,
                                'config_path': str(cp), 'output_path': str(out)})
    mp = root / 'results/study/manifest.json'
    write(mp, manifest)
    return root, mp


def test_relocation_preserves_poor_outcome_and_runs_frozen_source(study, tmp_path):
    root, mp = study
    original = json.loads(mp.read_text())
    output = tmp_path / 'bundle'
    assert exporter.export_study(root, mp, output)['runs'] == 2
    moved = tmp_path / 'moved bundle'
    output.rename(moved)
    manifest = runner.verify(moved)
    assert len(manifest['runs']) == 2
    for before, after in zip(original['runs'], manifest['runs']):
        for name in ('config.json', 'metrics.jsonl', 'summary.json', 'checkpoint.pt'):
            assert (Path(before['output_path']) / name).read_bytes() == (moved / after['output_path'] / name).read_bytes()
        assert not Path(after['output_path']).is_absolute()
    # Removal of the original inputs proves the command does not depend on them.
    shutil.rmtree(root)
    result = subprocess.run([sys.executable, '-S', str(moved / 'bundle.py'), 'verify'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    fresh = tmp_path / 'fresh training'
    result = subprocess.run([sys.executable, str(moved / 'bundle.py'), 'train', '1', '--output', str(fresh)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    invoked = json.loads((fresh / 'invocation.json').read_text())
    assert Path(invoked['source']).is_relative_to(moved / 'source')
    assert invoked['config']['seed'] == 1


@pytest.mark.parametrize('change', ['unfinished', 'missing_update', 'changed_source', 'missing_checkpoint', 'summary_mismatch'])
def test_invalid_or_incomplete_whole_study_creates_no_bundle(study, tmp_path, change):
    root, mp = study
    out = root / 'results/study/seed1'
    if change == 'unfinished':
        (out / 'summary.json').unlink()
    elif change == 'missing_update':
        p = out / 'metrics.jsonl'
        p.write_text(''.join(line for line in p.read_text().splitlines(keepends=True)
                             if not ('"event": "update"' in line and '"update": 1' in line)))
    elif change == 'changed_source':
        (mp.parent / 'source/bta_repro/__init__.py').write_text('# changed')
    elif change == 'missing_checkpoint':
        (out / 'checkpoint.pt').unlink()
    else:
        p = out / 'summary.json';v = json.loads(p.read_text());v['final_evaluation']['return_mean'] = 100.;write(p, v)
    destination = tmp_path / 'bundle'
    with pytest.raises((ValueError, OSError)):
        exporter.export_study(root, mp, destination)
    assert not destination.exists()


@pytest.mark.parametrize('change', ['metrics', 'extra_source', 'escape'])
def test_verification_rejects_modified_artifacts_and_paths(study, tmp_path, change):
    root, mp = study
    destination = tmp_path / 'bundle'
    exporter.export_study(root, mp, destination)
    if change == 'metrics':
        (destination / 'runs/0000/metrics.jsonl').write_text('{}\n')
    elif change == 'extra_source':
        (destination / 'source/bta_repro/extra.py').write_text('# not frozen')
    else:
        p = destination / 'inventory.json';v = json.loads(p.read_text());v['files']['../outside'] = 'bad';write(p, v)
    with pytest.raises((ValueError, OSError)):
        runner.verify(destination)


def test_outputs_cannot_overwrite_bundle_or_prior_work(study, tmp_path):
    root, mp = study
    destination = tmp_path / 'bundle'
    exporter.export_study(root, mp, destination)
    with pytest.raises(ValueError, match='already exists'):
        exporter.export_study(root, mp, destination)
    for path in (destination / 'new', mp):
        with pytest.raises(ValueError, match='new path outside'):
            runner.command(destination, 'train', 0, path)


def test_input_symlink_cannot_leave_project(study, tmp_path):
    root, mp = study
    cp = root / 'results/study/seed0.json'
    outside = tmp_path / 'outside.json'
    shutil.copyfile(cp, outside);cp.unlink();cp.symlink_to(outside)
    with pytest.raises(ValueError, match='leaves the project'):
        exporter.export_study(root, mp, tmp_path / 'bundle')
