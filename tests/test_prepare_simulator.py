"""Confirmation runs can retain selected code despite subsequent development."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[1] / 'scripts/prepare_simulator.py'
spec = importlib.util.spec_from_file_location('prepare_simulator', path)
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


@pytest.fixture
def project(tmp_path):
    package = tmp_path / 'src/bta_repro'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'simulator_train.py').write_text('# current version\n')
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    for name in ('run_simulator_array.py', 'run_simulator_worker.py'):
        (scripts / name).write_text('# launcher\n')
    return tmp_path


@pytest.fixture
def plan():
    return {'label': 'confirmation', 'scope': 'test', 'seeds': [1000, 1001],
            'base_config': {'environment': 'prey', 'steps': 16, 'num_envs': 2},
            'variants': {'bppo': {'algorithm': 'bppo'}, 'arppo': {'algorithm': 'arppo'}}}


def frozen(project):
    source = project / 'results/selected/source/bta_repro'
    source.mkdir(parents=True)
    (source / '__init__.py').write_text('')
    (source / 'simulator_train.py').write_text('# selected version\n')
    manifest = source.parents[1] / 'manifest.json'
    manifest.write_text(json.dumps({'source_sha256': {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}}))
    return manifest, source


def test_confirmation_copies_exact_selected_source_and_retains_origin(project, plan):
    manifest, source = frozen(project)
    output = preparation.prepare(plan, project, source_manifest=manifest, stamp='test')
    saved = json.loads((output / 'manifest.json').read_text())
    assert (output / 'source/bta_repro/simulator_train.py').read_bytes() == (source / 'simulator_train.py').read_bytes()
    assert saved['source_provenance']['manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert saved['source_sha256'] == json.loads(manifest.read_text())['source_sha256']
    assert [(r['algorithm'], r['seed']) for r in saved['runs']] == [
        ('bppo', 1000), ('bppo', 1001), ('arppo', 1000), ('arppo', 1001)]
    assert len({r['output_path'] for r in saved['runs']}) == 4
    assert 'source' in (output / 'submit.sbatch').read_text()


@pytest.mark.parametrize('change', ['modify', 'remove', 'add'])
def test_changed_frozen_source_rejected_before_creating_run(project, plan, change):
    manifest, source = frozen(project)
    if change == 'modify':
        (source / 'simulator_train.py').write_text('# changed\n')
    elif change == 'remove':
        (source / '__init__.py').unlink()
    else:
        (source / 'extra.py').write_text('# unexpected\n')
    with pytest.raises(ValueError, match='source|snapshot'):
        preparation.prepare(plan, project, source_manifest=manifest, stamp='test')
    assert not (project / 'results/confirmation_test').exists()


def test_default_still_copies_current_source(project, plan):
    output = preparation.prepare(plan, project, stamp='test')
    assert (output / 'source/bta_repro/simulator_train.py').read_text() == '# current version\n'
    assert json.loads((output / 'manifest.json').read_text())['source_provenance']['kind'] == 'working_tree'


@pytest.mark.parametrize('seeds', [[], [0, 0], [True], [-1]])
def test_bad_seeds_create_no_experiment(project, plan, seeds):
    plan['seeds'] = seeds
    with pytest.raises(ValueError):
        preparation.prepare(plan, project, stamp='test')
    assert not (project / 'results').exists()


def test_invalid_config_creates_no_experiment(project, plan):
    plan['base_config']['steps'] = 15
    with pytest.raises(ValueError, match='divisible'):
        preparation.prepare(plan, project, stamp='test')
    assert not (project / 'results').exists()


def test_external_source_manifest_is_rejected(project, plan):
    with pytest.raises(ValueError, match='belong to this repository'):
        preparation.prepare(plan, project, source_manifest=project.parent / 'other/manifest.json')


def test_escaping_source_symlink_is_rejected(project, plan):
    manifest, source = frozen(project)
    (source / 'escape.py').symlink_to(project / 'src/bta_repro/simulator_train.py')
    with pytest.raises(ValueError, match='outside the snapshot'):
        preparation.prepare(plan, project, source_manifest=manifest)


def test_existing_experiment_is_never_overwritten(project, plan):
    output = preparation.prepare(plan, project, stamp='test')
    before = (output / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        preparation.prepare(plan, project, stamp='test')
    assert (output / 'manifest.json').read_bytes() == before
