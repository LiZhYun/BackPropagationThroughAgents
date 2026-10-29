"""Export every run of a completed simulator study with its exact frozen source."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bta_repro.experiment_source import source_snapshot
from bta_repro.learning_curves import read_curve_run
from bta_repro.simulator_confirmation import check_training_interactions, expected_grid


def local(root, path):
    path = (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Input artifact leaves the project: {path}')
    return path


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def export_study(root, manifest_path, destination):
    root = Path(root).resolve()
    manifest_path = local(root, manifest_path)
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Destination already exists; choose a new directory')
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    if not manifest['runs']:
        raise ValueError('Study has no declared runs')
    source, hashes, _ = source_snapshot(root, manifest_path)
    blobs = {'source/bta_repro/' + name: data for name, data in source.items()}
    blobs['provenance/original_manifest.json'] = original
    portable = copy.deepcopy(manifest)
    files = {}
    expected = {}
    identities = set()
    for index, entry in enumerate(manifest['runs']):
        identity = (entry['variant'], entry['seed'])
        if identity in identities:
            raise ValueError('Duplicate variant/seed in manifest')
        identities.add(identity)
        config_path = local(root, entry['config_path'])
        output = local(root, entry['output_path'])
        actual = {**entry, 'config_path': str(config_path), 'output_path': str(output)}
        # Resolve and contain every artifact before the curve reader opens it.
        artifacts = {name: local(root, output / name) for name in
                     ('config.json', 'metrics.jsonl', 'summary.json', 'checkpoint.pt')}
        before = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for name, path in {'declared': config_path, **artifacts}.items()}
        curve = read_curve_run(actual, hashes, 'return_mean')
        if not curve['eligible'] or curve['errors']:
            raise ValueError(f"All declared runs must be complete and valid: {identity}: {curve['errors']}")
        metadata = json.loads(artifacts['config.json'].read_bytes())
        summary = json.loads(artifacts['summary.json'].read_bytes())
        rows = [json.loads(line) for line in artifacts['metrics.jsonl'].read_bytes().splitlines()]
        check_training_interactions(rows, metadata['config'], summary)
        evaluations = [row for row in rows if row.get('event') == 'evaluation']
        if [row['steps'] for row in evaluations] != expected_grid(metadata['config']):
            raise ValueError('Evaluation schedule differs from the declared configuration')
        if (summary['initial_evaluation'] != evaluations[0]['evaluation']
                or summary['final_evaluation'] != evaluations[-1]['evaluation']):
            raise ValueError('Summary and raw evaluation records disagree')
        if not artifacts['checkpoint.pt'].is_file() or artifacts['checkpoint.pt'].stat().st_size == 0:
            raise ValueError('Completed run is missing a nonempty final checkpoint')
        name = f'{index:04d}'  # Stable IDs avoid trusting labels as filesystem paths.
        cfg = f'configs/{name}.json'
        run = f'runs/{name}'
        files[cfg] = config_path
        expected[cfg] = before['declared']
        for filename, path in artifacts.items():
            files[f'{run}/{filename}'] = path
            expected[f'{run}/{filename}'] = before[filename]
        portable['runs'][index].update(config_path=cfg, output_path=run)
    portable['bundle_provenance'] = {
        'original_manifest': 'provenance/original_manifest.json',
        'original_manifest_sha256': hashlib.sha256(original).hexdigest(),
        'path_rule': 'Operational paths are relative to the bundle root; original evidence bytes are unchanged.',
        'qualification': 'Complete-study integrity export, not independent native replay or performance certification.'}
    blobs['manifest.json'] = encoded(portable)
    for source_name, bundled_name in [('scripts/bundle_runner.py', 'bundle.py'),
                                      ('scripts/evaluate_simulator.py', 'evaluate_simulator.py'),
                                      ('LICENSE', 'LICENSE')]:
        files[bundled_name] = local(root, source_name)
        expected[bundled_name] = hashlib.sha256(files[bundled_name].read_bytes()).hexdigest()
    blobs['README.md'] = b'''# Portable simulator study

Every declared run, including poor outcomes, is included with the exact frozen
training source, configuration, metadata, metrics, summary and final checkpoint.
Original evidence files are byte-identical. The original manifest is retained
under provenance; only the operational manifest uses relative paths.

From any working directory:

```sh
python /path/to/bundle/bundle.py verify
python /path/to/bundle/bundle.py list
python /path/to/bundle/bundle.py train 0 --output /new/path/training
python /path/to/bundle/bundle.py evaluate 0 --output /new/path/replay.json
```

Verification and listing use only the Python standard library. Training and
evaluation use the chosen Python interpreter and the bundled source. Install the
recorded dependencies and native engine separately; each runs/NNNN/config.json
contains the software versions, environment versions and task configuration.
Native engines, environments and licensed assets are not included. Engine paths
must be configured for the destination machine. No installer is run automatically.

Outputs must be new paths outside the bundle. Evaluation keeps the recorded
protocol and preprocessing; compare its raw records and versions with the saved
summary. Export alone does not prove native replay or successful reproduction.
Hash verification detects changed bytes, not the scientific truth of a result.
'''
    # Copy only bytes whose hashes were captured before validation.
    destination.mkdir(parents=True, exist_ok=False)
    try:
        inventory = {}
        for name, data in blobs.items():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            inventory[name] = hashlib.sha256(data).hexdigest()
        for name, source_path in files.items():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected[name]:
                raise ValueError(f'Input changed during export: {name}')
            inventory[name] = digest
        (destination / 'inventory.json').write_bytes(encoded({'format_version': 1, 'files': inventory}))
    except BaseException:
        shutil.rmtree(destination)  # Only the new directory created above belongs to this attempt.
        raise
    return {'runs': len(manifest['runs']), 'files': len(inventory), 'destination': str(destination),
            'original_manifest_sha256': hashlib.sha256(original).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = export_study(ROOT, args.manifest, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
