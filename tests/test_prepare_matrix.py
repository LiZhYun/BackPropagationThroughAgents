"""Matrix validation must preserve selected code and configuration budgets."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[1] / "scripts/prepare_matrix.py"
spec = importlib.util.spec_from_file_location("prepare_matrix", path)
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


@pytest.fixture
def project(tmp_path):
    package = tmp_path / "src/bta_repro"
    package.mkdir(parents=True)
    for name in (*preparation.MATRIX_MODULES, "unneeded.py"):
        (package / name).write_text("# working tree\n")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_matrix_array.py").write_text("# launcher\n")
    return tmp_path


def test_default_snapshot_is_minimal_and_json_budget_is_preserved(project):
    out = preparation.prepare(project, overrides={"steps": 200000}, stamp="test")
    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest["source_sha256"]) == set(preparation.MATRIX_MODULES)
    assert len(manifest["runs"]) == 24
    assert all(json.loads(Path(r["config_path"]).read_text())["steps"] == 200000
               for r in manifest["runs"])


def test_selected_source_is_exact_and_explicit_budget_overrides_json(project):
    source = project / "results/selected/source/bta_repro"
    source.mkdir(parents=True)
    for name in (*preparation.MATRIX_MODULES, "selected_extra.py"):
        (source / name).write_text("# selected code\n")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")}
    manifest_path = source.parents[1] / "manifest.json"
    manifest_path.write_text(json.dumps({"source_sha256": hashes}))
    out = preparation.prepare(project, algorithms=["agentmixer"], games=["climbing"], seeds=[2000, 2001],
                              overrides={"steps": 100000}, steps=200000, workers=1,
                              wall_time_minutes=45, source_manifest=manifest_path, stamp="test")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["source_sha256"] == hashes
    assert manifest["source_provenance"]["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert all((out / "source/bta_repro" / p.name).read_bytes() == p.read_bytes() for p in source.glob("*.py"))
    assert [r["seed"] for r in manifest["runs"]] == [2000, 2001]
    assert all(json.loads(Path(r["config_path"]).read_text())["steps"] == 200000 for r in manifest["runs"])
    assert "--array=0-0%2" in (out / "submit.sbatch").read_text()
    assert "--time=00:45:00" in (out / "submit.sbatch").read_text()


@pytest.mark.parametrize("arguments", [
    {"seeds": []}, {"seeds": [0, 0]}, {"seeds": [True]}, {"seeds": [-1]},
    {"algorithms": ["bppo", "bppo"]}, {"games": []}, {"games": ["unknown"]}, {"steps": 0},
])
def test_invalid_run_set_leaves_no_experiment(project, arguments):
    with pytest.raises(ValueError):
        preparation.prepare(project, stamp="test", **arguments)
    assert not (project / "results").exists()


def test_preparation_preserves_existing_experiment(project):
    out = preparation.prepare(project, stamp="test")
    before = (out / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        preparation.prepare(project, stamp="test")
    assert (out / "manifest.json").read_bytes() == before
