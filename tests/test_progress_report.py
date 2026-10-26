"""The public progress command remains usable without private run artifacts."""

import importlib.util
import json
from pathlib import Path


path = Path(__file__).resolve().parents[1] / "scripts/report_progress.py"
spec = importlib.util.spec_from_file_location("report_progress", path)
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def test_clean_checkout_reports_missing_manifest_without_inventing_run_counts(tmp_path):
    write(tmp_path / "experiments/jobs.jsonl", {"job_id": "42", "manifest": "results/absent/manifest.json"})
    record, = reporter.report(tmp_path, 1)
    assert record["artifact_status"] == "manifest_missing"
    assert record["declared_runs"] is None and record["counts"] is None
    assert record["last_recorded_training_steps"] is None
    assert record["runs"] == []


def test_bad_manifest_is_not_a_failed_training_run(tmp_path):
    write(tmp_path / "experiments/jobs.jsonl", {"job_id": "42", "manifest": "results/bad/manifest.json"})
    write(tmp_path / "results/bad/manifest.json", [])
    record, = reporter.report(tmp_path, 0)
    assert record["artifact_status"] == "manifest_unreadable"
    assert record["counts"] is None


def test_live_partial_line_keeps_last_complete_artifact_progress(tmp_path):
    write(tmp_path / "experiments/jobs.jsonl", {"job_id": "42", "manifest": "results/test/manifest.json"})
    write(tmp_path / "results/test/manifest.json", {"scope": "test", "runs": [
        {"output_path": "results/test/complete"}, {"output_path": "results/test/live"}]})
    write(tmp_path / "results/test/complete/summary.json", {"status": "completed", "steps": 100})
    path = tmp_path / "results/test/live/metrics.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"steps": 20}\n{"steps": 30}\n{"steps":')
    record, = reporter.report(tmp_path, 0)
    assert record["counts"]["completed"] == 1
    assert record["counts"]["incomplete"] == 1
    assert record["last_recorded_training_steps"] == 130


def test_copied_manifest_does_not_read_another_checkout(tmp_path):
    project = tmp_path / "checkout"
    write(project / "experiments/jobs.jsonl", {"job_id": "42", "manifest": "results/test/manifest.json"})
    external = tmp_path / "other/summary.json"
    write(external, {"status": "completed", "steps": 123})
    write(project / "results/test/manifest.json", {"runs": [{"output_path": str(external.parent)}]})
    record, = reporter.report(project, 0)
    assert record["counts"]["unavailable"] == 1
    assert record["counts"]["completed"] == 0
    assert record["last_recorded_training_steps"] == 0


def test_manifest_path_outside_checkout_is_unavailable(tmp_path):
    project = tmp_path / "checkout"
    write(project / "experiments/jobs.jsonl", {"job_id": "42", "manifest": "../other/manifest.json"})
    write(tmp_path / "other/manifest.json", {"runs": []})
    record, = reporter.report(project, 0)
    assert record["artifact_status"] == "manifest_unreadable"
    assert "outside this repository" in record["error"]
