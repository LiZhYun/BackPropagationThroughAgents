"""Evidence checks for full-budget, seed-level learning-curve reporting."""

import json
import math
from pathlib import Path
import random

import pytest

from bta_repro.learning_curves import seed_statistics, summarize_curves


def _evaluation(value, count=10):
    return {"return_mean": value, "count": count, "seed": 1_000_000,
            "execution": "decentralized", "deterministic": True,
            "win_rate": .5, "win_records": count, "capture_rate": .8, "capture_records": count,
            "fixed_window": {"reward_per_environment_step": value / 100, "window_count": 100,
                             "steps_per_window": 200, "environment_steps": 20_000,
                             "scope": "including_unfinished", "seed_scheme": "test_v1",
                             "seed": 1_000_000, "deterministic": True}}


def _manifest(tmp_path, curves=((0., 4., 10.), (0., 6., 20.))):
    manifest = {"scope": "test development fixture", "source_sha256": {"train.py": "frozen"}, "runs": []}
    for seed, values in enumerate(curves):
        output = tmp_path / f"variant_seed{seed}"
        output.mkdir()
        config = {"seed": seed, "algorithm": "agentmixer", "steps": 100, "hidden_dim": 64}
        config_path = tmp_path / f"seed{seed}.json"
        config_path.write_text(json.dumps(config))
        (output / "config.json").write_text(json.dumps({"config": config, "source_sha256": manifest["source_sha256"],
                                                        "environment_versions": {"test": "1"}}))
        points = [{"steps": step, "event": "evaluation", "stage": stage, "evaluation": _evaluation(value)}
                  for step, stage, value in zip((0, 20, 100), ("initial", "periodic", "final"), values)]
        (output / "metrics.jsonl").write_text("".join(json.dumps(point) + "\n" for point in points))
        (output / "summary.json").write_text(json.dumps({"status": "completed", "steps": 100,
                                                         "final_evaluation": _evaluation(values[-1])}))
        manifest["runs"].append({"variant": "variant", "algorithm": "agentmixer", "seed": seed,
                                 "output_path": str(output), "config_path": str(config_path)})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _edit_json(path, edit):
    value = json.loads(path.read_text())
    edit(value)
    path.write_text(json.dumps(value))


def _edit_curve(tmp_path, edit, seed=0):
    path = tmp_path / f"variant_seed{seed}" / "metrics.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    edit(records)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_area_uses_actual_steps_and_statistics_use_training_seeds(tmp_path):
    report = summarize_curves(_manifest(tmp_path))
    assert [run["full_budget_auc"]["area"] for run in report["runs"]] == [600., 1100.]
    result = report["groups"][0]["aggregate"]
    assert result["normalized_auc"]["mean"] == 8.5
    assert result["final"]["mean"] == result["final"]["median"] == 15.
    assert result["final"]["sample_sd_across_training_seeds"] == pytest.approx(math.sqrt(50))
    assert result["final"]["seed_count"] == 2
    assert result["final"]["bootstrap_95pct_mean"] is None
    assert [point["steps"] for point in result["curve"]] == [0, 20, 100]


def test_bootstrap_is_seed_level_repeatable_and_leaves_global_rng_unchanged():
    rng = random.getstate()
    constant = seed_statistics([7.] * 5, confidence_intervals=True)
    assert constant["bootstrap_95pct_mean"] == constant["bootstrap_95pct_median"] == [7., 7.]
    values = [1., 2., 3., 4., 100.]
    first = seed_statistics(values, confidence_intervals=True)
    assert first == seed_statistics(values, confidence_intervals=True)
    assert random.getstate() == rng
    lower, upper = first["bootstrap_95pct_mean"]
    assert min(values) <= lower < first["mean"] < upper <= max(values)
    assert seed_statistics(values[:4], confidence_intervals=True)["bootstrap_95pct_mean"] is None


@pytest.mark.parametrize("status", ["failed", "unfinished", "not_started"])
def test_missing_or_failed_seed_is_retained_and_blocks_whole_group(tmp_path, status):
    path = _manifest(tmp_path)
    output = tmp_path / "variant_seed1"
    if status == "failed":
        _edit_json(output / "summary.json", lambda x: x.update(status="failed"))
    else:
        (output / "summary.json").unlink()
        if status == "not_started":
            for file in output.iterdir():
                file.unlink()
            output.rmdir()
    report = summarize_curves(path)
    assert report["runs"][1]["status"] == status
    assert report["groups"][0]["declared_seeds"] == [0, 1]
    assert report["groups"][0]["eligible_seeds"] == [0]
    assert report["groups"][0]["aggregate"] is None


@pytest.mark.parametrize("defect", ["initial_missing", "final_missing", "duplicate_step", "nonfinite",
                                    "count_change", "seed_change", "summary_mismatch", "budget_mismatch",
                                    "source_mismatch", "config_mismatch", "bad_tail"])
def test_invalid_curve_cannot_supply_a_completed_score(tmp_path, defect):
    path = _manifest(tmp_path)
    output = tmp_path / "variant_seed0"
    if defect == "initial_missing":
        _edit_curve(tmp_path, lambda r: r.pop(0))
    elif defect == "final_missing":
        _edit_curve(tmp_path, lambda r: r.pop())
    elif defect == "duplicate_step":
        _edit_curve(tmp_path, lambda r: r.insert(1, r[0]))
    elif defect == "nonfinite":
        _edit_curve(tmp_path, lambda r: r[1]["evaluation"].update(return_mean=float("nan")))
    elif defect == "count_change":
        _edit_curve(tmp_path, lambda r: r[1]["evaluation"].update(count=2))
    elif defect == "seed_change":
        _edit_curve(tmp_path, lambda r: r[1]["evaluation"].update(seed=123))
    elif defect == "summary_mismatch":
        _edit_json(output / "summary.json", lambda r: r["final_evaluation"].update(return_mean=99))
    elif defect == "budget_mismatch":
        _edit_json(output / "summary.json", lambda r: r.update(steps=99))
    elif defect == "source_mismatch":
        _edit_json(output / "config.json", lambda r: r["source_sha256"].update(other="source"))
    elif defect == "config_mismatch":
        _edit_json(output / "config.json", lambda r: r["config"].update(hidden_dim=32))
    else:
        with (output / "metrics.jsonl").open("a") as f:
            f.write('{"event":')
    report = summarize_curves(path)
    assert report["runs"][0]["errors"]
    assert not report["runs"][0]["eligible"]
    assert report["runs"][0]["full_budget_auc"] is None
    assert report["groups"][0]["aggregate"] is None


def test_live_partial_json_tail_is_reported_without_extrapolation(tmp_path):
    path = _manifest(tmp_path)
    output = tmp_path / "variant_seed0"
    (output / "summary.json").unlink()
    _edit_curve(tmp_path, lambda r: r.pop())
    with (output / "metrics.jsonl").open("a") as f:
        f.write('{"event":')
    run = summarize_curves(path)["runs"][0]
    assert run["ignored_unfinished_tail"]
    assert not run["errors"] and not run["eligible"]
    assert run["points"][-1]["steps"] == 20
    assert run["full_budget_auc"] is None


@pytest.mark.parametrize("difference", ["grid", "config", "environment"])
def test_individually_valid_runs_with_different_protocols_are_not_pooled(tmp_path, difference):
    path = _manifest(tmp_path)
    if difference == "grid":
        _edit_curve(tmp_path, lambda r: r[1].update(steps=30), seed=1)
    elif difference == "config":
        _edit_json(tmp_path / "seed1.json", lambda r: r.update(hidden_dim=32))
        _edit_json(tmp_path / "variant_seed1/config.json", lambda r: r["config"].update(hidden_dim=32))
    else:
        _edit_json(tmp_path / "variant_seed1/config.json", lambda r: r["environment_versions"].update(test="2"))
    report = summarize_curves(path)
    assert all(run["eligible"] for run in report["runs"])
    assert report["groups"][0]["aggregation_errors"]
    assert report["groups"][0]["aggregate"] is None


def test_fixed_window_rate_uses_its_own_denominator_and_checks_it(tmp_path):
    path = _manifest(tmp_path)
    report = summarize_curves(path, "fixed_window_reward_per_step")
    assert report["groups"][0]["aggregate"]["final"]["mean"] == pytest.approx(.15)
    assert report["runs"][0]["evaluation_protocol"]["window_count"] == 100
    _edit_curve(tmp_path, lambda r: r[1]["evaluation"]["fixed_window"].update(environment_steps=200))
    assert not summarize_curves(path, "fixed_window_reward_per_step")["runs"][0]["eligible"]


@pytest.mark.parametrize("metric", ["win_rate", "capture_rate"])
def test_success_rate_needs_a_record_for_each_episode(tmp_path, metric):
    path = _manifest(tmp_path)
    assert summarize_curves(path, metric)["groups"][0]["aggregate"] is not None
    key = "win_records" if metric == "win_rate" else "capture_records"
    _edit_curve(tmp_path, lambda r: r[1]["evaluation"].update({key: 9}))
    assert not summarize_curves(path, metric)["runs"][0]["eligible"]


def test_reject_duplicate_training_seed_and_unknown_metric(tmp_path):
    path = _manifest(tmp_path)
    with pytest.raises(ValueError, match="metric"):
        summarize_curves(path, "training_reward")
    _edit_json(path, lambda m: m["runs"].append(m["runs"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        summarize_curves(path)
