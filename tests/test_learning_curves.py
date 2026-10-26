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
                             "scope": "all_steps_in_fixed_windows_including_unfinished_episodes",
                             "seed_scheme": "numpy_seed_sequence_bta_win_v1",
                             "seed": 1_000_000, "deterministic": True, "reward_sum": value * 200,
                             "windows": [{"window": index, "seed": 1000 + index,
                                          "steps": 200, "reward_sum": value * 2,
                                          "completed_episodes": 1, "captures": 0, "timeouts": 1,
                                          "unfinished_episode_steps": 72,
                                          "unfinished_episode_return": value * .72}
                                         for index in range(100)]}}


def _manifest(tmp_path, curves=((0., 4., 10.), (0., 6., 20.))):
    manifest = {"scope": "test development fixture", "source_sha256": {"train.py": "frozen"}, "runs": []}
    for seed, values in enumerate(curves):
        output = tmp_path / f"variant_seed{seed}"
        output.mkdir()
        config = {"seed": seed, "algorithm": "agentmixer", "steps": 100, "hidden_dim": 64,
                  "environment": "prey", "eval_windows": 100, "eval_window_steps": 200,
                  "eval_seed": 1_000_000}
        config_path = tmp_path / f"seed{seed}.json"
        config_path.write_text(json.dumps(config))
        (output / "config.json").write_text(json.dumps({"config": config, "source_sha256": manifest["source_sha256"],
                                                        "environment_versions": {"test": "1"}}))
        points = [{"steps": step, "event": "evaluation", "stage": stage, "evaluation": _evaluation(value)}
                  for step, stage, value in zip((0, 20, 100), ("initial", "periodic", "final"), values)]
        (output / "metrics.jsonl").write_text("".join(json.dumps(point) + "\n" for point in points))
        (output / "summary.json").write_text(json.dumps({"status": "completed", "steps": 100,
                                                         "initial_evaluation": _evaluation(values[0]),
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


@pytest.mark.parametrize("defect,expected", [
    ("missing_evaluation", "evaluation must be a JSON object"),
    ("null_windows", "fixed_window must be a JSON object"),
    ("nonobject_windows", "fixed_window must be a JSON object"),
    ("missing_records", "windows"),
    ("missing_window", "raw record count"),
    ("duplicate_index", "indices"),
    ("boolean_index", "window index"),
    ("duplicate_seed", "distinct uint32"),
    ("invalid_seed", "distinct uint32"),
    ("changed_seed_schedule", "convention changed"),
    ("wrong_length", "raw window length"),
    ("nonfinite_reward", "finite number"),
    ("boolean_reward", "finite number"),
    ("wrong_total", "raw window totals"),
    ("wrong_rate", "divided by all decisions"),
    ("fractional_denominator", "positive integer"),
    ("negative_partial_steps", "nonnegative integer"),
    ("excess_partial_steps", "counts exceed"),
    ("empty_partial_reward", "empty unfinished"),
    ("unaccounted_partial", "retain all steps"),
    ("extra_episodes", "counts exceed"),
    ("missing_endings", "account for every"),
    ("wrong_scope", "include unfinished"),
    ("unknown_seed_scheme", "unsupported"),
    ("wrong_execution", "execution disagrees"),
    ("stochastic_windows", "deterministic execution"),
    ("stochastic_episodes", "deterministic execution"),
])
def test_fixed_window_raw_corruption_blocks_aggregation(tmp_path, defect, expected):
    path = _manifest(tmp_path)

    def corrupt(records):
        evaluation = records[1]["evaluation"]
        window = evaluation["fixed_window"]
        first = window["windows"][0]
        changes = {
            "missing_evaluation": lambda: records[1].update(evaluation=None),
            "null_windows": lambda: evaluation.update(fixed_window=None),
            "nonobject_windows": lambda: evaluation.update(fixed_window=[]),
            "missing_records": lambda: window.pop("windows"),
            "missing_window": lambda: window["windows"].pop(),
            "duplicate_index": lambda: window["windows"][1].update(window=0),
            "boolean_index": lambda: first.update(window=False),
            "duplicate_seed": lambda: window["windows"][1].update(seed=first["seed"]),
            "invalid_seed": lambda: first.update(seed=2**32),
            "changed_seed_schedule": lambda: first.update(seed=3000),
            "wrong_length": lambda: first.update(steps=199),
            "nonfinite_reward": lambda: first.update(reward_sum=float("nan")),
            "boolean_reward": lambda: first.update(reward_sum=True),
            "wrong_total": lambda: window.update(reward_sum=999),
            "wrong_rate": lambda: window.update(reward_per_environment_step=999),
            "fractional_denominator": lambda: window.update(environment_steps=20000.0),
            "negative_partial_steps": lambda: first.update(unfinished_episode_steps=-1),
            "excess_partial_steps": lambda: first.update(unfinished_episode_steps=201),
            "empty_partial_reward": lambda: first.update(unfinished_episode_steps=0),
            "unaccounted_partial": lambda: first.update(completed_episodes=0, timeouts=0),
            "extra_episodes": lambda: first.update(completed_episodes=200),
            "missing_endings": lambda: first.update(timeouts=0),
            "wrong_scope": lambda: window.update(scope="completed_episodes_only"),
            "unknown_seed_scheme": lambda: window.update(seed_scheme="unspecified"),
            "wrong_execution": lambda: evaluation.update(execution="centralized"),
            "stochastic_windows": lambda: window.update(deterministic=False),
            "stochastic_episodes": lambda: evaluation.update(deterministic=False),
        }
        changes[defect]()

    _edit_curve(tmp_path, corrupt)
    report = summarize_curves(path, "fixed_window_reward_per_step")
    assert not report["runs"][0]["eligible"]
    assert any(expected in error for error in report["runs"][0]["errors"])
    assert report["runs"][1]["eligible"]
    assert report["groups"][0]["aggregate"] is None


@pytest.mark.parametrize("field,value", [("eval_windows", 99), ("eval_window_steps", 100),
                                         ("eval_seed", 123), ("environment", "football")])
def test_fixed_window_protocol_must_match_frozen_config(tmp_path, field, value):
    path = _manifest(tmp_path)
    _edit_json(tmp_path / "seed0.json", lambda c: c.update({field: value}))
    _edit_json(tmp_path / "variant_seed0/config.json", lambda m: m["config"].update({field: value}))
    report = summarize_curves(path, "fixed_window_reward_per_step")
    assert not report["runs"][0]["eligible"] and report["runs"][0]["errors"]
    assert report["groups"][0]["aggregate"] is None


@pytest.mark.parametrize("stage", ["initial_evaluation", "final_evaluation"])
def test_fixed_window_summary_must_retain_exact_raw_records(tmp_path, stage):
    path = _manifest(tmp_path)

    def exchange_rewards(summary):
        windows = summary[stage]["fixed_window"]["windows"]
        # Preserve the total and rate; only the underlying per-window evidence differs.
        windows[0]["reward_sum"] += 1
        windows[1]["reward_sum"] -= 1

    _edit_json(tmp_path / "variant_seed0/summary.json", exchange_rewards)
    report = summarize_curves(path, "fixed_window_reward_per_step")
    assert not report["runs"][0]["eligible"]
    assert "raw windows disagree" in report["runs"][0]["errors"][0]
    assert report["groups"][0]["aggregate"] is None


def test_fixed_windows_retain_reward_when_no_episode_finishes(tmp_path):
    path = _manifest(tmp_path, curves=((-3., 4., 10.),))

    def make_unfinished(evaluation):
        for window in evaluation["fixed_window"]["windows"]:
            window.update(completed_episodes=0, captures=0, timeouts=0,
                          unfinished_episode_steps=200,
                          unfinished_episode_return=window["reward_sum"])

    _edit_curve(tmp_path, lambda records: [make_unfinished(r["evaluation"]) for r in records])
    _edit_json(tmp_path / "variant_seed0/summary.json",
               lambda s: [make_unfinished(s[stage]) for stage in ("initial_evaluation", "final_evaluation")])
    report = summarize_curves(path, "fixed_window_reward_per_step")
    assert report["runs"][0]["eligible"]
    assert [p["value"] for p in report["runs"][0]["points"]] == [-.03, .04, .1]
    assert report["groups"][0]["aggregate"]["final"]["mean"] == .1


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
