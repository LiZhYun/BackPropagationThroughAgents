import json
from pathlib import Path

import pytest

from bta_repro.matrix_validation import digest, report_study, wilson


PAYOFFS = {"climbing": [[11., -30., 0.], [-30., 7., 0.], [0., 6., 5.]],
           "penalty": [[-100., 0., 10.], [0., 2., 0.], [10., 0., -100.]]}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def evaluation(game, *, uniform=False):
    values = sum(PAYOFFS[game], [])
    optimum = max(values)
    index = values.index(optimum)
    probabilities = [1 / 9] * 9 if uniform else [float(i == index) for i in range(9)]
    return {"execution_joint_probabilities": probabilities, "optimum_per_step": optimum,
            "execution_policy_expected_return": sum(p * r for p, r in zip(probabilities, values)),
            "optimal_action_probability": sum(p for p, r in zip(probabilities, values) if r == optimum),
            "greedy_actions": [index // 3, index % 3], "greedy_return": optimum}


def study(root, seeds=(100, 101), algorithms=("arppo", "bppo"), games=tuple(PAYOFFS), steps=100000):
    source = root / "runs/source/bta_repro/envs.py"
    source.parent.mkdir(parents=True)
    source.write_text(f"PAYOFFS = {PAYOFFS!r}\n")
    for name in ("run_array.py", "submit.sbatch"):
        (root / "runs" / name).write_text("fixture")
    source_hash = {"envs.py": digest(source)}
    plan = {"study_id": "fixture", "scope": "validation fixture", "training_steps": steps,
            "frozen_utc": "2026-01-01T00:00:00+00:00", "algorithms": list(algorithms),
            "games": list(games), "training_seeds": list(seeds),
            "all_prior_recorded_training_seeds": [0, 1, 2],
            "manifest": "runs/manifest.json", "source_sha256": source_hash,
            "launcher_sha256": digest(root / "runs/run_array.py"),
            "submission_script_sha256": digest(root / "runs/submit.sbatch"),
            "environment": {"python": "test", "torch": "test", "numpy": "test"},
            "evaluation": {"threshold": {"climbing": 10.45, "penalty": 9.5}}, "runs": []}
    for algorithm in plan["algorithms"]:
        for game in plan["games"]:
            for seed in seeds:
                name = f"{algorithm}_{game}_{seed}"
                config = {"algorithm": algorithm, "game": game, "seed": seed, "steps": steps,
                          "batch_size": 5000, "eval_every": 10}
                path = root / f"runs/{name}.json"
                write(path, config)
                entry = {k: config[k] for k in ("algorithm", "game", "seed")}
                entry.update(config_path=str(path.relative_to(root)), config_sha256=digest(path),
                             resolved_config=config, output_path=f"runs/{name}")
                plan["runs"].append(entry)
                out = root / entry["output_path"]
                metadata = {"config": config, "source_sha256": source_hash, **plan["environment"],
                            "created_utc": "2026-01-02T00:00:00+00:00"}
                write(out / "config.json", metadata)
                final = {"status": "completed", "steps": steps, **evaluation(game)}
                rows = [{"steps": step, **evaluation(game, uniform=step == 0)}
                        for step in [*range(0, steps, 50000), steps]]
                if algorithm == "agentmixer":
                    for row in [final, *rows]:
                        # An optimal local policy can coexist with a poor central policy.
                        central = evaluation(game, uniform=True)
                        row.update(training_joint_probabilities=central["execution_joint_probabilities"],
                                   training_policy_expected_return=central["execution_policy_expected_return"],
                                   training_optimal_action_probability=central["optimal_action_probability"])
                write(out / "summary.json", final)
                (out / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    manifest = {"source_sha256": source_hash, "runs": [
        {k: r[k] for k in ("algorithm", "game", "seed", "config_path", "output_path")}
        for r in plan["runs"]]}
    write(root / plan["manifest"], manifest)
    plan["manifest_sha256"] = digest(root / plan["manifest"])
    register(root, plan)
    return plan


def register(root, plan):
    write(root / "study.json", plan)
    write(root / "experiments/jobs.jsonl", {"study_plan": "study.json", "study_plan_sha256": digest(root / "study.json")})


def report(root):
    return report_study(root / "study.json", root)


def test_complete_study_checks_probabilities_and_computes_full_curve_statistics(tmp_path):
    study(tmp_path)
    result = report(tmp_path)
    assert result["errors"] == []
    assert len(result["runs"]) == 8
    assert all(r["eligible"] for r in result["runs"])
    assert all(g["complete"] and g["threshold_successes"] == 2 for g in result["groups"])
    run = result["runs"][0]
    initial = evaluation("climbing", uniform=True)["execution_policy_expected_return"]
    assert run["normalized_expected_return_auc"] == pytest.approx(.25 * initial + .75 * 11)
    assert run["first_observed_sustained_95pct_expected_return_step"] == 50000
    assert all(p["relative_mean_change_percent"] == 0 for p in result["paired_comparisons"])
    assert result["validation_decision"] == "not_assessed"  # Two seeds cannot pass the ten-seed rule.
    assert result["paper_reproduction"] == "not_established"


def test_pending_seed_is_retained_and_prevents_group_and_paired_aggregation(tmp_path):
    plan = study(tmp_path)
    (tmp_path / plan["runs"][0]["output_path"] / "summary.json").unlink()
    result = report(tmp_path)
    assert len(result["runs"]) == 8
    assert result["runs"][0]["status"] == "unfinished"
    assert result["groups"][0]["aggregate"] is None
    assert result["groups"][0]["threshold_success_wilson_95pct"] is None
    assert not result["paired_comparisons"][0]["eligible"]
    assert result["paired_comparisons"][1]["eligible"]


@pytest.mark.parametrize("corruption", ["source", "manifest", "launcher", "plan", "declared_config"])
def test_changed_frozen_inputs_cannot_pass_validation(tmp_path, corruption):
    plan = study(tmp_path)
    paths = {"source": "runs/source/bta_repro/envs.py", "manifest": plan["manifest"],
             "launcher": "runs/run_array.py", "plan": "study.json", "declared_config": plan["runs"][0]["config_path"]}
    path = tmp_path / paths[corruption]
    path.write_text(path.read_text() + "\n")  # Even a semantically identical edit changes provenance.
    result = report(tmp_path)
    assert len(result["runs"]) == 8
    assert result["errors"] or result["runs"][0]["errors"]
    assert result["validation_decision"] != "passed"
    if corruption != "declared_config":
        assert all(not r["eligible"] for r in result["runs"])


@pytest.mark.parametrize("corruption", ["config", "versions", "source", "timestamp", "budget", "failed",
                                         "expected", "probability", "nan", "missing_point", "summary_disagreement"])
def test_invalid_run_artifact_is_retained_without_aggregate(tmp_path, corruption):
    plan = study(tmp_path)
    out = tmp_path / plan["runs"][0]["output_path"]
    if corruption in ("config", "versions", "source", "timestamp"):
        path = out / "config.json"
        data = json.loads(path.read_text())
        if corruption == "config": data["config"]["seed"] += 1
        if corruption == "versions": data["torch"] = "different"
        if corruption == "source": data["source_sha256"]["envs.py"] = "different"
        if corruption == "timestamp": data["created_utc"] = "2025-01-01T00:00:00+00:00"
        write(path, data)
    elif corruption == "missing_point":
        path = out / "metrics.jsonl"
        rows = path.read_text().splitlines()
        path.write_text(rows[0] + "\n" + rows[-1] + "\n")
    else:
        path = out / "summary.json"
        data = json.loads(path.read_text())
        if corruption == "budget": data["steps"] = 99999
        if corruption == "failed": data["status"] = "failed"
        if corruption == "expected": data["execution_policy_expected_return"] = 100
        if corruption == "probability": data["execution_joint_probabilities"][0] = 0.5
        if corruption == "nan": data["execution_policy_expected_return"] = float("nan")
        if corruption == "summary_disagreement": data.update(evaluation("climbing", uniform=True))
        write(path, data)
    result = report(tmp_path)
    assert result["runs"][0]["errors"]
    assert not result["runs"][0]["eligible"]
    assert result["groups"][0]["aggregate"] is None


def test_greedy_optimum_without_expected_reward_does_not_meet_threshold(tmp_path):
    plan = study(tmp_path)
    out = tmp_path / plan["runs"][0]["output_path"]
    final = evaluation("climbing", uniform=True)
    write(out / "summary.json", {"status": "completed", "steps": 100000, **final})
    rows = [{"steps": step, **evaluation("climbing", uniform=step in (0, 100000))} for step in (0, 50000, 100000)]
    (out / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
    run = report(tmp_path)["runs"][0]
    assert run["eligible"] and not run["threshold_success"]
    assert run["final"]["greedy_return"] == 11
    assert run["first_observed_sustained_95pct_expected_return_step"] is None


def test_complete_ten_seed_validation_and_wilson_do_not_claim_paper_success(tmp_path, monkeypatch):
    # Confidence-interval resampling is covered in test_learning_curves; this
    # fixture tests the independent decision gate without repeating 10k draws.
    from bta_repro import matrix_validation as module
    from bta_repro.learning_curves import seed_statistics
    monkeypatch.setattr(module, "seed_statistics", lambda values, **kw: seed_statistics(values))
    study(tmp_path, seeds=range(100, 110))
    result = report(tmp_path)
    assert result["validation_decision"] == "passed"
    assert result["paper_reproduction"] == "not_established"
    assert wilson(10, 10) == pytest.approx([.7224672001371107, 1])
    plan = json.loads((tmp_path / "study.json").read_text())
    entry = next(r for r in plan["runs"] if r["algorithm"] == "bppo")
    out = tmp_path / entry["output_path"]
    failed_score = evaluation(entry["game"], uniform=True)
    write(out / "summary.json", {"status": "completed", "steps": 100000, **failed_score})
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    rows[-1].update(failed_score)
    (out / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
    assert report(tmp_path)["validation_decision"] == "failed"
    (out / "summary.json").unlink()
    assert report(tmp_path)["validation_decision"] == "incomplete"
    plan["all_prior_recorded_training_seeds"].append(100)
    register(tmp_path, plan)
    assert report(tmp_path)["validation_decision"] == "not_assessed"


def test_missing_planned_seed_is_rejected(tmp_path):
    plan = study(tmp_path)
    plan["runs"].pop()
    register(tmp_path, plan)
    result = report(tmp_path)
    assert "missing or duplicate planned run" in result["errors"][0]
    assert len(result["runs"]) == 7  # Retain every declaration, even when the study itself is invalid.


def test_duplicate_output_directory_is_rejected_even_if_manifest_agrees(tmp_path):
    plan = study(tmp_path)
    plan["runs"][1]["output_path"] = plan["runs"][0]["output_path"]
    manifest_path = tmp_path / plan["manifest"]
    manifest = json.loads(manifest_path.read_text())
    manifest["runs"][1]["output_path"] = manifest["runs"][0]["output_path"]
    write(manifest_path, manifest)
    plan["manifest_sha256"] = digest(manifest_path)
    register(tmp_path, plan)
    assert "duplicate run output" in report(tmp_path)["errors"][0]


def test_path_outside_project_is_rejected(tmp_path):
    plan = study(tmp_path)
    plan["runs"][0]["output_path"] = "../unrelated_project/run"
    register(tmp_path, plan)
    assert "leaves the reproduction" in report(tmp_path)["errors"][0]


@pytest.fixture
def agentmixer_study(tmp_path, monkeypatch):
    from bta_repro import matrix_validation as module
    from bta_repro.learning_curves import seed_statistics
    monkeypatch.setattr(module, "seed_statistics", lambda values, **kw: seed_statistics(values))
    plan = study(tmp_path, seeds=range(2000, 2010), algorithms=["agentmixer"], games=["climbing"], steps=200000)
    plan["validation_rule"] = {"kind": "all_seeds_meet_threshold", "algorithms": ["agentmixer"],
                               "games": ["climbing"], "required_seed_count": 10}
    register(tmp_path, plan)
    return plan


def test_agentmixer_rule_checks_local_execution_and_retains_central_gap(tmp_path, agentmixer_study):
    result = report(tmp_path)
    assert result["errors"] == []
    assert result["validation_decision"] == "passed"
    assert result["paired_comparisons"] == []
    assert result["groups"][0]["aggregate"]["expected_return"]["mean"] == 11
    assert result["groups"][0]["aggregate"]["training_expected_return"]["mean"] < 0
    assert result["paper_reproduction"] == "not_established"


def test_agentmixer_missing_or_failed_seed_cannot_pass(tmp_path, agentmixer_study):
    out = tmp_path / agentmixer_study["runs"][0]["output_path"]
    summary = json.loads((out / "summary.json").read_text())
    summary.update(evaluation("climbing", uniform=True))
    write(out / "summary.json", summary)
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    rows[-1].update(evaluation("climbing", uniform=True))
    (out / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
    result = report(tmp_path)
    assert result["validation_decision"] == "failed"
    assert result["groups"][0]["threshold_successes"] == 9
    (out / "summary.json").unlink()
    assert report(tmp_path)["validation_decision"] == "incomplete"


@pytest.mark.parametrize("change", ["empty_algorithms", "absent_game", "wrong_count", "prior_seed", "no_rule"])
def test_invalid_or_nonindependent_agentmixer_rule_cannot_pass(tmp_path, agentmixer_study, change):
    plan = agentmixer_study
    if change == "empty_algorithms": plan["validation_rule"]["algorithms"] = []
    if change == "absent_game": plan["validation_rule"]["games"] = ["penalty"]
    if change == "wrong_count": plan["validation_rule"]["required_seed_count"] = 9
    if change == "prior_seed": plan["all_prior_recorded_training_seeds"].append(2000)
    if change == "no_rule": del plan["validation_rule"]
    register(tmp_path, plan)
    assert report(tmp_path)["validation_decision"] == "not_assessed"


@pytest.mark.parametrize("change", ["central_expected", "central_probability", "central_summary_mismatch", "extra_source"])
def test_agentmixer_central_or_source_corruption_is_detected(tmp_path, agentmixer_study, change):
    out = tmp_path / agentmixer_study["runs"][0]["output_path"]
    if change == "extra_source":
        (tmp_path / "runs/source/bta_repro/extra.py").write_text("# unrecorded\n")
    else:
        data = json.loads((out / "summary.json").read_text())
        if change == "central_expected": data["training_policy_expected_return"] = 11
        if change == "central_probability": data["training_joint_probabilities"][0] = 1
        if change == "central_summary_mismatch":
            data.update(training_joint_probabilities=evaluation("climbing")["execution_joint_probabilities"],
                        training_policy_expected_return=11, training_optimal_action_probability=1)
        write(out / "summary.json", data)
    result = report(tmp_path)
    assert result["errors"] or result["runs"][0]["errors"]
    assert result["validation_decision"] != "passed"


def test_seed_inventory_hash_and_declared_seeds_are_checked(tmp_path, agentmixer_study):
    plan = agentmixer_study
    plan["prior_run_inventory"] = "prior_seeds.json"
    write(tmp_path / "prior_seeds.json", {"training_seeds": [0, 1, 2]})
    plan["prior_run_inventory_sha256"] = digest(tmp_path / "prior_seeds.json")
    register(tmp_path, plan)
    assert report(tmp_path)["validation_decision"] == "passed"
    write(tmp_path / "prior_seeds.json", {"training_seeds": [0, 1, 2, 2000]})
    assert "prior seed inventory changed" in report(tmp_path)["errors"][0]
    plan["prior_run_inventory_sha256"] = digest(tmp_path / "prior_seeds.json")
    register(tmp_path, plan)
    assert "inventory disagrees" in report(tmp_path)["errors"][0]


def test_empty_study_does_not_pass_vacuously(tmp_path):
    plan = study(tmp_path)
    plan["training_seeds"] = []
    plan["runs"] = []
    register(tmp_path, plan)
    assert report(tmp_path)["errors"] == ["training_seeds must be nonempty and unique"]
