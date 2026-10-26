import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("matrix_ablation_report", Path(__file__).parents[1] / "scripts/report_matrix_ablation.py")
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)
SEEDS = [4000, 4001, 4002, 4003, 4004]
GAMES = ["climbing", "penalty"]


def arm(offset=0):
    return {"errors": [], "unused_seeds_at_freeze": True, "runs": [
        {"algorithm": "agentmixer", "game": game, "seed": seed, "errors": [], "eligible": True,
         "final": {"execution_policy_expected_return": 5 + offset + i, "greedy_return": 11},
         "normalized_expected_return_auc": 3 + offset + i, "threshold_success": i == 4}
        for game in GAMES for i, seed in enumerate(SEEDS)]}


def test_pairing_uses_seed_identity_and_requires_the_entire_study():
    learned, identity = arm(2), arm()
    identity["runs"].reverse()
    result = reporter.paired_results(learned, identity, GAMES, SEEDS)
    for group in result:
        assert group["contrasts"]["execution_policy_expected_return"]["learned_minus_identity"] == [2] * 5
        assert group["contrasts"]["normalized_expected_return_auc"]["statistics"]["mean"] == 2
        assert group["contrasts"]["greedy_return"]["statistics"]["mean"] == 0
    identity["runs"][0]["eligible"] = False
    assert reporter.paired_results(learned, identity, GAMES, SEEDS) == []


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "invalid", "used_seeds"])
def test_incomplete_declaration_or_invalid_run_cannot_produce_a_contrast(mutation):
    learned, identity = arm(2), arm()
    if mutation == "missing": identity["runs"].pop()
    if mutation == "duplicate": identity["runs"].append(identity["runs"][0])
    if mutation == "invalid": identity["runs"][0]["errors"] = ["score mismatch"]
    if mutation == "used_seeds": identity["unused_seeds_at_freeze"] = False
    with pytest.raises(ValueError):
        reporter.paired_results(learned, identity, GAMES, SEEDS)


@pytest.fixture
def frozen_pair(tmp_path, monkeypatch):
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    (tmp_path / "configs").mkdir()
    targets = {"configurations": {
        "common": {"values": {"environment_steps": 200_000}},
        "matrix": {"inherits": "common", "values": {}}},
        "benchmarks": [{"id": "agentmixer:climbing", "paper": "agentmixer", "suite": "matrix",
                        "task": "climbing", "configuration": "matrix", "paper_scope": "appendix_commented"}]}
    (tmp_path / "configs/paper_targets.json").write_text(json.dumps(targets))
    plan_path = experiments / "pair.json"
    plan = {"study_id": "pair", "frozen_utc": "2026-01-01T12:00:00+00:00",
            "games": GAMES, "training_seeds": SEEDS, "training_steps": 1_000_000,
            "source_sha256": {"train.py": "source"}, "arms": {}}
    declarations = {}
    for name, enabled in (("learned", True), ("identity", False)):
        path = experiments / (name + ".json")
        declaration = {"games": GAMES, "training_seeds": SEEDS, "training_steps": 1_000_000,
            "source_sha256": plan["source_sha256"], "runs": [
                {"game": game, "seed": seed, "output_path": f"results/{name}_{game}_{seed}",
                 "resolved_config": {"game": game, "seed": seed, "modifier_enabled": enabled, "epochs": 4}}
                for game in GAMES for seed in SEEDS]}
        declarations[name] = declaration
        path.write_text(json.dumps(declaration))
        plan["arms"][name] = {"plan": str(path.relative_to(tmp_path)), "sha256": reporter.digest(path)}
    plan_path.write_text(json.dumps(plan))

    def register():
        records = [{"study_plan": entry["plan"], "ablation_plan": "experiments/pair.json",
                    "ablation_plan_sha256": reporter.digest(plan_path)} for entry in plan["arms"].values()]
        (experiments / "jobs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))

    register()
    monkeypatch.setattr(reporter, "report_study", lambda p, root: arm(2 if p.stem == "learned" else 0))
    return tmp_path, plan_path, plan, declarations, register


def test_verified_pair_reports_both_games(frozen_pair):
    root, path, *_ = frozen_pair
    report = reporter.report_ablation(path, root)
    assert report["status"] == "complete" and not report["errors"]
    assert len(report["paired_comparisons"]) == 2
    contexts = report["canonical_budget_context"]["tasks"]
    assert contexts[0]["paper_steps"] == 200_000
    assert contexts[0]["budget_relation"] == "longer_than"
    assert contexts[0]["study_to_paper_budget_ratio"] == 5
    assert contexts[1]["scope"] == "additional_task" and contexts[1]["paper_steps"] is None


@pytest.mark.parametrize("steps,relation", [(100_000, "shorter_than"), (200_000, "matches"), (1_000_000, "longer_than")])
def test_budget_annotation_resolves_canonical_inheritance(frozen_pair, steps, relation):
    root, *_ = frozen_pair
    context = reporter.canonical_budget_context(root, GAMES, steps)
    assert context["tasks"][0]["budget_relation"] == relation
    assert context["tasks"][0]["paper_steps"] == 200_000
    assert context["source_sha256"] == reporter.digest(root / "configs/paper_targets.json")


def test_configuration_change_beyond_the_modifier_is_rejected(frozen_pair):
    root, path, plan, declarations, register = frozen_pair
    declarations["identity"]["runs"][0]["resolved_config"]["epochs"] = 5
    arm_path = root / plan["arms"]["identity"]["plan"]
    arm_path.write_text(json.dumps(declarations["identity"]))
    plan["arms"]["identity"]["sha256"] = reporter.digest(arm_path)
    path.write_text(json.dumps(plan))
    register()
    result = reporter.report_ablation(path, root)
    assert result["status"] == "invalid"
    assert "beyond modifier_enabled" in result["errors"][0]


@pytest.mark.parametrize("changed", ["parent", "arm", "preexisting_run"])
def test_post_freeze_changes_or_preexisting_results_are_rejected(frozen_pair, changed):
    root, path, plan, declarations, _ = frozen_pair
    if changed == "parent":
        plan["training_steps"] += 1
        path.write_text(json.dumps(plan))
    elif changed == "arm":
        (root / plan["arms"]["identity"]["plan"]).write_text("{}")
    else:
        output = root / declarations["identity"]["runs"][0]["output_path"]
        output.mkdir(parents=True)
        (output / "config.json").write_text(json.dumps({"created_utc": "2025-01-01T00:00:00+00:00"}))
    result = reporter.report_ablation(path, root)
    assert result["status"] == "invalid" and result["errors"]
    assert not result["paired_comparisons"]
