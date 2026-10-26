"""Audit a frozen AgentMixer/identity pair and report paired seed differences."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bta_repro.learning_curves import seed_statistics
from bta_repro.matrix_validation import checked_path, digest, report_study, require
from bta_repro.benchmark_coverage import _settings


def canonical_budget_context(root, games, training_steps):
    """Derive paper scope from canonical targets, independently of study prose."""
    path = root / "configs/paper_targets.json"
    targets = json.loads(path.read_text())
    comparisons = []
    for game in games:
        matches = [b for b in targets["benchmarks"]
                   if b["paper"] == "agentmixer" and b["suite"] == "matrix" and b["task"] == game]
        require(len(matches) <= 1, "ambiguous canonical AgentMixer matrix target")
        item = {"game": game, "study_steps": training_steps}
        if not matches:
            item.update(scope="additional_task", benchmark=None, paper_steps=None)
        else:
            benchmark = matches[0]
            budget = _settings(targets["configurations"], benchmark["configuration"])["environment_steps"]
            require(type(budget) is int and budget > 0, "invalid canonical matrix budget")
            relation = "matches" if training_steps == budget else "longer_than" if training_steps > budget else "shorter_than"
            item.update(scope=benchmark["paper_scope"], benchmark=benchmark["id"], paper_steps=budget,
                        budget_relation=relation, study_to_paper_budget_ratio=training_steps / budget,
                        configuration=benchmark["configuration"])
        comparisons.append(item)
    return {"source": "configs/paper_targets.json", "source_sha256": digest(path), "tasks": comparisons,
            "qualification": "Budget matching alone does not establish protocol or performance equivalence."}


def paired_results(learned, identity, games, seeds):
    """Require every planned result before reporting either game's contrasts."""
    expected = {(game, seed) for game in games for seed in seeds}
    arms = []
    for report in (learned, identity):
        require(not report["errors"], "an arm failed its integrity audit")
        require(report["unused_seeds_at_freeze"], "paired study requires unused seeds")
        runs = report["runs"]
        indexed = {(r["game"], r["seed"]): r for r in runs}
        require(len(indexed) == len(runs) and set(indexed) == expected, "arm run set changed")
        require(all(r["algorithm"] == "agentmixer" for r in runs), "wrong ablation algorithm")
        require(not any(r["errors"] for r in runs), "an individual run failed its integrity audit")
        arms.append(indexed)
    if not all(r["eligible"] for arm in arms for r in arm.values()):
        return []
    comparisons = []
    for game in games:
        contrasts = {}
        for metric in ("execution_policy_expected_return", "greedy_return", "normalized_expected_return_auc"):
            values = []
            for seed in seeds:
                a, b = (arm[(game, seed)] for arm in arms)
                values.append(a[metric] - b[metric] if metric.endswith("auc")
                              else a["final"][metric] - b["final"][metric])
            contrasts[metric] = {"learned_minus_identity": values,
                                 "statistics": seed_statistics(values, confidence_intervals=True)}
        comparisons.append({"game": game, "paired_seeds": seeds, "contrasts": contrasts,
            "learned_threshold_successes": sum(arms[0][(game,s)]["threshold_success"] for s in seeds),
            "identity_threshold_successes": sum(arms[1][(game,s)]["threshold_success"] for s in seeds)})
    return comparisons


def report_ablation(plan_path, root=ROOT):
    root = Path(root).resolve()
    path = checked_path(root, plan_path)
    plan = json.loads(path.read_text())
    report = {"study_id": plan["study_id"], "plan_sha256": digest(path), "errors": [],
              "status": "incomplete", "arms": {}, "paired_comparisons": [],
              "paper_reproduction": "not_established", "reporter_sha256": digest(Path(__file__))}
    try:
        games, seeds = plan["games"], plan["training_seeds"]
        require(games and len(games) == len(set(games)), "empty or duplicate games")
        require(len(seeds) >= 5 and len(seeds) == len(set(seeds)), "insufficient or duplicate seeds")
        report["canonical_budget_context"] = canonical_budget_context(root, games, plan["training_steps"])
        submissions = [json.loads(line) for line in (root / "experiments/jobs.jsonl").read_text().splitlines()]
        registered = [event for event in submissions if event.get("ablation_plan") == str(path.relative_to(root))]
        require(registered and all(e["ablation_plan_sha256"] == report["plan_sha256"] for e in registered),
                "paired plan differs from its pre-run submission record")
        declarations = {}
        for arm, enabled in (("learned", True), ("identity", False)):
            entry = plan["arms"][arm]
            arm_path = checked_path(root, entry["plan"])
            require(digest(arm_path) == entry["sha256"], "arm plan changed")
            require(any(e.get("study_plan") == str(arm_path.relative_to(root)) for e in registered),
                    "arm lacks a paired-study submission record")
            declaration = json.loads(arm_path.read_text())
            require(declaration["games"] == games and declaration["training_seeds"] == seeds,
                    "arm games or seeds differ")
            require(declaration["training_steps"] == plan["training_steps"], "arm budget differs")
            require(declaration["source_sha256"] == plan["source_sha256"], "arm sources differ")
            arm_report = report_study(arm_path, root)
            require(not arm_report["errors"], f"{arm} integrity errors: {arm_report['errors']}")
            report["arms"][arm] = arm_report
            declarations[arm] = {}
            for run in declaration["runs"]:
                config = dict(run["resolved_config"])
                require(config.pop("modifier_enabled") is enabled, "wrong modifier setting")
                declarations[arm][(run["game"], run["seed"])] = config
                metadata = checked_path(root, run["output_path"]) / "config.json"
                if metadata.exists():
                    created = json.loads(metadata.read_text())["created_utc"]
                    require(datetime.fromisoformat(created) >= datetime.fromisoformat(plan["frozen_utc"]),
                            "run predates the paired plan")
        require(declarations["learned"] == declarations["identity"], "paired configs differ beyond modifier_enabled")
        report["paired_comparisons"] = paired_results(report["arms"]["learned"], report["arms"]["identity"], games, seeds)
        if report["paired_comparisons"]:
            report["status"] = "complete"
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["errors"].append(str(error))
        report["status"] = "invalid"
    report["interpretation"] = ("Paired differences use whole training seeds and the exact final budget. "
        "Intervals are pointwise descriptive bootstrap intervals, not family-wise or cross-paper superiority claims. "
        "Missing or invalid runs prevent every paired aggregate; central scores remain separate in each arm. "
        "Paper task and budget coverage comes from canonical_budget_context, not study-plan prose.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new file")
    report = report_ablation(args.plan)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report.get(k) for k in ("status", "errors", "canonical_budget_context", "paired_comparisons")}, indent=2))


if __name__ == "__main__":
    main()
