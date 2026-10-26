"""Audit a frozen matrix study and report every declared seed; standard library only."""

import ast
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from .learning_curves import seed_statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("study path leaves the reproduction repository")
    return path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name):
    require(type(value) is int and value >= 0, f"{name} must be a nonnegative integer")
    return value


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value), f"invalid {name}")
    return value


def wilson(successes, count):
    """Two-sided 95% interval for the observed threshold-success proportion."""
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    middle = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0., middle - half), min(1., middle + half)]


def check_evaluation(row, payoffs):
    """Recompute scores from all nine probabilities, not merely a logged argmax."""
    probabilities = row["execution_joint_probabilities"]
    require(len(probabilities) == 9, "expected nine joint-action probabilities")
    for p in probabilities:
        require(0 <= number(p, "probability") <= 1, "probability outside [0, 1]")
    require(math.isclose(sum(probabilities), 1, abs_tol=1e-5), "probabilities do not sum to one")
    optimum = max(payoffs)
    require(row["optimum_per_step"] == optimum, "payoff ceiling mismatch")
    expected = sum(p * reward for p, reward in zip(probabilities, payoffs))
    optimum_probability = sum(p for p, reward in zip(probabilities, payoffs) if reward == optimum)
    for key, value in (("execution_policy_expected_return", expected),
                       ("optimal_action_probability", optimum_probability)):
        require(math.isclose(number(row[key], key), value, rel_tol=1e-6, abs_tol=1e-4),
                f"{key} disagrees with joint probabilities")
    actions = row["greedy_actions"]
    require(len(actions) == 2 and all(type(a) is int and 0 <= a < 3 for a in actions),
            "invalid greedy actions")
    require(row["greedy_return"] == payoffs[3 * actions[0] + actions[1]], "greedy payoff mismatch")
    return {key: row[key] for key in ("execution_policy_expected_return", "greedy_return",
                                     "optimal_action_probability", "greedy_actions")}


def check_training_evaluation(row, payoffs):
    """Audit AgentMixer's central distribution separately from local execution."""
    check_evaluation({**row,
        "execution_joint_probabilities": row["training_joint_probabilities"],
        "execution_policy_expected_return": row["training_policy_expected_return"],
        "optimal_action_probability": row["training_optimal_action_probability"]}, payoffs)
    return {key: row[key] for key in ("training_policy_expected_return",
                                     "training_optimal_action_probability")}


def validation_rule(plan):
    """Explicit rules support new studies; preserve the original BPPO protocol."""
    rule = plan.get("validation_rule")
    if rule is None:
        if (plan["training_steps"] == 100000 and len(plan["training_seeds"]) == 10
                and set(plan["algorithms"]) == {"arppo", "bppo"}
                and set(plan["games"]) == {"climbing", "penalty"}):
            return {"algorithms": ["bppo"], "games": ["climbing", "penalty"],
                    "required_seed_count": 10}
        return None
    require(rule["kind"] == "all_seeds_meet_threshold", "unknown validation rule")
    for key in ("algorithms", "games"):
        values = rule[key]
        require(isinstance(values, list) and bool(values) and len(values) == len(set(values))
                and set(values).issubset(plan[key]), f"invalid validation {key}")
    count = integer(rule["required_seed_count"], "validation seed count")
    require(count >= 5 and count == len(plan["training_seeds"]), "validation seed count mismatch")
    return rule


def read_run(root, entry, plan, payoffs):
    record = {key: entry[key] for key in ("algorithm", "game", "seed", "output_path")}
    record.update(status="not_started", eligible=False, errors=[])
    try:
        config_path = checked_path(root, entry["config_path"])
        require(digest(config_path) == entry["config_sha256"], "declared configuration hash changed")
        declared = json.loads(config_path.read_text())
        resolved = entry["resolved_config"]
        require(all(resolved.get(k) == v for k, v in declared.items()), "resolved configuration mismatch")
        for key in ("algorithm", "game", "seed"):
            require(resolved[key] == entry[key], f"entry {key} mismatch")
        budget = integer(resolved["steps"], "budget")
        require(budget == plan["training_steps"] and budget > 0, "study budget mismatch")
        output = checked_path(root, entry["output_path"])
        if not output.exists():
            return record
        record["status"] = "unfinished"
        metadata_path = output / "config.json"
        if not metadata_path.exists():
            return record
        metadata = json.loads(metadata_path.read_text())
        require(metadata["config"] == resolved, "recorded configuration differs from frozen plan")
        require(metadata["source_sha256"] == plan["source_sha256"], "recorded source differs from frozen plan")
        require(all(metadata[key] == plan["environment"][key] for key in ("python", "torch", "numpy")),
                "software versions differ from frozen plan")
        require(datetime.fromisoformat(metadata["created_utc"]) >= datetime.fromisoformat(plan["frozen_utc"]),
                "run predates the frozen study")
        record["metadata_sha256"] = digest(metadata_path)
        summary_path = output / "summary.json"
        if not summary_path.exists():
            return record
        summary = json.loads(summary_path.read_text())
        record["status"] = summary.get("status", "unknown")
        record["summary_sha256"] = digest(summary_path)
        require(record["status"] == "completed", "run did not complete")
        require(integer(summary["steps"], "completed steps") == budget, "completed budget mismatch")
        final = check_evaluation(summary, payoffs)
        if entry["algorithm"] == "agentmixer":
            final.update(check_training_evaluation(summary, payoffs))
        metrics_path = output / "metrics.jsonl"
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        require(bool(rows), "empty learning curve")
        steps = [integer(row["steps"], "curve steps") for row in rows]
        require(steps[0] == 0 and steps[-1] == budget, "curve lacks initial or final budget endpoint")
        require(all(a < b for a, b in zip(steps, steps[1:])), "curve steps are not strictly increasing")
        batch, cadence = resolved["batch_size"], resolved["eval_every"]
        expected_steps = [0] + list(range(batch * cadence, budget, batch * cadence)) + [budget]
        require(steps == expected_steps, "curve does not match the predeclared evaluation grid")
        values = [check_evaluation(row, payoffs) for row in rows]
        if entry["algorithm"] == "agentmixer":
            for value, row in zip(values, rows):
                value.update(check_training_evaluation(row, payoffs))
        require(values[-1] == final, "summary disagrees with final curve evaluation")
        scores = [v["execution_policy_expected_return"] for v in values]
        area = sum((b - a) * (x + y) / 2 for a, b, x, y in zip(steps, steps[1:], scores, scores[1:]))
        threshold = plan["evaluation"]["threshold"][entry["game"]]
        crossing = None
        # Walk backwards: this requires every later logged score to pass too.
        for step, score in zip(reversed(steps), reversed(scores)):
            if score < threshold:
                break
            crossing = step
        success = final["greedy_return"] == max(payoffs) and scores[-1] >= threshold
        record.update(eligible=True, final=final, threshold_success=success,
                      normalized_expected_return_auc=area / budget,
                      first_observed_sustained_95pct_expected_return_step=crossing,
                      curve=[{"steps": step, "expected_return": score} for step, score in zip(steps, scores)],
                      metrics_sha256=digest(metrics_path))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        record["errors"].append(str(error))
    return record


def report_study(plan_path, root):
    root = Path(root).resolve()
    plan_path = checked_path(root, plan_path)
    plan = json.loads(plan_path.read_text())
    report = {"study_id": plan["study_id"], "scope": plan["scope"], "plan_sha256": digest(plan_path),
              "errors": [], "runs": [{**{k: r[k] for k in ("algorithm", "game", "seed", "output_path")},
                                      "status": "unverified", "eligible": False, "errors": []}
                                     for r in plan["runs"]], "groups": [], "paired_comparisons": [],
              "validation_decision": "not_assessed", "paper_reproduction": "not_established"}
    try:
        for key in ("algorithms", "games", "training_seeds"):
            values = plan[key]
            require(isinstance(values, list) and bool(values) and len(values) == len(set(values)),
                    f"{key} must be nonempty and unique")
        for seed in plan["training_seeds"]:
            require(integer(seed, "training seed") < 2**32, "training seed outside uint32")
        rule = validation_rule(plan)
        if "prior_run_inventory" in plan:
            inventory_path = checked_path(root, plan["prior_run_inventory"])
            require(digest(inventory_path) == plan["prior_run_inventory_sha256"], "prior seed inventory changed")
            inventory = json.loads(inventory_path.read_text())
            require(set(inventory["training_seeds"]) == set(plan["all_prior_recorded_training_seeds"]),
                    "prior seed inventory disagrees with plan")
        manifest_path = checked_path(root, plan["manifest"])
        require(digest(manifest_path) == plan["manifest_sha256"], "frozen manifest changed")
        manifest = json.loads(manifest_path.read_text())
        require(manifest["source_sha256"] == plan["source_sha256"], "manifest source hashes changed")
        require({p.name for p in (manifest_path.parent / "source/bta_repro").glob("*.py")}
                == set(plan["source_sha256"]), "frozen source file set changed")
        for name, expected in plan["source_sha256"].items():
            path = checked_path(root, manifest_path.parent / "source" / "bta_repro" / name)
            require(digest(path) == expected, f"frozen source changed: {name}")
        for name, key in (("run_array.py", "launcher_sha256"), ("submit.sbatch", "submission_script_sha256")):
            require(digest(manifest_path.parent / name) == plan[key], f"{name} changed")
        submissions = [json.loads(line) for line in (root / "experiments/jobs.jsonl").read_text().splitlines()]
        registered = [event for event in submissions if event.get("study_plan") == str(plan_path.relative_to(root))]
        require(bool(registered), "study has no recorded submission")
        require(all(event["study_plan_sha256"] == report["plan_sha256"] for event in registered),
                "plan differs from pre-run submission hash")
        expected_keys = {(a, g, s) for a in plan["algorithms"] for g in plan["games"] for s in plan["training_seeds"]}
        keys = [(r["algorithm"], r["game"], r["seed"]) for r in plan["runs"]]
        require(len(keys) == len(set(keys)) and set(keys) == expected_keys, "missing or duplicate planned run")
        manifest_runs = {(r["algorithm"], r["game"], r["seed"]): r for r in manifest["runs"]}
        require(len(manifest_runs) == len(manifest["runs"]) == len(keys) and set(manifest_runs) == expected_keys,
                "manifest run set differs from plan")
        outputs = set()
        for entry in plan["runs"]:
            counterpart = manifest_runs[(entry["algorithm"], entry["game"], entry["seed"])]
            for key in ("config_path", "output_path"):
                require(checked_path(root, entry[key]) == checked_path(root, counterpart[key]), "manifest path mismatch")
            output = checked_path(root, entry["output_path"])
            require(output not in outputs, "duplicate run output directory")
            outputs.add(output)
        # Read literal accepted payoffs without importing torch or executing frozen code.
        tree = ast.parse((manifest_path.parent / "source/bta_repro/envs.py").read_text())
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "PAYOFFS" for t in node.targets)]
        require(len(assignments) == 1, "cannot identify frozen payoff tables")
        tables = ast.literal_eval(assignments[0].value)
        for index, entry in enumerate(plan["runs"]):
            payoffs = [value for row in tables[entry["game"]] for value in row]
            report["runs"][index] = read_run(root, entry, plan, payoffs)
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as error:
        report["errors"].append(str(error))
        return report

    independent = set(plan["training_seeds"]).isdisjoint(plan["all_prior_recorded_training_seeds"])
    report["unused_seeds_at_freeze"] = independent
    groups = defaultdict(list)
    for run in report["runs"]:
        groups[(run["algorithm"], run["game"])].append(run)
    for (algorithm, game), runs in sorted(groups.items()):
        eligible = [r for r in runs if r["eligible"]]
        successes = sum(r["threshold_success"] for r in eligible)
        complete = len(eligible) == len(runs)
        group = {"algorithm": algorithm, "game": game, "declared_seeds": [r["seed"] for r in runs],
                 "eligible_seeds": [r["seed"] for r in eligible], "threshold_successes": successes,
                 "declared_seed_count": len(runs), "complete": complete, "aggregate": None,
                 "threshold_success_wilson_95pct": wilson(successes, len(runs)) if complete else None}
        if complete:
            group["aggregate"] = {
                "expected_return": seed_statistics([r["final"]["execution_policy_expected_return"] for r in runs], confidence_intervals=independent),
                "normalized_auc": seed_statistics([r["normalized_expected_return_auc"] for r in runs], confidence_intervals=independent)}
            if algorithm == "agentmixer":
                group["aggregate"]["training_expected_return"] = seed_statistics(
                    [r["final"]["training_policy_expected_return"] for r in runs],
                    confidence_intervals=independent)
        report["groups"].append(group)
    for game in plan["games"]:
        if not {"arppo", "bppo"}.issubset(plan["algorithms"]):
            continue
        ar, bp = groups[("arppo", game)], groups[("bppo", game)]
        if not all(r["eligible"] for r in ar + bp):
            report["paired_comparisons"].append({"game": game, "eligible": False})
            continue
        baseline = {r["seed"]: r for r in ar}
        differences = [r["final"]["execution_policy_expected_return"] - baseline[r["seed"]]["final"]["execution_policy_expected_return"] for r in bp]
        ar_mean = sum(r["final"]["execution_policy_expected_return"] for r in ar) / len(ar)
        delta = sum(differences) / len(differences)
        report["paired_comparisons"].append({"game": game, "eligible": True,
            "seeds": [r["seed"] for r in bp], "expected_return_differences": differences,
            "difference_statistics": seed_statistics(differences, confidence_intervals=independent),
            "relative_mean_change_percent": 100 * delta / abs(ar_mean) if ar_mean else None})
    if independent and rule is not None:
        selected = [r for r in report["runs"] if r["algorithm"] in rule["algorithms"]
                    and r["game"] in rule["games"]]
        if all(r["eligible"] for r in report["runs"]):
            report["validation_decision"] = "passed" if selected and all(r["threshold_success"] for r in selected) else "failed"
        else:
            report["validation_decision"] = "incomplete"
    return report
