"""Seed-level learning curves from frozen simulator manifests (standard library only).

Curve area uses the actual logged training-step coordinates. No run is selected
by its best checkpoint, and no missing endpoint or failed seed is filled in.
"""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


METRICS = ("return_mean", "win_rate", "capture_rate", "fixed_window_reward_per_step")
BOOTSTRAP_SEED = 1729
BOOTSTRAP_SAMPLES = 10_000


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _metric(evaluation, metric):
    value = (evaluation.get("fixed_window", {}).get("reward_per_environment_step")
             if metric == "fixed_window_reward_per_step" else evaluation.get(metric))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{metric} is unavailable or nonfinite")
    if metric in ("win_rate", "capture_rate") and not 0 <= value <= 1:
        raise ValueError(f"{metric} must be in [0, 1]")
    return float(value)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _evaluation_protocol(evaluation, metric):
    """Preserve the denominator, seed stream and execution convention."""
    if metric == "fixed_window_reward_per_step":
        window = evaluation["fixed_window"]
        count = _positive_integer(window["window_count"], "window_count")
        length = _positive_integer(window["steps_per_window"], "steps_per_window")
        if window["environment_steps"] != count * length:
            raise ValueError("fixed-window denominator disagrees with count × length")
        return {key: window[key] for key in (
            "window_count", "steps_per_window", "scope", "seed_scheme", "seed", "deterministic")}
    count = _positive_integer(evaluation["count"], "evaluation episode count")
    if metric in ("win_rate", "capture_rate"):
        key = "win_records" if metric == "win_rate" else "capture_records"
        if evaluation[key] != count:
            raise ValueError(f"{metric} is not recorded for every evaluation episode")
    return {key: evaluation[key] for key in ("count", "seed", "execution", "deterministic")}


def _percentile(sorted_values, fraction):
    index = fraction * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    return sorted_values[lower] + (index - lower) * (sorted_values[upper] - sorted_values[lower])


def seed_statistics(values, *, confidence_intervals=False):
    result = {"seed_count": len(values), "mean": statistics.mean(values),
              "median": statistics.median(values),
              "sample_sd_across_training_seeds": statistics.stdev(values) if len(values) > 1 else None}
    if confidence_intervals:
        result["bootstrap_95pct_mean"] = None
        result["bootstrap_95pct_median"] = None
        # The project requires at least five independent final training seeds.
        # A CI alone does not establish that a supplied manifest is independent.
        if len(values) >= 5:
            rng = random.Random(BOOTSTRAP_SEED)
            means, medians = [], []
            for _ in range(BOOTSTRAP_SAMPLES):
                sample = rng.choices(values, k=len(values))
                means.append(statistics.mean(sample))
                medians.append(statistics.median(sample))
            for name, samples in (("mean", means), ("median", medians)):
                samples.sort()
                result[f"bootstrap_95pct_{name}"] = [_percentile(samples, p) for p in (.025, .975)]
    return result


def _read_run(entry, source_hashes, metric):
    output = Path(entry["output_path"])
    result = {key: entry[key] for key in ("variant", "algorithm", "seed")}
    result.update(output_path=str(output), status="unfinished" if output.exists() else "not_started",
                  eligible=False, errors=[], points=[], full_budget_auc=None)
    try:
        declared = json.loads(Path(entry["config_path"]).read_text())
        budget = result["budget"] = _positive_integer(declared["steps"], "training budget")
        if declared["seed"] != entry["seed"] or declared["algorithm"] != entry["algorithm"]:
            raise ValueError("declared config disagrees with manifest seed or algorithm")
        summary_path = output / "summary.json"
        summary = None
        if summary_path.exists():
            data = summary_path.read_bytes()
            result["summary_sha256"] = _hash(data)
            summary = json.loads(data)
            result["status"] = summary.get("status", "unknown")
        if not output.exists():
            return result
        metadata = json.loads((output / "config.json").read_text())
        config = metadata["config"]
        if any(config.get(key) != value for key, value in declared.items()):
            raise ValueError("recorded config disagrees with frozen run config")
        if metadata["source_sha256"] != source_hashes:
            raise ValueError("recorded source hashes disagree with frozen manifest")
        result["environment_versions"] = metadata["environment_versions"]
        config_without_seed = {key: value for key, value in config.items() if key != "seed"}
        result["configuration_sha256_without_seed"] = _hash(_canonical(config_without_seed).encode())
        data = (output / "metrics.jsonl").read_bytes()
        result["metrics_snapshot_sha256"] = _hash(data)
        result["ignored_unfinished_tail"] = False
        lines = data.splitlines(keepends=True)
        protocol = None
        stage = None
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1 and not line.endswith(b"\n") and result["status"] != "completed":
                    result["ignored_unfinished_tail"] = True
                    break
                raise ValueError(f"malformed metrics JSON at line {index + 1}") from None
            if record.get("event") != "evaluation":
                continue
            step = record["steps"]
            if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= budget:
                raise ValueError("evaluation step is outside the declared training budget")
            if result["points"] and step <= result["points"][-1]["steps"]:
                raise ValueError("evaluation steps must strictly increase; duplicate checkpoints are not averaged")
            evaluation = record["evaluation"]
            value = _metric(evaluation, metric)
            current_protocol = _evaluation_protocol(evaluation, metric)
            if protocol is not None and current_protocol != protocol:
                raise ValueError("evaluation denominator, seed or execution convention changed within run")
            protocol = current_protocol
            result["points"].append({"steps": step, "value": value})
            stage = record["stage"]
        result["evaluation_protocol"] = protocol
        if result["status"] != "completed":
            return result
        if summary["steps"] != budget:
            raise ValueError("completed run did not reach its declared training budget")
        points = result["points"]
        if len(points) < 2 or points[0]["steps"] != 0 or points[-1]["steps"] != budget or stage != "final":
            raise ValueError("full-budget curve needs initial step 0 and the final budget endpoint")
        if _metric(summary["final_evaluation"], metric) != points[-1]["value"]:
            raise ValueError("final metric disagrees between summary and curve")
        if _evaluation_protocol(summary["final_evaluation"], metric) != protocol:
            raise ValueError("final evaluation protocol disagrees between summary and curve")
        area = math.fsum((right["steps"] - left["steps"]) * (left["value"] + right["value"]) / 2
                         for left, right in zip(points, points[1:]))
        if not math.isfinite(area):
            raise ValueError("curve area overflowed; no full-budget aggregate is valid")
        result.update(eligible=True, full_budget_auc={"area": area, "normalized_by_training_steps": area / budget})
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        result["errors"].append(str(error))
    return result


def summarize_curves(manifest_path, metric="return_mean"):
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    data = Path(manifest_path).read_bytes()
    manifest = json.loads(data)
    seen = set()
    groups = defaultdict(list)
    runs = []
    for entry in manifest["runs"]:
        key = (entry["variant"], entry["seed"])
        if key in seen:
            raise ValueError(f"duplicate variant/training seed in manifest: {key}")
        seen.add(key)
        run = _read_run(entry, manifest["source_sha256"], metric)
        runs.append(run)
        groups[entry["variant"]].append(run)
    comparisons = []
    for variant, records in groups.items():
        eligible = [run for run in records if run["eligible"]]
        reasons = []
        if len(eligible) != len(records):
            reasons.append("some declared seeds lack a valid completed full-budget curve")
        for field in ("configuration_sha256_without_seed", "evaluation_protocol", "environment_versions"):
            if len({_canonical(run.get(field)) for run in eligible}) > 1:
                reasons.append(f"training seeds differ in {field}")
        grids = {tuple(point["steps"] for point in run["points"]) for run in eligible}
        if len(grids) > 1:
            reasons.append("evaluation grids differ; no implicit interpolation or common-prefix selection")
        group = {"variant": variant, "declared_seeds": [r["seed"] for r in records],
                 "eligible_seeds": [r["seed"] for r in eligible], "aggregation_errors": reasons,
                 "aggregate": None}
        if not reasons:
            group["aggregate"] = {
                "final": seed_statistics([r["points"][-1]["value"] for r in records], confidence_intervals=True),
                "normalized_auc": seed_statistics([r["full_budget_auc"]["normalized_by_training_steps"]
                                                   for r in records], confidence_intervals=True),
                "curve": [{"steps": points[0]["steps"], **seed_statistics([p["value"] for p in points])}
                          for points in zip(*(r["points"] for r in records))],
            }
        comparisons.append(group)
    return {
        "scope": manifest["scope"], "metric": metric, "manifest_sha256": _hash(data),
        "reporter_sha256": _hash(Path(__file__).read_bytes()),
        "area_rule": "trapezoidal integral of logged evaluation means over actual training steps, divided by full budget",
        "uncertainty": {"unit": "training-seed mean; evaluation episodes are not independent training seeds",
                        "interval": "percentile bootstrap, linear percentile interpolation; omitted below five seeds",
                        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_samples": BOOTSTRAP_SAMPLES},
        "interpretation": "All declared seeds must complete at the same settings and evaluation grid before aggregation. "
                          "Intervals do not certify independent final seeds, protocol equivalence or paper superiority.",
        "runs": runs, "groups": comparisons,
    }
