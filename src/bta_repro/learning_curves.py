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
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be a JSON object")
    if metric == "fixed_window_reward_per_step" and not isinstance(evaluation.get("fixed_window"), dict):
        raise ValueError("fixed_window must be a JSON object")
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


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _fixed_window_protocol(evaluation, config):
    """Recompute Prey reductions from all recorded windows, including partials.

    This checks window bookkeeping and the recorded seed schedule. Without raw
    transition traces it cannot independently reconstruct environment rewards;
    NumPy seed derivation and policy execution require checkpoint replay.
    """
    window = evaluation["fixed_window"]
    count = _positive_integer(window["window_count"], "window_count")
    length = _positive_integer(window["steps_per_window"], "steps_per_window")
    denominator = _positive_integer(window["environment_steps"], "window environment_steps")
    if denominator != count * length:
        raise ValueError("fixed-window denominator disagrees with count × length")
    if (config["environment"] != "prey"
            or _positive_integer(config["eval_windows"], "configured eval_windows") != count
            or _positive_integer(config["eval_window_steps"], "configured eval_window_steps") != length):
        raise ValueError("fixed-window protocol disagrees with the frozen training configuration")
    seed = _nonnegative_integer(window["seed"], "fixed-window evaluation seed")
    if (seed >= 2**32 or seed != _nonnegative_integer(config["eval_seed"], "configured evaluation seed")
            or seed != _nonnegative_integer(evaluation["seed"], "episode evaluation seed")):
        raise ValueError("fixed-window evaluation seed disagrees with configuration or episode evaluation")
    execution = {"agentmixer": "decentralized", "bppo": "autoregressive",
                 "arppo": "autoregressive", "mappo": "independent"}[config["algorithm"]]
    if evaluation["execution"] != execution:
        raise ValueError("fixed-window execution disagrees with the algorithm")
    if window["deterministic"] is not True or evaluation["deterministic"] is not True:
        raise ValueError("fixed-window comparison requires deterministic execution")
    if window["scope"] != "all_steps_in_fixed_windows_including_unfinished_episodes":
        raise ValueError("fixed-window scope must include unfinished episode rewards")
    if window["seed_scheme"] != "numpy_seed_sequence_bta_win_v1":
        raise ValueError("unsupported fixed-window seed scheme")
    records = window["windows"]
    if not isinstance(records, list) or len(records) != count:
        raise ValueError("fixed-window raw record count disagrees with window_count")
    seeds, rewards = [], []
    for index, record in enumerate(records):
        if _nonnegative_integer(record["window"], "window index") != index:
            raise ValueError("fixed-window indices must be consecutive and ordered")
        child_seed = _nonnegative_integer(record["seed"], "window seed")
        if child_seed >= 2**32 or child_seed in seeds:
            raise ValueError("window seeds must be distinct uint32 values")
        seeds.append(child_seed)
        if _positive_integer(record["steps"], "window steps") != length:
            raise ValueError("raw window length disagrees with steps_per_window")
        reward = _finite_number(record["reward_sum"], "window reward_sum")
        rewards.append(reward)
        partial_steps = _nonnegative_integer(record["unfinished_episode_steps"], "unfinished episode steps")
        partial_reward = _finite_number(record["unfinished_episode_return"], "unfinished episode return")
        completed = _nonnegative_integer(record["completed_episodes"], "completed window episodes")
        captures = _nonnegative_integer(record["captures"], "window captures")
        timeouts = _nonnegative_integer(record["timeouts"], "window timeouts")
        if partial_steps > length or completed > length - partial_steps:
            raise ValueError("window episode counts exceed the recorded decisions")
        if captures + timeouts != completed:
            raise ValueError("Prey window captures and timeouts must account for every completed episode")
        if partial_steps == 0 and partial_reward != 0:
            raise ValueError("an empty unfinished episode must have zero reward")
        if completed == 0 and (partial_steps != length or not math.isclose(
                partial_reward, reward, rel_tol=1e-10, abs_tol=1e-9)):
            raise ValueError("a window with no completed episode must retain all steps and reward as unfinished")
    total = math.fsum(rewards)
    reported_total = _finite_number(window["reward_sum"], "total window reward_sum")
    reported_rate = _finite_number(window["reward_per_environment_step"], "window reward rate")
    if not math.isclose(total, reported_total, rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError("fixed-window reward_sum disagrees with raw window totals")
    if not math.isclose(total / denominator, reported_rate, rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError("fixed-window reward rate disagrees with raw reward divided by all decisions")
    protocol = {key: window[key] for key in (
        "window_count", "steps_per_window", "scope", "seed_scheme", "seed", "deterministic")}
    return {**protocol, "execution": execution, "window_seeds": seeds}


def _evaluation_protocol(evaluation, metric, config):
    """Preserve the denominator, seed stream and execution convention."""
    if metric == "fixed_window_reward_per_step":
        return _fixed_window_protocol(evaluation, config)
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


def read_curve_run(entry, source_hashes, metric):
    """Read one declared run, retaining incomplete artifacts and validation errors."""
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
        first_evaluation = last_evaluation = None
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
            current_protocol = _evaluation_protocol(evaluation, metric, config)
            if protocol is not None and current_protocol != protocol:
                raise ValueError("evaluation denominator, seed or execution convention changed within run")
            protocol = current_protocol
            result["points"].append({"steps": step, "value": value})
            stage = record["stage"]
            if first_evaluation is None:
                first_evaluation = evaluation
            last_evaluation = evaluation
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
        if _evaluation_protocol(summary["final_evaluation"], metric, config) != protocol:
            raise ValueError("final evaluation protocol disagrees between summary and curve")
        if metric == "fixed_window_reward_per_step":
            for name, recorded in (("initial_evaluation", first_evaluation), ("final_evaluation", last_evaluation)):
                if (_evaluation_protocol(summary[name], metric, config) != protocol
                        or summary[name]["fixed_window"] != recorded["fixed_window"]):
                    raise ValueError(f"{name} raw windows disagree between summary and curve")
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
        run = read_curve_run(entry, manifest["source_sha256"], metric)
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
