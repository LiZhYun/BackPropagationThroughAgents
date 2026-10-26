"""Plot the completed matrix validation records, retaining every declared seed."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_report(path, count):
    report = json.loads(path.read_text())
    if report["errors"] or len(report["runs"]) != count or not all(
            r["eligible"] and not r["errors"] for r in report["runs"]):
        raise ValueError("figures require every declared run to have passed its integrity audit")
    if not report["unused_seeds_at_freeze"]:
        raise ValueError("these validation figures require unused training seeds")
    for run in report["runs"]:
        if run["curve"][-1]["expected_return"] != run["final"]["execution_policy_expected_return"]:
            raise ValueError("curve endpoint disagrees with the final score")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for PNG, PDF and provenance")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory must be new")
    paths = {"bppo_ar": ROOT / "experiments/matrix_confirmation1m_results_20261026.json",
             "agentmixer": ROOT / "experiments/matrix_agentmixer_confirmation200k_results_20261026.json"}
    reports = {name: read_report(path, 20 if name == "bppo_ar" else 10) for name, path in paths.items()}
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = []
    verified_groups = []

    def curve(axis, runs, color, label, budget, expected_seeds, individual_seeds=False):
        runs = sorted(runs, key=lambda r: r["seed"])
        assert [r["seed"] for r in runs] == expected_seeds
        grid = [point["steps"] for point in runs[0]["curve"]]
        assert grid[0] == 0 and grid[-1] == budget and all(a < b for a, b in zip(grid, grid[1:]))
        assert all([p["steps"] for p in r["curve"]] == grid for r in runs)
        values = np.array([[p["expected_return"] for p in r["curve"]] for r in runs])
        assert np.isfinite(values).all()
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        scale = 1e6 if budget >= 1e6 else 1e3
        x = np.asarray(grid) / scale
        if individual_seeds:
            for values_for_seed in values:
                axis.plot(x, values_for_seed, color=color, linewidth=.9, alpha=.25)
        axis.plot(x, mean, color=color, linewidth=2, label=label)
        if not individual_seeds:
            axis.fill_between(x, mean - sd, mean + sd, color=color, alpha=.14, linewidth=0)
        axis.set_xlim(0, budget / scale)
        axis.grid(alpha=.18)
        verified_groups.append({"label": label, "seeds": expected_seeds, "points": len(grid),
                                "budget": budget, "final_mean": float(mean[-1]), "final_sample_sd": float(sd[-1]),
                                "variability_display": "individual seed curves" if individual_seeds else "mean plus/minus sample SD"})
        return float(mean[-1])

    def save(fig, name):
        for suffix in ("png", "pdf"):
            destination = args.output_dir / (name + "." + suffix)
            options = {"metadata": {"Creator": "bta-repro", "CreationDate": None, "ModDate": None}} if suffix == "pdf" else {}
            fig.savefig(destination, dpi=180, **options)
            outputs.append(destination)
        plt.close(fig)

    report = reports["bppo_ar"]
    assert report["validation_decision"] == "passed"
    assert {(r["algorithm"], r["game"], r["seed"]) for r in report["runs"]} == {
        (a, g, s) for a in ("bppo", "arppo") for g in ("climbing", "penalty") for s in range(3000, 3005)}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for axis, game, optimum in zip(axes, ("climbing", "penalty"), (11, 10)):
        for method, color, label in (("bppo", "#0072B2", "BPPO"), ("arppo", "#D55E00", "AR-PPO")):
            runs = [r for r in report["runs"] if r["algorithm"] == method and r["game"] == game]
            mean = curve(axis, runs, color, label, 1_000_000, list(range(3000, 3005)))
            group = next(g for g in report["groups"] if g["algorithm"] == method and g["game"] == game)
            assert math.isclose(mean, group["aggregate"]["expected_return"]["mean"], abs_tol=1e-12)
        axis.axhline(optimum, color="#666666", linestyle=":", linewidth=1, label="Payoff ceiling")
        axis.set(title=game.title(), xlabel="Training decisions (millions)", ylabel="Execution-policy expected reward")
        axis.legend(loc="lower right", fontsize=9)
    fig.suptitle("BPPO and AR-PPO · 1M decisions · five independent training seeds", fontsize=13)
    fig.text(.5, .025, "Lines: seed means; shading: ±1 sample SD. Both methods pass 5/5 on each game; superiority is not established.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .94))
    save(fig, "bppo_ar_1m")

    report = reports["agentmixer"]
    assert report["validation_decision"] == "failed"
    runs = sorted(report["runs"], key=lambda r: r["seed"])
    assert all(r["algorithm"] == "agentmixer" and r["game"] == "climbing" for r in runs)
    assert sum(r["threshold_success"] for r in runs) == 8
    assert all(r["final"]["greedy_return"] == 11 for r in runs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    mean = curve(axes[0], runs, "#0072B2", "Local execution mean", 200_000, list(range(2000, 2010)), individual_seeds=True)
    assert math.isclose(mean, report["groups"][0]["aggregate"]["expected_return"]["mean"], abs_tol=1e-12)
    axes[0].axhline(10.45, color="#555555", linestyle="--", linewidth=1, label="Declared threshold: 10.45")
    axes[0].set(title="Local expected-return learning curve", xlabel="Training decisions (thousands)", ylabel="Expected reward")
    axes[0].legend(loc="lower right", fontsize=8)
    x = np.arange(len(runs))
    local = np.array([r["final"]["execution_policy_expected_return"] for r in runs])
    central = np.array([r["final"]["training_policy_expected_return"] for r in runs])
    assert np.isfinite(local).all() and np.isfinite(central).all()
    for i, a, b in zip(x, local, central):
        axes[1].plot([i - .12, i + .12], [a, b], color="#999999", linewidth=1, zorder=1)
    axes[1].scatter(x - .12, local, color="#0072B2", s=35, label="Local execution", zorder=3)
    axes[1].scatter(x + .12, central, color="#D55E00", marker="s", s=30, label="Central training policy", zorder=3)
    axes[1].axhline(10.45, color="#555555", linestyle="--", linewidth=1)
    axes[1].set(xticks=x, xticklabels=[str(r["seed"]) for r in runs], ylim=(min(local.min(), central.min()) - 1.3, 13.5),
                title="Every seed at the final 200k endpoint", xlabel="Training seed", ylabel="Expected reward")
    axes[1].tick_params(axis="x", labelrotation=45, labelsize=8)
    axes[1].grid(axis="y", alpha=.18)
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].text(.5, .92, "Greedy payoff = 11 for all 10 seeds", transform=axes[1].transAxes, ha="center", fontsize=9)
    fig.suptitle("AgentMixer Climbing · 200k decisions · local-return criterion fails (8/10)", fontsize=13)
    fig.text(.5, .025, "All ten seeds retained. Left: thin lines show individual seeds; bold line shows their mean. Central scores do not replace local execution.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .94))
    save(fig, "agentmixer_climbing_200k")

    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    provenance = {"reports": {name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for name, path in paths.items()},
        "script": "scripts/plot_matrix_evidence.py", "script_sha256": digest(Path(__file__)),
        "matplotlib": matplotlib.__version__, "numpy": np.__version__, "groups": verified_groups,
        "aggregation": "Arithmetic mean and sample SD across whole training seeds; identical logged step grids; no smoothing or interpolation beyond straight plotted segments.",
        "agentmixer_endpoint_seeds": [r["seed"] for r in runs], "omitted_runs": [],
        "outputs": {p.name: digest(p) for p in outputs}}
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
