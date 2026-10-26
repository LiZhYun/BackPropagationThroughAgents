"""Plot validated curve reports; requires Matplotlib, not the simulator engines."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New PDF, PNG or SVG figure")
    parser.add_argument("--title", default="Development learning curves")
    args = parser.parse_args()
    if args.output.suffix.lower() not in (".pdf", ".png", ".svg"):
        parser.error("output must be a PDF, PNG or SVG")
    sidecar = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() or sidecar.exists():
        parser.error("figure or provenance sidecar already exists; choose a new output path")
    data = args.report.read_bytes()
    report = json.loads(data)
    groups = [group for group in report["groups"] if group["aggregate"] is not None]
    if not groups:
        parser.error("no complete validated group can be plotted")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        parser.error("plotting requires Matplotlib; the JSON reporter needs only the standard library")
    maximum_step = max(group["aggregate"]["curve"][-1]["steps"] for group in groups)
    divisor, unit = ((1e6, "millions") if maximum_step >= 1e6 else
                     (1e3, "thousands") if maximum_step >= 1e3 else (1., "steps"))
    labels = {"return_mean": "Evaluation return", "win_rate": "Evaluation win rate",
              "capture_rate": "Evaluation capture rate",
              "fixed_window_reward_per_step": "Fixed-window evaluation reward / decision"}
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555", "#E69F00", "#56B4E9")
    fig, axis = plt.subplots(figsize=(8, 5))
    has_band = False
    for index, group in enumerate(groups):
        curve = group["aggregate"]["curve"]
        x = [point["steps"] / divisor for point in curve]
        y = [point["mean"] for point in curve]
        color = colors[index % len(colors)]
        n = group["aggregate"]["final"]["seed_count"]
        axis.plot(x, y, color=color, linewidth=1.8, marker="o", markersize=3,
                  label=f"{group['variant'].replace('_', ' ')} (n={n})")
        if n > 1:
            sd = [point["sample_sd_across_training_seeds"] for point in curve]
            axis.fill_between(x, [mean - spread for mean, spread in zip(y, sd)],
                              [mean + spread for mean, spread in zip(y, sd)], color=color, alpha=.12)
            has_band = True
    axis.set(title=args.title, xlabel=f"Training environment steps ({unit})", ylabel=labels[report["metric"]])
    axis.grid(alpha=.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=8, loc="best")
    note = ("Lines: training-seed means. Shading: ±1 sample SD, where n ≥ 2."
            if has_band else "One training seed per variant; no uncertainty band.")
    omitted = [group["variant"] for group in report["groups"] if group["aggregate"] is None]
    if omitted:
        note += f" {len(omitted)} incomplete or invalid groups omitted; see report."
    fig.text(.5, .025, note, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, 1))
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    sidecar.write_text(json.dumps({"report": str(args.report),
        "report_sha256": hashlib.sha256(data).hexdigest(), "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matplotlib": matplotlib.__version__, "statistic": "mean", "band": "sample standard deviation across training seeds",
        "omitted_groups": omitted, "title": args.title}, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
