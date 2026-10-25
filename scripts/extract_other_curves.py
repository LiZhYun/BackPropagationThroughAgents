#!/usr/bin/env python3
"""Extract non-MuJoCo vector curves with explicit source and reward conventions.

Uses PyMuPDF==1.26.5 and the audited geometry helpers in extract_curve_targets.py.
Does not modify canonical targets, reference assets, or MuJoCo outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import zipfile

import pymupdf as fitz

import extract_curve_targets as geometry


SMAC = tuple((f"10gen_{race} {size}", f"10gen_{race}_{size}")
             for race in ("protoss", "zerg", "terran") for size in ("5v5", "10v10", "10v11"))
FOOTBALL = (("Football 3 vs 1 academy", "academy_3_vs_1_with_keeper"),
            ("Football counterattack academy", "academy_counterattack_hard"),
            ("Football corner academy", "academy_corner"))
MATRIX = (("Climbing Game", "climbing"), ("Penalty Game", "penalty"))


def spec(name, pdf, panels, *, paper="bpta", label="BPPO", metric="episode_return",
         split="not_stated", primary=True, scope="active_main", quantum=.001,
         signed_y=None, source_lines=None):
    return dict(name=name, pdf=pdf, panels=panels, paper=paper, label=label, metric=metric,
                split=split, primary=primary, scope=scope, quantum=quantum,
                signed_y=signed_y or {}, source_lines=source_lines)


# Some negative signs are drawn as glyph paths, absent from extracted text.
# These signed tick sequences are transcribed from the reviewed original render,
# ordered from top to bottom. Magnitudes and positions remain machine-checked.
FIGURES = (
    spec("bppo_matrix", "references/bpta/Matrix.pdf", MATRIX, quantum=10,
         signed_y={"penalty": (2000, 0, -2000, -4000)}, source_lines=[318, 326]),
    spec("bppo_matrix_projection", "references/bpta/Matrix_proj.pdf", MATRIX,
         label="ARMAPPO w/ PROJ", primary=False, scope="active_appendix_ablation", quantum=10,
         signed_y={"climbing": (2000, 1500, 1000, 500, 0, -500),
                   "penalty": (2000, 1000, 0, -1000, -2000, -3000, -4000)}, source_lines=[711, 717]),
    spec("bppo_smac_test", "references/bpta/SMAC.pdf", SMAC, metric="test_win_rate",
         split="test", scope="active_appendix", source_lines=[675, 681]),
    spec("bppo_smac_train", "references/bpta/SMAC_incre.pdf", SMAC, metric="train_win_rate",
         split="train", primary=False, source_lines=[268, 272]),
    spec("bppo_football_test", "references/bpta/Football.pdf", FOOTBALL, metric="test_win_rate",
         split="test", scope="active_appendix", source_lines=[683, 689]),
    spec("bppo_football_train", "references/bpta/Football_train.pdf", FOOTBALL, metric="train_win_rate",
         split="train", primary=False, source_lines=[331, 336]),
    spec("bppo_football_5v5", "references/bpta/Football_5v5.pdf", (("Football 5 vs 5", "5_vs_5"),),
         quantum=.01, signed_y={"5_vs_5": (0, -.5, -1)}, source_lines=[352, 359]),
    spec("agentmixer_matrix", "references/agentmixer/figs/Matrix.pdf", MATRIX[:1],
         paper="agentmixer", label="AgentMixer", scope="commented_appendix_asset", quantum=.01,
         signed_y={"climbing": (10, 0, -10, -20)}, source_lines=[978, 1005]),
    spec("agentmixer_prey", "references/agentmixer/figs/Prey_main.pdf",
         (("Predator-Prey Penalty = 0", "predator_prey_penalty_0"),
          ("Predator-Prey Penalty = -1", "predator_prey_penalty_-1")),
         paper="agentmixer", label="AgentMixer", scope="commented_appendix_asset",
         metric="average_test_episode_return", split="test_per_commented_prose", quantum=.01,
         source_lines=[1009, 1016]),
)


def axes(rect, text, drawings, signed_y):
    """Calibrate numeric labels against full-length grid strokes, never curve bounds."""
    horizontal, vertical = [], []
    for index, path in enumerate(drawings):
        if path["type"] != "s" or len(path["items"]) != 1 or path["items"][0][0] != "l":
            continue
        if abs(path["width"] - .5) > .01:
            continue
        bounds = path["rect"]
        if bounds.height < .001 and abs(bounds.x0 - rect.x0) < .001 and abs(bounds.x1 - rect.x1) < .001:
            if rect.y0 <= bounds.y0 <= rect.y1:
                horizontal.append((bounds.y0, index))
        if bounds.width < .001 and abs(bounds.y0 - rect.y0) < .001 and abs(bounds.y1 - rect.y1) < .001:
            if rect.x0 <= bounds.x0 <= rect.x1:
                vertical.append((bounds.x0, index))
    exponent = [s for s in text if re.fullmatch(r"1e\d+", s["text"])
                and abs(s["bbox"][2] - rect.x1) < 2 and rect.y1 < s["origin"][1] < rect.y1 + 65]
    if len(exponent) != 1 or not horizontal or not vertical:
        raise ValueError("Missing or ambiguous axis grids/scientific exponent.")
    x_ticks, y_ticks = [], []
    for entry in text:
        if not geometry.NUMBER.fullmatch(entry["text"]):
            continue
        x0, y0, x1, y1 = entry["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        value = float(entry["text"].replace("\N{MINUS SIGN}", "-"))
        if rect.x0 - 1 <= cx <= rect.x1 + 1 and rect.y1 - 1 <= cy <= rect.y1 + 22:
            coordinate, index = min(vertical, key=lambda item: abs(item[0] - cx))
            if abs(coordinate - cx) < 1:
                x_ticks.append(dict(label=entry["text"], value=value * float(exponent[0]["text"]),
                                    pdf_position=coordinate, grid_path_index=index, text_bbox=entry["bbox"]))
        if rect.x0 - 90 <= x0 < x1 < rect.x0 and rect.y0 - 3 <= cy <= rect.y1 + 3:
            coordinate, index = min(horizontal, key=lambda item: abs(item[0] - cy))
            if abs(coordinate - cy) < 2:
                y_ticks.append(dict(label=entry["text"], value=value, pdf_position=coordinate,
                                    grid_path_index=index, text_bbox=entry["bbox"]))
    y_ticks.sort(key=lambda tick: tick["pdf_position"])
    if signed_y:
        if len(y_ticks) != len(signed_y):
            raise ValueError("Signed tick recipe differs from observed tick count.")
        for tick, signed_value in zip(y_ticks, signed_y):
            if abs(abs(tick["value"]) - abs(signed_value)) > 1e-8:
                raise ValueError("Visually transcribed signed tick disagrees with PDF text magnitude.")
            tick["raw_text_value"] = tick["value"]
            tick["value"] = signed_value
            tick["sign_source"] = "explicit_recipe_verified_against_original_render"
    x_axis, y_axis = geometry.fit_axis(x_ticks), geometry.fit_axis(y_ticks)
    if x_axis["slope"] <= 0 or y_axis["slope"] >= 0:
        raise ValueError("Axis signs disagree with the displayed orientation.")
    return x_axis, y_axis, exponent[0]


def conventions(figure, task):
    matrix = task in ("climbing", "penalty")
    if matrix:
        optimal = 11 if task == "climbing" else 10
        scale = 200 if figure["paper"] == "bpta" else 1
        return dict(axis_unit="plotted_episode_return", per_decision_optimal_payoff=optimal,
                    plotted_scale_equivalent_optimum=scale * optimal,
                    inferred_payoff_to_plot_multiplier=scale,
                    normalization_is_inference=True,
                    observation_protocol="The manuscripts describe stateless games but do not specify exact observation/state tensors, previous-action features, or time features. The plotted figures cannot establish equivalence with constant-observation one-step training.",
                    episode_convention="BPTA plots approximately 200 times the one-decision payoff; its table has episode length 200, but environment episode versus rollout/aggregation semantics are unresolved."
                    if scale == 200 else "AgentMixer plots the one-decision payoff scale near 11 despite a commented episode-length-200 table; normalization and episode semantics are unresolved.")
    if "win_rate" in figure["metric"]:
        return dict(axis_unit="win_rate_fraction_0_to_1", per_decision_optimal_payoff=None,
                    episode_convention="Train and test axes are preserved separately; original evaluation episode count and deterministic/stochastic rule are unspecified.")
    if task == "5_vs_5":
        return dict(axis_unit="plotted_episode_return", per_decision_optimal_payoff=None,
                    episode_convention="Caption specifies averaged return, not win rate. Train/test split and precise reward aggregation are not stated by this figure.")
    return dict(axis_unit="plotted_average_test_episode_return", per_decision_optimal_payoff=None,
                episode_convention="Commented prose describes reward 10 for capture, but plotted returns near 2.2/0.35 do not determine episode length, team-versus-agent reduction, or reward normalization.")


def crossing_summary(xs, ys, threshold, x_error, y_error):
    """Crossing brackets use neighboring plotted vertices, without linear interpolation."""
    def first_index(values, sustained=False):
        for index, value in enumerate(values):
            if value >= threshold and (not sustained or min(values[index:]) >= threshold):
                return index
        return None

    def crossing(values, sustained=False):
        index = first_index(values, sustained)
        if index is None:
            return None
        bracket = [None if index == 0 else xs[index - 1] - x_error, xs[index] + x_error]
        return dict(vertex_index=index, first_plotted_steps=xs[index], previous_plotted_steps=None if index == 0 else xs[index - 1],
                    crossing_steps_bracket=bracket, left_censored_at_first_vertex=index == 0)

    return dict(threshold=threshold, first_observed_crossing=crossing(ys),
                first_observed_sustained_crossing=crossing(ys, True),
                first_possible_crossing_with_reading_allowance=crossing([y + y_error for y in ys]),
                first_certain_crossing_with_reading_allowance=crossing([y - y_error for y in ys]),
                first_certain_sustained_crossing_with_reading_allowance=crossing([y - y_error for y in ys], True),
                interpretation="Thresholds are analysis conventions, not thresholds reported by the paper. Sustained means all remaining plotted centerline vertices meet the threshold, not successful individual seeds.")


def extract_figure(root, out, figure):
    document = fitz.open(root / figure["pdf"])
    if len(document) != 1:
        raise ValueError("Expected one-page archived figure.")
    page = document[0]
    drawings, text = page.get_drawings(), geometry.spans(page)
    legend = geometry.find_legend(text, drawings, figure["label"])
    frames = [(i, d["rect"]) for i, d in enumerate(drawings)
              if d["type"] == "f" and len(d["items"]) == 1 and d["items"][0][0] == "re"
              and d["fill"] and all(.90 < c < .96 for c in d["fill"])
              and d["rect"].width > 150 and d["rect"].height > 80]
    frames.sort(key=lambda entry: (round(entry[1].y0, 2), entry[1].x0))
    if len(frames) != len(figure["panels"]):
        raise ValueError(f"Panel count mismatch: {figure['name']}")
    page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(out / (figure["name"] + "_original.png"))
    targets, csv_rows = [], []
    for panel_index, ((frame_index, rect), (title, task)) in enumerate(zip(frames, figure["panels"])):
        titles = [entry for entry in text if entry["text"] == title
                  and rect.x0 < (entry["bbox"][0] + entry["bbox"][2]) / 2 < rect.x1
                  and rect.y0 - 35 < entry["origin"][1] < rect.y0]
        if len(titles) != 1:
            raise ValueError(f"Could not identify title: {figure['name']} / {title}")
        paths = [(i, d) for i, d in enumerate(drawings) if geometry.rgb_close(d["color"], legend["rgb"])
                 and len(d["items"]) > 10 and rect.contains(d["rect"])]
        curves = [(i, d) for i, d in paths if d["type"] == "s"]
        bands = [(i, d) for i, d in paths if d["type"] == "fs" and d["fill_opacity"] < .5]
        if len(curves) != 1 or len(bands) != 1:
            raise ValueError(f"Ambiguous curve/band: {title}: {len(curves)}/{len(bands)}")
        curve_index, curve = curves[0]
        band_index, band = bands[0]
        points, band_points = geometry.line_points(curve), geometry.line_points(band)
        if any(b[0] <= a[0] for a, b in zip(points, points[1:])):
            raise ValueError("Curve x vertices do not increase strictly.")
        x_axis, y_axis, exponent = axes(rect, text, drawings, figure["signed_y"].get(task))
        xs = [geometry.map_axis(x_axis, x) for x, _ in points]
        ys = [geometry.map_axis(y_axis, y) for _, y in points]
        band_values = []
        for vertex, ((x, y), step, central) in enumerate(zip(points, xs, ys)):
            matching_y = [py for px, py in band_points if abs(px - x) <= geometry.COORDINATE_TOLERANCE_PT]
            if not matching_y:
                raise ValueError("Band vertices do not match centerline x positions.")
            lower, upper = sorted(geometry.map_axis(y_axis, y) for y in (min(matching_y), max(matching_y)))
            band_values.append((lower, upper))
            csv_rows.append(dict(figure=figure["name"], paper=figure["paper"], algorithm=figure["label"], task=task,
                                 metric=figure["metric"], primary_target=figure["primary"], vertex=vertex,
                                 pdf_x_pt=x, pdf_y_pt=y, environment_steps_estimate=step,
                                 central_curve_estimate=central, band_lower_estimate=lower, band_upper_estimate=upper))
        allowance_pt = geometry.COORDINATE_TOLERANCE_PT + .25 + curve["width"] / 2
        y_error = abs(y_axis["slope"]) * (allowance_pt + y_axis["max_fit_residual_pt"])
        x_error = abs(x_axis["slope"]) * (allowance_pt + x_axis["max_fit_residual_pt"])
        quantum = figure["quantum"]
        rounded = round(round(ys[-1] / quantum) * quantum, 6)
        reading_range = [round(math.floor((ys[-1] - y_error) / quantum) * quantum, 6),
                         round(math.ceil((ys[-1] + y_error) / quantum) * quantum, 6)]
        convention = conventions(figure, task)
        record = dict(paper=figure["paper"], algorithm=figure["label"], task=task,
                      figure=figure["name"], primary_benchmark_target=figure["primary"],
                      metric=figure["metric"], evaluation_split=figure["split"], source_scope=figure["scope"],
                      source_pdf=figure["pdf"], source_pdf_sha256=geometry.sha256(root / figure["pdf"]),
                      manuscript="references/bpta/Formatting-Instructions-LaTeX-2024.tex" if figure["paper"] == "bpta" else "references/agentmixer/aaai25.tex",
                      manuscript_lines=figure["source_lines"], page=1, panel_index=panel_index,
                      title=title, title_bbox=titles[0]["bbox"], plot_frame=list(rect), plot_frame_path_index=frame_index,
                      legend=legend, curve_path_index=curve_index, band_path_index=band_index,
                      curve_width_pt=curve["width"], curve_dashes=curve["dashes"], curve_vertices=len(points),
                      axis_x=x_axis, axis_y=y_axis, x_exponent_text=exponent,
                      endpoint_pdf_xy=list(points[-1]), last_plotted_steps_estimate=xs[-1],
                      first_plotted_steps_estimate=xs[0], median_vertex_spacing_steps=statistics.median(b - a for a, b in zip(xs, xs[1:])),
                      last_plotted_steps_reading_range=[xs[-1] - x_error, xs[-1] + x_error],
                      central_curve_endpoint_estimate=ys[-1], reported_endpoint_rounded=rounded,
                      digitization_reading_range=reading_range, reporting_quantum=quantum,
                      digitization_uncertainty=dict(coordinate_precision_tolerance_pt=geometry.COORDINATE_TOLERANCE_PT,
                                                   registration_allowance_pt=.25, curve_half_width_pt=curve["width"] / 2,
                                                   y_axis_fit_residual_pt=y_axis["max_fit_residual_pt"],
                                                   total_y_reading_allowance_pt=allowance_pt + y_axis["max_fit_residual_pt"],
                                                   return_reading_allowance_before_rounding=y_error,
                                                   step_reading_allowance=x_error, is_statistical_confidence_interval=False),
                      shaded_band_endpoint_estimates=list(band_values[-1]),
                      shaded_band_midpoint_minus_centerline=statistics.mean(band_values[-1]) - ys[-1],
                      central_curve_aggregation="mean_over_five_seeds_per_source_text" if figure["paper"] == "bpta" else "mean_or_median_not_resolved_from_supplied_source",
                      shaded_band_interpretation="source_text_states_standard_deviation_over_five_seeds_but_rendered_construction_unverified" if figure["paper"] == "bpta" else "not_defined_for_this_figure_in_supplied_source",
                      last_five_plotted_vertices=dict(steps=xs[-5:], central_curve_values=ys[-5:], mean=statistics.mean(ys[-5:]),
                                                     range=[min(ys[-5:]), max(ys[-5:])], is_original_evaluation_aggregation=False),
                      reward_and_episode_convention=convention,
                      caveats=["Plotted centerline estimate, not original seed-level data.",
                               "Reading ranges are not experimental confidence intervals.",
                               "No endpoint extrapolation or best-checkpoint selection."])
        if task in ("climbing", "penalty"):
            record["sample_efficiency_thresholds"] = []
            for fraction in (.90, .95, .99):
                entry = crossing_summary(xs, ys, fraction * convention["plotted_scale_equivalent_optimum"], x_error, y_error)
                entry["fraction_of_scale_equivalent_optimum"] = fraction
                record["sample_efficiency_thresholds"].append(entry)
            record["caveats"].append("Return normalization is inferred from axis scale; raw episode semantics cannot be recovered.")
        if task.startswith("10gen_"):
            record["caveats"].append("Figure labels 5v5/10v10/10v11 conflict with BPTA prose mentioning 20v20/20v23; do not relabel these panels.")
        targets.append(record)
        page.draw_polyline([fitz.Point(*p) for p in points], color=(0, .35, .8), width=.35, overlay=True)
        page.draw_circle(fitz.Point(*points[-1]), 2.5, color=(0, .15, .8), width=.8, overlay=True)
        # Put annotations near the frame's top edge, away from lower legends.
        page.insert_text((rect.x0 + 4, rect.y0 + 10), f"endpoint ~{rounded:g}", fontsize=7, color=(0, .15, .8), overlay=True)
    overlay = out / (figure["name"] + "_verified_overlay.png")
    page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(overlay)
    provenance = dict(source=figure["pdf"], sha256=geometry.sha256(root / figure["pdf"]), metadata=document.metadata,
                      embedded_file_count=document.embfile_count(), page_rect=list(page.rect), drawing_count=len(drawings),
                      panel_count=len(frames), legend=legend, source_scope=figure["scope"],
                      original_render=figure["name"] + "_original.png", overlay_render=overlay.name,
                      overlay_sha256=geometry.sha256(overlay))
    return targets, csv_rows, provenance


def verify_sources(root):
    """Verify both archive integrity and selected extracted member byte identity."""
    inventory = []
    selected = [figure["pdf"] for figure in FIGURES] + list(geometry.MANUSCRIPTS)
    for entry in json.loads((root / "references/sources.json").read_text()):
        archive = root.parent / entry["archive"]
        if not archive.exists():
            raise FileNotFoundError(f"Provenance verification requires original archive: {archive}")
        actual_hash = geometry.sha256(archive)
        if actual_hash != entry["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {archive}")
        assets = []
        with zipfile.ZipFile(archive) as bundle:
            names = [name for name in bundle.namelist() if not name.endswith("/")]
            for relative in selected:
                if relative.startswith(entry["extracted_to"] + "/"):
                    member = str(Path(relative).relative_to(entry["extracted_to"]))
                    checksum = geometry.sha256(root / relative)
                    if hashlib.sha256(bundle.read(member)).hexdigest() != checksum:
                        raise ValueError(f"Extracted asset differs from archive: {relative}")
                    assets.append(dict(path=relative, archive_member=member, sha256=checksum))
        candidates = [name for name in names if Path(name).suffix.lower() in
                      {".csv", ".json", ".npy", ".npz", ".pkl", ".pickle", ".mat", ".h5", ".hdf5", ".xlsx", ".py", ".ipynb", ".r"}]
        inventory.append({**entry, "verified_sha256": actual_hash, "file_count": len(names),
                          "data_or_plotting_script_candidates": candidates, "verified_extracted_assets": assets})
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/target_extraction_other"))
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output if args.output.is_absolute() else root / args.output
    out.mkdir(parents=True, exist_ok=True)
    inventory = verify_sources(root)
    targets, rows, figures = [], [], []
    for figure in FIGURES:
        extracted, csv_rows, provenance = extract_figure(root, out, figure)
        targets.extend(extracted)
        rows.extend(csv_rows)
        figures.append(provenance)
    primary_count = sum(target["primary_benchmark_target"] for target in targets)
    if primary_count != 18 or len(targets) != 32:
        raise ValueError("Coverage changed; review the expected target inventory.")
    report = dict(schema_version=1, status="digitized_curve_estimates_not_exact_published_measurements",
                  script="scripts/extract_other_curves.py", script_sha256=geometry.sha256(Path(__file__).resolve()),
                  geometry_helpers="scripts/extract_curve_targets.py", geometry_helpers_sha256=geometry.sha256(Path(geometry.__file__)),
                  python=sys.version.split()[0], pymupdf=fitz.VersionBind, source_archive_inventory=inventory,
                  figures=figures, targets=targets, primary_benchmark_target_count=primary_count,
                  secondary_curve_count=len(targets) - primary_count,
                  coverage_note="18 benchmark targets here + 21 MuJoCo curve estimates + 6 AgentMixer SMACv2 tabulated targets cover the 45-row benchmark inventory. This is source-target coverage, not completed reproduction.",
                  unresolved_prose_only_targets=[f"10gen_{race}_{size}" for race in ("protoss", "terran", "zerg") for size in ("20v20", "20v23")],
                  pdf_coordinate_system="points (1/72 inch), origin top-left, y increasing downwards",
                  render_scale=1.5, render_dpi=108,
                  visual_review="Original and annotated renders require manual visual verification; see docs/other_curve_targets.md.",
                  canonical_manifest_modified=False)
    (out / "curve_targets.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / "digitized_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / "endpoint_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("paper", "algorithm", "task", "metric", "primary_target", "last_plotted_steps", "endpoint_estimate", "reading_lower", "reading_upper", "source_scope"))
        for target in targets:
            writer.writerow((target["paper"], target["algorithm"], target["task"], target["metric"], target["primary_benchmark_target"],
                             round(target["last_plotted_steps_estimate"]), target["reported_endpoint_rounded"],
                             *target["digitization_reading_range"], target["source_scope"]))
            print(target["figure"], target["task"], "x", round(target["last_plotted_steps_estimate"]),
                  "endpoint", target["reported_endpoint_rounded"], target["digitization_reading_range"])
    with (out / "matrix_thresholds.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("paper", "algorithm", "task", "fraction", "threshold_plotted_units", "first_observed_steps", "previous_observed_steps", "sustained_observed_steps", "first_certain_steps", "sustained_certain_steps"))
        for target in targets:
            for threshold in target.get("sample_efficiency_thresholds", []):
                def time(kind):
                    value = threshold[kind]
                    return round(value["first_plotted_steps"]) if value else None
                first = threshold["first_observed_crossing"]
                previous = first["previous_plotted_steps"] if first else None
                writer.writerow((target["paper"], target["algorithm"], target["task"], threshold["fraction_of_scale_equivalent_optimum"],
                                 threshold["threshold"], time("first_observed_crossing"), round(previous) if previous is not None else None,
                                 time("first_observed_sustained_crossing"), time("first_certain_crossing_with_reading_allowance"),
                                 time("first_certain_sustained_crossing_with_reading_allowance")))
    print(f"Wrote {primary_count} primary targets, {len(targets) - primary_count} secondary curves, {len(rows)} vertices.")


if __name__ == "__main__":
    main()
