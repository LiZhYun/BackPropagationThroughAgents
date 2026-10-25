#!/usr/bin/env python3
"""Digitize selected accepted-source vector PDFs without changing references.

Optional extraction dependency: PyMuPDF 1.26.5. Run with that package on
PYTHONPATH; it is deliberately not a training-package dependency. Output
numbers are estimates of plotted central curves, never original run data.
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


FIGURES = (
    ("bppo", "bpta", "BPPO", "references/bpta/MuJoCo.pdf", "active_main", 9),
    ("agentmixer_main", "agentmixer", "AgentMixer", "references/agentmixer/figs/MuJoCo_main.pdf", "active_main", 6),
    ("agentmixer_app", "agentmixer", "AgentMixer", "references/agentmixer/figs/MuJoCo_app.pdf", "commented_appendix_asset", 6),
)
MANUSCRIPTS = (
    "references/bpta/Formatting-Instructions-LaTeX-2024.tex",
    "references/agentmixer/aaai25.tex",
)
TITLE = re.compile(r"^(Ant|HalfCheetah|Walker2d|Humanoid|HumanoidStandup)-v2 ([0-9]+x[0-9]+d?)$")
NUMBER = re.compile(r"^[\-\N{MINUS SIGN}]?\d+(?:\.\d+)?$")
COORDINATE_TOLERANCE_PT = 0.001


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spans(page):
    return [dict(text=s["text"], bbox=list(s["bbox"]), origin=list(s["origin"]), size=s["size"])
            for block in page.get_text("dict")["blocks"] if "lines" in block
            for line in block["lines"] for s in line["spans"]]


def rgb_close(first, second):
    return first is not None and second is not None and max(abs(a - b) for a, b in zip(first, second)) < 1e-5


def line_points(path):
    if not path["items"] or any(item[0] != "l" for item in path["items"]):
        raise ValueError("Selected curve must be a vector polyline, not a raster or Bezier fit.")
    points = [tuple(path["items"][0][1])]
    for _, start, end in path["items"]:
        if max(abs(a - b) for a, b in zip(points[-1], start)) > COORDINATE_TOLERANCE_PT:
            raise ValueError("Discontinuous path cannot be interpreted as one plotted curve.")
        points.append(tuple(end))
    return points


def fit_axis(anchors):
    positions = [entry["pdf_position"] for entry in anchors]
    values = [entry["value"] for entry in anchors]
    if len(positions) < 2 or len(set(positions)) != len(positions):
        raise ValueError("At least two distinct tick-grid anchors are required.")
    mean_p, mean_v = statistics.mean(positions), statistics.mean(values)
    slope = sum((p - mean_p) * (v - mean_v) for p, v in zip(positions, values)) / sum((p - mean_p) ** 2 for p in positions)
    intercept = mean_v - slope * mean_p
    residual = max(abs(p - (v - intercept) / slope) for p, v in zip(positions, values))
    if residual > 0.01:
        raise ValueError(f"Tick mapping is not consistently linear: residual {residual} PDF points.")
    return {"slope": slope, "intercept": intercept, "max_fit_residual_pt": residual, "anchors": anchors}


def map_axis(axis, coordinate):
    return axis["slope"] * coordinate + axis["intercept"]


def find_legend(text, drawings, label):
    labels = [s for s in text if s["text"] == label]
    if len(labels) != 1:
        raise ValueError(f"Expected one unambiguous {label} legend label, found {len(labels)}.")
    entry = labels[0]
    x0, y0, _, y1 = entry["bbox"]
    candidates = []
    for index, path in enumerate(drawings):
        rect = path["rect"]
        if (path["type"] == "s" and 1 <= len(path["items"]) <= 3
                and all(item[0] == "l" for item in path["items"])
                and rect.height < 0.001 and 8 <= rect.width <= 40
                and 0 <= x0 - rect.x1 <= 15 and y0 <= rect.y0 <= y1
                and path["color"] is not None and max(path["color"]) - min(path["color"]) > 0.1):
            candidates.append((index, path))
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous legend line for {label}: {len(candidates)} candidates.")
    index, path = candidates[0]
    return {"text": label, "text_bbox": entry["bbox"], "line_path_index": index,
            "line_bbox": list(path["rect"]), "rgb": list(path["color"])}


def panel_axes(rect, text, drawings):
    horizontal, vertical = [], []
    for index, path in enumerate(drawings):
        if path["type"] != "s" or len(path["items"]) != 1 or path["items"][0][0] != "l":
            continue
        if abs(path["width"] - 0.5) > 0.01:
            continue
        bounds = path["rect"]
        if bounds.height < 0.001 and abs(bounds.x0 - rect.x0) < 0.001 and abs(bounds.x1 - rect.x1) < 0.001:
            if rect.y0 <= bounds.y0 <= rect.y1:
                horizontal.append((bounds.y0, index))
        if bounds.width < 0.001 and abs(bounds.y0 - rect.y0) < 0.001 and abs(bounds.y1 - rect.y1) < 0.001:
            if rect.x0 <= bounds.x0 <= rect.x1:
                vertical.append((bounds.x0, index))
    exponents = [s for s in text if re.fullmatch(r"1e\d+", s["text"])
                 and abs(s["bbox"][2] - rect.x1) < 2 and rect.y1 < s["origin"][1] < rect.y1 + 65]
    if len(exponents) != 1:
        raise ValueError("Every panel must expose one verified scientific x-axis multiplier.")
    multiplier = float(exponents[0]["text"])
    x_ticks, y_ticks = [], []
    for entry in text:
        if not NUMBER.fullmatch(entry["text"]):
            continue
        x0, y0, x1, y1 = entry["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        value = float(entry["text"].replace("\N{MINUS SIGN}", "-"))
        if rect.x0 - 1 <= cx <= rect.x1 + 1 and rect.y1 - 1 <= cy <= rect.y1 + 22:
            coordinate, index = min(vertical, key=lambda item: abs(item[0] - cx))
            if abs(coordinate - cx) < 1:
                x_ticks.append({"label": entry["text"], "value": value * multiplier,
                                "pdf_position": coordinate, "grid_path_index": index, "text_bbox": entry["bbox"]})
        if rect.x0 - 90 <= x0 < x1 < rect.x0 and rect.y0 - 3 <= cy <= rect.y1 + 3:
            coordinate, index = min(horizontal, key=lambda item: abs(item[0] - cy))
            if abs(coordinate - cy) < 2:
                y_ticks.append({"label": entry["text"], "value": value,
                                "pdf_position": coordinate, "grid_path_index": index, "text_bbox": entry["bbox"]})
    return fit_axis(x_ticks), fit_axis(y_ticks), exponents[0]


def archive_inventory(root):
    inventory = []
    suffixes = {".csv", ".json", ".npy", ".npz", ".pkl", ".pickle", ".mat", ".h5", ".hdf5", ".xlsx", ".py", ".ipynb", ".r"}
    for entry in json.loads((root / "references/sources.json").read_text()):
        archive = root.parent / entry["archive"]
        if not archive.exists():
            inventory.append({**entry, "archive_available": False})
            continue
        actual_hash = sha256(archive)
        if actual_hash != entry["sha256"]:
            raise ValueError(f"Archive hash mismatch: {archive}")
        with zipfile.ZipFile(archive) as bundle:
            names = [name for name in bundle.namelist() if not name.endswith("/")]
            verified_assets = []
            for relative in [item[3] for item in FIGURES] + list(MANUSCRIPTS):
                if not relative.startswith(entry["extracted_to"] + "/"):
                    continue
                member = str(Path(relative).relative_to(entry["extracted_to"]))
                extracted_hash = sha256(root / relative)
                if hashlib.sha256(bundle.read(member)).hexdigest() != extracted_hash:
                    raise ValueError(f"Extracted source differs from its archive member: {relative}")
                verified_assets.append({"path": relative, "archive_member": member, "sha256": extracted_hash})
        inventory.append({**entry, "archive_available": True, "verified_sha256": actual_hash,
                          "file_count": len(names), "files": names,
                          "verified_extracted_assets": verified_assets,
                          "data_or_plotting_script_candidates": [name for name in names if Path(name).suffix.lower() in suffixes]})
    return inventory


def extract_figure(root, out, short, paper, algorithm, relative, scope, expected_panels):
    document = fitz.open(root / relative)
    if len(document) != 1:
        raise ValueError("This extraction recipe expects the supplied single-page figure assets.")
    page = document[0]
    drawings, text = page.get_drawings(), spans(page)
    legend = find_legend(text, drawings, algorithm)
    panels = [(index, path["rect"]) for index, path in enumerate(drawings)
              if path["type"] == "f" and len(path["items"]) == 1 and path["items"][0][0] == "re"
              and path["fill"] is not None and all(0.90 < c < 0.96 for c in path["fill"])
              and path["rect"].width > 150 and path["rect"].height > 80]
    panels.sort(key=lambda item: (round(item[1].y0, 2), item[1].x0))
    if len(panels) != expected_panels:
        raise ValueError(f"Expected {expected_panels} plot frames, found {len(panels)} in {relative}.")
    page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(out / f"{short}_original.png")
    targets, csv_rows = [], []
    for panel_index, (frame_index, rect) in enumerate(panels):
        titles = [entry for entry in text if TITLE.fullmatch(entry["text"])
                  and rect.x0 < (entry["bbox"][0] + entry["bbox"][2]) / 2 < rect.x1
                  and rect.y0 - 35 < entry["origin"][1] < rect.y0]
        if len(titles) != 1:
            raise ValueError("Panel title could not be uniquely identified.")
        title = titles[0]["text"]
        matching = [(index, path) for index, path in enumerate(drawings)
                    if rgb_close(path["color"], legend["rgb"]) and len(path["items"]) > 20
                    and rect.contains(path["rect"])]
        curves = [(index, path) for index, path in matching if path["type"] == "s"]
        bands = [(index, path) for index, path in matching if path["type"] == "fs" and path["fill_opacity"] < .5]
        if len(curves) != 1 or len(bands) != 1:
            raise ValueError(f"Ambiguous curve or band for {title}: {len(curves)}, {len(bands)}.")
        curve_index, curve = curves[0]
        band_index, band = bands[0]
        points, band_points = line_points(curve), line_points(band)
        if any(b[0] <= a[0] for a, b in zip(points, points[1:])):
            raise ValueError("Curve x coordinates must increase strictly.")
        x_axis, y_axis, exponent = panel_axes(rect, text, drawings)
        xs = [map_axis(x_axis, x) for x, _ in points]
        ys = [map_axis(y_axis, y) for _, y in points]
        bands_data = []
        for point_index, ((x, y), step, value) in enumerate(zip(points, xs, ys)):
            band_y = [py for px, py in band_points if abs(px - x) <= COORDINATE_TOLERANCE_PT]
            if not band_y:
                raise ValueError("Band polygon and centerline do not share x vertices.")
            lower, upper = sorted((map_axis(y_axis, min(band_y)), map_axis(y_axis, max(band_y))))
            bands_data.append((lower, upper))
            csv_rows.append({"paper": paper, "algorithm": algorithm, "task": title.replace(" ", "_"),
                             "vertex": point_index, "pdf_x_pt": x, "pdf_y_pt": y,
                             "environment_steps_estimate": step, "central_curve_estimate": value,
                             "shaded_band_lower_estimate": lower, "shaded_band_upper_estimate": upper})

        # Conservative reading tolerance: vector coordinate quantization,
        # half a grid stroke for registration, and half the curve stroke.
        registration_pt = .25
        y_tolerance_pt = COORDINATE_TOLERANCE_PT + registration_pt + curve["width"] / 2 + y_axis["max_fit_residual_pt"]
        x_tolerance_pt = COORDINATE_TOLERANCE_PT + registration_pt + curve["width"] / 2 + x_axis["max_fit_residual_pt"]
        y_error = abs(y_axis["slope"]) * y_tolerance_pt
        x_error = abs(x_axis["slope"]) * x_tolerance_pt
        quantum = 100 if abs(ys[-1]) >= 10000 else 10
        rounded = round(ys[-1] / quantum) * quantum
        reading_range = [math.floor((ys[-1] - y_error) / quantum) * quantum,
                         math.ceil((ys[-1] + y_error) / quantum) * quantum]
        band_center_offset = statistics.mean(bands_data[-1]) - ys[-1]
        shared = title.startswith(("Ant-v2 ", "HalfCheetah-v2 ", "Walker2d-v2 ")) and not title.endswith("2x4d")
        record = {
            "paper": paper, "algorithm": algorithm, "task": title.replace(" ", "_"),
            "shared_task_name_across_papers": shared, "source_scope": scope,
            "source_pdf": relative, "source_pdf_sha256": sha256(root / relative), "page": 1,
            "panel_index": panel_index, "title": title, "title_bbox": titles[0]["bbox"],
            "plot_frame": list(rect), "plot_frame_path_index": frame_index,
            "legend": legend, "curve_path_index": curve_index, "band_path_index": band_index,
            "curve_width_pt": curve["width"], "curve_vertices": len(points),
            "axis_x": x_axis, "axis_y": y_axis, "x_exponent_text": exponent,
            "endpoint_pdf_xy": list(points[-1]), "last_plotted_steps_estimate": xs[-1],
            "last_plotted_steps_reading_range": [xs[-1] - x_error, xs[-1] + x_error],
            "paper_training_budget_steps": 10000000,
            "central_curve_endpoint_estimate": ys[-1], "reported_endpoint_rounded": rounded,
            "digitization_reading_range": reading_range, "reporting_quantum": quantum,
            "digitization_uncertainty": {
                "coordinate_precision_tolerance_pt": COORDINATE_TOLERANCE_PT,
                "registration_allowance_pt": registration_pt,
                "curve_half_width_pt": curve["width"] / 2,
                "y_axis_fit_residual_pt": y_axis["max_fit_residual_pt"],
                "total_y_reading_allowance_pt": y_tolerance_pt,
                "return_reading_allowance_before_rounding": y_error,
                "is_statistical_confidence_interval": False,
            },
            "shaded_band_endpoint_estimates": list(bands_data[-1]),
            "shaded_band_interpretation": "source_text_states_standard_deviation_over_five_seeds_but_polygon_asymmetry_is_unresolved" if paper == "bpta" else "not_explicitly_defined_for_MuJoCo_in_active_source",
            "shaded_band_midpoint_minus_centerline": band_center_offset,
            "central_curve_aggregation": "mean_over_five_seeds" if paper == "bpta" else "mean_or_median_not_resolved_from_supplied_source",
            "last_five_plotted_vertices": {"steps": xs[-5:], "central_curve_values": ys[-5:],
                                           "mean": statistics.mean(ys[-5:]), "range": [min(ys[-5:]), max(ys[-5:])],
                                           "is_original_evaluation_aggregation": False},
            "caveats": ["Digitized plotted centerline, not original seed-level data.",
                        "Final plotted vertex is earlier than the 10M budget; no extrapolation is performed.",
                        "Evaluation policy, smoothing and checkpoint selection are not recoverable from the figure."],
        }
        if title.startswith("Ant-v2 "):
            record["caveats"].append("Ant observation settings differ across papers: BPTA obsk=None, AgentMixer obsk=0.")
        if abs(band_center_offset) > y_error:
            record["caveats"].append("Band midpoint differs from the centerline by more than the digitization reading allowance; its statistical construction is unresolved.")
        targets.append(record)
        # Blue trace follows the selected red line; a circle marks the extracted
        # last vertex. All source files remain unchanged: only render the copy.
        page.draw_polyline([fitz.Point(*point) for point in points], color=(0, .35, .8), width=.35, overlay=True)
        page.draw_circle(fitz.Point(*points[-1]), 2.5, color=(0, .15, .8), width=.8, overlay=True)
        text_y = min(max(points[-1][1] - 6, rect.y0 + 12), rect.y1 - 6)
        page.insert_text((rect.x0 + 5, text_y), f"endpoint ~{rounded:,}", fontsize=7, color=(0, .15, .8), overlay=True)
    overlay = out / f"{short}_verified_overlay.png"
    page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(overlay)
    provenance = {"source": relative, "sha256": sha256(root / relative), "metadata": document.metadata,
                  "embedded_file_count": document.embfile_count(), "page_rect": list(page.rect),
                  "drawing_count": len(drawings), "panel_count": len(panels), "legend": legend,
                  "original_render": f"{short}_original.png", "overlay_render": overlay.name,
                  "overlay_sha256": sha256(overlay)}
    return targets, csv_rows, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/target_extraction"))
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output if args.output.is_absolute() else root / args.output
    out.mkdir(parents=True, exist_ok=True)
    targets, rows, figures = [], [], []
    for specification in FIGURES:
        extracted, csv_rows, provenance = extract_figure(root, out, *specification)
        targets.extend(extracted)
        rows.extend(csv_rows)
        figures.append(provenance)
    report = {
        "schema_version": 1, "status": "digitized_curve_estimates_not_exact_published_measurements",
        "script": "scripts/extract_curve_targets.py", "script_sha256": sha256(Path(__file__).resolve()),
        "python": sys.version.split()[0], "pymupdf": fitz.VersionBind,
        "source_archive_inventory": archive_inventory(root), "figures": figures, "targets": targets,
        "pdf_coordinate_system": "points (1/72 inch), origin at top left, y increasing downwards",
        "render_scale": 1.5, "render_dpi": 108,
        "visual_review": "Original and annotated renders require manual visual verification; see docs/curve_targets.md.",
        "canonical_manifest_modified": False,
    }
    (out / "curve_targets.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / "digitized_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / "endpoint_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("paper", "task", "last_plotted_steps", "endpoint_estimate", "reading_lower", "reading_upper", "band_lower", "band_upper", "aggregation"))
        for target in targets:
            writer.writerow((target["paper"], target["task"], round(target["last_plotted_steps_estimate"]),
                             target["reported_endpoint_rounded"], *target["digitization_reading_range"],
                             *target["shaded_band_endpoint_estimates"], target["central_curve_aggregation"]))
            print(f"{target['paper']:10} {target['title']:26} x={target['last_plotted_steps_estimate']/1e6:.6f}M "
                  f"y~{target['reported_endpoint_rounded']:,} reading_range={target['digitization_reading_range']}")
    print(f"Wrote {len(targets)} endpoints and {len(rows)} vector vertices to {out}")


if __name__ == "__main__":
    main()
