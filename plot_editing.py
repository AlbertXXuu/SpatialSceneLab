"""Rebuild the S1 measured editing figure and per-scene table. MIT.

Matplotlib is optional; the native editing workflow does not depend on it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = Path(args.results).resolve()
    assets = Path(args.assets).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("figure output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads(results.read_text(encoding="utf-8"))
    combined = next(task for task in summary["tasks"] if task["scene_id"] == "dev-combined-study")
    images = [assets / name for name in ("before.png", "after.png")]
    for path in images:
        if sha256(path) != combined["assets"][path.name]["sha256"]:
            raise ValueError("figure input image changed from the measured task")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), dpi=200, facecolor="#f6f8fc")
    fig.subplots_adjust(left=.015, right=.985, top=.82, bottom=.17, wspace=.025)
    for axis, path, title in zip(axes, images, ("Before · observed B0 regions", "After · desk translation / chair rotation")):
        axis.imshow(plt.imread(path))
        axis.set_axis_off()
        axis.set_title(title, fontsize=10, color="#17243c", pad=12)
    fig.suptitle("Observed surfaces: local ownership and rigid edits", x=.025, y=.965,
                 ha="left", fontsize=17, fontweight="bold", color="#111d34")
    fig.text(.025, .888, "Combined study · native Blender 5.1.2 · fixed review camera · CPU Cycles", fontsize=10, color="#52627a")
    fig.text(.025, .105, "12 + 8 explicitly selected faces transferred; desk +0.25 m X / -0.10 m Y; chair 32° about world Z.",
             fontsize=9, color="#17243c")
    fig.text(.025, .05, "Holes and residual surfaces remain visible. This verifies editing and export, not complete-object recovery.",
             fontsize=9, color="#52627a")
    png, pdf = output / "native-editing-comparison.png", output / "native-editing-comparison.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf, metadata={"Title": "SpatialSceneLab S1 native editing", "Author": "AlvenX", "CreationDate": None})
    plt.close(fig)
    rows = []
    for task in summary["tasks"]:
        native = task.get("native_task", {})
        rows.append({"scene": task["scene_id"], "status": task["status"],
                     "source_faces": native.get("initial_source_faces"),
                     "committed_operations": len(native.get("operations", [])),
                     "native_undo_redo_cycles": sum(op.get("undo_redo", {}).get("status") == "pass" for op in native.get("operations", [])),
                     "new_process_reopen": task.get("reopen", {}).get("status", "not_completed"),
                     "task_process_seconds": task["process"]["elapsed_seconds"],
                     "sampled_process_tree_peak_rss_bytes": task["process"]["sampled_process_tree_peak_rss_bytes"],
                     "maximum_native_corner_distance_m": max((op["state"]["maximum_corner_distance_m"] for op in native.get("operations", [])), default=None),
                     "maximum_glb_world_vertex_distance_m": native.get("glb_geometry", {}).get("maximum_world_vertex_distance_m")})
    csv_path = output / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {"schema": "spatial-scene-lab.editing-figures.v1", "measured_results_sha256": sha256(results),
                "script_sha256": sha256(__file__), "matplotlib": matplotlib.__version__,
                "inputs": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in images},
                "outputs": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in (png, pdf, csv_path)},
                "image_dimensions": [2400, 1080],
                "scope": "actual observed-surface Blender renders; edited regions are incomplete B0 selections"}
    (output / "figure-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(output)}))


if __name__ == "__main__":
    main()
