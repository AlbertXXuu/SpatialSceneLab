"""Rebuild publication figures from recorded development results (optional Matplotlib).

Reference overview shows authored geometry, not inferred reconstruction. MIT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


PALETTE = ["#2563EB", "#4F46E5", "#7C3AED"]
NAMES = {"dev-contact-chair": "Contact chair", "dev-same-color-dining": "Same-color dining",
         "dev-curved-sofa-wall": "Curved sofa / wall", "dev-occluded-bookshelf": "Occluded bookshelf",
         "dev-thin-rack-noise": "Thin rack / noise", "dev-combined-study": "Combined study"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_figures(results_path, suite_path, output_path):
    results_path, suite_path = Path(results_path).resolve(), Path(suite_path).resolve()
    results = json.loads(results_path.read_text(encoding="utf-8"))
    manifest = json.loads((suite_path / "suite-manifest.json").read_text(encoding="utf-8"))
    if sha256(suite_path / "suite-manifest.json") != results["input_manifest_sha256"]:
        raise ValueError("figure suite does not match the recorded experiment input")
    requested = Path(output_path).absolute()
    if any(path.is_symlink() or getattr(path, "is_junction", lambda: False)()
           for path in (requested, *requested.parents)):
        raise ValueError("figure output chain contains a link")
    output = requested.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("figure output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    names = [scene["scene_id"] for scene in results["scenes"]]
    x = np.arange(len(names))
    methods = ("b0", "b1", "b2")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for ax, key, title in zip(axes,
                             ("area_iou", "false_negative_ratio", "false_positive_ratio"),
                             ("Target area IoU (higher is better)",
                              "Missed target area / target area", "Non-target selected area / target area")):
        for index, method in enumerate(methods):
            values = []
            for scene in results["scenes"]:
                result = next((item for item in scene["methods"] if item["method"] == method), {})
                value = result.get("target", {}).get(key)
                values.append(np.nan if value is None else value)
            ax.bar(x + (index - 1) * .24, values, width=.23, color=PALETTE[index],
                   label=method.upper(), hatch=("", "//", "..")[index], linewidth=.35)
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_xticks(x, [NAMES.get(name, name).replace(" / ", "\n").replace(" ", "\n", 1)
                         for name in names], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Area ratio")
        ax.set_ylim(bottom=0)
        if key != "false_positive_ratio":
            ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, ncols=3, loc="upper left")
    fig.suptitle("Fixed-mesh B0 / B1 / B2 — six authored development scenes", fontsize=16)
    fig.savefig(output / "baseline-comparison.png", dpi=160)
    fig.savefig(output / "baseline-comparison.pdf")
    plt.close(fig)

    fig = plt.figure(figsize=(14, 9.5), constrained_layout=True)
    entries = {scene["scene_id"]: scene for scene in manifest["scenes"]}
    for index, name in enumerate(names):
        record = entries[name]
        reference = (suite_path / record["reference_labels"]).resolve()
        if suite_path not in reference.parents:
            raise ValueError("reference path escapes suite")
        measured = next(scene for scene in results["scenes"] if scene["scene_id"] == name)
        if sha256(reference) != measured.get("reference_sha256"):
            raise ValueError("figure reference differs from the scored physical geometry")
        with np.load(reference, allow_pickle=False) as data:
            vertices, faces, owners = data["vertices"], data["triangles"], data["triangle_owner"]
            ids = data["object_ids"].tolist()
        target = ids.index(record["target_id"])
        ax = fig.add_subplot(2, 3, index + 1, projection="3d")
        # Display coordinates X, Z, Y so the original Y-up remains vertical.
        corners = vertices[faces][:, :, [0, 2, 1]]
        for label in np.unique(owners):
            selected = owners == label
            color = PALETTE[0] if label == target else ("#CBD5E1" if label >= 0 else "#E2E8F0")
            alpha = .9 if label >= 0 else .12
            collection = Poly3DCollection(corners[selected], facecolors=color,
                                          alpha=alpha, shade=True, lightsource=matplotlib.colors.LightSource(315, 45))
            ax.add_collection3d(collection)
        furniture = vertices[np.unique(faces[owners >= 0])][:, [0, 2, 1]]
        lower, upper = furniture.min(0), furniture.max(0)
        center, span = (lower + upper) / 2, max(upper - lower) * 1.15
        ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
        ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
        ax.set_zlim(0, max(upper[2] + .1, span * .6))
        ax.set_box_aspect((1, 1, .7))
        ax.view_init(elev=24, azim=55)
        ax.set_title(f"{NAMES.get(name, name)} · {record['frames']} views", fontsize=11)
        ax.set_axis_off()
    fig.suptitle("Authored physical references — blue marks each task target", fontsize=16)
    fig.savefig(output / "development-scene-overview.png", dpi=150)
    fig.savefig(output / "development-scene-overview.pdf")
    plt.close(fig)
    provenance = {"schema": "spatial-scene-lab.development-figures.v1",
                  "results_sha256": sha256(results_path),
                  "input_manifest_sha256": sha256(suite_path / "suite-manifest.json"),
                  "plot_script_sha256": sha256(__file__),
                  "matplotlib": matplotlib.__version__, "numpy": np.__version__,
                  "python": platform.python_version(),
                  "mkl_threading_layer": os.environ.get("MKL_THREADING_LAYER"),
                  "scope": "development comparisons and authored physical reference overview; no held-out result",
                  "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                            for path in sorted(output.iterdir()) if path.is_file()}}
    (output / "figure-manifest.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build_figures(args.results, args.suite, args.output), indent=2))
