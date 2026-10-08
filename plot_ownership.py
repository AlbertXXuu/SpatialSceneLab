"""Rebuild the S2a diagnostic figure from recorded reason areas. MIT."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.results), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("figure output must be new or empty")
    result = json.loads(source.read_text(encoding="utf-8"))
    if result["status"] != "completed" or result["completed_scenes"] != 6:
        raise ValueError("requires all six completed diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    keys = ["fn_target_box_overlap", "fn_structure_exclusion", "fn_outside_target_box_environment",
            "fn_outside_target_box_other_object", "fn_outside_target_box_other_overlap"]
    labels = ["Target box overlap", "Structure exclusion", "Outside target box / environment",
              "Outside target box / other object", "Outside target box / other overlap"]
    colors = ["#4f46e5", "#7c3aed", "#2563eb", "#3b82f6", "#52647a"]
    names = ["Contact chair", "Same-color dining", "Curved sofa / wall", "Occluded shelf", "Thin rack / noise", "Combined study"]
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=180)
    left = np.zeros(6)
    for key, label, color in zip(keys, labels, colors):
        values = np.array([next(t for t in s["targets"] if t["object_id"] == s["target_id"])
                           ["fn_reason_fraction"][key] or 0 for s in result["scenes"]]) * 100
        ax.barh(names, values, left=left, label=label, color=color, height=.65)
        for i, value in enumerate(values):
            if value > 10:
                ax.text(left[i] + value / 2, i, f"{value:.1f}%", va="center", ha="center", color="white", fontsize=10)
        left += values
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of missed, scorable target surface area (%)")
    ax.set_title("The same omission metric hides different prior-assignment failures", loc="left", pad=18, weight="bold")
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.20), ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.19, right=.98, top=.87, bottom=.29)
    fig.text(.02, .035, "Six frozen S0 development meshes. Diagnostic replay only; B1 predictions and the reference matcher are unchanged.", fontsize=9, color="#52647a")
    outputs = []
    for suffix in ("png", "pdf"):
        path = output / f"ownership-failure-causes.{suffix}"
        fig.savefig(path, metadata={"CreationDate": None} if suffix == "pdf" else None)
        outputs.append(path)
    plt.close(fig)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (output / "figure-manifest.json").write_text(json.dumps({
        "results_sha256": sha(source), "generator_sha256": sha(Path(__file__)),
        "matplotlib": matplotlib.__version__, "outputs": {p.name: sha(p) for p in outputs},
        "scope": "decision-path attribution of existing B1 false-negative area; not a new method improvement"
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
