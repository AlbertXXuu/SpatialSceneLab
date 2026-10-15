"""Replay raw Table0 diagnosis, bounded pose control and safe output locally. MIT.

Uses prepared input and installed CPU dependencies. No acquisition, dependency
installation, service or Blender is needed. Exit 2 preserves a completed replay
whose geometric adoption gate failed; it is not a runtime success claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from fit_stool import sha256, validate_output
from reproduce_real_tearoom import ROOT, TARGET, run_stage, verify_source


CODE_FILES = ("reproduce_table0_diagnosis.py", "reproduce_real_tearoom.py", "scan_pipeline.py",
              "audit_real_tabletop.py", "audit_real_table_alignment.py", "balanced_table_pose.py",
              "audit_table0_inputs.py", "diagnose_table0.py", "reconstruct_mesh.py",
              "offline_run.py", "fit_stool.py")


def verify_exports(points, candidate):
    """Reject a changed default output or point/camera mismatch on a failed gate."""
    report = json.loads((candidate / "alignment.json").read_text(encoding="utf-8"))
    original = json.loads((points / "scene.json").read_text(encoding="utf-8"))
    canonical = json.loads((candidate / "scene.json").read_text(encoding="utf-8"))
    diagnostic = json.loads((candidate / "candidate-scene.json").read_text(encoding="utf-8"))
    if (diagnostic.get("schema") != "alvenx.real-table-alignment-candidate.v1" or
            diagnostic.get("observations") != "candidate-observations.npz" or
            canonical.get("observations") != "observations.npz"):
        raise ValueError("Candidate must have an explicit diagnostic schema")
    unchanged = (sha256(points / "observations.npz") == sha256(candidate / "observations.npz")
                 and original["frames"] == canonical["frames"])
    if report["passed"] is False and (not unchanged or
            canonical.get("alignment_control", {}).get("disposition") != "identity_fallback"):
        raise ValueError("Rejected alignment changed the default observation bundle")
    with np.load(points / "observations.npz", allow_pickle=False) as source:
        raw = {key: source[key] for key in source.files}
    with np.load(candidate / "candidate-observations.npz", allow_pickle=False) as source:
        corrected = {key: source[key] for key in source.files}
    if set(raw) != set(corrected) or any(not np.array_equal(raw[k], corrected[k]) for k in raw if k != "points"):
        raise ValueError("Candidate changed source identities, colors, pixels or denominator")
    before_frames = {row["frame_index"]: row for row in original["frames"]}
    after_frames = {row["frame_index"]: row for row in diagnostic["frames"]}
    if len(after_frames) != len(diagnostic["frames"]) or set(before_frames) != set(after_frames):
        raise ValueError("Candidate must preserve every original camera frame")
    maximum, changed = 0., 0
    for frame, metadata in before_frames.items():
        ids = raw["frame_index"] == frame
        first = np.asarray(metadata["camera_to_world_cv"])
        second = np.asarray(after_frames[frame]["camera_to_world_cv"])
        before_camera = np.linalg.solve(first[:3, :3], (raw["points"][ids] - first[:3, 3]).T).T
        after_camera = np.linalg.solve(second[:3, :3], (corrected["points"][ids] - second[:3, 3]).T).T
        if len(before_camera):
            maximum = max(maximum, float(np.max(np.abs(before_camera - after_camera))))
        changed += int(not np.array_equal(first, second))
    if not np.isfinite(maximum) or maximum > 1e-9:
        raise ValueError("Candidate points and cameras do not preserve the original camera rays/depth")
    return {"canonical_original_bundle_unchanged": unchanged,
            "candidate_camera_space_max_component_error_m": maximum,
            "candidate_changed_camera_count": changed, "source_point_count": len(raw["points"]),
            "candidate_adopted": report["passed"], "candidate_diagnostic_schema": diagnostic["schema"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    scan = args.input.resolve(strict=True)
    output = validate_output(args.output, [scan])
    if (not scan.is_dir() or output.exists() or scan in output.parents or output in scan.parents or
            output == ROOT or ROOT in output.parents and not (ROOT / ".local") in output.parents):
        raise ValueError("Use a new output tree separate from input; repository output belongs under .local")
    manifest = ROOT / "docs/tea-room-source-manifest.json"
    initial = verify_source(scan, manifest)
    fingerprints = {name: sha256(ROOT / name) for name in CODE_FILES}
    output.mkdir(parents=True)
    stages = []
    receipt = {"schema": "alvenx.table0-diagnosis-replay.v1", "completed": False,
               "source": initial, "source_code_sha256": fingerprints, "stages": stages,
               "runtime": "Prepared local input and CPU dependencies; no acquisition or installation",
               "failure": None}

    def guarded(name, script, arguments, allowed=(0,)):
        return run_stage(name, [sys.executable, "-B", str(ROOT / "offline_run.py"), "--script", script,
                         "--report", str(output / f"{name}-guard.json"), "--", *map(str, arguments)],
                         output, stages, allowed)

    try:
        points = output / "points"
        guarded("points", "scan_pipeline.py", ["reconstruct", "--input", scan, "--output", points,
                "--frame-step", 1, "--pixel-step", 2, "--min-confidence", 1, "--max-depth", 6,
                "--voxel-size", .015, "--box-margin", .025, "--structure-tolerance", .025])
        inputs = ["--scene", points / "scene.json", "--observations", points / "observations.npz"]
        guarded("input-contract", "audit_table0_inputs.py", ["--input", scan, *inputs,
                "--output", output / "input-contract"])
        guarded("residual-diagnosis", "diagnose_table0.py", [*inputs, "--output", output / "diagnosis"])
        guarded("baseline", "audit_real_tabletop.py", [*inputs, "--object", TARGET,
                "--output", output / "baseline"])
        common = [*inputs, "--object", TARGET, "--background-record", output / "baseline/background-icp.json"]
        for method in ("normal_translation", "balanced_pose"):
            protocol = output / f"{method}-protocol.json"
            guarded("freeze-" + method, "audit_real_table_alignment.py", [*common, "--method", method,
                    "--freeze-protocol", protocol])
            guarded(method, "audit_real_table_alignment.py", [*common, "--protocol", protocol,
                    "--output", output / method], allowed=(0, 2))
        receipt["exports"] = {method: verify_exports(points, output / method)
                              for method in ("normal_translation", "balanced_pose")}
        # Exercise the actual TSDF consumer: safe identity output must reproduce
        # the original mesh, while the diagnostic schema must be refused early.
        for name, scene in (("original-surface", points / "scene.json"),
                            ("fallback-surface", output / "balanced_pose/scene.json")):
            guarded(name, "reconstruct_mesh.py", ["--scene", scene, "--output", output / name])
        first = json.loads((output / "original-surface/surface-scene.json").read_text(encoding="utf-8"))
        second = json.loads((output / "fallback-surface/surface-scene.json").read_text(encoding="utf-8"))
        first_parts = {p["id"]: p["sha256"] for p in first["parts"]}
        second_parts = {p["id"]: p["sha256"] for p in second["parts"]}
        if (first_parts != second_parts or first["cameras"] != second["cameras"] or
                sha256(output / "original-surface/surface.ply") != sha256(output / "fallback-surface/surface.ply")):
            raise ValueError("Identity fallback changed the integrated surface or camera metadata")
        guarded("refuse-candidate-surface", "reconstruct_mesh.py", ["--scene", output / "balanced_pose/candidate-scene.json",
                "--output", output / "candidate-surface-must-not-exist"], allowed=(1,))
        if (output / "candidate-surface-must-not-exist").exists():
            raise ValueError("Rejected candidate created a surface output")
        receipt["surface_contract"] = {"original_and_fallback_mesh_bytes_equal": True,
            "all_part_hashes_equal": True, "part_count": len(first_parts), "camera_metadata_equal": True,
            "diagnostic_candidate_rejected_before_output": True}
        for stage in stages:
            guard = json.loads((output / f"{stage['name']}-guard.json").read_text(encoding="utf-8"))
            if not guard["dns_block_probe_passed"] or guard["runtime_denied_network_events"]:
                raise RuntimeError("A stage made an outbound Python network attempt")
        receipt["offline_guards"] = {"count": len(stages), "blocked_probes": len(stages),
            "runtime_outbound_events": 0, "scope": "CPython socket/DNS; not OS/native subprocess isolation"}
        if initial != verify_source(scan, manifest) or fingerprints != {name: sha256(ROOT / name) for name in CODE_FILES}:
            raise ValueError("Source or implementation changed during replay")
        alignment = json.loads((output / "balanced_pose/alignment.json").read_text(encoding="utf-8"))
        receipt["alignment_gate_passed"] = alignment["passed"]
        receipt["completed"] = True
    except BaseException as error:
        receipt["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "replay.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"completed": True, "alignment_gate_passed": receipt["alignment_gate_passed"],
                      "stages": len(stages)}))
    return 0 if receipt["alignment_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
