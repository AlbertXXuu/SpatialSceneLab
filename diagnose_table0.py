"""Local Table0 residual strata and shared-cell diagnostics, without exclusions. MIT.

The frozen plane is a comparison model, not measured truth. This writes numeric
evidence only; it neither corrects a pose nor changes the scoring population.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import audit_real_tabletop as baseline
from fit_stool import sha256, validate_output
from reproduce_real_tearoom import TARGET, verify_source
from scan_pipeline import load_frame


PARAMETERS = {"depth_edge_step_pixels": 2, "depth_edge_threshold_m": .05,
              "common_cell_m": .05, "common_cell_minimum_points": 3,
              "reference_frames": [4, 12], "tail_frames": [56, 57, 58]}


def depth_edges(depth, uv):
    """Fixed raw four-neighbour discontinuity; not a calibrated error label."""
    depth, uv = np.asarray(depth), np.asarray(uv)
    if (depth.ndim != 2 or not np.isfinite(depth).all() or uv.ndim != 2 or uv.shape[1] != 2
            or not np.issubdtype(uv.dtype, np.integer)):
        raise ValueError("Finite depth and integer Nx2 pixel coordinates required")
    h, w = depth.shape
    if min(h, w) < 1 or np.any(uv < 0) or np.any(uv >= [w, h]):
        raise ValueError("Pixels must remain inside the raw depth image")
    x, y = uv.T
    z = depth[y, x]
    step = PARAMETERS["depth_edge_step_pixels"]
    jumps = [np.abs(z - depth[np.clip(y + dy, 0, h - 1), np.clip(x + dx, 0, w - 1)])
             for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step))]
    return np.maximum.reduce(jumps) > PARAMETERS["depth_edge_threshold_m"]


def summary(mask, residual, depth, incidence, confidence, edge, membership_count):
    values = residual[mask]
    if not len(values):
        return {"count": 0}
    absolute = np.abs(values)
    return {"count": len(values), "signed_median_m": float(np.median(values)),
            "mean_m": float(absolute.mean()), "p95_m": float(np.quantile(absolute, .95)),
            "within10mm": float(np.mean(absolute <= baseline.PARAMETERS["support_distance_m"])),
            "depth_median_m": float(np.median(depth[mask])),
            "incidence_cosine_median": float(np.median(incidence[mask])),
            "confidence2_fraction": float(np.mean(confidence[mask] == 2)),
            "depth_edge_fraction": float(np.mean(edge[mask])),
            "ambiguous_prior_fraction": float(np.mean(membership_count[mask] > 1))}


def shared_cells(local, frames, band, residual):
    """Compare medians in the same spatial cells, without adjusting a plane."""
    cells = np.floor(local[:, [0, 2]] / PARAMETERS["common_cell_m"]).astype(np.int64)
    by_frame = {}
    for frame in np.unique(frames[band]):
        ids = np.flatnonzero(band & (frames == frame))
        by_frame[int(frame)] = {}
        for cell in np.unique(cells[ids], axis=0):
            selected = ids[np.all(cells[ids] == cell, axis=1)]
            if len(selected) >= PARAMETERS["common_cell_minimum_points"]:
                by_frame[int(frame)][tuple(cell)] = float(np.median(residual[selected]))
    result = []
    for reference in PARAMETERS["reference_frames"]:
        if reference not in by_frame:
            continue
        for frame in sorted(by_frame):
            if frame == reference:
                continue
            common = sorted(set(by_frame[reference]) & set(by_frame[frame]))
            delta = np.array([by_frame[frame][key] - by_frame[reference][key] for key in common])
            result.append({"reference": reference, "other": frame, "common_cells": len(common),
                           "signed_median_delta_m": float(np.median(delta)) if len(delta) else None,
                           "positive_fraction": float(np.mean(delta > 0)) if len(delta) else None})
    return result


def diagnose(scene_path, observations_path, output, source_manifest=None):
    paths = [Path(scene_path), Path(observations_path)]
    output = validate_output(output, paths)
    if output.exists():
        raise ValueError("Use a new output directory; retain previous evidence")
    code = [Path(__file__), Path(baseline.__file__), Path(load_frame.__code__.co_filename),
            Path(verify_source.__code__.co_filename), Path(__file__).with_name("fit_stool.py")]
    before = {p.name: sha256(p) for p in paths + code}
    scene, index, points, frames, candidates, local, band, train, valid = baseline.load_inputs(
        scene_path, observations_path, TARGET)
    source = Path(scene["source"]).resolve(strict=True)
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Diagnostic output must be separate from raw input")
    manifest = source_manifest or Path(__file__).parent / "docs/tea-room-source-manifest.json"
    source_check = verify_source(source, manifest)
    coefficients, _ = baseline.fit_plane(local, frames, train)
    normal = np.array([-coefficients[0], 1., -coefficients[1]])
    normal /= np.linalg.norm(normal)
    world_normal = np.asarray(scene["objects"][index]["transform_world"])[:3, :3] @ normal
    residual = (local @ normal - coefficients[2] / np.sqrt(1 + sum(coefficients[:2] ** 2)))
    with np.load(observations_path, allow_pickle=False) as data:
        uv = data["pixel_uv"]
    if uv.shape != (len(points), 2) or not np.issubdtype(uv.dtype, np.integer):
        raise ValueError("Original observations require one integer pixel_uv for every point")
    n = len(points)
    depth, incidence = np.zeros(n), np.zeros(n)
    confidence, edge = np.zeros(n, np.uint8), np.zeros(n, bool)
    for frame in scene["frames"]:
        ids = np.flatnonzero(band & (frames == frame["frame_index"]))
        if not len(ids):
            continue
        member = (source / frame["source"]).resolve(strict=True)
        if source not in member.parents:
            raise ValueError("Frame source escapes the raw scan")
        raw = load_frame(member)
        if (not np.allclose(raw["camera_to_world_cv"], frame["camera_to_world_cv"], atol=1e-9, rtol=0)
                or not np.allclose(raw["intrinsics_depth"], frame["intrinsics_depth"], atol=1e-9, rtol=0)):
            raise ValueError("Residual sensor diagnosis requires the original raw frame camera and intrinsics")
        pixels = uv[ids]
        edge[ids] = depth_edges(raw["depth_mm"].astype(float) / 1000., pixels)
        x, y = pixels.T
        depth[ids] = raw["depth_mm"][y, x] / 1000.
        confidence[ids] = raw["confidence"][y, x]
        rays = np.column_stack([pixels, np.ones(len(ids))]) @ np.linalg.inv(raw["intrinsics_depth"]).T
        world_rays = rays @ raw["camera_to_world_cv"][:3, :3].T
        incidence[ids] = np.abs(world_rays @ world_normal) / np.linalg.norm(world_rays, axis=1)
    membership_count = candidates.sum(axis=1)

    def stats(mask):
        return summary(mask, residual, depth, incidence, confidence, edge, membership_count)

    tail = valid & np.isin(frames, PARAMETERS["tail_frames"])
    legacy_p95 = float(np.quantile(np.abs(residual[valid]), .95))
    sensitivity = []
    for frame in PARAMETERS["reference_frames"]:
        if not np.any(train & (frames == frame)):
            continue
        alternative, _ = baseline.fit_plane(local, frames, train & (frames != frame))
        shift = (local[train, 0] * (alternative[0] - coefficients[0])
                 + local[train, 2] * (alternative[1] - coefficients[1]) + alternative[2] - coefficients[2])
        sensitivity.append({"left_out_training_frame": frame,
                            "training_prediction_change_p95_m": float(np.quantile(np.abs(shift), .95))})
    result = {"schema": "alvenx.table0-source-diagnosis.v1", "target": TARGET,
              "parameters": PARAMETERS, "source_check": source_check, "source_and_code_sha256": before,
              "training": stats(train), "all_original_validation": stats(valid),
              "frames": [{"frame": int(frame), "training": stats(train & (frames == frame)),
                          "validation": stats(valid & (frames == frame))} for frame in np.unique(frames[band])],
              "groups": {"confidence2": stats(valid & (confidence == 2)),
                         "unique_owned_prior": stats(valid & (membership_count == 1)),
                         "ambiguous_prior": stats(valid & (membership_count > 1)),
                         "depth_boundary": stats(valid & edge), "not_depth_boundary": stats(valid & ~edge)},
              "tail_attribution": {"frames": PARAMETERS["tail_frames"], "frame_observations": int(tail.sum()),
                  "above50mm_all": int(np.sum(valid & (np.abs(residual) > .05))),
                  "above50mm_in_tail_frames": int(np.sum(tail & (np.abs(residual) > .05))),
                  "above_original_p95_all": int(np.sum(valid & (np.abs(residual) > legacy_p95))),
                  "above_original_p95_in_tail_frames": int(np.sum(tail & (np.abs(residual) > legacy_p95)))},
              "shared_cells": shared_cells(local, frames, band, residual),
              "training_plane_sensitivity": sensitivity,
              "validation_indices_sha256": hashlib.sha256(np.flatnonzero(valid).astype("<i8").tobytes()).hexdigest(),
              "causal_identification": "unresolved: depth, pose, timing and scene motion are not independently observed",
              "scope": "Previously inspected development scan; strata preserve every original validation observation. "
                       "The comparison plane is not physical ground truth; boundary association is not a sensor-cause proof."}
    if before != {p.name: sha256(p) for p in paths + code} or source_check != verify_source(source, manifest):
        raise ValueError("Source or code changed during diagnosis")
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.scene, args.observations, args.output)
    print(json.dumps({"training": report["training"]["count"],
                      "original_validation": report["all_original_validation"]["count"],
                      "physical_cause_identified": False}))


if __name__ == "__main__":
    main()
