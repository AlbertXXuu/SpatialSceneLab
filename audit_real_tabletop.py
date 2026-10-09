"""Reproduce the frozen real-table plane failure and background-only ICP control. MIT.

Writes two detailed local diagnostic JSON files and one aggregate comparison.
Original points, frames, candidate membership, and the training plane stay fixed;
failed registrations retain identity. Only comparison.json is intended for publication.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from fit_stool import sha256, validate_output


PARAMETERS = {
    "training_frame_max": 30, "top_band_below_m": .15, "top_band_above_m": .05,
    "plane_seed": 20261009, "plane_draws": 1000, "plane_max_tilt_degrees": 10,
    "plane_inlier_m": .01, "plane_refinements": 5,
    "background_box_margin_m": .08, "background_voxel_m": .03,
    "normal_radius_m": .12, "normal_max_nn": 30, "tukey_m": .03,
    "correspondence_m": .06, "icp_iterations": 60,
    "minimum_correspondences": 150, "minimum_fitness": .5,
    "minimum_eigenvalue_ratio": 1e-4, "maximum_translation_m": .08,
    "maximum_rotation_degrees": 3, "support_distance_m": .01,
    "support_fraction": .8, "support_frame_count": 5,
    "common_cell_m": .05, "common_cell_minimum_points": 3,
    "common_cell_reference_frame": 4, "common_cell_other_frames": [46, 47, 48, 49],
}
LIMITATIONS = (
    "Candidate membership is a RoomPlan box prior, not instance ground truth. "
    "Frame and registration residuals are diagnostic consistency measurements, "
    "not calibrated correctness or independent physical accuracy. No complete "
    "table or unobserved geometry is recovered by this audit."
)


def rigid_transform(value, label):
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{label} must be a finite 4x4 rigid transform")
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6, rtol=0) or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0) or
            not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError(f"{label} must be rigid without reflection or scale")
    return transform


def local_points(points, transform):
    return (points - transform[:3, 3]) @ transform[:3, :3]


def load_inputs(scene_path, observations_path, object_uuid):
    scene = json.loads(Path(scene_path).read_text(encoding="utf-8"))
    if not isinstance(scene, dict) or scene.get("units") != "m" or scene.get("up_axis") != "Y":
        raise ValueError("Expected Y-up metre observations")
    objects = scene.get("objects")
    if not isinstance(objects, list) or not objects or any(not isinstance(o, dict) for o in objects):
        raise ValueError("Expected nonempty object records")
    for obj in objects:
        try:
            dimensions = np.asarray(obj["dimensions_m"], dtype=float)
            rigid_transform(obj["transform_world"], "Object")
        except (KeyError, TypeError) as error:
            raise ValueError("Every object needs numeric dimensions and a rigid transform") from error
        if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
            raise ValueError("Object dimensions must be three finite positive values")
    matches = [i for i, obj in enumerate(objects) if obj.get("uuid") == object_uuid]
    if len(matches) != 1 or objects[matches[0]].get("category") != "Table":
        raise ValueError("Target UUID must identify exactly one Table")
    index = matches[0]
    transform = np.asarray(objects[index]["transform_world"], dtype=float)
    if not np.allclose(transform[:3, 1], [0, 1, 0], atol=1e-6, rtol=0):
        raise ValueError("Target Table must have an upright local Y axis")
    records = scene.get("frames")
    if not isinstance(records, list) or not records or any(not isinstance(f, dict) for f in records):
        raise ValueError("Expected frame records")
    ids = [f.get("frame_index") for f in records]
    if any(type(f) is not int or f < 0 for f in ids) or len(set(ids)) != len(ids):
        raise ValueError("Frame IDs must be unique nonnegative integers")
    for frame in records:
        if "camera_to_world_cv" in frame:
            rigid_transform(frame["camera_to_world_cv"], "Camera")
    with np.load(observations_path, allow_pickle=False) as data:
        if not {"points", "frame_index", "candidates"}.issubset(data.files):
            raise ValueError("Observations require points, frame_index and candidates")
        points, frames, candidates = (data[k] for k in ("points", "frame_index", "candidates"))
    if (points.ndim != 2 or points.shape[1] != 3 or
            not np.issubdtype(points.dtype, np.number) or np.iscomplexobj(points) or
            not np.isfinite(points).all()):
        raise ValueError("Points must be finite numeric Nx3 coordinates")
    n = len(points)
    if (frames.shape != (n,) or not np.issubdtype(frames.dtype, np.integer) or
            not np.isin(frames, ids).all()):
        raise ValueError("Observation frame IDs must be declared integer frame IDs")
    if candidates.shape != (n, len(objects)) or candidates.dtype != np.bool_:
        raise ValueError("Candidates must be a boolean array with one column per object")
    points = points.astype(float, copy=False)
    local = local_points(points, transform)
    top = float(objects[index]["dimensions_m"][1]) / 2
    band = candidates[:, index] & (local[:, 1] >= top - PARAMETERS["top_band_below_m"])
    band &= local[:, 1] <= top + PARAMETERS["top_band_above_m"]
    train = band & (frames <= PARAMETERS["training_frame_max"])
    valid = band & (frames > PARAMETERS["training_frame_max"])
    if np.sum(train) < 3 or not valid.any() or len(np.unique(frames[train])) < 2:
        raise ValueError("Need nonempty validation and at least three training points from two frames")
    design = np.column_stack([local[train, 0], local[train, 2], np.ones(np.sum(train))])
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("Training plane support is geometrically degenerate")
    return scene, index, points, frames, candidates, local, band, train, valid


def distances(points, coefficients):
    return np.abs(points[:, 0] * coefficients[0] + points[:, 2] * coefficients[1]
                  + coefficients[2] - points[:, 1]) / np.sqrt(1 + sum(coefficients[:2] ** 2))


def summarize(points, frames, coefficients):
    residual = distances(points, coefficients)
    if not len(residual):
        raise ValueError("Cannot summarize an empty validation denominator")
    return {"count": len(points), "mean_m": float(residual.mean()),
            "p95_m": float(np.quantile(residual, .95)),
            "within10mm": float(np.mean(residual <= PARAMETERS["support_distance_m"])),
            "frames": [{"frame": int(f), "count": int(np.sum(frames == f)),
                        "within10mm": float(np.mean(residual[frames == f] <= PARAMETERS["support_distance_m"])),
                        "median_m": float(np.median(residual[frames == f])),
                        "mean_m": float(np.mean(residual[frames == f]))} for f in np.unique(frames)]}


def support_gate(summary):
    return (summary["within10mm"] >= PARAMETERS["support_fraction"] and
            len(summary["frames"]) >= PARAMETERS["support_frame_count"])


def fit_plane(local, frames, train):
    """The frozen frame-balanced triplet fit; no validation observations read."""
    points, selected_frames = local[train], frames[train]
    design = np.column_stack([points[:, 0], points[:, 2], np.ones(len(points))])
    unique, counts = np.unique(selected_frames, return_counts=True)
    weight = np.array([1 / counts[np.searchsorted(unique, f)] for f in selected_frames])
    rng = np.random.default_rng(PARAMETERS["plane_seed"])
    best = None
    for _ in range(PARAMETERS["plane_draws"]):
        ids = rng.choice(len(points), 3, replace=False)
        if len(np.unique(selected_frames[ids])) < 2:
            continue
        try:
            coefficients = np.linalg.solve(design[ids], points[ids, 1])
        except np.linalg.LinAlgError:
            continue
        if np.linalg.norm(coefficients[:2]) > np.tan(np.deg2rad(PARAMETERS["plane_max_tilt_degrees"])):
            continue
        residual = np.abs(design @ coefficients - points[:, 1]) / np.sqrt(1 + sum(coefficients[:2] ** 2))
        score = np.sum(weight * (residual <= PARAMETERS["plane_inlier_m"]))
        if best is None or score > best[0]:
            best = (score, coefficients)
    if best is None:
        raise ValueError("No nondegenerate near-horizontal training plane was found")
    coefficients = best[1]
    for _ in range(PARAMETERS["plane_refinements"]):
        residual = np.abs(design @ coefficients - points[:, 1]) / np.sqrt(1 + sum(coefficients[:2] ** 2))
        keep = residual <= PARAMETERS["plane_inlier_m"]
        if np.sum(keep) < 3 or np.linalg.matrix_rank(design[keep]) < 3:
            raise ValueError("Training inlier geometry is insufficient or degenerate")
        weight_sqrt = np.sqrt(weight[keep])
        coefficients = np.linalg.lstsq(design[keep] * weight_sqrt[:, None],
                                       points[keep, 1] * weight_sqrt, rcond=None)[0]
    return coefficients, np.quantile(points[keep][:, [0, 2]], [.01, .99], axis=0).tolist()


def background_mask(points, objects, candidates):
    mask = ~candidates.any(axis=1)
    for obj in objects:
        local = local_points(points, np.asarray(obj["transform_world"]))
        mask &= ~np.all(np.abs(local) <= np.asarray(obj["dimensions_m"]) / 2
                        + PARAMETERS["background_box_margin_m"], axis=1)
    return mask


def cloud(points, o3d):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(
        PARAMETERS["background_voxel_m"])


def register_background(source, target, o3d):
    """A rejected or failed estimate returns identity, retaining its diagnostics."""
    identity = np.eye(4)
    record = {"source_background_count": len(source.points), "transform_applied": False,
              "estimated_transform": None, "checks": {}}
    if len(source.points) < 3 or len(target.points) < 3:
        record.update(status="insufficient_geometry", failure="Fewer than three background points")
        return identity, record
    reg = o3d.pipelines.registration
    try:
        before = reg.evaluate_registration(source, target, PARAMETERS["correspondence_m"], identity)
        estimate = reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=PARAMETERS["tukey_m"]))
        outcome = reg.registration_icp(source, target, PARAMETERS["correspondence_m"], identity, estimate,
            reg.ICPConvergenceCriteria(max_iteration=PARAMETERS["icp_iterations"]))
        transform = rigid_transform(outcome.transformation, "Estimated registration")
        angle = float(np.degrees(np.arccos(np.clip((np.trace(transform[:3, :3]) - 1) / 2, -1, 1))))
        delta = float(np.linalg.norm(transform[:3, 3]))
        pairs = np.asarray(outcome.correspondence_set)
        eigenvalues = np.zeros(6)
        if len(pairs):
            aligned = np.asarray(source.points)[pairs[:, 0]] @ transform[:3, :3].T + transform[:3, 3]
            normals = np.asarray(target.normals)[pairs[:, 1]]
            jacobian = np.column_stack([np.cross(aligned, normals), normals])
            eigenvalues = np.linalg.eigvalsh(jacobian.T @ jacobian / len(jacobian))
        ratio = float(max(eigenvalues[0], 0) / max(eigenvalues[-1], 1e-12))
        checks = {"enough_background_points": len(pairs) >= PARAMETERS["minimum_correspondences"],
                  "background_fitness": outcome.fitness >= PARAMETERS["minimum_fitness"],
                  "pose_observability": ratio >= PARAMETERS["minimum_eigenvalue_ratio"],
                  "bounded_translation": delta <= PARAMETERS["maximum_translation_m"],
                  "bounded_rotation": angle <= PARAMETERS["maximum_rotation_degrees"],
                  "rmse_not_worse": outcome.inlier_rmse <= before.inlier_rmse}
        accepted = all(checks.values())
        record.update(status="accepted" if accepted else "rejected", transform_applied=accepted,
                      fitness_before=float(before.fitness), fitness_after=float(outcome.fitness),
                      rmse_before_m=float(before.inlier_rmse), rmse_after_m=float(outcome.inlier_rmse),
                      matched_count=len(pairs), normal_jacobian_eigenvalues=eigenvalues.tolist(),
                      eigenvalue_ratio=ratio, translation_norm_m=delta, rotation_degrees=angle,
                      estimated_transform=transform.tolist(), checks=checks)
        return transform if accepted else identity, record
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        record.update(status="registration_error", failure=f"{type(error).__name__}: {error}")
        return identity, record


def apply_transforms(points, frames, transforms):
    changed = points.copy()
    for frame, transform in transforms.items():
        mask = frames == frame
        changed[mask] = points[mask] @ transform[:3, :3].T + transform[:3, 3]
    return changed


def background_control(scene, points, frames, candidates, valid, transform, coefficients):
    import open3d as o3d

    background = background_mask(points, scene["objects"], candidates)
    target = cloud(points[background & (frames <= PARAMETERS["training_frame_max"])], o3d)
    if len(target.points) >= 3:
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=PARAMETERS["normal_radius_m"], max_nn=PARAMETERS["normal_max_nn"]))
    transforms, records = {}, []
    for frame in sorted(f["frame_index"] for f in scene["frames"]
                        if f["frame_index"] > PARAMETERS["training_frame_max"]):
        source = cloud(points[background & (frames == frame)], o3d)
        transforms[frame], record = register_background(source, target, o3d)
        record["frame_index"] = frame
        records.append(record)
    # The original mask is never recomputed after registration, even on failures.
    aligned = apply_transforms(points[valid], frames[valid], transforms)
    summary = summarize(local_points(aligned, transform), frames[valid], coefficients)
    return {"schema": "alvenx.real-table-background-control.v1",
            "method": "background-only point-to-plane ICP; fixed training plane and validation indices",
            "target_background_count": len(target.points), "frames": records, "validation": summary,
            "support_gate_passed": support_gate(summary), "passed": support_gate(summary),
            "open3d_version": o3d.__version__}


def common_cells(local, frames, band):
    def heights(frame):
        points = local[band & (frames == frame)]
        cells = np.floor(points[:, [0, 2]] / PARAMETERS["common_cell_m"]).astype(np.int64)
        result = {}
        for cell in np.unique(cells, axis=0):
            mask = np.all(cells == cell, axis=1)
            if np.sum(mask) >= PARAMETERS["common_cell_minimum_points"]:
                result[tuple(cell)] = float(np.median(points[mask, 1]))
        return result
    reference = heights(PARAMETERS["common_cell_reference_frame"])
    rows = []
    for frame in PARAMETERS["common_cell_other_frames"]:
        other = heights(frame)
        shared = sorted(set(reference) & set(other))
        delta = np.array([other[cell] - reference[cell] for cell in shared])
        rows.append({"reference_frame": PARAMETERS["common_cell_reference_frame"], "other_frame": frame,
                     "common_cell_count": len(shared),
                     "median_height_delta_m": float(np.median(delta)) if len(delta) else None,
                     "p10_p90_height_delta_m": np.quantile(delta, [.1, .9]).tolist() if len(delta) else [],
                     "positive_delta_fraction": float(np.mean(delta > 0)) if len(delta) else None})
    return rows


def run(scene_path, observations_path, object_uuid, output):
    output = validate_output(output, [scene_path, observations_path])
    if output.exists():
        raise ValueError("Output must be a new directory; existing directories are never overwritten")
    scene, index, points, frames, candidates, local, band, train, valid = load_inputs(
        scene_path, observations_path, object_uuid)
    coefficients, bounds = fit_plane(local, frames, train)
    target = scene["objects"][index]
    baseline = summarize(local[valid], frames[valid], coefficients)
    plane = {"schema": "alvenx.real-table-plane-fit.v1", "target": object_uuid,
             "candidate_count": int(np.sum(candidates[:, index])), "band_count": int(np.sum(band)),
             "coefficients": coefficients.tolist(), "inlier_bounds_local": bounds,
             "tilt_degrees": float(np.rad2deg(np.arctan(np.linalg.norm(coefficients[:2])))),
             "training": summarize(local[train], frames[train], coefficients), "validation": baseline,
             "prior_validation": summarize(local[valid], frames[valid],
                 np.array([0, 0, target["dimensions_m"][1] / 2])),
             "passed": support_gate(baseline)}
    registered = background_control(scene, points, frames, candidates, valid,
                                    np.asarray(target["transform_world"]), coefficients)
    corrected = registered["validation"]
    before_frames = {r["frame"]: r for r in baseline["frames"]}
    comparison = {"schema": "alvenx.real-table-comparison.v1", "passed": registered["passed"],
                  "plane_support_gate_passed": plane["passed"],
                  "background_support_gate_passed": registered["passed"],
                  "validation_denominator_unchanged": corrected["count"] == baseline["count"],
                  "validation_count": baseline["count"], "prior_validation": plane["prior_validation"],
                  "plane_validation": baseline,
                  "background_validation": corrected, "common_cells": common_cells(local, frames, band),
                  "target_background_count": registered["target_background_count"],
                  "open3d_version": registered["open3d_version"],
                  "registration_frames": [{
                      **{key: row[key] for key in ("frame_index", "status", "source_background_count",
                          "transform_applied", "fitness_before", "fitness_after", "rmse_before_m",
                          "rmse_after_m", "matched_count", "eigenvalue_ratio", "translation_norm_m",
                          "rotation_degrees", "failure") if key in row},
                      "rejected_checks": [key for key, passed in row["checks"].items() if not passed]}
                      for row in registered["frames"]],
                  "per_frame_changes": [{"frame": row["frame"], "count": row["count"],
                      "mean_delta_m": row["mean_m"] - before_frames[row["frame"]]["mean_m"],
                      "within10mm_delta": row["within10mm"] - before_frames[row["frame"]]["within10mm"]}
                      for row in corrected["frames"]]}
    provenance = {"source_scene_sha256": sha256(scene_path),
                  "source_observations_sha256": sha256(observations_path),
                  "code_sha256": sha256(__file__),
                  "shared_helpers_sha256": sha256(Path(__file__).with_name("fit_stool.py")),
                  "parameters": PARAMETERS, "numpy_version": np.__version__,
                  "python_version": sys.version.split()[0], "limitations": LIMITATIONS,
                  "denominator": "Every original validation-band observation; repeated frames retained; failed ICP uses identity",
                  "background_exclusion": "All original object OBBs expanded by 80 mm on each local axis and all object candidates; candidate exclusion is redundant on the frozen scan"}
    output.mkdir(parents=True, exist_ok=False)
    for name, report in [("plane-fit.json", plane), ("background-icp.json", registered),
                         ("comparison.json", comparison)]:
        report.update(provenance)
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--object", required=True, help="UUID of the measured Table")
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for two detailed local diagnostic JSON files and one aggregate comparison")
    args = parser.parse_args()
    report = run(args.scene, args.observations, args.object, args.output)
    print(json.dumps({"passed": report["passed"], "validation_count": report["validation_count"],
                      "plane_within10mm": report["plane_validation"]["within10mm"],
                      "background_within10mm": report["background_validation"]["within10mm"]}))


if __name__ == "__main__":
    main()
