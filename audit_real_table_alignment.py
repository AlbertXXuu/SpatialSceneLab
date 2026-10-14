"""Frozen one-DOF target-aware real-table alignment control. MIT.

An estimate uses only unique-owned even-index later-view observations in training
support cells. Odd observations score it; every original validation observation
is additionally retained. This is a normal-offset control, not recovered camera
pose or hidden-shape accuracy. Freeze a protocol before running evaluation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

import audit_real_tabletop as baseline
from fit_stool import sha256, validate_output


PARAMETERS = {
    "index_period": 2, "estimation_remainder": 0, "score_remainder": 1,
    "training_cell_m": .05, "training_support_distance_m": .01,
    "minimum_estimation_points": 12, "minimum_common_cells": 3,
    "maximum_offset_m": .04, "maximum_mad_m": .006,
    "estimator_support_distance_m": .01, "minimum_estimator_support": .6,
    "minimum_frame_extent_m": .10,
    "target_score_support_m": .01, "minimum_target_score_support": .8,
    "minimum_validation_frames": 5,
    "control_voxel_m": .03, "control_normal_radius_m": .12,
    "control_normal_max_nn": 30, "control_match_distance_m": .06,
    "minimum_control_count": 100, "maximum_control_mean_worsening_m": .002,
    "maximum_control_p95_worsening_m": .005,
    "minimum_per_frame_control_count": 30,
}
LIMITATIONS = [
    "Previously inspected real scan: development evidence, not an untouched final test.",
    "RoomPlan candidate boxes and unique ownership are priors, not semantic ground truth.",
    "Only plane-normal translation is estimated; a single plane cannot establish tangential translation or yaw.",
    "Plane tilt rotations are deliberately not estimated. Normal offset does not identify depth bias, pose error or scene motion.",
    "Scoring observations are disjoint from later-view estimation samples but share the same frames and sensor.",
    "Neighbour/background nearest-training-plane residuals measure consistency, not physical accuracy.",
    "Raw scan, camera records and derived observation bundles stay local because their source has no redistribution license.",
]


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def fingerprints(scene_path, observations_path):
    return {"source_scene_sha256": sha256(scene_path),
            "source_observations_sha256": sha256(observations_path),
            "code_sha256": sha256(__file__),
            "baseline_code_sha256": sha256(baseline.__file__),
            "shared_helpers_sha256": sha256(Path(__file__).with_name("fit_stool.py"))}


def partition(valid, candidates, target_index):
    indices = np.arange(len(valid))
    estimate = valid & candidates[:, target_index] & (candidates.sum(axis=1) == 1)
    estimate &= indices % PARAMETERS["index_period"] == PARAMETERS["estimation_remainder"]
    score = valid & (indices % PARAMETERS["index_period"] == PARAMETERS["score_remainder"])
    if np.any(estimate & score) or not np.any(score):
        raise ValueError("Estimation and nonempty score masks must be disjoint")
    return estimate, score


def protocol_value(scene_path, observations_path, object_uuid, background_path):
    before = fingerprints(scene_path, observations_path)
    loaded = baseline.load_inputs(scene_path, observations_path, object_uuid)
    _, index, _, frames, candidates, local, _, train, valid = loaded
    coefficients, bounds = baseline.fit_plane(local, frames, train)
    estimate, score = partition(valid, candidates, index)
    if fingerprints(scene_path, observations_path) != before:
        raise ValueError("Source or input changed while freezing protocol")
    return {"schema": "alvenx.real-table-alignment-protocol.v1", "target": object_uuid,
            **before, "parameters": PARAMETERS, "baseline_parameters": baseline.PARAMETERS,
            "background_record_sha256": sha256(background_path),
            "training_plane_coefficients": coefficients.tolist(),
            "training_inlier_bounds_local": bounds,
            "counts": {"training": int(train.sum()), "original_validation": int(valid.sum()),
                       "estimation_candidates": int(estimate.sum()), "disjoint_score": int(score.sum())},
            "gate": "Target support AND every required neighbour/background consistency control. Missing or failed controls fail closed.",
            "limitations": LIMITATIONS}


def freeze(scene_path, observations_path, object_uuid, background_path, output):
    output = validate_output(output, [scene_path, observations_path, background_path])
    if output.exists():
        raise ValueError("Protocol must be a new file")
    value = protocol_value(scene_path, observations_path, object_uuid, background_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, value)
    return value


def signed_distances(points, coefficients):
    normal = np.array([-coefficients[0], 1., -coefficients[1]])
    return (points @ normal - coefficients[2]) / np.linalg.norm(normal)


def estimate_offset(points, cells, coefficients, world_rotation):
    """Only estimate samples enter this function; a rejection applies identity."""
    identity = np.eye(4)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all() or
            np.asarray(world_rotation).shape != (3, 3)):
        raise ValueError("Estimate needs finite Nx3 samples and a rigid world rotation")
    check_rotation = np.eye(4)
    check_rotation[:3, :3] = world_rotation
    baseline.rigid_transform(check_rotation, "World rotation")
    point_cells = np.floor(points[:, [0, 2]] / PARAMETERS["training_cell_m"]).astype(np.int64)
    shared = np.array([tuple(cell) in cells for cell in point_cells], dtype=bool)
    samples = points[shared]
    occupied = {tuple(cell) for cell in point_cells[shared]}
    record = {"estimation_candidate_count": len(points), "estimation_common_count": len(samples),
              "common_cell_count": len(occupied), "transform_applied": False,
              "estimated_transform": None, "applied_transform": identity.tolist()}
    if not len(samples):
        record.update(status="insufficient_support", rejected_checks=["nonempty_common_support"], checks={"nonempty_common_support": False})
        return identity, record
    residual = signed_distances(samples, coefficients)
    offset = float(np.median(residual))
    mad = float(np.median(np.abs(residual - offset)))
    fraction = float(np.mean(np.abs(residual - offset) <= PARAMETERS["estimator_support_distance_m"]))
    extent = np.ptp(samples[:, [0, 2]], axis=0)
    local_normal = np.array([-coefficients[0], 1., -coefficients[1]])
    local_normal /= np.linalg.norm(local_normal)
    transform = np.eye(4)
    transform[:3, 3] = -offset * (world_rotation @ local_normal)
    baseline.rigid_transform(transform, "Normal translation")
    checks = {"minimum_points": len(samples) >= PARAMETERS["minimum_estimation_points"],
              "minimum_cells": len(occupied) >= PARAMETERS["minimum_common_cells"],
              "bounded_normal_offset": abs(offset) <= PARAMETERS["maximum_offset_m"],
              "robust_mad": mad <= PARAMETERS["maximum_mad_m"],
              "estimator_support": fraction >= PARAMETERS["minimum_estimator_support"],
              "support_extent": float(max(extent)) >= PARAMETERS["minimum_frame_extent_m"]}
    accepted = all(checks.values())
    record.update(status="accepted" if accepted else "rejected", transform_applied=accepted,
                  signed_normal_offset_m=offset, mad_m=mad, estimator_within10mm=fraction,
                  support_extent_xz_m=extent.tolist(), estimated_transform=transform.tolist(),
                  applied_transform=(transform if accepted else identity).tolist(),
                  checks=checks, rejected_checks=[key for key, passed in checks.items() if not passed],
                  rotation_determinant=float(np.linalg.det(transform[:3, :3])),
                  translation_norm_m=float(np.linalg.norm(transform[:3, 3])))
    return (transform if accepted else identity), record


def estimate_transforms(scene, index, frames, candidates, local, train, valid, coefficients):
    estimate, score = partition(valid, candidates, index)
    training_support = local[train & (baseline.distances(local, coefficients) <= PARAMETERS["training_support_distance_m"])]
    cells = {tuple(cell) for cell in np.floor(training_support[:, [0, 2]] / PARAMETERS["training_cell_m"]).astype(np.int64)}
    rotation = np.asarray(scene["objects"][index]["transform_world"])[:3, :3]
    transforms, records = {}, []
    for frame in sorted(f["frame_index"] for f in scene["frames"] if f["frame_index"] > baseline.PARAMETERS["training_frame_max"]):
        transform, record = estimate_offset(local[estimate & (frames == frame)], cells, coefficients, rotation)
        record["frame_index"] = frame
        transforms[frame] = transform
        records.append(record)
    return transforms, records, estimate, score


def background_transforms(path, source_fingerprints):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("source_scene_sha256", "source_observations_sha256", "parameters"):
        expected = baseline.PARAMETERS if key == "parameters" else source_fingerprints[key]
        if data.get(key) != expected:
            raise ValueError("Frozen background control source or parameters differ")
    if data.get("schema") != "alvenx.real-table-background-control.v1":
        raise ValueError("Expected detailed frozen background registration record")
    transforms = {}
    for row in data["frames"]:
        frame = row["frame_index"]
        if type(frame) is not int or frame in transforms:
            raise ValueError("Background frame IDs must be unique integers")
        transforms[frame] = (baseline.rigid_transform(row["estimated_transform"], "Background")
                             if row["transform_applied"] else np.eye(4))
    return transforms


def residual_summary(values, frames):
    if len(values) == 0:
        return {"count": 0, "mean_m": None, "p95_m": None, "frames": []}
    return {"count": len(values), "mean_m": float(np.mean(values)), "p95_m": float(np.quantile(values, .95)),
            "frames": [{"frame": int(f), "count": int(np.sum(frames == f)),
                        "mean_m": float(np.mean(values[frames == f])),
                        "p95_m": float(np.quantile(values[frames == f], .95))} for f in np.unique(frames)]}


def consistency_controls(scene, points, frames, candidates, target_index, aligned):
    """Fixed original correspondences, normals and denominator; no rematching."""
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Full neighbour/background audit requires existing optional Open3D environment") from error
    odd = np.arange(len(points)) % PARAMETERS["index_period"] == PARAMETERS["score_remainder"]
    target_candidates = candidates[:, target_index]
    groups = [(f"neighbour_{i}", candidates[:, i] & ~target_candidates, obj["category"], obj["uuid"])
              for i, obj in enumerate(scene["objects"]) if i != target_index and
              np.any(candidates[:, i] & target_candidates)]
    groups.append(("background", baseline.background_mask(points, scene["objects"], candidates), "background", None))
    records = []
    for name, mask, category, uuid in groups:
        train = mask & (frames <= baseline.PARAMETERS["training_frame_max"])
        score = mask & odd & (frames > baseline.PARAMETERS["training_frame_max"])
        record = {"name": name, "category": category, "uuid": uuid, "training_count": int(train.sum()),
                  "candidate_score_count": int(score.sum()), "passed": False}
        if int(train.sum()) < 3 or not np.any(score):
            record.update(status="missing_support")
            records.append(record)
            continue
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[train]))
        cloud = cloud.voxel_down_sample(PARAMETERS["control_voxel_m"])
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=PARAMETERS["control_normal_radius_m"], max_nn=PARAMETERS["control_normal_max_nn"]))
        positions, normals = np.asarray(cloud.points), np.asarray(cloud.normals)
        index = o3d.geometry.KDTreeFlann(cloud)
        score_ids = np.flatnonzero(score)
        matched_ids, matched_train = [], []
        for original in score_ids:
            count, ids, squared = index.search_knn_vector_3d(points[original], 1)
            if count and squared[0] <= PARAMETERS["control_match_distance_m"] ** 2 and np.linalg.norm(normals[ids[0]]) > .5:
                matched_ids.append(int(original))
                matched_train.append(ids[0])
        ids = np.array(matched_ids, dtype=int)
        refs = np.array(matched_train, dtype=int)
        before = np.abs(np.einsum("ij,ij->i", points[ids] - positions[refs], normals[refs]))
        after = np.abs(np.einsum("ij,ij->i", aligned[ids] - positions[refs], normals[refs]))
        first, second = residual_summary(before, frames[ids]), residual_summary(after, frames[ids])
        enough = len(ids) >= PARAMETERS["minimum_control_count"]
        first_frames = {row["frame"]: row for row in first["frames"]}
        frame_checks = [{"frame": row["frame"], "count": row["count"],
                         "scorable": row["count"] >= PARAMETERS["minimum_per_frame_control_count"],
                         "mean_delta_m": row["mean_m"] - first_frames[row["frame"]]["mean_m"],
                         "p95_delta_m": row["p95_m"] - first_frames[row["frame"]]["p95_m"]}
                        for row in second["frames"]]
        for row in frame_checks:
            row["passed"] = (row["scorable"] and row["mean_delta_m"] <= PARAMETERS["maximum_control_mean_worsening_m"] and
                             row["p95_delta_m"] <= PARAMETERS["maximum_control_p95_worsening_m"])
        checks = {"enough_fixed_matches": enough,
                  "mean_not_materially_worse": bool(len(ids) and second["mean_m"] <= first["mean_m"] + PARAMETERS["maximum_control_mean_worsening_m"]),
                  "p95_not_materially_worse": bool(len(ids) and second["p95_m"] <= first["p95_m"] + PARAMETERS["maximum_control_p95_worsening_m"]),
                  "scorable_frames_not_materially_worse": any(row["scorable"] for row in frame_checks) and all(row["passed"] for row in frame_checks if row["scorable"])}
        record.update(status="scored", before=first, after=second, checks=checks, passed=all(checks.values()),
                      rejected_checks=[key for key, passed in checks.items() if not passed],
                      per_frame_changes=frame_checks,
                      fixed_correspondence_indices_sha256=hashlib.sha256(np.column_stack([ids, refs]).astype("<i8").tobytes()).hexdigest())
        records.append(record)
    return records, o3d.__version__


def evaluate(scene_path, observations_path, object_uuid, background_path, protocol_path, output):
    output = validate_output(output, [scene_path, observations_path, background_path, protocol_path])
    if output.exists():
        raise ValueError("Evaluation output must be a new directory")
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if protocol != protocol_value(scene_path, observations_path, object_uuid, background_path):
        raise ValueError("Frozen protocol, inputs or code changed; do not tune against scores")
    started = time.perf_counter()
    scene, index, points, frames, candidates, local, _, train, valid = baseline.load_inputs(scene_path, observations_path, object_uuid)
    coefficients = np.array(protocol["training_plane_coefficients"])
    transforms, records, estimate, score = estimate_transforms(scene, index, frames, candidates, local, train, valid, coefficients)
    prior_transforms = background_transforms(background_path, protocol)
    expected_frames = {f["frame_index"] for f in scene["frames"] if f["frame_index"] > baseline.PARAMETERS["training_frame_max"]}
    if set(prior_transforms) != expected_frames:
        raise ValueError("Background control must preserve every later frame")
    aligned = baseline.apply_transforms(points, frames, transforms)
    bg_aligned = baseline.apply_transforms(points, frames, prior_transforms)
    object_transform = np.asarray(scene["objects"][index]["transform_world"])
    aligned_local = baseline.local_points(aligned, object_transform)
    bg_local = baseline.local_points(bg_aligned, object_transform)
    comparisons = {}
    for name, mask in (("disjoint_score", score), ("all_original_validation", valid)):
        comparisons[name] = {"baseline": baseline.summarize(local[mask], frames[mask], coefficients),
                             "background_icp": baseline.summarize(bg_local[mask], frames[mask], coefficients),
                             "normal_translation": baseline.summarize(aligned_local[mask], frames[mask], coefficients)}
    controls, version = consistency_controls(scene, points, frames, candidates, index, aligned)
    score_stats = comparisons["disjoint_score"]["normal_translation"]
    original_stats = comparisons["all_original_validation"]["normal_translation"]
    original_target_passed = (original_stats["within10mm"] >= PARAMETERS["minimum_target_score_support"] and
                             len(original_stats["frames"]) >= PARAMETERS["minimum_validation_frames"])
    disjoint_target_passed = (score_stats["within10mm"] >= PARAMETERS["minimum_target_score_support"] and
                     len(score_stats["frames"]) >= PARAMETERS["minimum_validation_frames"])
    target_passed = original_target_passed and disjoint_target_passed
    controls_passed = bool(controls) and all(row["passed"] for row in controls)
    report = {"schema": "alvenx.real-table-alignment.v1", **fingerprints(scene_path, observations_path),
              "protocol_sha256": sha256(protocol_path), "parameters": PARAMETERS,
              "training_plane_coefficients": coefficients.tolist(),
              "target": object_uuid, "target_support_gate_passed": target_passed,
              "original_target_support_gate_passed": original_target_passed,
              "disjoint_target_support_gate_passed": disjoint_target_passed,
              "neighbour_background_gate_passed": controls_passed, "passed": target_passed and controls_passed,
              "counts": protocol["counts"], "comparisons": comparisons, "frames": records,
              "controls": controls, "estimated_degrees_of_freedom": ["plane_normal_translation"],
              "single_plane_unobservable_degrees_of_freedom": ["tangent_translation_1", "tangent_translation_2", "normal_axis_yaw"],
              "deliberately_unestimated_degrees_of_freedom": ["tilt_rotation_1", "tilt_rotation_2"],
              "estimation_scoring_overlap": int(np.sum(estimate & score)),
              "validation_denominator_unchanged": int(valid.sum()) == comparisons["all_original_validation"]["normal_translation"]["count"],
              "numpy_version": np.__version__, "open3d_version": version,
              "python_version": sys.version.split()[0], "runtime_seconds": time.perf_counter() - started,
              "limitations": LIMITATIONS}
    current = fingerprints(scene_path, observations_path)
    if (current != {k: protocol[k] for k in current} or
            sha256(background_path) != protocol["background_record_sha256"] or
            json.loads(Path(protocol_path).read_text(encoding="utf-8")) != protocol):
        raise ValueError("Inputs, frozen controls, protocol or code changed during evaluation")
    output.mkdir(parents=True, exist_ok=False)
    with np.load(observations_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["points"] = aligned
    np.savez_compressed(output / "observations.npz", **arrays)
    corrected_scene = copy.deepcopy(scene)
    for frame in corrected_scene["frames"]:
        if frame["frame_index"] in transforms and "camera_to_world_cv" in frame:
            frame["camera_to_world_cv"] = (transforms[frame["frame_index"]] @ np.asarray(frame["camera_to_world_cv"])).tolist()
    corrected_scene["alignment_control"] = {"schema": report["schema"], "passed": report["passed"],
                                           "protocol_sha256": report["protocol_sha256"], "is_camera_pose_estimate": False}
    write_json(output / "scene.json", corrected_scene)
    write_json(output / "alignment.json", report)
    write_json(output / "summary.json", public_summary(report))
    return report


def public_summary(report):
    """Aggregate numeric evidence without coordinates, cameras or transforms."""
    result = {key: copy.deepcopy(value) for key, value in report.items()
              if key not in ("training_plane_coefficients", "frames")}
    result["schema"] = "alvenx.real-table-alignment-summary.v1"
    result["frames"] = [{key: copy.deepcopy(value) for key, value in row.items()
                         if key not in ("estimated_transform", "applied_transform", "support_extent_xz_m")}
                        for row in report["frames"]]
    result["full_denominator_includes_estimation_samples"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--background-record", type=Path, required=True)
    parser.add_argument("--freeze-protocol", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze_protocol:
        if args.protocol or args.output:
            parser.error("Freeze protocol in a separate invocation before evaluation")
        report = freeze(args.scene, args.observations, args.object, args.background_record, args.freeze_protocol)
        print(json.dumps({"frozen": True, "counts": report["counts"]}))
        return 0
    if not args.protocol or not args.output:
        parser.error("Evaluation requires --protocol and --output")
    report = evaluate(args.scene, args.observations, args.object, args.background_record, args.protocol, args.output)
    print(json.dumps({key: report[key] for key in ("passed", "target_support_gate_passed", "neighbour_background_gate_passed", "counts")}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
