"""Fixed-correspondence, group-balanced real-table pose control. MIT.

The caller freezes the training plane and the original target-quality records.
Only training geometry and even-index later observations enter the fit. This
module returns bounded candidate rigid motions, not an acceptance verdict or a
physical explanation of the scan. The caller scores independent observations,
preserves original inputs, and applies accepted motions to points AND cameras.
"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np

import audit_real_tabletop as baseline


PARAMETERS = {
    "huber_m": .01, "irls_iterations": 6,
    "maximum_translation_m": .08, "maximum_rotation_degrees": 3.,
    "minimum_eigenvalue_ratio": 1e-4, "minimum_rank": 6,
    "training_cell_m": .05, "training_support_distance_m": .01,
    "control_voxel_m": .03, "control_normal_radius_m": .12,
    "control_normal_max_nn": 30, "control_match_distance_m": .06,
    "minimum_normal_norm": .5, "estimation_index_period": 2,
    "estimation_index_remainder": 0,
    "group_weight": "Each nonempty group has total weight one before Huber weighting",
    "motion_model": "Fixed linear Jacobian, min-norm IRLS; Rodrigues rotation about original camera centre",
    "frame_policy": "Only frames accepted by the original target-normal quality checks may change",
}


def _indices_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def _rotation(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1e-15:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def _huber(residual):
    threshold = PARAMETERS["huber_m"]
    absolute = np.abs(residual)
    return np.where(absolute <= threshold, .5 * residual ** 2,
                    threshold * (absolute - .5 * threshold))


def _solve_pose(blocks, origin):
    """Solve one fixed linearization; never rematch or inspect score points."""
    jacobians, residuals, weights, groups, indices = [], [], [], [], []
    for block in blocks:
        count = len(block["ids"])
        if not count:
            continue
        normals = block["normals"]
        jacobians.append(np.column_stack([
            np.cross(block["positions"] - origin, normals), normals]))
        residuals.append(block["residuals"])
        weights.append(np.full(count, 1 / count))
        indices.append(block["ids"])
        groups.append({"name": block["name"], "count": count, "group_total_weight": 1.,
                       "estimation_indices_sha256": _indices_hash(block["ids"])})
    if not jacobians:
        raise ValueError("A pose solve needs nonempty target estimation support")
    jacobian = np.vstack(jacobians)
    residual = np.concatenate(residuals)
    base_weight = np.concatenate(weights)
    delta = np.zeros(6)
    iterations = []
    for step in range(PARAMETERS["irls_iterations"]):
        current = residual + jacobian @ delta
        robust = np.minimum(1., PARAMETERS["huber_m"] / np.maximum(np.abs(current), 1e-15))
        square_root = np.sqrt(base_weight * robust)
        weighted = jacobian * square_root[:, None]
        delta, _, rank, _ = np.linalg.lstsq(weighted, -residual * square_root, rcond=None)
        iterations.append({"iteration": step + 1, "rank": int(rank),
                           "rotation_degrees": float(np.degrees(np.linalg.norm(delta[:3]))),
                           "translation_m": float(np.linalg.norm(delta[3:])),
                           "group_balanced_huber_objective_m2": float(
                               np.dot(base_weight, _huber(residual + jacobian @ delta)))})
    eigenvalues = np.linalg.eigvalsh(weighted.T @ weighted)
    ratio = float(max(eigenvalues[0], 0) / max(eigenvalues[-1], 1e-15))
    checks = {
        "finite": bool(np.isfinite(delta).all()),
        "full_rank": int(rank) == PARAMETERS["minimum_rank"],
        "pose_observable": ratio >= PARAMETERS["minimum_eigenvalue_ratio"],
        "bounded_translation": float(np.linalg.norm(delta[3:])) <= PARAMETERS["maximum_translation_m"],
        "bounded_rotation": float(np.degrees(np.linalg.norm(delta[:3]))) <= PARAMETERS["maximum_rotation_degrees"],
    }
    transform = np.eye(4)
    transform[:3, :3] = _rotation(delta[:3])
    transform[:3, 3] = origin - transform[:3, :3] @ origin + delta[3:]
    baseline.rigid_transform(transform, "Balanced candidate")
    accepted = all(checks.values())
    record = {"status": "accepted" if accepted else "rejected", "transform_applied": accepted,
              "groups": groups, "total_estimation_count": len(residual),
              "estimation_indices_sha256": _indices_hash(np.concatenate(indices)),
              "checks": checks, "rejected_checks": [k for k, passed in checks.items() if not passed],
              "rank": int(rank), "eigenvalues": eigenvalues.tolist(), "eigenvalue_ratio": ratio,
              "delta_rotation_radians": delta[:3].tolist(), "delta_translation_m": delta[3:].tolist(),
              "iterations": iterations, "estimated_transform": transform.tolist(),
              "applied_transform": (transform if accepted else np.eye(4)).tolist(),
              "original_group_balanced_huber_objective_m2": float(np.dot(base_weight, _huber(residual))),
              "linear_estimated_group_balanced_huber_objective_m2": float(
                  np.dot(base_weight, _huber(residual + jacobian @ delta)))}
    return transform if accepted else np.eye(4), record


def _independent_groups(scene, target_index, points, frames, candidates, active_frames):
    """Freeze matches against training geometry before estimating any motion."""
    training = frames <= baseline.PARAMETERS["training_frame_max"]
    even = np.arange(len(points)) % PARAMETERS["estimation_index_period"] == PARAMETERS["estimation_index_remainder"]
    discover = training | even  # Later odd membership cannot introduce a fitted group.
    target = candidates[:, target_index]
    masks = [(f"neighbour_{i}", candidates[:, i] & ~target)
             for i in range(len(scene["objects"])) if i != target_index and
             np.any(candidates[discover, i] & target[discover])]
    masks.append(("background", baseline.background_mask(points, scene["objects"], candidates)))
    groups = []
    for name, mask in masks:
        train = mask & training
        source_ids = np.flatnonzero(mask & even & np.isin(frames, list(active_frames)))
        if int(train.sum()) < 3 or not len(source_ids):
            continue
        try:
            import open3d as o3d
        except ImportError as error:
            raise RuntimeError("Balanced pose control requires the existing optional Open3D environment") from error
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[train]))
        cloud = cloud.voxel_down_sample(PARAMETERS["control_voxel_m"])
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=PARAMETERS["control_normal_radius_m"], max_nn=PARAMETERS["control_normal_max_nn"]))
        positions, normals = np.asarray(cloud.points), np.asarray(cloud.normals)
        tree = o3d.geometry.KDTreeFlann(cloud)
        matched_ids, matched_train = [], []
        for source in source_ids:
            count, ids, squared = tree.search_knn_vector_3d(points[source], 1)
            if (count and squared[0] <= PARAMETERS["control_match_distance_m"] ** 2 and
                    np.linalg.norm(normals[ids[0]]) > PARAMETERS["minimum_normal_norm"]):
                matched_ids.append(int(source))
                matched_train.append(ids[0])
        ids, references = np.asarray(matched_ids, dtype=int), np.asarray(matched_train, dtype=int)
        groups.append({"name": name, "ids": ids, "positions": points[ids],
                       "reference_positions": positions[references], "normals": normals[references],
                       "residuals": np.einsum("ij,ij->i", points[ids] - positions[references], normals[references])})
    return groups


def estimate_transforms(scene, target_index, points, frames, candidates, local,
                        coefficients, estimate_mask, normal_records):
    """Return one rigid transform/record per declared later frame, failing closed.

    Rejected original target estimates remain identity. Missing independent
    geometry is never declared verified here; the caller retains its explicit
    neighbour/background score gates. Original inputs are not mutated.
    """
    points, frames, candidates, local, coefficients, estimate_mask = (
        np.asarray(value) for value in (points, frames, candidates, local, coefficients, estimate_mask))
    objects, frame_records = scene["objects"], scene["frames"]
    if (type(target_index) is not int or not 0 <= target_index < len(objects) or
            points.ndim != 2 or points.shape[1:] != (3,) or local.shape != points.shape or
            not np.issubdtype(points.dtype, np.number) or np.iscomplexobj(points) or
            not np.isfinite(points).all() or not np.issubdtype(local.dtype, np.number) or
            np.iscomplexobj(local) or not np.isfinite(local).all() or
            frames.shape != (len(points),) or not np.issubdtype(frames.dtype, np.integer) or
            candidates.shape != (len(points), len(objects)) or candidates.dtype != np.bool_ or
            estimate_mask.shape != (len(points),) or estimate_mask.dtype != np.bool_ or
            coefficients.shape != (3,) or not np.issubdtype(coefficients.dtype, np.number) or
            np.iscomplexobj(coefficients) or not np.isfinite(coefficients).all()):
        raise ValueError("Balanced pose inputs need finite points/plane, integer frames and boolean membership masks")
    frame_ids = [row["frame_index"] for row in frame_records]
    if (any(type(frame) is not int or frame < 0 for frame in frame_ids) or
            len(set(frame_ids)) != len(frame_ids) or not np.isin(frames, frame_ids).all()):
        raise ValueError("Observation frames must have unique declared integer frame records")
    frame_lookup = {row["frame_index"]: row for row in frame_records}
    training_max = baseline.PARAMETERS["training_frame_max"]
    later = {frame for frame in frame_ids if frame > training_max}
    if (not isinstance(normal_records, list) or any(not isinstance(row, dict) for row in normal_records) or
            len(normal_records) != len(later) or
            any(type(row.get("frame_index")) is not int or type(row.get("transform_applied")) is not bool
                for row in normal_records) or {row["frame_index"] for row in normal_records} != later):
        raise ValueError("Original normal quality records must preserve every later frame exactly once")
    rotation = baseline.rigid_transform(objects[target_index]["transform_world"], "Target")[:3, :3]
    if np.any(estimate_mask & ((frames <= training_max) | (np.arange(len(points)) % 2 != 0) |
                              ~candidates[:, target_index] | (candidates.sum(axis=1) != 1))):
        raise ValueError("Target estimation mask must contain only unique-owned even later observations")
    active = {row["frame_index"] for row in normal_records if row["transform_applied"]}
    for row in normal_records:
        if row["transform_applied"] and (not row.get("checks") or not all(value is True for value in row["checks"].values())):
            raise ValueError("Accepted original estimates require passed explicit quality checks")
    dimensions = np.asarray(objects[target_index]["dimensions_m"], dtype=float)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError("Target dimensions must be finite and positive")
    band = candidates[:, target_index] & (local[:, 1] >= dimensions[1] / 2 - baseline.PARAMETERS["top_band_below_m"])
    band &= local[:, 1] <= dimensions[1] / 2 + baseline.PARAMETERS["top_band_above_m"]
    support = local[band & (frames <= training_max) &
                    (baseline.distances(local, coefficients) <= PARAMETERS["training_support_distance_m"])]
    cells = {tuple(cell) for cell in np.floor(support[:, [0, 2]] / PARAMETERS["training_cell_m"]).astype(np.int64)}
    ids = np.flatnonzero(estimate_mask)
    common = [tuple(cell) in cells for cell in np.floor(local[ids][:, [0, 2]] / PARAMETERS["training_cell_m"]).astype(np.int64)]
    target_ids = ids[np.asarray(common, dtype=bool)]
    normal_raw = np.array([-coefficients[0], 1., -coefficients[1]])
    normal_length = np.linalg.norm(normal_raw)
    normal_local = normal_raw / normal_length
    normal_world = rotation @ normal_local
    independent = _independent_groups(scene, target_index, points, frames, candidates, active) if active else []
    transforms, records = {}, []
    for original in sorted(normal_records, key=lambda row: row["frame_index"]):
        frame = original["frame_index"]
        quality = {key: copy.deepcopy(original[key]) for key in (
            "frame_index", "estimation_candidate_count", "estimation_common_count",
            "common_cell_count", "transform_applied", "status", "checks", "rejected_checks",
            "signed_normal_offset_m", "mad_m", "estimator_within10mm",
            "rotation_determinant", "translation_norm_m") if key in original}
        selected = target_ids[frames[target_ids] == frame]
        record = {"frame_index": frame, "original_normal_quality": quality, "groups": [],
                  "total_estimation_count": 0, "estimation_indices_sha256": _indices_hash([]),
                  "transform_applied": False, "estimated_transform": None,
                  "applied_transform": np.eye(4).tolist(), "checks": {},
                  "status": "target_quality_rejected" if frame not in active else "missing_common_support"}
        transform = np.eye(4)
        if frame in active and len(selected):
            camera = baseline.rigid_transform(frame_lookup[frame]["camera_to_world_cv"], "Camera")
            normal = np.broadcast_to(normal_world, (len(selected), 3))
            target_residual = (local[selected] @ normal_raw - coefficients[2]) / normal_length
            blocks = [{"name": "target_table", "ids": selected, "positions": points[selected],
                       "normals": normal, "residuals": target_residual}]
            for group in independent:
                chosen = frames[group["ids"]] == frame
                blocks.append({key: (value[chosen] if key != "name" else value)
                               for key, value in group.items() if key != "reference_positions"})
            try:
                transform, solved = _solve_pose(blocks, camera[:3, 3])
                record.update(solved)
            except np.linalg.LinAlgError:
                record.update(status="numerical_failure", rejected_checks=["finite_solvable_system"])
        records.append(record)
        transforms[frame] = transform
    return transforms, records
