"""Area-weighted partition development evaluation against independent mesh labels.

Samples lie on the unchanged reconstructed mesh. The nearest point on the
independent physical reference mesh supplies a label only inside the recorded
distance tolerance. No object boxes, camera visibility test, or physical scan
accuracy claim are involved. MIT.
"""
from __future__ import annotations

import numpy as np

from partition_baselines import SAMPLING_RULE, barycentric_samples, mesh_arrays, triangle_areas


def _owners(value, count, name, minimum):
    value = np.asarray(value)
    if value.shape != (count,) or value.dtype.kind not in "iu" or np.any(value < minimum):
        raise ValueError(f"{name} must contain {count} integer labels >= {minimum}")
    if value.size and np.any(value > np.iinfo(np.int32).max):
        raise ValueError(f"{name} exceeds int32 labels")
    return value.astype(np.int32, copy=False)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def _label_areas(labels, weights):
    return {str(int(label)): float(weights[labels == label].sum()) for label in np.unique(labels)}


def _add_pairs(destination, reference, predicted, weights):
    if not len(reference):
        return
    pairs, inverse = np.unique(np.column_stack((reference, predicted)), axis=0, return_inverse=True)
    sums = np.bincount(inverse, weights=weights, minlength=len(pairs))
    for pair, area in zip(pairs, sums):
        key = tuple(map(int, pair))
        destination[key] = destination.get(key, 0.0) + float(area)


def _measure(vertices, triangles, face_owner, reference_vertices, reference_triangles,
             reference_owner, matching_tolerance, samples_per_face, candidate_mask,
             method_failure, ambiguity_margin):
    vertices, triangles = mesh_arrays(vertices, triangles)
    reference_vertices, reference_triangles = mesh_arrays(reference_vertices, reference_triangles)
    face_owner = _owners(face_owner, len(triangles), "face_owner", -2)
    reference_owner = _owners(reference_owner, len(reference_triangles), "reference_owner", -1)
    if not np.isfinite(matching_tolerance) or matching_tolerance < 0:
        raise ValueError("matching_tolerance must be finite and nonnegative")
    if not np.isfinite(ambiguity_margin) or ambiguity_margin < 0:
        raise ValueError("ambiguity_margin must be finite and nonnegative")
    barycentric = barycentric_samples(samples_per_face)
    if candidate_mask is not None:
        candidate_mask = np.asarray(candidate_mask)
        if candidate_mask.shape != (len(triangles),) or candidate_mask.dtype.kind != "b":
            raise ValueError("candidate_mask must be one boolean per reconstructed face")
    areas = triangle_areas(vertices, triangles)
    reference_areas = triangle_areas(reference_vertices, reference_triangles)
    if not np.isfinite(areas).all() or not np.isfinite(reference_areas).all():
        raise ValueError("triangle area is not finite")
    positive_reference = np.flatnonzero(reference_areas > 0)
    scene = None
    owner_scenes = []
    if len(positive_reference) and len(triangles):
        # Keep Open3D optional for pure prior partitioning; no package installation.
        import open3d as o3d
        if (np.any(np.abs(vertices) > np.finfo(np.float32).max)
                or np.any(np.abs(reference_vertices) > np.finfo(np.float32).max)):
            raise ValueError("raycasting coordinates exceed float32 range")
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(reference_vertices, dtype=np.float32)),
                            o3d.core.Tensor(np.ascontiguousarray(
                                reference_triangles[positive_reference], dtype=np.uint32)))
        # A second surface with the same owner is not a label ambiguity. Build
        # per-owner BVHs to find the nearest competing label, including room.
        if len(np.unique(reference_owner[positive_reference])) > 1:
            for label in np.unique(reference_owner[positive_reference]):
                selected = positive_reference[reference_owner[positive_reference] == label]
                owner_scene = o3d.t.geometry.RaycastingScene()
                owner_scene.add_triangles(
                    o3d.core.Tensor(np.ascontiguousarray(reference_vertices, dtype=np.float32)),
                    o3d.core.Tensor(np.ascontiguousarray(reference_triangles[selected], dtype=np.uint32)))
                owner_scenes.append((int(label), owner_scene))
    confusion, candidate_confusion, unscorable = {}, {}, {}
    scorable_area = 0.0
    scorable_samples = 0
    candidate_scorable_area = 0.0
    distance_area_sum = 0.0
    distance_available_area = 0.0
    distance_max = None
    ambiguous_area = 0.0
    distance_rejected_area = 0.0
    # Temporary raycast queries and candidate arrays stay below ~64k samples.
    chunk_faces = max(1, 65536 // samples_per_face)
    for start in range(0, len(triangles), chunk_faces):
        stop = min(start + chunk_faces, len(triangles))
        points = np.einsum("sk,fkc->fsc", barycentric, vertices[triangles[start:stop]])
        points = points.reshape(-1, 3)
        weights = np.repeat(areas[start:stop] / samples_per_face, samples_per_face)
        predicted = np.repeat(face_owner[start:stop], samples_per_face)
        matched = np.zeros(len(points), dtype=bool)
        truth = np.full(len(points), -2, dtype=np.int32)
        if scene is not None:
            queries = o3d.core.Tensor(np.ascontiguousarray(points, dtype=np.float32))
            nearest = scene.compute_closest_points(queries)
            primitive = nearest["primitive_ids"].numpy().astype(np.int64)
            distance = np.linalg.norm(nearest["points"].numpy().astype(np.float64) - points, axis=1)
            valid = ((primitive < len(positive_reference)) & np.isfinite(distance)
                     & (weights > 0))
            nearest_owner = np.full(len(points), -2, dtype=np.int32)
            nearest_owner[valid] = reference_owner[positive_reference[primitive[valid]]]
            second_distance = np.full(len(points), np.inf)
            for label, owner_scene in owner_scenes:
                competing = owner_scene.compute_closest_points(queries)
                competing_distance = np.linalg.norm(
                    competing["points"].numpy().astype(np.float64) - points, axis=1)
                other_label = valid & (nearest_owner != label)
                second_distance[other_label] = np.minimum(second_distance[other_label],
                                                         competing_distance[other_label])
            in_tolerance = valid & (distance <= matching_tolerance)
            ambiguous = in_tolerance & ((second_distance - distance) <= ambiguity_margin)
            matched = in_tolerance & ~ambiguous
            truth[matched] = nearest_owner[matched]
            ambiguous_area += float(weights[ambiguous].sum())
            distance_rejected_area += float(weights[valid & ~in_tolerance].sum())
            if np.any(valid):
                distance_area_sum += float(np.dot(distance[valid], weights[valid]))
                distance_available_area += float(weights[valid].sum())
                maximum = float(distance[valid].max())
                distance_max = maximum if distance_max is None else max(distance_max, maximum)
        scorable_area += float(weights[matched].sum())
        scorable_samples += int(np.count_nonzero(matched))
        _add_pairs(confusion, truth[matched], predicted[matched], weights[matched])
        for label, area in _label_areas(predicted[~matched], weights[~matched]).items():
            unscorable[label] = unscorable.get(label, 0.0) + area
        if candidate_mask is not None:
            in_candidate = np.repeat(candidate_mask[start:stop], samples_per_face)
            candidate_matched = matched & in_candidate
            candidate_scorable_area += float(weights[candidate_matched].sum())
            _add_pairs(candidate_confusion, truth[candidate_matched], predicted[candidate_matched],
                       weights[candidate_matched])
    total_area = float(areas.sum())
    unscorable_area = max(0.0, total_area - scorable_area)
    prediction_area = _label_areas(face_owner, areas)
    reference_area = _label_areas(reference_owner, reference_areas)
    candidate_domain = None
    if candidate_mask is not None:
        candidate_area = float(areas[candidate_mask].sum())
        accepted_area = sum(area for (truth, predicted), area in candidate_confusion.items()
                            if predicted != -2)
        error_area = sum(area for (truth, predicted), area in candidate_confusion.items()
                         if predicted != -2 and truth != predicted)
        candidate_domain = {
            "definition": "caller-supplied input candidate faces; primary metrics use the full mesh",
            "area_m2": candidate_area, "scorable_area_m2": candidate_scorable_area,
            "reference_label_available_fraction": _ratio(candidate_scorable_area, candidate_area),
            "accepted_scorable_area_m2": accepted_area,
            "accepted_fraction_of_scorable_candidates": _ratio(accepted_area, candidate_scorable_area),
            "accepted_error_area_m2": error_area,
            "risk": _ratio(error_area, accepted_area),
            "accepted_rule": "predicted owner != -2, including environment",
        }
    result = {
        "schema": "spatial-scene-lab.partition-evaluation.v1",
        "scope": "development partition area on a fixed reconstructed mesh against independent reference geometry",
        "physical_scan_accuracy_verified": False,
        "parameters": {"matching_tolerance_m": float(matching_tolerance),
                       "samples_per_face": int(samples_per_face), "sampling_rule": SAMPLING_RULE,
                       "sample_weight": "reconstructed face area / samples_per_face",
                       "matching_rule": "closest point on a positive-area reference triangle; distance <= tolerance",
                       "ambiguity_margin_m": float(ambiguity_margin),
                       "ambiguity_rule": "reject when nearest different-owner distance minus nearest distance <= margin",
                       "normal_filter": None, "visibility_filter": None,
                       "candidate_domain_restricts_primary_metrics": False},
        "coverage": {"mesh_area_m2": total_area, "scorable_area_m2": scorable_area,
                     "unscorable_area_m2": unscorable_area,
                     "ambiguous_area_m2": ambiguous_area,
                     "distance_rejected_area_m2": distance_rejected_area,
                     "scorable_fraction": _ratio(scorable_area, total_area),
                     "faces": len(triangles), "zero_area_faces": int(np.count_nonzero(areas == 0)),
                     "samples": len(triangles) * int(samples_per_face),
                     "scorable_samples": scorable_samples,
                     "unscorable_samples": len(triangles) * int(samples_per_face) - scorable_samples},
        "matching": {"distance_area_weighted_mean_m": _ratio(distance_area_sum, distance_available_area)
                     if scene is not None else None,
                     "distance_max_m": distance_max,
                     "ambiguity_check": "nearest different-owner reference surface distance gap",
                     "visibility_verified": False},
        "reference": {"area_m2": float(reference_areas.sum()),
                      "by_owner_area_m2": reference_area,
                      "zero_area_faces": int(np.count_nonzero(reference_areas == 0)),
                      "visible_reference_surface_coverage": None,
                      "area_role": "physical reference metadata; excluded from partition denominators"},
        "labels": {"predicted_mesh_area_m2": prediction_area,
                   "scorable_confusion_area_m2": [
                       {"reference_owner": truth, "predicted_owner": predicted, "area_m2": area}
                       for (truth, predicted), area in sorted(confusion.items())],
                   "unscorable_by_predicted_owner_area_m2": unscorable},
        "candidate_domain": candidate_domain,
        "failure_counts": {"method_failures": int(method_failure is not None),
                           "empty_mesh": int(len(triangles) == 0),
                           "zero_scorable_area": int(scorable_area == 0)},
        "method_failure": None if method_failure is None else str(method_failure),
    }
    return result, confusion, candidate_confusion


def _target(result, confusion, candidate_confusion, target_index, object_ids):
    if (isinstance(target_index, bool) or not isinstance(target_index, (int, np.integer))
            or target_index < 0):
        raise ValueError("target_index must be a nonnegative integer")
    target_index = int(target_index)
    if object_ids is not None and target_index >= len(object_ids):
        raise ValueError("target_index is outside object_ids")
    tp = sum(area for (truth, predicted), area in confusion.items()
             if truth == target_index and predicted == target_index)
    fn = sum(area for (truth, predicted), area in confusion.items()
             if truth == target_index and predicted != target_index)
    fp = sum(area for (truth, predicted), area in confusion.items()
             if truth != target_index and predicted == target_index)
    target_area = tp + fn
    target_exists = result["reference"]["by_owner_area_m2"].get(str(target_index), 0.0) > 0
    unknown_predicted = result["labels"]["unscorable_by_predicted_owner_area_m2"].get(str(target_index), 0.0)
    unknown_other = max(0.0, result["coverage"]["unscorable_area_m2"] - unknown_predicted)
    scores = {name: None for name in ("area_iou", "precision", "recall",
                                      "false_negative_ratio", "false_positive_ratio")}
    bounds = {name: {"lower": None, "upper": None} for name in scores}
    if target_area > 0 and result["method_failure"] is None:
        scores.update(area_iou=_ratio(tp, tp + fp + fn), precision=_ratio(tp, tp + fp),
                      recall=_ratio(tp, target_area), false_negative_ratio=_ratio(fn, target_area),
                      false_positive_ratio=_ratio(fp, target_area))
        bounds["area_iou"] = {"lower": _ratio(tp, tp + fp + fn + unknown_predicted + unknown_other),
                              "upper": _ratio(tp + unknown_predicted, tp + fp + fn + unknown_predicted)}
        bounds["precision"] = {"lower": _ratio(tp, tp + fp + unknown_predicted),
                               "upper": _ratio(tp + unknown_predicted, tp + fp + unknown_predicted)}
        bounds["recall"] = {"lower": _ratio(tp, target_area + unknown_other),
                            "upper": _ratio(tp + unknown_predicted, target_area + unknown_predicted)}
        bounds["false_negative_ratio"] = {"lower": 1 - bounds["recall"]["upper"],
                                          "upper": 1 - bounds["recall"]["lower"]}
        bounds["false_positive_ratio"] = {
            "lower": _ratio(fp, target_area + unknown_predicted + unknown_other),
            "upper": _ratio(fp + unknown_predicted, target_area)}
    if result["method_failure"] is not None:
        status = "method_failure"
    elif not target_exists:
        status = "no_target"
    elif result["coverage"]["mesh_area_m2"] == 0:
        status = "empty_mesh"
    elif target_area == 0:
        status = "no_scorable_target"
    else:
        status = "evaluated"
    candidate_target = sum(area for (truth, _), area in candidate_confusion.items() if truth == target_index)
    return {"index": target_index, "object_id": str(object_ids[target_index]) if object_ids is not None else None,
            "status": status, **scores,
            "scorable_target_area_m2": target_area, "true_positive_area_m2": tp,
            "false_positive_area_m2": fp, "false_negative_area_m2": fn,
            "unresolved_target_area_m2": sum(area for (truth, predicted), area in confusion.items()
                                             if truth == target_index and predicted == -2),
            "unscorable_predicted_target_area_m2": unknown_predicted,
            "target_outside_candidate_area_m2": max(0.0, target_area - candidate_target)
            if result["candidate_domain"] is not None else None,
            "false_positive_by_reference_owner_area_m2": {
                str(truth): area for (truth, predicted), area in confusion.items()
                if predicted == target_index and truth != target_index},
            "conservative_bounds": bounds,
            "bounds_definition": "assign all unscorable mesh area adversarially to target/non-target; finite sampling error is not bounded",
            "denominator": "all scorable target area on the fixed reconstructed mesh, including outside candidates",
            "failure_counts": {"method_failures": int(result["method_failure"] is not None),
                               "no_target": int(not target_exists),
                               "no_scorable_target": int(target_exists and target_area == 0)}}


def evaluate(vertices, triangles, face_owner, reference_vertices, reference_triangles,
             reference_owner, target_index, matching_tolerance=0.08, samples_per_face=7,
             *, object_ids=None, candidate_mask=None, method_failure=None, ambiguity_margin=0.002):
    """Evaluate one target using full-mesh area; all results are JSON serializable.

    Unknown predictions on matched target samples count as missed target area.
    Unmatched samples and competing labels within ambiguity_margin metres are
    not scored and receive explicit conservative bounds.
    A target absent from the reference, or with no scorable target surface,
    produces null scores. This evaluation does not measure reference visibility,
    complete object reconstruction, or real-world physical scan accuracy.
    """
    result, confusion, candidate_confusion = _measure(
        vertices, triangles, face_owner, reference_vertices, reference_triangles,
        reference_owner, matching_tolerance, samples_per_face, candidate_mask, method_failure,
        ambiguity_margin)
    result["target"] = _target(result, confusion, candidate_confusion, target_index, object_ids)
    result["status"] = result["target"]["status"]
    return result


def evaluate_all_objects(vertices, triangles, face_owner, reference_vertices, reference_triangles,
                         reference_owner, matching_tolerance=0.08, samples_per_face=7,
                         *, object_ids=None, candidate_mask=None, method_failure=None,
                         ambiguity_margin=0.002):
    """Match the independent reference once, then report each object and macro IoU.

    ``object_ids`` orders labels exactly like the scene's objects. Without it,
    indices are the union of nonnegative prediction/reference labels. Missing
    targets retain null scores, remain in the attempt counts, and are reported
    separately from the macro average's explicit scored-target denominator.
    """
    result, confusion, candidate_confusion = _measure(
        vertices, triangles, face_owner, reference_vertices, reference_triangles,
        reference_owner, matching_tolerance, samples_per_face, candidate_mask, method_failure,
        ambiguity_margin)
    if object_ids is None:
        indices = sorted({int(label) for field in (result["reference"]["by_owner_area_m2"],
                                                   result["labels"]["predicted_mesh_area_m2"])
                          for label in field if int(label) >= 0})
    else:
        indices = range(len(object_ids))
        for labels in (reference_owner, face_owner):
            labels = np.asarray(labels)
            if np.any(labels >= len(object_ids)):
                raise ValueError("owner label is outside object_ids")
    targets = [_target(result, confusion, candidate_confusion, index, object_ids) for index in indices]
    scored = [target["area_iou"] for target in targets if target["area_iou"] is not None]
    result.update(targets=targets, macro_target_area_iou=float(np.mean(scored)) if scored else None,
                  object_attempts=len(targets), scored_objects=len(scored),
                  unscored_objects=len(targets) - len(scored),
                  status="method_failure" if method_failure is not None else "evaluated" if scored else "no_scorable_target")
    result["failure_counts"].update(no_target=sum(target["failure_counts"]["no_target"] for target in targets),
                                     no_scorable_target=sum(target["failure_counts"]["no_scorable_target"] for target in targets))
    return result
