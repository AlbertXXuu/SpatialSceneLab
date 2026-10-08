"""Explain frozen B1 ownership errors without changing its partition or scorer. MIT.

Reference labels enter this diagnostic only after the prior-only B1 replay.
The existing evaluator groups sampled reference area by temporary face IDs in
bounded batches. Its per-face barycentric points, weights and reference search
are unchanged; these temporary IDs are never written as editable ownership.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

import evaluate_partition
from partition_baselines import mesh_arrays, partition, triangle_areas
from scan_pipeline import box_candidates, structure_mask


REASONS = (
    "tp", "fn_structure_exclusion", "fn_target_box_overlap",
    "fn_outside_target_box_other_object", "fn_outside_target_box_other_overlap",
    "fn_outside_target_box_environment", "fp_other_object", "fp_environment",
    "tn", "unscorable_predicted_target", "unscorable_other",
)
DEFINITIONS = {
    "tp": "scorable reference target area selected as this target",
    "fn_structure_exclusion": "target centroid box hit removed by the structure prior (takes precedence over overlap)",
    "fn_target_box_overlap": "target centroid box hit and another box hit, with no structure exclusion; B1 is -2",
    "fn_outside_target_box_other_object": "centroid outside the target box plus margin and uniquely assigned to another object",
    "fn_outside_target_box_other_overlap": "centroid outside the target box plus margin and unresolved in multiple other boxes",
    "fn_outside_target_box_environment": "centroid outside the target box plus margin and assigned environment; raw structure flag is also retained",
    "fp_other_object": "uniquely accepted target centroid prior covers sampled area matched to another reference object",
    "fp_environment": "uniquely accepted target centroid prior covers sampled area matched to reference environment",
    "tn": "scorable non-target reference area not selected as this target",
    "unscorable_predicted_target": "unscorable sampled area on faces selected as this target",
    "unscorable_other": "unscorable sampled area on faces not selected as this target",
}
AREA_FIELDS = {
    "true_positive_area_m2": (0,), "false_negative_area_m2": (1, 2, 3, 4, 5),
    "false_positive_area_m2": (6, 7), "unscorable_predicted_target_area_m2": (9,),
    "unresolved_target_area_m2": (2, 4),
}
REL_TOL, ABS_TOL = 1e-9, 1e-10


def _check_close(name, actual, expected):
    if not math.isclose(float(actual), float(expected), rel_tol=REL_TOL, abs_tol=ABS_TOL):
        raise ValueError(f"{name} conservation mismatch: {actual} != {expected}")
    return abs(float(actual) - float(expected))


def _deadline(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("diagnostic suite exceeded its recorded time budget")


def prior_evidence(vertices, triangles, owners, objects, structures, ownership):
    """Replay only input priors and require exact equality with frozen B1 owners."""
    vertices, triangles = mesh_arrays(vertices, triangles)
    owners = np.asarray(owners)
    if (owners.shape != (len(triangles),) or owners.dtype.kind not in "iu"
            or np.any(owners < -2) or np.any(owners >= len(objects))):
        raise ValueError("frozen owners must be valid integer B1 labels for every face")
    replay = partition(vertices, triangles, objects, structures, ownership, "b1")["face_owner"]
    if not np.array_equal(replay, owners):
        raise ValueError("B1 prior replay differs from frozen face ownership")
    centroids = vertices[triangles].mean(axis=1)
    candidates = box_candidates(centroids, objects, ownership["box_margin_m"])
    excluded = structure_mask(centroids, structures, ownership["structure_tolerance_m"])
    return {"centroid_m": centroids, "raw_box_candidates": candidates,
            "structure_excluded": excluded}


def trace_reference_area(vertices, triangles, reference_vertices, reference_triangles,
                         reference_owner, object_count, parameters, *, batch_faces=1024,
                         deadline=None):
    """Recover per-source-face area with the unmodified evaluator in small batches.

    Face IDs are diagnostic grouping keys, not semantic predictions. Batching
    bounds the evaluator's per-label boolean scans, which are quadratic for
    unique labels. Sampling depends only on each triangle and samples_per_face,
    never a global face index, subset size, or random state.
    """
    if isinstance(batch_faces, bool) or not isinstance(batch_faces, int) or not 1 <= batch_faces <= 4096:
        raise ValueError("batch_faces must be an integer between 1 and 4096")
    vertices, triangles = mesh_arrays(vertices, triangles)
    reference_owner = np.asarray(reference_owner)
    if np.any(reference_owner >= object_count):
        raise ValueError("reference owner is outside the source object order")
    areas = np.zeros((len(triangles), object_count + 1), dtype=np.float64)
    unknown = np.zeros(len(triangles), dtype=np.float64)
    coverage = {key: 0 for key in ("mesh_area_m2", "scorable_area_m2", "unscorable_area_m2",
                "ambiguous_area_m2", "distance_rejected_area_m2", "faces", "zero_area_faces",
                "samples", "scorable_samples", "unscorable_samples")}
    for start in range(0, len(triangles), batch_faces):
        _deadline(deadline)
        stop = min(start + batch_faces, len(triangles))
        measured = evaluate_partition.evaluate(
            vertices, triangles[start:stop], np.arange(stop - start, dtype=np.int32),
            reference_vertices, reference_triangles, reference_owner, target_index=0,
            matching_tolerance=parameters["matching_tolerance_m"],
            samples_per_face=parameters["samples_per_face"],
            ambiguity_margin=parameters["ambiguity_margin_m"])
        for pair in measured["labels"]["scorable_confusion_area_m2"]:
            areas[start + pair["predicted_owner"], pair["reference_owner"] + 1] += pair["area_m2"]
        for face, area in measured["labels"]["unscorable_by_predicted_owner_area_m2"].items():
            unknown[start + int(face)] = area
        for key in coverage:
            coverage[key] += measured["coverage"][key]
    _deadline(deadline)
    np.testing.assert_allclose(areas.sum(axis=1) + unknown, triangle_areas(vertices, triangles),
                               rtol=REL_TOL, atol=ABS_TOL, err_msg="per-face reference area conservation")
    return areas, unknown, coverage


def target_reason_areas(owners, target, evidence, reference_area, unknown):
    """Partition each target's whole mesh area into mutually exclusive reasons.

    One triangle may contain multiple sampled reference labels. Therefore its
    reason *areas*, rather than a single majority reason, preserve exact scores.
    """
    owners = np.asarray(owners)
    raw = evidence["raw_box_candidates"]
    selected = owners == target
    hit = raw[:, target]
    excluded = evidence["structure_excluded"]
    masks = [selected, hit & excluded, hit & ~excluded & (raw.sum(axis=1) > 1),
             ~hit & (owners >= 0), ~hit & (owners == -2), ~hit & (owners == -1)]
    if not np.all(np.stack(masks).sum(axis=0) == 1):
        raise ValueError("prior reasons must assign every face exactly once")
    result = np.zeros((len(owners), len(REASONS)), dtype=np.float64)
    truth_target = reference_area[:, target + 1]
    for column, mask in enumerate(masks):
        result[mask, column] = truth_target[mask]
    other_objects = reference_area[:, 1:].sum(axis=1) - truth_target
    result[selected, 6] = np.maximum(0, other_objects[selected])
    result[selected, 7] = reference_area[selected, 0]
    result[~selected, 8] = np.maximum(0, reference_area[~selected].sum(axis=1) - truth_target[~selected])
    result[selected, 9] = unknown[selected]
    result[~selected, 10] = unknown[~selected]
    np.testing.assert_allclose(result.sum(axis=1), reference_area.sum(axis=1) + unknown,
                               rtol=REL_TOL, atol=ABS_TOL, err_msg="per-face reason area conservation")
    return result


def check_original(owners, reference_area, unknown, coverage, original):
    """Require the entire original confusion and unknown areas, not just IoU."""
    maximum = 0.0
    actual = {}
    for predicted in np.unique(owners):
        totals = reference_area[owners == predicted].sum(axis=0)
        for column, area in enumerate(totals):
            if area:
                actual[(column - 1, int(predicted))] = float(area)
    expected = {(row["reference_owner"], row["predicted_owner"]): row["area_m2"]
                for row in original["labels"]["scorable_confusion_area_m2"]}
    for pair in actual.keys() | expected.keys():
        maximum = max(maximum, _check_close(f"confusion {pair}", actual.get(pair, 0), expected.get(pair, 0)))
    old_unknown = original["labels"]["unscorable_by_predicted_owner_area_m2"]
    for owner in set(map(int, old_unknown)) | set(map(int, np.unique(owners))):
        maximum = max(maximum, _check_close(f"unscorable owner {owner}", unknown[owners == owner].sum(),
                                          old_unknown.get(str(owner), 0)))
    for key, value in coverage.items():
        maximum = max(maximum, _check_close(key, value, original["coverage"][key]))
    return maximum


def _save(output, summary):
    from run_development import save_json
    save_json(output / "diagnostics-results.json", summary)
    fields = ("scene_id", "object_id", "task_target", "reason", "area_m2", "faces")
    with (output / "diagnostics-metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scene in summary["scenes"]:
            for target in scene.get("targets", []):
                for reason, area in target["area_m2"].items():
                    writer.writerow({"scene_id": scene["scene_id"], "object_id": target["object_id"],
                                     "task_target": target["object_id"] == scene["target_id"], "reason": reason,
                                     "area_m2": area, "faces": target["face_count_by_reason"][reason]})


def diagnose(results_path, suite_path, published_path, output_path, *, batch_faces=1024, max_seconds=1200):
    from reevaluate_development import _check_hash, _check_scene, _member, _no_links, _read_json
    from run_development import peak_ram_bytes, save_json, sha256
    import open3d as o3d

    requested = [Path(path).absolute() for path in (results_path, suite_path, published_path, output_path)]
    for path in requested:
        _no_links(path)
    results, suite, published, output = [path.resolve() for path in requested]
    for source in (results, suite, published):
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("diagnostic output must not overlap input artifacts")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("diagnostic output must be a new or empty directory")
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 1200:
        raise ValueError("max_seconds must be in (0, 1200]")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    deadline = started + max_seconds
    summary = {"schema": "spatial-scene-lab.partition-diagnostics.v1", "status": "integrity_failure",
               "scope": "frozen B1 development error attribution; reference used only for diagnosis",
               "method_changed": False, "reference_used_for_partition": False,
               "reason_order": list(REASONS), "reason_definitions": DEFINITIONS,
               "source": {"results_directory": str(results), "suite_directory": str(suite),
                          "published_results": str(published)},
               "parameters": {"batch_faces": batch_faces, "max_seconds": max_seconds,
                              "relative_tolerance": REL_TOL, "absolute_tolerance": ABS_TOL,
                              "trace_rule": "unchanged evaluator; temporary batch-local face IDs as grouping keys"},
               "limitations": ["nearest reference matching has no normal or visibility filter",
                               "unreconstructed physical surfaces are not partition errors",
                               "finite barycentric area estimates; no real-scene or held-out generalization",
                               "prior decision traces explain selection mechanics, not colour or ultimate physical causality"],
               "code_sha256": {name: sha256(Path(__file__).parent / name) for name in
                               ("partition_diagnostics.py", "evaluate_partition.py", "partition_baselines.py",
                                "scan_pipeline.py", "reevaluate_development.py", "run_development.py")},
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "open3d": o3d.__version__, "compute": "CPU", "new_paid_allocation": 0},
               "resources": {}, "findings": [], "scenes": []}
    frozen = []
    try:
        original = _read_json(_member(results, "development-results.json"))
        published_result = _read_json(published)
        if original != published_result:
            raise ValueError("source development results differ from the supplied published anchor")
        manifest_path = _member(suite, "suite-manifest.json")
        manifest = _read_json(manifest_path)
        if (original.get("schema") != "spatial-scene-lab.development.v1"
                or original.get("status") != "completed"
                or manifest.get("schema") != "spatial-scene-lab.fixture-suite.v1"):
            raise ValueError("requires a completed development result and fixture suite")
        _check_hash(manifest_path, original.get("input_manifest_sha256"), "input manifest")
        if original.get("input_generator_sha256") != manifest.get("generator_sha256"):
            raise ValueError("source generator hash differs from fixture manifest")
        for name in ("evaluate_partition.py", "partition_baselines.py", "scan_pipeline.py", "run_development.py"):
            _check_hash(Path(__file__).parent / name, original["code_sha256"].get(name), name)
        records, outcomes = manifest["scenes"], original["scenes"]
        record_ids, outcome_ids = [x["scene_id"] for x in records], [x["scene_id"] for x in outcomes]
        if (not records or len(set(record_ids)) != len(record_ids) or len(set(outcome_ids)) != len(outcome_ids)
                or set(record_ids) != set(outcome_ids)):
            raise ValueError("suite and published scene identities must correspond exactly")
        summary["source"].update(development_results_sha256=sha256(results / "development-results.json"),
                                  published_results_sha256=sha256(published), input_manifest_sha256=sha256(manifest_path))
        by_id = {outcome["scene_id"]: outcome for outcome in outcomes}
        for record in records:
            _deadline(deadline)
            try:
                scene = _check_scene(suite, results, record, by_id[record["scene_id"]])
                source_scene = _read_json(_member(results, f"{record['scene_id']}/points/scene.json"))
                method = next(item for item in scene["methods"] if item["method"] == "b1")
                evidence = prior_evidence(scene["vertices"], scene["triangles"], method["owners"],
                                          source_scene["objects"], source_scene["structures"], source_scene["ownership"])
                frozen.append((scene, method, evidence))
            except Exception as error:
                summary["findings"].append({"scene_id": record["scene_id"], "stage": "integrity",
                                            "error": f"{type(error).__name__}: {error}"})
    except Exception as error:
        summary["findings"].append({"scene_id": None, "stage": "integrity", "error": f"{type(error).__name__}: {error}"})
    if not summary["findings"]:
        summary["status"] = "running"
        _save(output, summary)
        for scene, method, evidence in frozen:
            tick = time.perf_counter()
            result = {"scene_id": scene["scene_id"], "target_id": scene["target_id"], "status": "failed",
                      "source": {**scene["hashes"], "reference_mesh_sha256": scene["reference_mesh_sha256"],
                                 "face_owner_sha256": method["face_owner_sha256"], "metrics_sha256": method["metrics_sha256"]}}
            summary["scenes"].append(result)
            try:
                reference_area, unknown, coverage = trace_reference_area(
                    scene["vertices"], scene["triangles"], scene["reference_vertices"], scene["reference_triangles"],
                    scene["reference_owner"], len(scene["model_ids"]), method["original"]["parameters"],
                    batch_faces=batch_faces, deadline=deadline)
                maximum = check_original(method["owners"], reference_area, unknown, coverage, method["original"])
                arrays = {"face_id": np.arange(len(scene["triangles"]), dtype=np.int64),
                          "face_owner": method["owners"], "face_area_m2": triangle_areas(scene["vertices"], scene["triangles"]),
                          "object_ids": np.asarray(scene["model_ids"]), "reason_names": np.asarray(REASONS),
                          "reference_owner_order": np.arange(-1, len(scene["model_ids"]), dtype=np.int32),
                          "scorable_reference_area_m2": reference_area, "unscorable_area_m2": unknown, **evidence}
                targets = []
                for original_target in method["original"]["targets"]:
                    target = original_target["index"]
                    values = target_reason_areas(method["owners"], target, evidence, reference_area, unknown)
                    areas = values.sum(axis=0)
                    for field, columns in AREA_FIELDS.items():
                        maximum = max(maximum, _check_close(field, areas[list(columns)].sum(), original_target[field]))
                    maximum = max(maximum, _check_close("whole mesh by reasons", areas.sum(), coverage["mesh_area_m2"]))
                    arrays[f"target_{target}_reason_area_m2"] = values
                    fn, fp = areas[1:6].sum(), areas[6:8].sum()
                    targets.append({"index": target, "object_id": original_target["object_id"],
                                    "original_metrics": original_target,
                                    "area_m2": dict(zip(REASONS, map(float, areas))),
                                    "face_count_by_reason": dict(zip(REASONS, map(int, (values > 0).sum(axis=0)))),
                                    "fn_reason_fraction": {REASONS[i]: float(areas[i] / fn) if fn else None for i in range(1, 6)},
                                    "fp_reason_fraction": {REASONS[i]: float(areas[i] / fp) if fp else None for i in (6, 7)}})
                directory = output / scene["scene_id"]
                directory.mkdir()
                trace = directory / "face-diagnostics.npz"
                np.savez_compressed(trace, **arrays)
                result.update(status="completed", coverage=coverage, targets=targets,
                              conservation={"status": "pass", "maximum_absolute_difference_m2": maximum,
                                            "b1_replay_exact": True, "reason_partition_complete": True},
                              face_trace={"path": str(trace.relative_to(output)).replace("\\", "/"), "sha256": sha256(trace),
                                          "face_index_basis": "zero-based triangles in the hash-bound frozen surface.ply",
                                          "reference_area_columns": "reference_owner_order; -1 environment then source object order",
                                          "reason_area_arrays": "target_<index>_reason_area_m2; columns follow reason_names",
                                          "contains_evaluation_reference": True, "use_for_partition_or_edit_ownership": False})
                result["seconds"] = time.perf_counter() - tick
                save_json(directory / "diagnostics.json", result)
            except Exception as error:
                result["seconds"] = time.perf_counter() - tick
                summary["findings"].append({"scene_id": scene["scene_id"], "stage": "diagnosis",
                                            "error": f"{type(error).__name__}: {error}"})
            print(json.dumps({"scene_id": scene["scene_id"], "status": result["status"], "seconds": result["seconds"]}), flush=True)
            _save(output, summary)
        summary["status"] = "diagnostic_failure" if summary["findings"] else "completed"
    summary["resources"] = {"wall_seconds": time.perf_counter() - started, "process_lifetime_peak_ram_bytes": peak_ram_bytes(),
                            "new_paid_allocation": 0, "model_calls": 0}
    summary["completed_scenes"] = sum(scene["status"] == "completed" for scene in summary["scenes"])
    _save(output, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--published-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-faces", type=int, default=1024)
    parser.add_argument("--max-seconds", type=float, default=1200)
    args = parser.parse_args(argv)
    summary = diagnose(args.results, args.suite, args.published_results, args.output,
                       batch_faces=args.batch_faces, max_seconds=args.max_seconds)
    print(json.dumps({"status": summary["status"], "completed_scenes": summary["completed_scenes"],
                      "findings": len(summary["findings"]), "resources": summary["resources"]}), flush=True)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
