"""Re-evaluate frozen development partitions for area/matching sensitivity. MIT.

This reads existing surfaces and B0/B1/B2 labels; reconstruction and partition
algorithms are never invoked. Integrity findings are saved before any scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import stat

import numpy as np
import open3d as o3d

import evaluate_partition
from partition_baselines import SAMPLING_RULE
from run_development import inside, reference_arrays, save_json, sha256


METHODS = ("b0", "b1", "b2")
TOLERANCES = (.04, .08, .12)
SAMPLE_COUNTS = (7, 49)
TARGET_FIELDS = ("area_iou", "precision", "recall", "false_positive_ratio", "false_negative_ratio",
                 "scorable_target_area_m2", "true_positive_area_m2", "false_positive_area_m2",
                 "false_negative_area_m2", "unresolved_target_area_m2")
COVERAGE_FIELDS = ("mesh_area_m2", "scorable_area_m2", "unscorable_area_m2", "scorable_fraction",
                   "ambiguous_area_m2", "distance_rejected_area_m2")


def _no_links(path):
    for member in (path, *path.parents):
        if not member.exists() and not member.is_symlink():
            continue
        if (member.is_symlink() or getattr(member, "is_junction", lambda: False)()
                or getattr(member.lstat(), "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            raise ValueError(f"path must not contain links or junctions: {member}")


def _member(root, relative):
    path = root / relative
    _no_links(path)
    return inside(root, relative)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _check_hash(path, expected, name):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"missing or invalid {name} hash anchor")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{name} SHA256 mismatch: {path}")
    return actual


def _check_scene(suite, results, record, outcome):
    identity = record["scene_id"]
    if not re.fullmatch(r"[a-z0-9_-]+", identity):
        raise ValueError("unsafe scene_id")
    if outcome.get("status") != "completed" or outcome.get("target_id") != record["target_id"]:
        raise ValueError("scene must be completed with the manifest's target UUID")
    directory = _member(results, identity)
    if _read_json(_member(directory, "result.json")) != outcome:
        raise ValueError("scene result.json differs from development-results.json")
    scene_path = _member(directory, "points/scene.json")
    surface_path = _member(directory, "surface/surface-scene.json")
    mesh_path = _member(directory, "surface/surface.ply")
    labels_path = _member(suite, record["reference_labels"])
    hashes = {
        "scene_sha256": _check_hash(scene_path, outcome.get("scene_sha256"), "scene"),
        "surface_manifest_sha256": _check_hash(surface_path, outcome.get("surface_manifest_sha256"), "surface manifest"),
        "surface_sha256": _check_hash(mesh_path, outcome.get("surface_sha256"), "surface mesh"),
        "reference_sha256": _check_hash(labels_path, outcome.get("reference_sha256"), "reference labels"),
    }
    scene, surface = _read_json(scene_path), _read_json(surface_path)
    if (scene.get("schema") != "spatial-scene-lab.observations.v1"
            or scene.get("units") != "m" or scene.get("up_axis") != "Y"):
        raise ValueError("source scene must be metre/Y-up observations")
    if surface.get("input_scene_sha256") != hashes["scene_sha256"]:
        raise ValueError("surface manifest does not bind the source scene hash")
    scan = _member(suite, record["scan"])
    if Path(scene["source"]).resolve() != scan:
        raise ValueError("source scene scan differs from suite manifest")
    if not scene.get("input_files"):
        raise ValueError("source scene is missing frozen input file hashes")
    for name, expected in scene["input_files"].items():
        _check_hash(_member(scan, name), expected, f"raw input {name}")
    model_ids = [obj["uuid"] for obj in scene["objects"]]
    if len(set(model_ids)) != len(model_ids) or record["target_id"] not in model_ids:
        raise ValueError("scene object UUIDs must be unique and include the task target")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices, triangles = np.asarray(mesh.vertices).copy(), np.asarray(mesh.triangles).copy()
    for metadata in (surface, outcome):
        if metadata.get("vertices") != len(vertices) or metadata.get("triangles") != len(triangles):
            raise ValueError("mesh counts differ from their frozen result records")
    ref_vertices, ref_triangles, ref_owner = reference_arrays(labels_path, model_ids)
    reference_mesh_path = _member(suite, record["reference_mesh"])
    reference_mesh = o3d.io.read_triangle_mesh(str(reference_mesh_path))
    if (not np.array_equal(np.asarray(reference_mesh.vertices), ref_vertices)
            or not np.array_equal(np.asarray(reference_mesh.triangles), ref_triangles)):
        raise ValueError("physical reference PLY geometry differs from frozen reference label arrays")
    methods = outcome.get("methods", [])
    if {item["method"] for item in methods} != set(METHODS) or len(methods) != len(METHODS):
        raise ValueError("scene needs exactly the frozen B0/B1/B2 outcomes")
    frozen = []
    for item in methods:
        method = item["method"]
        if item.get("status") != "completed":
            raise ValueError(f"{method} was not completed in the original run")
        owner_path = _member(directory, f"{method}-face-owner.npy")
        metrics_path = _member(results, item["metrics_path"])
        if metrics_path != _member(directory, f"{method}-metrics.json"):
            raise ValueError(f"{method} metrics_path differs from its scene/method")
        owner_hash = _check_hash(owner_path, item.get("face_owner_sha256"), f"{method} face_owner")
        metrics_hash = _check_hash(metrics_path, item.get("metrics_sha256"), f"{method} metrics")
        owners = np.load(owner_path, allow_pickle=False)
        if (owners.shape != (len(triangles),) or owners.dtype != np.int32
                or np.any(owners < -2) or np.any(owners >= len(model_ids))):
            raise ValueError(f"{method} face labels must be int32 and match the fixed mesh/object UUID order")
        original = _read_json(metrics_path)
        parameters = original["parameters"]
        if (parameters.get("matching_tolerance_m") != .08 or parameters.get("samples_per_face") != 7
                or parameters.get("sampling_rule") != SAMPLING_RULE):
            raise ValueError(f"{method} original metrics must use the frozen 7/.08 sampling rule")
        if "ambiguity_margin_m" not in parameters:
            raise ValueError(f"{method} original metrics are missing the ambiguity margin")
        target = next(target for target in original["targets"] if target["object_id"] == record["target_id"])
        if target != item["target"] or original["coverage"] != item["coverage"]:
            raise ValueError(f"{method} original metric values differ from the result records")
        if ({target["object_id"] for target in original["targets"]} != set(model_ids)
                or len(original["targets"]) != len(model_ids)):
            raise ValueError(f"{method} original metrics do not contain each source object UUID once")
        frozen.append({"method": method, "owners": owners, "original": original,
                       "ambiguity_margin_m": parameters["ambiguity_margin_m"],
                       "face_owner_sha256": owner_hash, "metrics_sha256": metrics_hash})
    return {"scene_id": identity, "target_id": record["target_id"], "model_ids": model_ids,
            "vertices": vertices, "triangles": triangles,
            "reference_vertices": ref_vertices, "reference_triangles": ref_triangles,
            "reference_owner": ref_owner, "methods": frozen, "hashes": hashes,
            "reference_mesh_sha256": sha256(reference_mesh_path)}


def _compact(metrics):
    return {"targets": [{"index": item["index"], "object_id": item["object_id"],
                         "status": item["status"], **{key: item[key] for key in TARGET_FIELDS}}
                        for item in metrics["targets"]],
            "coverage": {key: metrics["coverage"][key] for key in COVERAGE_FIELDS}}


def _compare(original, recomputed, scene_id, method):
    mismatches = []
    original = _compact(original)
    recomputed = _compact(recomputed)

    def compare_field(field, expected, actual):
        if isinstance(expected, (float, int)) and not isinstance(expected, bool):
            equal = isinstance(actual, (float, int)) and math.isclose(
                expected, actual, rel_tol=1e-9, abs_tol=1e-10)
        else:
            equal = expected == actual
        if not equal:
            mismatches.append({"scene_id": scene_id, "method": method, "field": field,
                               "original": expected, "recomputed": actual})
    old_targets = {item["object_id"]: item for item in original["targets"]}
    new_targets = {item["object_id"]: item for item in recomputed["targets"]}
    compare_field("object_ids", sorted(old_targets), sorted(new_targets))
    for identity in old_targets.keys() & new_targets.keys():
        for key, expected in old_targets[identity].items():
            compare_field(f"targets.{identity}.{key}", expected, new_targets[identity][key])
    for key, expected in original["coverage"].items():
        compare_field(f"coverage.{key}", expected, recomputed["coverage"][key])
    return mismatches


def _save(output, summary):
    save_json(output / "sensitivity-results.json", summary)
    fields = ("scene_id", "method", "matching_tolerance_m", "samples_per_face", "object_id", "target_status",
              *TARGET_FIELDS, *COVERAGE_FIELDS)
    with (output / "sensitivity-metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for comparison in summary["comparisons"]:
            for target in comparison["targets"]:
                writer.writerow({"scene_id": comparison["scene_id"], "method": comparison["method"],
                                 "matching_tolerance_m": comparison["matching_tolerance_m"],
                                 "samples_per_face": comparison["samples_per_face"],
                                 "object_id": target["object_id"], "target_status": target["status"],
                                 **{key: target[key] for key in TARGET_FIELDS}, **comparison["coverage"]})


def reevaluate(results_path, suite_path, output_path):
    """Strict preflight, then the fixed 3x2 sensitivity grid on existing labels.

    A missing hash, changed file, or incomplete source run yields a saved
    integrity_failure report without scoring. Original .08/7 measurements are
    compared for every object, including nulls and statuses, at relative 1e-9 /
    absolute 1e-10 tolerance. A mismatch remains a reported failure.
    """
    requested = [Path(path).absolute() for path in (results_path, suite_path, output_path)]
    for path in requested:
        _no_links(path)
    results, suite, output = [path.resolve() for path in requested]
    for source in (results, suite):
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("sensitivity output must not overlap the suite or source results")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("sensitivity output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "spatial-scene-lab.development-sensitivity.v1", "status": "integrity_failure",
        "scope": "fixed development mesh/partition sensitivity; no algorithms rerun or physical accuracy claim",
        "findings": [], "comparisons": [],
        "configurations": [{"matching_tolerance_m": tolerance, "samples_per_face": count}
                           for tolerance in TOLERANCES for count in SAMPLE_COUNTS],
        "baseline_consistency": {"status": "not_run", "checked_method_tasks": 0, "mismatches": [],
                                 "relative_tolerance": 1e-9, "absolute_tolerance": 1e-10},
        "source": {"results_directory": str(results), "suite_directory": str(suite)},
        "code_sha256": {"reevaluate_development.py": sha256(Path(__file__)),
                        "evaluate_partition.py": sha256(Path(evaluate_partition.__file__)),
                        "run_development.py": sha256(Path(__file__).parent / "run_development.py")},
        "environment": {"numpy": np.__version__, "open3d": o3d.__version__},
    }
    frozen = []
    try:
        manifest_path = _member(suite, "suite-manifest.json")
        results_manifest_path = _member(results, "development-results.json")
        manifest, original = _read_json(manifest_path), _read_json(results_manifest_path)
        if (manifest.get("schema") != "spatial-scene-lab.fixture-suite.v1"
                or original.get("schema") != "spatial-scene-lab.development.v1"
                or original.get("status") != "completed"):
            raise ValueError("requires a completed development run and its fixture suite")
        manifest_hash = _check_hash(manifest_path, original.get("input_manifest_sha256"), "suite manifest")
        if original.get("input_generator_sha256") != manifest.get("generator_sha256"):
            raise ValueError("suite generator hash differs from the original run")
        for name in ("evaluate_partition.py", "partition_baselines.py", "run_development.py"):
            _check_hash(Path(__file__).parent / name, original["code_sha256"].get(name),
                        f"frozen evaluation dependency {name}")
        records = manifest["scenes"]
        outcomes = original["scenes"]
        scene_ids = [record["scene_id"] for record in records]
        outcome_ids = [record["scene_id"] for record in outcomes]
        if (not records or len(set(scene_ids)) != len(scene_ids)
                or len(set(outcome_ids)) != len(outcome_ids) or set(scene_ids) != set(outcome_ids)):
            raise ValueError("suite and completed result scenes must correspond exactly")
        if original["configuration"]["methods"] != list(METHODS):
            raise ValueError("original methods must be the frozen B0/B1/B2 controls")
        summary["source"].update(input_manifest_sha256=manifest_hash,
                                  development_results_sha256=sha256(results_manifest_path),
                                  input_generator_sha256=manifest.get("generator_sha256"))
        by_scene = {record["scene_id"]: record for record in outcomes}
        for record in records:
            try:
                frozen.append(_check_scene(suite, results, record, by_scene[record["scene_id"]]))
            except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
                summary["findings"].append({"scene_id": record["scene_id"], "method": None,
                                            "check": "scene_integrity", "message": f"{type(error).__name__}: {error}"})
    except (OSError, ValueError, KeyError, TypeError) as error:
        summary["findings"].append({"scene_id": None, "method": None, "check": "run_integrity",
                                    "message": f"{type(error).__name__}: {error}"})
    if summary["findings"]:
        _save(output, summary)
        return summary
    summary["frozen_artifacts"] = [
        {"scene_id": scene["scene_id"], **scene["hashes"], "reference_mesh_sha256": scene["reference_mesh_sha256"],
         "methods": [{key: method[key] for key in ("method", "face_owner_sha256", "metrics_sha256")}
                     for method in scene["methods"]]} for scene in frozen]
    summary["status"] = "running"
    summary["attempted_evaluations"] = len(frozen) * len(METHODS) * len(summary["configurations"])
    _save(output, summary)
    consistency = summary["baseline_consistency"]
    for scene in frozen:
        for method in scene["methods"]:
            for configuration in summary["configurations"]:
                try:
                    metrics = evaluate_partition.evaluate_all_objects(
                        scene["vertices"], scene["triangles"], method["owners"],
                        scene["reference_vertices"], scene["reference_triangles"], scene["reference_owner"],
                        object_ids=scene["model_ids"], matching_tolerance=configuration["matching_tolerance_m"],
                        samples_per_face=configuration["samples_per_face"], ambiguity_margin=method["ambiguity_margin_m"])
                    compact = _compact(metrics)
                except Exception as error:
                    summary["findings"].append({"scene_id": scene["scene_id"], "method": method["method"],
                                                "check": "evaluation", **configuration,
                                                "message": f"{type(error).__name__}: {error}"})
                    continue
                summary["comparisons"].append({"scene_id": scene["scene_id"], "method": method["method"],
                                                "task_target_id": scene["target_id"], **configuration, **compact})
                if configuration == {"matching_tolerance_m": .08, "samples_per_face": 7}:
                    consistency["checked_method_tasks"] += 1
                    consistency["mismatches"].extend(_compare(method["original"], metrics, scene["scene_id"], method["method"]))
            print(json.dumps({"scene_id": scene["scene_id"], "method": method["method"],
                              "sensitivity_configurations": len(summary["configurations"])}), flush=True)
            _save(output, summary)
    consistency["status"] = ("incomplete" if consistency["checked_method_tasks"] != len(frozen) * len(METHODS)
                             else "fail" if consistency["mismatches"] else "pass")
    summary["status"] = ("evaluation_failure" if summary["findings"] else
                         "consistency_failure" if consistency["mismatches"] else "completed")
    summary["completed_method_tasks"] = consistency["checked_method_tasks"]
    summary["completed_evaluations"] = len(summary["comparisons"])
    summary["failed_evaluations"] = summary["attempted_evaluations"] - summary["completed_evaluations"]
    _save(output, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = reevaluate(args.results, args.suite, args.output)
    print(json.dumps({"status": result["status"], "findings": len(result["findings"]),
                      "evaluations": len(result["comparisons"]),
                      "baseline_consistency": result["baseline_consistency"]["status"]}), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
