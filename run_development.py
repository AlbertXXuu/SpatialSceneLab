"""Run independent authored development scenes and fixed-mesh B0/B1/B2 comparisons.

The reference labels are loaded only after partitions have been computed. This
is development evidence, not the protocol's held-out experiment. MIT.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import open3d as o3d

import evaluate_partition
import partition_baselines
import reconstruct_mesh
import scan_pipeline


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def inside(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"manifest path must stay inside its suite: {relative}")
    return path


def peak_ram_bytes():
    """Process lifetime peak, not per-phase incremental memory or GPU memory."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def reference_arrays(labels_path, model_ids):
    with np.load(labels_path, allow_pickle=False) as data:
        vertices = data["vertices"].copy()
        triangles = data["triangles"].copy()
        owners = data["triangle_owner"].copy()
        reference_ids = data["object_ids"].tolist()
    if len(set(reference_ids)) != len(reference_ids) or set(reference_ids) != set(model_ids):
        raise ValueError("reference and input object UUIDs must agree; map by identity, not ordering")
    if owners.shape != (len(triangles),) or owners.dtype.kind not in 'iu':
        raise ValueError("reference owner labels must be one integer per triangle")
    if np.any(owners < -1) or np.any(owners >= len(reference_ids)):
        raise ValueError("reference has an invalid owner label")
    mapped = np.full(owners.shape, -1, dtype=np.int32)
    for index, identity in enumerate(reference_ids):
        mapped[owners == index] = model_ids.index(identity)
    return vertices, triangles, mapped


def run_suite(suite_path, output_path, *, scene_ids=None, voxel_size=.035,
              truncation=.105, matching_tolerance=.08, samples_per_face=7):
    started = time.perf_counter()
    suite = Path(suite_path).resolve()
    manifest_path = suite / "suite-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["scenes"]
    names = [record["scene_id"] for record in records]
    if not records or len(set(names)) != len(names):
        raise ValueError("suite must have unique, nonempty scenes")
    if scene_ids:
        if set(scene_ids) - set(names):
            raise ValueError("requested scene not in manifest")
        records = [record for record in records if record["scene_id"] in scene_ids]
    for record in records:
        identity = record["scene_id"]
        if not isinstance(identity, str) or not identity or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in identity):
            raise ValueError("unsafe scene_id")
    requested_output = Path(output_path).absolute()
    for member in (requested_output, *requested_output.parents):
        if member.is_symlink() or getattr(member, "is_junction", lambda: False)():
            raise ValueError("output directory chain must not contain symlinks or junctions")
    output = requested_output.resolve()
    if output == suite or suite in output.parents or output in suite.parents:
        raise ValueError("suite and output directories must not overlap")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("experiment output must be a new or empty directory; retain prior runs")
    output.mkdir(parents=True, exist_ok=True)
    methods = ("b0", "b1", "b2")
    summary = {
        "schema": "spatial-scene-lab.development.v1",
        "scope": "six-scene development comparisons; no held-out or real-scene generalization claim",
        "input_manifest_sha256": sha256(manifest_path),
        "input_generator_sha256": manifest.get("generator_sha256"),
        "configuration": {"voxel_size_m": voxel_size, "sdf_truncation_m": truncation,
                          "matching_tolerance_m": matching_tolerance,
                          "samples_per_face": samples_per_face, "methods": list(methods)},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "open3d": o3d.__version__,
                        "compute": "CPU; no learned-model inference", "new_paid_allocation": 0},
        "code_sha256": {name: sha256(Path(__file__).parent / name) for name in (
            "run_development.py", "scan_pipeline.py", "reconstruct_mesh.py",
            "partition_baselines.py", "evaluate_partition.py",
            "fixtures/make_complex_fixture.py", "fixtures/complex-development-scenes.json")},
        "scenes": [],
    }
    rows = []
    for record in records:
        scene_started = time.perf_counter()
        identity = record["scene_id"]
        directory = output / identity
        directory.mkdir(parents=True, exist_ok=True)
        result = {"scene_id": identity, "target_id": record["target_id"],
                  "factors": record["factors"], "status": "running", "methods": []}
        stage = "input"
        try:
            scan = inside(suite, record["scan"])
            labels_path = inside(suite, record["reference_labels"])
            stage = "observations"
            observation = scan_pipeline.reconstruct(scan, directory / "points")
            source_scene = directory / "points" / "scene.json"
            scene = json.loads(source_scene.read_text(encoding="utf-8"))
            model_ids = [obj["uuid"] for obj in scene["objects"]]
            if record["target_id"] not in model_ids:
                raise ValueError("task target missing from input objects")
            stage = "tsdf"
            surface = reconstruct_mesh.reconstruct(source_scene, directory / "surface",
                                                  voxel_size, truncation, require_editable=False)
            mesh_path = directory / "surface" / "surface.ply"
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            result.update(frames=observation["frames"], observations=observation["observation_count"],
                          vertices=len(vertices), triangles=len(triangles),
                          observation_seconds=observation["elapsed_seconds"],
                          tsdf_seconds=surface["elapsed_seconds"],
                          scene_sha256=sha256(source_scene),
                          surface_manifest_sha256=sha256(directory / "surface" / "surface-scene.json"),
                          surface_sha256=sha256(mesh_path), reference_sha256=sha256(labels_path))
            stage = "partition"
            partitions = {}
            for method in methods:
                tick = time.perf_counter()
                try:
                    value = partition_baselines.partition(vertices, triangles, scene["objects"],
                                                         scene["structures"], scene["ownership"], method)
                    owner = np.asarray(value["face_owner"], dtype=np.int32)
                    if owner.shape != (len(triangles),):
                        raise ValueError("partition owner count differs from fixed mesh")
                    np.save(directory / f"{method}-face-owner.npy", owner)
                    partitions[method] = {**value, "partition_seconds": time.perf_counter() - tick}
                except Exception as error:
                    result["methods"].append({"method": method, "status": "failed",
                                              "stage": "partition", "error": f"{type(error).__name__}: {error}"})
            # Independent reference loading happens after all methods have run.
            stage = "reference"
            ref_vertices, ref_triangles, ref_owner = reference_arrays(labels_path, model_ids)
            stage = "evaluation"
            for method, value in partitions.items():
                tick = time.perf_counter()
                try:
                    metrics = evaluate_partition.evaluate_all_objects(
                        vertices, triangles, value["face_owner"], ref_vertices, ref_triangles,
                        ref_owner, object_ids=model_ids, matching_tolerance=matching_tolerance,
                        samples_per_face=samples_per_face)
                    save_json(directory / f"{method}-metrics.json", metrics)
                    target = next(item for item in metrics["targets"] if item["object_id"] == record["target_id"])
                    outcome = {"method": method, "status": "completed", "parameters": value["parameters"],
                               "face_owner_sha256": sha256(directory / f"{method}-face-owner.npy"),
                               "metrics_sha256": sha256(directory / f"{method}-metrics.json"),
                               "partition_seconds": value["partition_seconds"],
                               "evaluation_seconds": time.perf_counter() - tick,
                               "target": target, "coverage": metrics["coverage"],
                               "metrics_path": f"{identity}/{method}-metrics.json"}
                    result["methods"].append(outcome)
                    rows.append({"scene_id": identity, "method": method, "target_id": record["target_id"],
                                 "target_status": target["status"],
                                 **{key: target[key] for key in ("area_iou", "precision", "recall",
                                    "false_positive_ratio", "false_negative_ratio")},
                                 "partition_seconds": outcome["partition_seconds"]})
                except Exception as error:
                    result["methods"].append({"method": method, "status": "failed",
                                              "stage": "evaluation", "error": f"{type(error).__name__}: {error}"})
            result["status"] = "completed" if len(result["methods"]) == len(methods) and all(
                item["status"] == "completed" for item in result["methods"]) else "failed"
        except Exception as error:
            result.update(status="failed", stage=stage, error=f"{type(error).__name__}: {error}")
            (directory / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        result.update(elapsed_seconds=time.perf_counter() - scene_started,
                      process_lifetime_peak_ram_bytes=peak_ram_bytes())
        summary["scenes"].append(result)
        save_json(directory / "result.json", result)
        save_json(output / "development-results.json", summary)
        print(json.dumps({"scene_id": identity, "status": result["status"],
                          "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    aggregate = []
    for method in methods:
        valid = [row for row in rows if row["method"] == method and row["area_iou"] is not None]
        completed = sum(any(item["method"] == method and item["status"] == "completed"
                            for item in scene["methods"]) for scene in summary["scenes"])
        aggregate.append({"method": method, "attempted_scenes": len(records),
                          "completed_scenes": completed, "evaluable_target_scenes": len(valid),
                          **{f"mean_{key}": float(np.mean([row[key] for row in valid])) if valid else None
                             for key in ("area_iou", "false_positive_ratio", "false_negative_ratio")}})
    summary.update(status="completed" if all(item["status"] == "completed" for item in summary["scenes"]) else "failed",
                   attempted_scenes=len(records), completed_scenes=sum(item["status"] == "completed" for item in summary["scenes"]),
                   attempted_method_tasks=len(records) * len(methods),
                   completed_method_tasks=sum(item["completed_scenes"] for item in aggregate),
                   aggregate=aggregate, elapsed_seconds=time.perf_counter() - started,
                   process_lifetime_peak_ram_bytes=peak_ram_bytes(),
                   memory_scope="single-process lifetime peak through each marker; not per-case incremental or full process tree")
    save_json(output / "development-results.json", summary)
    with (output / "development-metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ("scene_id", "method", "target_id", "target_status", "area_iou", "precision", "recall",
                  "false_positive_ratio", "false_negative_ratio", "partition_seconds")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene", action="append", dest="scene_ids")
    parser.add_argument("--voxel-size", type=float, default=.035)
    parser.add_argument("--truncation", type=float, default=.105)
    parser.add_argument("--matching-tolerance", type=float, default=.08)
    parser.add_argument("--samples-per-face", type=int, default=7)
    args = parser.parse_args()
    result = run_suite(args.suite, args.output, scene_ids=args.scene_ids, voxel_size=args.voxel_size,
                       truncation=args.truncation, matching_tolerance=args.matching_tolerance,
                       samples_per_face=args.samples_per_face)
    print(json.dumps({key: result[key] for key in ("status", "attempted_scenes", "completed_scenes",
                                                 "completed_method_tasks", "elapsed_seconds")}), flush=True)
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
