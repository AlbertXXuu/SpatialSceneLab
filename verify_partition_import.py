"""Compare saved native initial face identities directly with frozen S0 geometry. MIT.

This independent check reopens the S1 runner's before.blend; it does not rerun or
change the edits. Each native face ID must identify the same directed source
triangle and RGB corner values as its B1 bundle's source-index mapping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


POSITION_TOLERANCE = 1e-6
COLOR_TOLERANCE = 1e-6


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def compare_initial(expected, actual):
    for key in ("ids", "owners", "corners", "colors"):
        if expected[key].shape != actual[key].shape:
            return {"status": "fail", "reason": f"{key} shape differs"}
    for key in ("ids", "owners"):
        if not np.array_equal(expected[key], actual[key]):
            return {"status": "fail", "reason": f"{key} differ"}
    if not all(np.isfinite(values[key]).all() for values in (expected, actual) for key in ("corners", "colors")):
        return {"status": "fail", "reason": "nonfinite corner or color"}
    position = float(np.max(np.linalg.norm(expected["corners"] - actual["corners"], axis=-1)))
    color = float(np.max(np.abs(expected["colors"] - actual["colors"])))
    return {"status": "pass" if position <= POSITION_TOLERANCE and color <= COLOR_TOLERANCE else "fail",
            "source_faces": len(expected["ids"]), "owners_match": True, "native_ids_match": True,
            "maximum_directed_corner_distance_m": position, "position_tolerance_m": POSITION_TOLERANCE,
            "maximum_linear_rgba_difference": color, "linear_rgba_tolerance": COLOR_TOLERANCE,
            "comparison": "same mapped source face and corner order; no nearest-point or triangle reassignment"}


def expected_initial(source, record, bundle_path):
    import export_partition as export
    import open3d as o3d
    _, observation, mesh, owners, _ = export.validate_source(source, record)
    bundle_path = Path(bundle_path).resolve()
    bundle = export.read_json(bundle_path)
    if (bundle.get("schema") != "spatial-scene-lab.surface.v1" or bundle.get("scene_id") != record["scene_id"]
            or bundle["provenance"]["source_surface_sha256"] != record["surface_sha256"]):
        raise ValueError("bundle source mesh binding differs")
    source_vertices = np.asarray(mesh.vertices)
    source_triangles = np.asarray(mesh.triangles)
    source_colors = np.asarray(mesh.vertex_colors)
    values = {key: [] for key in ("ids", "owners", "corners", "colors", "source_face_indices")}
    registry, offset = [], 0
    for part in bundle["parts"]:
        if part["native_source_face_offset"] != offset:
            raise ValueError("native source offset is not contiguous")
        indices = {}
        for kind, limit in (("face", len(source_triangles)), ("vertex", len(source_vertices))):
            entry = part["source_indices"][kind]
            path = export.checked_file(bundle_path.parent, entry["path"], entry["sha256"])
            indices[kind] = np.load(path, allow_pickle=False)
            mapping = indices[kind]
            if (mapping.ndim != 1 or mapping.dtype.kind not in "iu" or len(mapping) != entry["count"]
                    or np.any(mapping < 0) or np.any(mapping >= limit) or len(np.unique(mapping)) != len(mapping)):
                raise ValueError("source mapping must contain unique in-range integer indices")
        faces, vertices = indices["face"], indices["vertex"]
        if len(faces) != part["triangles"] or len(vertices) != part["vertices"]:
            raise ValueError("source mapping count differs from part")
        label = part["owner_label"]
        identity = observation["objects"][label]["uuid"] if label >= 0 else {-1: "environment", -2: "unresolved"}[label]
        if part["id"] != identity or not np.all(owners[faces] == label):
            raise ValueError("part identity or owner label differs from frozen ownership")
        mesh_path = export.checked_file(bundle_path.parent, part["path"], part["sha256"])
        part_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if (not np.array_equal(vertices[np.asarray(part_mesh.triangles)], source_triangles[faces])
                or not np.array_equal(np.asarray(part_mesh.vertices), source_vertices[vertices])
                or not np.array_equal(np.asarray(part_mesh.vertex_colors), source_colors[vertices])):
            raise ValueError("part geometry or color differs from declared source indices")
        corners = source_vertices[source_triangles[faces]]
        world = corners[:, :, [0, 2, 1]].copy()
        world[:, :, 1] *= -1  # Existing importer's metre Y-up -> Blender Z-up matrix.
        srgb = source_colors[source_triangles[faces]]
        linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
        rgba = np.concatenate((linear, np.ones((*linear.shape[:2], 1))), axis=-1)
        values["ids"].append(np.arange(offset, offset + len(faces), dtype=np.int32))
        values["owners"].append(np.full(len(faces), part["id"]))
        values["corners"].append(world)
        values["colors"].append(rgba)
        values["source_face_indices"].append(faces)
        registry.append({"id": part["id"], "sha256": part["sha256"], "offset": offset, "faces": len(faces)})
        offset += len(faces)
    merged = {key: np.concatenate(value) for key, value in values.items()}
    if not np.array_equal(np.sort(merged["source_face_indices"]), np.arange(len(source_triangles))):
        raise ValueError("source mappings do not retain every source triangle exactly once")
    return merged, registry


def native_worker(args):
    import bpy
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import verify_blender_edit as verify
    output = Path(args.output)
    metadata = json.loads((output / "expected-metadata.json").read_text(encoding="utf-8"))
    result = {"status": "fail", "execution": "scripted fresh-process native initial lineage comparison"}
    try:
        if sha256(args.blend) != metadata["before_blend_sha256"]:
            raise ValueError("before.blend changed after input validation")
        if sha256(output / "expected-source.npz") != metadata["expected_source_sha256"]:
            raise ValueError("expected source arrays changed after preparation")
        bpy.ops.wm.open_mainfile(filepath=str(Path(args.blend).resolve()), load_ui=False)
        with np.load(output / "expected-source.npz", allow_pickle=False) as archive:
            expected = {key: archive[key] for key in archive.files}
        result.update(compare_initial(expected, verify.snapshot()))
        result["registry_match"] = json.loads(bpy.context.scene["ax_sources"]) == metadata["registry"]
        result["bundle_manifest_match"] = bpy.context.scene["ax_source_manifest_sha256"] == metadata["bundle_sha256"]
        result["blender"] = bpy.app.version_string
        result["source_surface_sha256"] = metadata["source_surface_sha256"]
        result["before_blend_sha256"] = metadata["before_blend_sha256"]
        if not result["registry_match"] or not result["bundle_manifest_match"]:
            result["status"] = "fail"
    except Exception as error:
        result.update(status="fail", error=f"{type(error).__name__}: {error}")
    finally:
        save_json(output / "native-result.json", result)
    if result["status"] != "pass":
        raise RuntimeError(json.dumps(result))


def run_verification(results, published_results, bundles, editing, output, blender):
    import export_partition as export
    from run_edit_suite import run_process
    results, bundles, editing, output = (Path(value).resolve() for value in (results, bundles, editing, output))
    for source in (results, bundles, editing):
        export.separate_output(output, source)
    baseline_hash = sha256(published_results)
    if sha256(results / "development-results.json") != baseline_hash:
        raise ValueError("prepared results differ from requested baseline")
    records = export.read_json(published_results)["scenes"]
    exported = export.read_json(bundles / "partition-export-results.json")
    edited = export.read_json(editing / "edit-suite-results.json")
    if exported["published_results_sha256"] != baseline_hash:
        raise ValueError("bundle export used a different baseline")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "spatial-scene-lab.partition-native-lineage.v1", "status": "running",
               "verifier_sha256": sha256(__file__), "published_results_sha256": baseline_hash,
               "color_conversion": "source PLY RGB8 sRGB to linear using IEC 61966-2-1; alpha = 1",
               "scope": "initial native face ID to frozen directed source triangle and corner colors",
               "scenes": []}
    started = time.perf_counter()
    for record in records:
        name = export.safe_id(record["scene_id"])
        item = {"scene_id": name, "status": "fail"}
        directory = output / name
        directory.mkdir()
        try:
            entry = next(entry for entry in exported["scenes"] if entry["scene_id"] == name)
            bundle = export.checked_file(bundles, entry["bundle"], entry["bundle_sha256"])
            task = next(task for task in edited["tasks"] if task["scene_id"] == name)
            if task["input_scene_sha256"] != entry["bundle_sha256"]:
                raise ValueError("native editing task used a different bundle")
            blend = export.checked_file(editing, f"{name}/before.blend", task["assets"]["before.blend"]["sha256"])
            expected, registry = expected_initial(results / name, record, bundle)
            np.savez_compressed(directory / "expected-source.npz", **expected)
            save_json(directory / "expected-metadata.json", {
                "registry": registry, "bundle_sha256": sha256(bundle), "before_blend_sha256": sha256(blend),
                "source_surface_sha256": record["surface_sha256"],
                "expected_source_sha256": sha256(directory / "expected-source.npz")})
            item["process"] = run_process([str(blender), "--background", "--factory-startup", "--python-exit-code", "1",
                                           "--python", str(Path(__file__).resolve()), "--", "--blend", str(blend),
                                           "--output", str(directory)], output / f"{name}.log", 1200)
            item["native"] = export.read_json(directory / "native-result.json")
            if item["process"]["status"] == "pass" and item["native"]["status"] == "pass":
                item["status"] = "pass"
        except Exception as error:
            item["error"] = f"{type(error).__name__}: {error}"
        summary["scenes"].append(item)
        summary["elapsed_seconds"] = time.perf_counter() - started
        save_json(output / "native-lineage-results.json", summary)
        print(json.dumps(item), flush=True)
    summary["status"] = "pass" if summary["scenes"] and all(item["status"] == "pass" for item in summary["scenes"]) else "fail"
    save_json(output / "native-lineage-results.json", summary)
    return summary


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results")
    parser.add_argument("--published-results")
    parser.add_argument("--bundles")
    parser.add_argument("--editing")
    parser.add_argument("--blender")
    parser.add_argument("--output", required=True)
    parser.add_argument("--blend", help=argparse.SUPPRESS)
    args = parser.parse_args(args)
    if args.blend:
        native_worker(args)
    else:
        if not all((args.results, args.published_results, args.bundles, args.editing, args.blender)):
            parser.error("results, published-results, bundles, editing and blender are required")
        result = run_verification(args.results, args.published_results, args.bundles, args.editing, args.output, args.blender)
        raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
