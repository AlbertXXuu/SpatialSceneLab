"""Export frozen B1 ownership on unchanged S0 surfaces for native editing. MIT.

The published result hashes bind the mesh, observations and saved owner array.
No reference geometry or labels are read. Output uses the existing S1 bundle
layout and records a reversible link from native source IDs to S0 face indices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import open3d as o3d

from partition_baselines import mesh_arrays


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("manifest contains nonfinite numeric data")
        if isinstance(value, dict):
            for member in value.values():
                finite(member)
        elif isinstance(value, list):
            for member in value:
                finite(member)
    finite(doc)
    return doc


def safe_id(value):
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value)
            or value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL",
                                             *[f"COM{i}" for i in range(1, 10)],
                                             *[f"LPT{i}" for i in range(1, 10)]}):
        raise ValueError("identity must be a safe portable filename")
    return value


def member(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("input path must stay inside its bundle")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("input path escaped its bundle")
    return path


def checked_file(root, relative, expected):
    path = member(root, relative)
    if (not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected)
            or not path.is_file() or sha256(path) != expected):
        raise ValueError(f"input SHA256 mismatch: {relative}")
    return path


def separate_output(output, source):
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("output and source directories must not overlap")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("retain existing runs; output must be new or empty")


def validate_source(directory, record):
    """Verify frozen metadata before loading only the mesh and ownership array."""
    directory = Path(directory).resolve()
    safe_id(record["scene_id"])
    if directory.name != record["scene_id"] or record.get("status") != "completed":
        raise ValueError("source scene identity or completion state differs")
    surface_path = checked_file(directory, "surface/surface-scene.json", record["surface_manifest_sha256"])
    mesh_path = checked_file(directory, "surface/surface.ply", record["surface_sha256"])
    observation_path = checked_file(directory, "points/scene.json", record["scene_sha256"])
    methods = [method for method in record["methods"] if method.get("method") == "b1"]
    if len(methods) != 1 or methods[0].get("status") != "completed":
        raise ValueError("requires one completed frozen B1 method")
    method = methods[0]
    parameters = method["parameters"]
    if (parameters.get("method") != "b1" or parameters.get("reference_used") is not False
            or parameters.get("mesh_changed") is not False):
        raise ValueError("B1 must preserve the mesh and use no reference")
    owner_path = checked_file(directory, "b1-face-owner.npy", method["face_owner_sha256"])
    surface, observation = read_json(surface_path), read_json(observation_path)
    for doc, schema in ((surface, "surface"), (observation, "observations")):
        if (doc.get("schema") != f"spatial-scene-lab.{schema}.v1"
                or doc.get("units") != "m" or doc.get("up_axis") != "Y"):
            raise ValueError("requires metre / Y-up surface and observation manifests")
    if surface.get("status") != "pass" or surface.get("input_scene_sha256") != record["scene_sha256"]:
        raise ValueError("surface input manifest binding differs")
    objects = observation.get("objects")
    if not isinstance(objects, list):
        raise ValueError("observations must declare ordered objects")
    ids = [safe_id(obj["uuid"]) for obj in objects]
    if len({identity.casefold() for identity in ids}) != len(ids) or any(
            identity.casefold() in {"environment", "unresolved", "surface", "surface-scene"} for identity in ids):
        raise ValueError("object identities are duplicated or reserved")
    for obj in objects:
        if not all(isinstance(obj.get(key), str) and obj[key] for key in ("name", "category")):
            raise ValueError("objects require nonempty name and category")
    part_ids, paths, part_faces = set(), set(), 0
    for part in surface.get("parts", []):
        identity = safe_id(part["id"])
        if identity.casefold() in part_ids or identity not in {"environment", *ids}:
            raise ValueError("source part identity is duplicated or absent from objects")
        path = checked_file(surface_path.parent, part["path"], part["sha256"])
        if path in paths:
            raise ValueError("source part path is duplicated")
        if type(part.get("triangles")) is not int or part["triangles"] <= 0:
            raise ValueError("source part requires a positive triangle count")
        part_ids.add(identity.casefold())
        paths.add(path)
        part_faces += part["triangles"]
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices, triangles = mesh_arrays(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    colors = np.asarray(mesh.vertex_colors)
    normals = np.asarray(mesh.vertex_normals)
    if (len(triangles) == 0 or colors.shape != vertices.shape or not np.isfinite(colors).all()
            or np.any(colors < 0) or np.any(colors > 1)
            or (len(normals) and (normals.shape != vertices.shape or not np.isfinite(normals).all()))):
        raise ValueError("surface requires nonempty finite triangles and vertex colors/normals")
    for doc in (record, surface):
        if doc.get("vertices") != len(vertices) or doc.get("triangles") != len(triangles):
            raise ValueError("surface vertex or triangle count differs from manifest")
    if part_faces != len(triangles):
        raise ValueError("source parts do not account for the triangle count")
    owners = np.load(owner_path, allow_pickle=False)
    if owners.ndim != 1 or owners.dtype.kind not in "iu" or len(owners) != len(triangles):
        raise ValueError("owners must be one integer label per source triangle")
    if np.any(owners < -2) or np.any(owners >= len(objects)):
        raise ValueError("owner label is outside the declared objects/environment/unresolved")
    return surface, observation, mesh, owners, method


def export_scene(directory, record, output, published_results_sha256):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    separate_output(output, directory)
    surface, observation, mesh, owners, method = validate_source(directory, record)
    started = time.perf_counter()
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    colors, normals = np.asarray(mesh.vertex_colors), np.asarray(mesh.vertex_normals)
    output.mkdir(parents=True, exist_ok=True)
    parts, all_faces, offset = [], [], 0
    for label in [-1, -2, *range(len(observation["objects"]))]:
        faces = np.flatnonzero(owners == label)
        if not len(faces):
            continue
        obj = observation["objects"][label] if label >= 0 else None
        identity = obj["uuid"] if obj else ("environment" if label == -1 else "unresolved")
        source_vertices, local_triangles = np.unique(triangles[faces], return_inverse=True)
        part = o3d.geometry.TriangleMesh()
        part.vertices = o3d.utility.Vector3dVector(vertices[source_vertices].copy())
        part.triangles = o3d.utility.Vector3iVector(local_triangles.reshape(-1, 3).astype(np.int32))
        part.vertex_colors = o3d.utility.Vector3dVector(colors[source_vertices].copy())
        if len(normals):
            part.vertex_normals = o3d.utility.Vector3dVector(normals[source_vertices].copy())
        path = output / f"{identity}.ply"
        if not o3d.io.write_triangle_mesh(str(path), part, write_ascii=False):
            raise OSError(f"could not write surface part: {identity}")
        # Verify the serialized artifact, including topology order and exact RGB8.
        readback = o3d.io.read_triangle_mesh(str(path))
        if not (np.array_equal(np.asarray(readback.vertices), vertices[source_vertices])
                and np.array_equal(np.asarray(readback.triangles), np.asarray(part.triangles))
                and np.array_equal(np.asarray(readback.vertex_colors), colors[source_vertices])
                and (not len(normals) or np.array_equal(np.asarray(readback.vertex_normals), normals[source_vertices]))):
            raise ValueError(f"serialized part changed source geometry, color or normals: {identity}")
        lineage = {}
        for kind, values in (("face", faces), ("vertex", source_vertices)):
            mapping = output / f"{identity}.source-{kind}-indices.npy"
            np.save(mapping, values, allow_pickle=False)
            lineage[kind] = {"path": mapping.name, "sha256": sha256(mapping), "count": len(values)}
        parts.append({"id": identity, "name": obj["name"] if obj else (
                          "Environment" if label == -1 else "Unresolved (-2)"),
                      "category": obj["category"] if obj else (
                          "Environment" if label == -1 else "Unresolved"),
                      "path": path.name, "sha256": sha256(path),
                      "role": "environment" if label == -1 else "observed_object_region",
                      "ownership_state": "object" if obj else identity, "owner_label": label,
                      "vertices": len(source_vertices), "triangles": len(faces),
                      "source_indices": lineage, "native_source_face_offset": offset})
        all_faces.append(faces)
        offset += len(faces)
    if not np.array_equal(np.sort(np.concatenate(all_faces)), np.arange(len(triangles))):
        raise AssertionError("source faces were lost or duplicated")
    np.save(output / "b1-face-owner.npy", owners, allow_pickle=False)
    if sha256(output / "b1-face-owner.npy") != method["face_owner_sha256"]:
        raise ValueError("saved owner array bytes differ from the frozen input")
    eligible = [part for part in parts if part["owner_label"] >= 0]
    result = {"schema": "spatial-scene-lab.surface.v1", "status": "pass",
              "scene_id": record["scene_id"], "units": "m", "up_axis": "Y",
              "method": "frozen B1 centroid ownership on unchanged S0 surface",
              "input_scene_sha256": record["scene_sha256"],
              "vertices": len(vertices), "triangles": len(triangles), "parts": parts,
              "cameras": surface.get("cameras", []), "structures": surface.get("structures", []),
              "preferred_editable_object": max(eligible, key=lambda p: p["triangles"])["id"] if eligible else None,
              "provenance": {"published_results_sha256": published_results_sha256,
                             "source_surface_sha256": record["surface_sha256"],
                             "source_surface_manifest_sha256": record["surface_manifest_sha256"],
                             "source_observation_sha256": record["scene_sha256"],
                             "face_owner": {"path": "b1-face-owner.npy", "sha256": method["face_owner_sha256"]},
                             "exporter_sha256": sha256(__file__), "reference_used": False, "mesh_changed": False},
              "partition": {"parameters": method["parameters"], "all_triangles_accounted_for": True,
                            "environment_triangles": int(np.count_nonzero(owners == -1)),
                            "unresolved_triangles": int(np.count_nonzero(owners == -2)),
                            "object_triangles": int(np.count_nonzero(owners >= 0)),
                            "geometry_color_normals_exact": True, "owner_array_bytes_exact": True},
              "identity_mapping": "native source face = part offset + local face index; source_indices.face maps that local index to the frozen source_surface_sha256 triangle",
              "unresolved_representation": "separate unresolved region (-2); observed_object_region role is the existing Blender transport type",
              "elapsed_seconds": time.perf_counter() - started}
    save_json(output / "surface-scene.json", result)
    return result


def export_bundles(results, published_results, output, selected=None):
    results, output = Path(results).resolve(), Path(output).resolve()
    published_results = Path(published_results).resolve()
    separate_output(output, results)
    if published_results == output or output in published_results.parents:
        raise ValueError("output must not contain published evidence")
    published = read_json(published_results)
    if published.get("schema") != "spatial-scene-lab.development.v1":
        raise ValueError("requires published development result manifest")
    digest = sha256(published_results)
    if sha256(results / "development-results.json") != digest:
        raise ValueError("prepared results differ from published result manifest")
    records = published["scenes"]
    names = [safe_id(record["scene_id"]) for record in records]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("published scene identities are duplicated")
    if selected and set(selected) - set(names):
        raise ValueError("requested scene is absent from published results")
    records = [record for record in records if not selected or record["scene_id"] in selected]
    if not records:
        raise ValueError("requires at least one published scene")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "spatial-scene-lab.partition-export.v1", "status": "running",
               "method": "b1", "published_results_sha256": digest,
               "exporter_sha256": sha256(__file__), "reference_used": False,
               "scope": "frozen source geometry and ownership export; scripted engineering acceptance",
               "scenes": []}
    started = time.perf_counter()
    for record in records:
        item = {"scene_id": record["scene_id"], "status": "fail"}
        try:
            bundle = output / record["scene_id"] / "surface"
            result = export_scene(member(results, record["scene_id"]), record, bundle, digest)
            item.update(status="pass", vertices=result["vertices"], triangles=result["triangles"],
                        partition=result["partition"], bundle=f"{record['scene_id']}/surface/surface-scene.json",
                        bundle_sha256=sha256(bundle / "surface-scene.json"))
        except Exception as error:
            item["error"] = f"{type(error).__name__}: {error}"
        summary["scenes"].append(item)
        summary["elapsed_seconds"] = time.perf_counter() - started
        save_json(output / "partition-export-results.json", summary)
        print(json.dumps(item), flush=True)
    summary["status"] = "pass" if all(item["status"] == "pass" for item in summary["scenes"]) else "fail"
    save_json(output / "partition-export-results.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--published-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene", action="append")
    args = parser.parse_args()
    result = export_bundles(args.results, args.published_results, args.output, args.scene)
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
