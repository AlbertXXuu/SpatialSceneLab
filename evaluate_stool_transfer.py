"""Evaluate frozen stool attempts, retaining incompatible, missing and failed outputs.

References are evaluation-only. No fitting is performed or imported here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import evaluate_stool_quality as q


PART_NAMES = ("seat", "leg_1", "leg_2", "leg_3", "brace_1_2", "brace_2_3", "brace_3_1")


def load_reference(directory, object_uuid="same-stool"):
    """Read physical components and captured rays without a fixed primitive count."""
    directory = Path(directory)
    metadata = json.loads((directory / "physical-scene.json").read_text(encoding="utf-8"))
    if metadata.get("units") != "m" or metadata.get("up_axis") != "Y":
        raise ValueError("reference requires metres and Y up")
    with np.load(directory / "labels.npz", allow_pickle=False) as labels:
        vertices, faces = labels["vertices"], labels["triangles"]
        owners, ids = labels["triangle_owner"], labels["object_ids"].tolist()
    q.arrays_mesh(vertices, faces)
    if owners.shape != (len(faces),) or ids.count(object_uuid) != 1:
        raise ValueError("invalid physical object ownership")
    owner_index = ids.index(object_uuid)
    target_indices = np.flatnonzero(owners == owner_index)
    target = q.arrays_mesh(vertices, faces[target_indices])
    if not len(target_indices):
        raise ValueError("target has no reference triangles")
    labels, _, _ = target.cluster_connected_triangles()
    labels = np.asarray(labels)
    components = sorted(np.unique(labels), key=lambda label: int(np.flatnonzero(labels == label)[0]))
    parts, part_by_face = [], np.full(len(faces), -1, dtype=int)
    declared = metadata.get("target_primitives", [])
    for index, component in enumerate(components):
        indices = target_indices[labels == component]
        part_by_face[indices] = index
        name = f"physical_component_{index}"
        if declared:
            if len(declared) != len(components):
                raise ValueError("declared physical primitives differ from connected geometry")
            record = declared[index]
            name = record["id"]
            if "global_triangle_indices" in record and sorted(record["global_triangle_indices"]) != indices.tolist():
                raise ValueError("declared primitive triangles differ from actual component")
            if "triangle_start" in record and list(range(record["triangle_start"], record["triangle_start"] +
                                                        record["triangle_count"])) != indices.tolist():
                raise ValueError("declared primitive triangle range differs from actual component")
        part_vertices = vertices[np.unique(faces[indices])]
        parts.append({"id": name, "reference_triangles": len(indices),
                      "global_triangle_index_first": int(indices.min()),
                      "global_triangle_index_last": int(indices.max()),
                      "world_bounds_m": [part_vertices.min(0).tolist(), part_vertices.max(0).tolist()]})
    intrinsic = np.asarray(metadata["intrinsics_depth"], dtype=float)
    width, height = metadata["depth_resolution"]
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or min(intrinsic[0, 0], intrinsic[1, 1]) <= 0:
        raise ValueError("invalid depth intrinsics")
    yy, xx = np.mgrid[:height, :width]
    directions = np.stack([(xx - intrinsic[0, 2]) / intrinsic[0, 0],
                           -(yy - intrinsic[1, 2]) / intrinsic[1, 1], -np.ones_like(xx)], axis=-1)
    files = sorted((directory / "frame_labels").glob("frame_*.npz"))
    poses = metadata["true_camera_poses_arkit"]
    if len(files) != len(poses) or not files:
        raise ValueError("reference frame count differs from camera poses or is empty")
    points, point_parts, frames, frame_hashes = [], [], [], []
    for index, path in enumerate(files):
        if path.name != f"frame_{index:05d}.npz":
            raise ValueError("reference frames are not contiguous")
        with np.load(path, allow_pickle=False) as frame:
            valid, ownership = frame["measurement_valid"], frame["instance_owner"]
            depth, triangles = frame["exact_depth_m"], frame["triangle_index"]
            if any(array.shape != (height, width) for array in (valid, ownership, depth, triangles)):
                raise ValueError("reference image resolution mismatch")
            mask = valid.astype(bool) & (ownership == owner_index)
            hit_triangles = triangles[mask]
            if (np.any(hit_triangles < 0) or np.any(hit_triangles >= len(faces)) or
                    np.any(owners[hit_triangles] != owner_index)):
                raise ValueError("inconsistent target ray labels")
            pose = np.asarray(poses[index], dtype=float)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError("invalid true camera pose")
            if np.any(~np.isfinite(depth[mask])) or np.any(depth[mask] <= 0):
                raise ValueError("invalid exact reference depth")
            local = directions[mask] * depth[mask, None]
            points.append(local @ pose[:3, :3].T + pose[:3, 3])
            point_parts.append(part_by_face[hit_triangles])
        frames.append({"frame_index": index, "valid_target_ray_hits": int(mask.sum())})
        frame_hashes.append({"path": str(path.resolve()), "sha256": q.sha256(path)})
    points, point_parts = np.concatenate(points), np.concatenate(point_parts)
    if "object_statistics" in metadata:
        expected = next(item["valid_ray_count"] for item in metadata["object_statistics"]
                        if item["object_id"] == object_uuid)
    elif declared and all("valid_ray_count" in part for part in declared):
        expected = sum(part["valid_ray_count"] for part in declared)
    else:
        raise ValueError("reference requires declared object or primitive ray counts")
    if len(points) != expected:
        raise ValueError("captured ray count differs from reference metadata")
    for index, part in enumerate(declared):
        if "valid_ray_count" in part and int(np.count_nonzero(point_parts == index)) != part["valid_ray_count"]:
            raise ValueError("captured primitive ray count differs from reference metadata")
    physical_scene = q.ray_scene(target)
    consistency = q.distances(physical_scene, points)
    if len(consistency) and consistency.max() > 1e-4:
        raise ValueError("reference rays disagree with physical triangles")
    provenance = {"directory": str(directory.resolve()), "object_uuid": object_uuid,
                  "files": [{"path": str((directory / name).resolve()), "sha256": q.sha256(directory / name)}
                            for name in ("labels.npz", "physical-scene.json")],
                  "frame_files": frame_hashes, "frames": frames, "valid_target_ray_hits": len(points),
                  "primitive_count": len(parts), "primitives": parts,
                  "ray_to_mesh_consistency": q.distance_summary(consistency)}
    return physical_scene, points, point_parts, parts, provenance


def convex_planes(mesh):
    """Actual outward triangle halfspaces, or None if they do not bound this solid."""
    if mesh is None or not len(mesh.triangles) or not mesh.is_watertight():
        return None
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    corners = vertices[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    sizes = np.linalg.norm(normals, axis=1)
    if np.any(sizes < 1e-12):
        return None
    normals /= sizes[:, None]
    offsets = -np.einsum("ij,ij->i", normals, corners[:, 0])
    if np.max(vertices @ normals.T + offsets) > 2e-8:
        return None
    return np.column_stack((normals, offsets))


def check_components(fit_dir, fit, assembly):
    """Retain each failed check rather than abandoning a low-quality candidate."""
    records = fit.get("editable_components", [])
    exact_names = [record.get("name") for record in records] == list(PART_NAMES)
    by_name = {record.get("name"): record for record in records}
    result, meshes, planes, hashes = [], {}, {}, {}
    vertices_list, faces_list, offset = [], [], 0
    for name in PART_NAMES:
        path = fit_dir / (name + ".ply")
        try:
            mesh, status = q.read_mesh(path)
        except (ValueError, RuntimeError) as error:
            mesh, status = None, str(error)
        if path.is_file():
            hashes[str(path.resolve())] = q.sha256(path)
        meshes[name] = mesh
        planes[name] = convex_planes(mesh)
        closed = bool(mesh is not None and len(mesh.triangles) and mesh.is_watertight())
        endpoint_errors = []
        if mesh is not None and len(mesh.vertices):
            vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            vertices_list.append(vertices)
            faces_list.append(faces + offset)
            offset += len(vertices)
            for endpoint in ("start_m", "end_m"):
                point = np.asarray(by_name.get(name, {}).get(endpoint, []), dtype=float)
                endpoint_errors.append(float(np.linalg.norm(vertices - point, axis=1).min())
                                       if point.shape == (3,) and np.isfinite(point).all() else None)
        result.append({"id": name, "status": status, "topology": q.topology(mesh),
                       "watertight": closed, "outward_convex": planes[name] is not None,
                       "record_endpoint_to_vertex_errors_m": endpoint_errors})
    same = bool(assembly is not None and len(vertices_list) == 7 and
                np.array_equal(np.concatenate(vertices_list), np.asarray(assembly.vertices)) and
                np.array_equal(np.concatenate(faces_list), np.asarray(assembly.triangles)))
    expected = [(f"leg_{i}", "end_m", "seat") for i in range(1, 4)]
    for first, second in ((1, 2), (2, 3), (3, 1)):
        expected.extend([(f"brace_{first}_{second}", "start_m", f"leg_{first}"),
                         (f"brace_{first}_{second}", "end_m", f"leg_{second}")])
    joints = []
    for source, endpoint, target in expected:
        point = np.asarray(by_name.get(source, {}).get(endpoint, []), dtype=float)
        signed, endpoint_error = None, None
        if (point.shape == (3,) and np.isfinite(point).all() and meshes[source] is not None and
                len(meshes[source].vertices)):
            endpoint_error = float(np.linalg.norm(np.asarray(meshes[source].vertices) - point, axis=1).min())
            if planes[target] is not None:
                signed = float(np.max(planes[target][:, :3] @ point + planes[target][:, 3]))
        passed = bool(signed is not None and signed < -1e-6 and endpoint_error is not None and
                      endpoint_error <= 2e-8 and planes[source] is not None)
        joints.append({"source": source, "endpoint": endpoint, "target": target,
                       "signed_target_face_plane_distance_m": signed,
                       "endpoint_to_actual_source_vertex_m": endpoint_error, "positive_solid_contact": passed})
    return {"components": result, "exact_seven_named_components": exact_names,
            "all_seven_closed": bool(exact_names and all(item["watertight"] for item in result)),
            "assembly_equals_component_files": same, "joints": joints,
            "all_nine_positive_solid_contacts": bool(all(item["positive_solid_contact"] for item in joints)),
            "frozen_files": hashes}


def evaluate(reference, scene_path, observations_path, fit_dir, object_uuid="same-stool"):
    physical_scene, points, point_parts, parts, provenance = load_reference(reference, object_uuid)
    scene_data = json.loads(scene_path.read_text(encoding="utf-8"))
    if scene_data.get("units") != "m" or scene_data.get("up_axis") != "Y":
        raise ValueError("observed scene requires metres and Y up")
    object_ids = [obj["uuid"] for obj in scene_data["objects"]]
    if object_ids.count(object_uuid) != 1:
        raise ValueError("target object must occur exactly once in scene")
    owner_index = object_ids.index(object_uuid)
    with np.load(observations_path, allow_pickle=False) as observation:
        all_points, all_frames, owners = observation["points"], observation["frame_index"], observation["owner"]
        if (all_points.ndim != 2 or all_points.shape[1] != 3 or not np.isfinite(all_points).all() or
                all_frames.shape != (len(all_points),) or owners.shape != (len(all_points),) or
                all_frames.dtype.kind not in "iu" or np.any(all_frames < 0)):
            raise ValueError("malformed observation arrays")
        chosen = owners == owner_index
        observed, frames = all_points[chosen], all_frames[chosen]
    inputs = {name: {"path": str(path.resolve()), "sha256": q.sha256(path)}
              for name, path in (("scene", scene_path), ("observations", observations_path))}
    fit_path, assembly_path = fit_dir / "fit.json", fit_dir / "fitted_stool.ply"
    fit, errors = {}, []
    if fit_path.is_file():
        try:
            fit = json.loads(fit_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as error:
            errors.append(f"invalid fit record: {error}")
    else:
        errors.append("fit.json absent; no candidate acceptance can be established")
    try:
        mesh, status = q.read_mesh(assembly_path)
    except (ValueError, RuntimeError) as error:
        mesh, status = None, "invalid"
        errors.append(f"invalid assembly: {error}")
    frozen = {str(path.resolve()): q.sha256(path) for path in (fit_path, assembly_path) if path.is_file()}
    input_match = bool(fit and all(fit.get("input_files", {}).get(name, {}).get("sha256") == record["sha256"]
                                  for name, record in inputs.items()))
    identity_match = bool(fit.get("object_uuid") == object_uuid and fit.get("source_owner_index") == owner_index and
                          fit.get("units") == "m" and fit.get("up_axis") == "Y")
    structure = check_components(fit_dir, fit, mesh)
    frozen.update(structure.pop("frozen_files"))
    candidate_scene = q.ray_scene(mesh)
    reference_distances = q.distances(candidate_scene, points)
    observed_distances = q.distances(candidate_scene, observed)
    validation = q.distance_summary(observed_distances[frames % 2 == 1])
    per_part = [{"id": part["id"], **q.distance_summary(reference_distances[point_parts == index])}
                for index, part in enumerate(parts)]
    supported_parts = [part for part in per_part if part["sample_count"] > 0]
    reverse = q.distance_summary(q.distances(physical_scene, q.sample_mesh_area(mesh)))
    p95_pass = validation["p95_distance_m"] is not None and validation["p95_distance_m"] <= .010
    checks = {"input_content_matches_fit_record": input_match, "target_and_units_match": identity_match,
              "validation_p95_at_most_10mm": bool(p95_pass),
              "seven_closed_components": structure["all_seven_closed"],
              "nine_solid_contacts": structure["all_nine_positive_solid_contacts"],
              "assembly_equals_components": structure["assembly_equals_component_files"]}
    engineering_pass = bool(not errors and all(checks.values()))
    fitter_pass = fit.get("quality_gate", {}).get("passed") is True
    report = {
        "schema": "alvenx.stool-transfer-evaluation.v1", "object_uuid": object_uuid,
        "script_sha256": q.sha256(__file__), "shared_evaluator_sha256": q.sha256(q.__file__),
        "open3d_version": q.o3d.__version__, "inputs": inputs, "reference": provenance,
        "fit_directory": str(fit_dir.resolve()), "fit_status": status, "errors": errors, "frozen_files": frozen,
        "declared_fitter_sha256": fit.get("input_files", {}).get("fitter", {}).get("sha256"),
        "fitter_quality_gate_passed": fitter_pass,
        "engineering_acceptance": {"passed": engineering_pass, "checks": checks,
                                   "criteria": "Validation triangle P95 <=10mm; seven closed named components; nine positive solid contacts; exact assembly and matching input identities."},
        "candidate_accepted": bool(engineering_pass and fitter_pass),
        "thresholds_m": list(q.THRESHOLDS),
        "valid_reference_ray_hit_recall": q.distance_summary(reference_distances),
        "per_reference_primitive_ray_hit_recall": per_part,
        "observed_primitive_balanced_recall": {
            "observed_primitive_count": len(supported_parts), "total_primitive_count": len(parts),
            "unobserved_primitive_ids": [part["id"] for part in per_part if part["sample_count"] == 0],
            "within": [{"distance_m": threshold,
                        "fraction": float(np.mean([part["within"][index]["fraction"] for part in supported_parts]))
                        if supported_parts else None} for index, threshold in enumerate(q.THRESHOLDS)],
        },
        "source_mesh_area_samples_to_physical_target": reverse,
        "source_area_sampling": {"sample_count_per_nonempty_mesh": q.SAMPLE_COUNT, "seed": q.SAMPLE_SEED,
                                 "method": "uniform barycentric point after triangle selection proportional to area"},
        "observed_point_mesh_distance": {"training_even": q.distance_summary(observed_distances[frames % 2 == 0]),
                                         "validation_odd": validation},
        "assembly_topology": q.topology(mesh), "structure": structure,
        "limitations": [
            "Ray recall counts captured valid rays, including repeated observations; it is not full or visible surface-area completeness.",
            "Unobserved physical components have null recall and are listed, never silently treated as perfect or removed from the total primitive count.",
            "Reverse accuracy samples all fitted primitive shells, including internal overlap faces, against authored reference shells; it is not exposed-union accuracy.",
            "Odd frames are a validation split; references are evaluation-only. This evaluator does not establish what files an upstream fitter read.",
            "Engineering pass is a bounded geometry criterion, not automatic family identification. Manifest labels must separately identify incompatible and insufficient cases and count any accepted negative as a failure.",
            "A missing fit remains an unaccepted row; its absence alone does not prove the fitter explicitly refused the input. Consult the retained attempt log.",
            "Joint proof checks actual closed convex source and target triangles and cap-center identity; overlapping editable solids are not a welded union.",
        ],
    }
    for path, digest in frozen.items():
        if q.sha256(path) != digest:
            raise ValueError(f"candidate changed during evaluation: {path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--fit-dir", type=Path)
    parser.add_argument("--object-uuid", default="same-stool")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(q.self_check()))
        if not args.reference:
            return
    if any(value is None for value in (args.reference, args.scene, args.observations, args.fit_dir, args.output)):
        parser.error("--reference --scene --observations --fit-dir --output are required")
    output = q.new_report_path(args.output)
    report = evaluate(args.reference, args.scene, args.observations, args.fit_dir, args.object_uuid)
    q.write_new_report(output, report)
    print(json.dumps({"output": str(output.resolve()), "fit_status": report["fit_status"],
                      "candidate_accepted": report["candidate_accepted"],
                      "engineering_acceptance": report["engineering_acceptance"],
                      "validation": report["observed_point_mesh_distance"]["validation_odd"]}, indent=2))


if __name__ == "__main__":
    main()
