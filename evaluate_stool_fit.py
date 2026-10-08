"""Evaluate the frozen fitted stool; never fit or alter reconstruction geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import evaluate_stool_quality as q


def balanced(parts):
    if any(part["sample_count"] == 0 for part in parts):
        raise ValueError("all seven reference components require valid rays")
    return {
        "definition": "Arithmetic mean of seven per-reference-primitive statistics; each primitive has equal weight",
        "primitive_count": len(parts),
        "mean_distance_m": float(np.mean([part["mean_distance_m"] for part in parts])),
        "within": [{"distance_m": threshold,
                    "fraction": float(np.mean([part["within"][index]["fraction"] for part in parts]))}
                   for index, threshold in enumerate(q.THRESHOLDS)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True, help="frozen fit.json; component PLY files must be alongside it")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--fitter", type=Path, help="optional exact historical fitter source for hash verification")
    parser.add_argument("--previous-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = q.new_report_path(args.output)
    previous = json.loads(args.previous_metrics.read_text(encoding="utf-8"))
    fit_path, fit_dir = args.fit, args.fit.parent
    mesh_path = fit_dir / "fitted_stool.ply"
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    frozen_hashes = {str(path.resolve()): q.sha256(path) for path in [fit_path, mesh_path]}
    _, _, physical_scene, points, point_parts, parts, reference = q.load_reference(args.reference)
    def reference_signature(value):
        # Content identities stay comparable when historical files move to a
        # different checkout or OS. Do not resolve old machine-specific paths.
        files = [(item["path"].replace("\\", "/").rsplit("/", 1)[-1], item["sha256"])
                 for item in value["files"] + value["frame_files"]]
        return sorted(files), value["frames"], value["valid_stool_ray_hits"]

    if (reference_signature(reference) != reference_signature(previous["reference"]) or
            list(q.THRESHOLDS) != previous["thresholds_m"]):
        raise ValueError("reference contents or thresholds changed since TSDF evaluation")
    if (previous["source_area_sampling"]["sample_count_per_nonempty_stool_mesh"] != q.SAMPLE_COUNT or
            previous["source_area_sampling"]["seed"] != q.SAMPLE_SEED):
        raise ValueError("source area sampling differs from previous metrics")
    mesh, status = q.read_mesh(mesh_path)
    if status != "ok":
        raise ValueError("fitted mesh is missing or empty")
    scene = q.ray_scene(mesh)
    ray_distances = q.distances(scene, points)
    per_part = [{"id": part["id"], **q.distance_summary(ray_distances[point_parts == index])}
                for index, part in enumerate(parts)]
    source_samples = q.sample_mesh_area(mesh)
    source_distances = q.distances(physical_scene, source_samples)
    reverse = q.distance_summary(source_distances)
    component_records = fit["editable_components"]
    expected_names = ["seat", "leg_1", "leg_2", "leg_3", "brace_1_2", "brace_2_3", "brace_3_1"]
    if [part["name"] for part in component_records] != expected_names:
        raise ValueError("expected the stool's seven named components in assembly order")
    component_meshes, hulls, component_checks = {}, {}, []
    concatenated_vertices, concatenated_faces, offset = [], [], 0
    floor = float(fit["contact_checks"]["floor_y_m"])
    for index, record in enumerate(component_records):
        name = record["name"]
        path = fit_dir / (name + ".ply")
        part_mesh, part_status = q.read_mesh(path)
        if part_status != "ok":
            raise ValueError(f"invalid component: {name}")
        frozen_hashes[str(path.resolve())] = q.sha256(path)
        component_meshes[name] = part_mesh
        vertices, faces = np.asarray(part_mesh.vertices), np.asarray(part_mesh.triangles)
        corners = vertices[faces]
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        offsets = -np.einsum("ij,ij->i", normals, corners[:, 0])
        if np.max(vertices @ normals.T + offsets) > 2e-8:
            raise ValueError(f"component is not convex with consistent outward faces: {name}")
        hulls[name] = np.column_stack((normals, offsets))
        start, end = np.asarray(record["start_m"]), np.asarray(record["end_m"])
        axis = (end - start) / np.linalg.norm(end - start)
        axial = (vertices - start) @ axis
        radial = np.linalg.norm(vertices - start - axial[:, None] * axis, axis=1)
        rim = radial > record["radius_m"] * .5
        endpoint_errors = [float(np.linalg.norm(vertices - endpoint, axis=1).min()) for endpoint in (start, end)]
        volume = float(np.einsum("ij,ij->i", vertices[faces[:, 0]],
                               np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]])).sum() / 6)
        part_checks = {
            "id": name, "provenance_role": "fitted/structural", "path": str(path.resolve()),
            "sha256": frozen_hashes[str(path.resolve())], "topology": q.topology(part_mesh),
            "watertight_individual_mesh": bool(part_mesh.is_watertight()),
            "positive_signed_volume_m3": volume,
            "endpoint_centers_nearest_vertex_error_m": endpoint_errors,
            "max_rim_radius_error_against_fit_record_m": float(np.abs(radial[rim] - record["radius_m"]).max()),
            "vertex_min_y_m": float(vertices[:, 1].min()),
            "floor_clearance_m": float(vertices[:, 1].min() - floor) if name.startswith("leg_") else None,
            "source_area_samples_to_physical_stool": q.distance_summary(q.distances(physical_scene, q.sample_mesh_area(part_mesh))),
        }
        if max(endpoint_errors) > 2e-8 or not part_checks["watertight_individual_mesh"] or volume <= 0:
            raise ValueError(f"invalid closed component or mismatched endpoint record: {name}")
        component_checks.append(part_checks)
        concatenated_vertices.append(vertices)
        concatenated_faces.append(faces + offset)
        offset += len(vertices)
    same_assembly = (np.array_equal(np.concatenate(concatenated_vertices), np.asarray(mesh.vertices)) and
                     np.array_equal(np.concatenate(concatenated_faces), np.asarray(mesh.triangles)))
    if not same_assembly:
        raise ValueError("assembly mesh differs from concatenation of the seven component files")
    records_by_name = {part["name"]: part for part in component_records}
    expected_joints = [(f"leg_{i}", "end_m", "seat") for i in range(1, 4)]
    for first, second in [(1, 2), (2, 3), (3, 1)]:
        expected_joints += [(f"brace_{first}_{second}", "start_m", f"leg_{first}"),
                            (f"brace_{first}_{second}", "end_m", f"leg_{second}")]
    joints = []
    for source, endpoint, target in expected_joints:
        point = np.asarray(records_by_name[source][endpoint])
        equations = hulls[target]
        signed = float(np.max(equations[:, :3] @ point + equations[:, 3]))
        joints.append({"source_part": source, "source_endpoint": endpoint, "target_part": target,
                       "max_signed_target_convex_face_plane_distance_m": signed,
                       "strictly_inside_actual_target_polyhedron": bool(signed < -1e-6)})
    if not all(joint["strictly_inside_actual_target_polyhedron"] for joint in joints):
        raise ValueError("one or more intended joints lack proven positive solid overlap")
    observation_path, scene_path = args.observations, args.scene
    input_paths = [("observations", observation_path), ("scene", scene_path)]
    if args.fitter:
        input_paths.append(("fitter", args.fitter))
    for name, path in input_paths:
        if q.sha256(path) != fit["input_files"][name]["sha256"]:
            raise ValueError(f"fitted input or code has changed: {name}")
    observed_scene = json.loads(scene_path.read_text(encoding="utf-8"))
    measured_floor = max(np.asarray(item["transform_world"])[1, 3] + item["dimensions_m"][1] / 2
                         for item in observed_scene["structures"] if item["category"] == "Floor")
    if abs(measured_floor - floor) > 1e-10:
        raise ValueError("fit floor differs from supplied scene Floor prior")
    with np.load(observation_path, allow_pickle=False) as observations:
        chosen = observations["owner"] == fit["source_owner_index"]
        observation_points = observations["points"][chosen]
        frames = observations["frame_index"][chosen]
    observation_distances = q.distances(scene, observation_points)
    if not np.any(frames % 2 == 0) or not np.any(frames % 2 == 1):
        raise ValueError("both even training and odd validation observations are required")
    outside_5mm = source_distances > .005
    seat_planes = hulls["seat"]
    inside_seat = np.max(source_samples @ seat_planes[:, :3].T + seat_planes[:, 3], axis=1) < -1e-6
    # Reference ray order is frame order from load_reference; no fitting uses these data.
    reference_frames = np.concatenate([np.full(frame["valid_stool_ray_hits"], frame["frame_index"])
                                       for frame in reference["frames"]])
    frame_groups = {"even_frames_used_by_fitter": reference_frames % 2 == 0,
                    "odd_frames_used_for_fitter_validation": reference_frames % 2 == 1}
    output = {
        "schema": "local-frozen-stool-fit-geometry-audit.v1", "provenance_role": "fitted/structural",
        "self_check": q.self_check(), "script_sha256": q.sha256(__file__),
        "shared_evaluator_sha256": q.sha256(q.__file__), "same_reference_as_tsdf_evaluation": True,
        "prior_evaluator_sha256": previous["script_sha256"],
        "evaluator_source_file_identical_to_prior": q.sha256(q.__file__) == previous["script_sha256"],
        "fitter_source_hash_verified": bool(args.fitter),
        "declared_fitter_sha256": fit["input_files"]["fitter"]["sha256"],
        "prior_metrics_path": str(args.previous_metrics.resolve()), "prior_metrics_sha256": q.sha256(args.previous_metrics),
        "reference": reference, "thresholds_m": list(q.THRESHOLDS),
        "frozen_files": frozen_hashes,
        "valid_reference_ray_hit_recall": q.distance_summary(ray_distances),
        "per_reference_primitive_ray_hit_recall": per_part,
        "reference_primitive_balanced_ray_hit_recall": balanced(per_part),
        "reference_ray_recall_by_frame_group": {name: q.distance_summary(ray_distances[mask]) for name, mask in frame_groups.items()},
        "source_mesh_area_samples_to_physical_stool": reverse,
        "reverse_error_tail_location": {
            "samples_outside_5mm": int(outside_5mm.sum()),
            "outside_5mm_samples_inside_fitted_seat": int(np.count_nonzero(outside_5mm & inside_seat)),
            "definition": "Inside uses strict halfspaces of the actual convex fitted seat; this localizes discrepancies without removing them from accuracy",
        },
        "source_area_sampling": previous["source_area_sampling"],
        "assembly_topology": q.topology(mesh), "assembly_equals_seven_component_files": same_assembly,
        "components": component_checks,
        "joint_forensics": {"method": "Source cap-center endpoint strictly inside actual target component convex face halfspaces; each target is a closed convex 80-sided cylinder. Positive interior margin proves solid overlap for these frozen components, not welding.",
                            "intended_joint_count": len(joints), "all_nine_positive_solid_overlaps": True, "joints": joints},
        "observed_point_mesh_distance": {"definition": "Unsigned distance to actual exported primitive triangle shells, independently replacing the fitter's min-SDF residual",
                                         "training_even": q.distance_summary(observation_distances[frames % 2 == 0]),
                                         "validation_odd": q.distance_summary(observation_distances[frames % 2 == 1])},
        "interpretation": {
            "reference_scope": "The reference is used only by this evaluator. This report cannot establish what any upstream fitter read or whether a model family was selected from prior knowledge.",
            "train_validation": "Odd frames are validation because the fitter prototype gate uses their errors for acceptance. This report cannot establish an untouched holdout across earlier iterations.",
            "residual_limit": "Fitter abs(min signed cylinder distances) is not exact distance to union boundary in overlap interiors. Closest-part labels use argmin(abs signed distances), which can differ from the primitive providing the residual. Independent metrics here query triangles.",
            "gate_limit": "Fitter gate checks balanced mean errors and >=5 training supports per primitive, not tail error or a minimum validation support per primitive. It is correctly labeled a bounded prototype gate, not product quality or unique recovery.",
        },
        "limitations": [
            *previous["limitations"][:1],
            "All reverse samples include hidden/internal primitive surfaces in both fitted and physical reference geometry. Accuracy is proximity to authored shells, not exposed union surface accuracy.",
            "Same development fixture and operator-selected seven-part family; these metrics do not establish generalization or unseen-region correctness. Geometry was frozen before this reference evaluation.",
            "Seven individually watertight overlapping shells constitute an editable assembly, not a welded watertight union mesh or a completed full scene.",
            "Reference primitive-balanced recall gives each of the seven reference components equal weight; reverse accuracy remains uniform fitted-source-area sampling.",
        ],
    }
    for path, expected in frozen_hashes.items():
        if q.sha256(path) != expected:
            raise ValueError(f"fitted geometry changed during evaluation: {path}")
    q.write_new_report(args.output, output)
    print(json.dumps({"output": str(args.output.resolve()), "ray_recall": output["valid_reference_ray_hit_recall"],
                      "balanced": output["reference_primitive_balanced_ray_hit_recall"], "reverse_accuracy": reverse,
                      "joints": len(joints), "topology": output["assembly_topology"]}, indent=2))


if __name__ == "__main__":
    main()
