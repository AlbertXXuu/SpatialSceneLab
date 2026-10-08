"""Stool geometry diagnostic; reference is read only and used for evaluation.

This is an audit script for the rejected same-color dining stool, not a benchmark
or a reconstruction input. Ray-hit recall counts valid captured pixel rays (with
repeat observations); it does not estimate full or visible surface area.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d


THRESHOLDS = (0.005, 0.010, 0.020)
SAMPLE_COUNT = 100_000
SAMPLE_SEED = 20261009


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_report_path(path):
    """A report must have a new path, including when an existing path is a link."""
    path = Path(path).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"report already exists; choose a new output path: {path}")
    return path


def write_new_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents a file appearing after the initial check
    # from being overwritten. Existing source files and reports are never replaced.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


def arrays_mesh(vertices, triangles):
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite N x 3")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in "iu":
        raise ValueError("triangles must be integer M x 3")
    if len(triangles) and (triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError("triangle vertex outside vertex array")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh


def read_mesh(path):
    if not path.is_file():
        return None, "missing"
    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh = arrays_mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    return mesh, "ok" if len(mesh.triangles) else "empty"


def ray_scene(mesh):
    if mesh is None or not len(mesh.triangles):
        return None
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.Dtype.Float32),
                        o3d.core.Tensor(np.asarray(mesh.triangles), dtype=o3d.core.Dtype.UInt32))
    return scene


def distances(scene, points):
    if not len(points):
        return np.empty(0, dtype=float)
    if scene is None:
        return np.full(len(points), np.inf)
    result = scene.compute_distance(o3d.core.Tensor(np.asarray(points), dtype=o3d.core.Dtype.Float32),
                                    nthreads=1).numpy().astype(float)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("raycasting returned invalid unsigned distances")
    return result


def distance_summary(values):
    values = np.asarray(values)
    count = len(values)
    finite = bool(count and np.isfinite(values).all())
    return {
        "sample_count": count,
        "finite_distance_count": int(np.isfinite(values).sum()),
        "mean_distance_m": float(values.mean()) if finite else None,
        "median_distance_m": float(np.median(values)) if finite else None,
        "p95_distance_m": float(np.quantile(values, .95)) if finite else None,
        "max_distance_m": float(values.max()) if finite else None,
        "within": [{"distance_m": threshold, "count": int(np.count_nonzero(values <= threshold)),
                    "fraction": float(np.mean(values <= threshold)) if count else None}
                   for threshold in THRESHOLDS],
    }


def topology(mesh):
    if mesh is None or not len(mesh.triangles):
        return {"vertices": 0 if mesh is None else len(mesh.vertices), "triangles": 0,
                "connected_triangle_components": 0, "boundary_edges": 0,
                "nonmanifold_edges": 0, "mesh_triangle_area_m2": 0.0,
                "largest_component_area_fraction": None, "largest_components": []}
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    _, component_counts, component_areas = mesh.cluster_connected_triangles()
    component_counts, component_areas = np.asarray(component_counts), np.asarray(component_areas)
    area = float(component_areas.sum())
    ordered = np.argsort(-component_areas, kind="stable")
    return {"vertices": len(vertices), "triangles": len(faces),
            "connected_triangle_components": len(component_counts),
            "component_method": "Open3D cluster_connected_triangles; original vertex indices, no welding",
            "boundary_edges": int(np.count_nonzero(counts == 1)),
            "boundary_edge_length_m": float(np.linalg.norm(vertices[edges[counts == 1, 0]] -
                                                          vertices[edges[counts == 1, 1]], axis=1).sum()),
            "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
            "mesh_triangle_area_m2": area,
            "largest_component_area_fraction": float(component_areas.max() / area) if area else None,
            "largest_components": [{"triangles": int(component_counts[index]),
                                    "area_m2": float(component_areas[index])} for index in ordered[:10]]}


def sample_mesh_area(mesh):
    if mesh is None or not len(mesh.triangles):
        return np.empty((0, 3), dtype=float)
    corners = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
    areas = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1) / 2
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("mesh has no finite positive-area triangles")
    rng = np.random.default_rng(SAMPLE_SEED)
    sampled = corners[rng.choice(len(corners), SAMPLE_COUNT, p=areas / areas.sum())]
    u, v = rng.random((2, SAMPLE_COUNT))
    root_u = np.sqrt(u)
    return (sampled[:, 0] * (1 - root_u[:, None]) +
            sampled[:, 1] * (root_u * (1 - v))[:, None] +
            sampled[:, 2] * (root_u * v)[:, None])


def load_reference(directory):
    metadata = json.loads((directory / "physical-scene.json").read_text(encoding="utf-8"))
    with np.load(directory / "labels.npz", allow_pickle=False) as labels:
        vertices, faces = labels["vertices"], labels["triangles"]
        owner, ids = labels["triangle_owner"], labels["object_ids"].tolist()
    owner_index = ids.index("same-stool")
    stool_indices = np.flatnonzero(owner == owner_index)
    stool_mesh = arrays_mesh(vertices, faces[stool_indices])
    labels, counts, _ = stool_mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    components = sorted(np.unique(labels), key=lambda label: int(np.flatnonzero(labels == label)[0]))
    if len(components) != 7 or not np.all(counts == 64):
        raise ValueError("this audit expects the original seven 64-triangle stool primitives")
    names = ["seat", "leg_local_angle_0", "leg_local_angle_120", "leg_local_angle_240",
             "brace_local_angles_0_120", "brace_local_angles_120_240", "brace_local_angles_240_360"]
    part_by_face = np.full(len(faces), -1, dtype=int)
    part_metadata = []
    for index, (component, name) in enumerate(zip(components, names)):
        global_indices = stool_indices[labels == component]
        part_by_face[global_indices] = index
        part_vertices = vertices[np.unique(faces[global_indices])]
        part_metadata.append({"id": name, "reference_triangles": len(global_indices),
                              "global_triangle_index_first": int(global_indices.min()),
                              "global_triangle_index_last": int(global_indices.max()),
                              "world_bounds_m": [part_vertices.min(0).tolist(), part_vertices.max(0).tolist()]})
    intrinsic = np.asarray(metadata["intrinsics_depth"])
    width, height = metadata["depth_resolution"]
    yy, xx = np.mgrid[:height, :width]
    directions = np.stack([(xx - intrinsic[0, 2]) / intrinsic[0, 0],
                           -(yy - intrinsic[1, 2]) / intrinsic[1, 1], -np.ones_like(xx)], axis=-1)
    points, point_parts, frame_counts, frame_digests = [], [], [], []
    files = sorted((directory / "frame_labels").glob("frame_*.npz"))
    poses = metadata["true_camera_poses_arkit"]
    if len(files) != len(poses):
        raise ValueError("reference frame count differs from true camera pose count")
    for index, path in enumerate(files):
        if path.name != f"frame_{index:05d}.npz":
            raise ValueError("reference frames are not contiguous")
        with np.load(path, allow_pickle=False) as frame:
            mask = frame["measurement_valid"] & (frame["instance_owner"] == owner_index)
            if mask.shape != (height, width):
                raise ValueError("reference resolution mismatch")
            triangles = frame["triangle_index"][mask]
            if np.any(triangles < 0) or np.any(owner[triangles] != owner_index):
                raise ValueError("inconsistent stool ray labels")
            pose = np.asarray(poses[index])
            local_points = directions[mask] * frame["exact_depth_m"][mask, None]
            world_points = local_points @ pose[:3, :3].T + pose[:3, 3]
        points.append(world_points)
        point_parts.append(part_by_face[triangles])
        frame_counts.append({"frame_index": index, "valid_stool_ray_hits": int(mask.sum())})
        frame_digests.append({"path": str(path.resolve()), "sha256": sha256(path)})
    points, point_parts = np.concatenate(points), np.concatenate(point_parts)
    expected = next(item["valid_ray_count"] for item in metadata["object_statistics"]
                    if item["object_id"] == "same-stool")
    if len(points) != expected or not len(points):
        raise ValueError("valid ray-hit denominator differs from reference metadata")
    physical_scene = ray_scene(stool_mesh)
    self_distances = distances(physical_scene, points)
    if self_distances.max() > 1e-4:
        raise ValueError("reference ray-hit reconstruction failed physical mesh consistency")
    return metadata, stool_mesh, physical_scene, points, point_parts, part_metadata, {
        "reference_directory": str(directory.resolve()),
        "files": [{"path": str((directory / name).resolve()), "sha256": sha256(directory / name)}
                  for name in ("labels.npz", "physical-scene.json")],
        "frame_files": frame_digests, "frames": frame_counts,
        "valid_stool_ray_hits": len(points), "reference_ray_consistency_tolerance_m": 1e-4,
        "reference_rays_to_physical_mesh": distance_summary(self_distances),
        "stool_primitives": part_metadata,
    }


def evaluate_variant(name, directory, physical_scene, points, point_parts, parts):
    result = {"name": name, "surface_directory": str(directory.resolve())}
    metadata_path = directory / "surface-scene.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        result["reconstruction_parameters"] = metadata.get("parameters")
        result["input_scene_sha256"] = metadata.get("input_scene_sha256")
    for key, filename in [("full_tsdf_surface", "surface.ply"), ("assigned_stool_mesh", "same-stool.ply")]:
        path = directory / filename
        mesh, status = read_mesh(path)
        values = distances(ray_scene(mesh), points)
        entry = {"path": str(path.resolve()), "status": status,
                 "sha256": sha256(path) if path.is_file() else None,
                 "valid_reference_ray_hit_recall": distance_summary(values),
                 "per_primitive_ray_hit_recall": [
                     {"id": part["id"], **distance_summary(values[point_parts == index])}
                     for index, part in enumerate(parts)]}
        if key == "assigned_stool_mesh":
            entry["assignment"] = "B0 original reconstruction split: three vertices share the same nonnegative hard-prior owner; not B1"
            entry["topology"] = topology(mesh)
            samples = sample_mesh_area(mesh)
            entry["source_mesh_area_samples_to_physical_stool"] = distance_summary(distances(physical_scene, samples))
        result[key] = entry
    return result


def self_check():
    mesh = arrays_mesh([[0., 0, 0], [1, 0, 0], [0, 1, 0]], np.asarray([[0, 1, 2]]))
    values = distances(ray_scene(mesh), [[.2, .2, 0], [.2, .2, .01], [.2, .2, .025]])
    np.testing.assert_allclose(values, [0, .01, .025], atol=1e-7)
    summary = distance_summary(np.asarray([0., .01, .025]))
    assert [item["count"] for item in summary["within"]] == [1, 2, 2]
    assert topology(mesh)["boundary_edges"] == 3
    assert topology(mesh)["connected_triangle_components"] == 1
    np.testing.assert_allclose(sample_mesh_area(mesh)[:, 2], 0)
    empty = arrays_mesh(np.empty((0, 3)), np.empty((0, 3), dtype=int))
    assert ray_scene(empty) is None and topology(empty)["triangles"] == 0
    missing_summary = distance_summary(distances(None, [[0., 0, 0]]))
    assert missing_summary["mean_distance_m"] is None
    assert all(item["fraction"] == 0 for item in missing_summary["within"])
    assert distance_summary([])["within"][0]["fraction"] is None
    try:
        arrays_mesh([[np.nan, 0, 0]], np.asarray([[0, 0, 0]]))
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite mesh input accepted")
    return {"status": "pass", "checks": ["analytic unsigned distances", "inclusive distance thresholds",
            "single-triangle topology", "area sample plane", "empty mesh", "missing mesh",
            "empty sample denominator", "non-finite input rejected"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--variant", action="append", default=[], help="name=surface-directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    checks = self_check()
    if args.self_check and not args.variant:
        print(json.dumps(checks))
        return
    if not args.reference or not args.output or not args.variant:
        parser.error("--reference, --variant, and --output are required for evaluation")
    args.output = new_report_path(args.output)
    variants = []
    for specification in args.variant:
        if "=" not in specification:
            parser.error("each --variant must be name=surface-directory")
        name, path = specification.split("=", 1)
        if not name or not path:
            parser.error("variant name and surface directory must be nonempty")
        variants.append((name, Path(path)))
    if len({name for name, _ in variants}) != len(variants):
        parser.error("variant names must be unique")
    metadata, stool_mesh, physical_scene, points, point_parts, parts, provenance = load_reference(args.reference)
    report = {
        "schema": "local-stool-geometry-audit.v1", "self_check": checks,
        "script_sha256": sha256(__file__), "open3d_version": o3d.__version__,
        "thresholds_m": list(THRESHOLDS), "reference": provenance,
        "reference_stool_topology": topology(stool_mesh),
        "definitions": {
            "ray_hit_denominator": "Every reference frame pixel with measurement_valid=true and instance_owner=same-stool, unweighted; repeated views/rays are retained. Coordinates use exact_depth_m, depth intrinsics and true camera poses.",
            "recall": "Fraction of these exact physical hit points whose unsigned nearest triangle distance is <= threshold. Full TSDF surface ignores ownership; assigned stool tests the delivered object region.",
            "accuracy": "100000 pseudorandom uniform-area source stool mesh samples, seed 20261009, distance to any physical stool triangle; sample fraction, not an exact surface-area integral.",
            "primitive_names": "Generator make_complex_fixture.py _furniture stool order: seat, legs at local 0/120/240 degrees, then braces between those angles. Seven disconnected reference primitives each have 64 triangles. Connectivity does not describe physical assembly integrity.",
        },
        "limitations": [
            "Ray-hit recall is observation-weighted coverage of captured valid rays, not visible-area coverage, full-surface completeness, or unseen-surface accuracy.",
            "Full TSDF nearest-surface matches can include nearby non-stool geometry, especially near the floor; assigned-stool metrics are reported alongside.",
            "Physical reference contains overlapping closed primitives and hidden internal triangles. Reverse accuracy can match those faces, so it is proximity to authored primitive geometry, not a physical outer-shell metric.",
            "Boundary edges and connected components use raw mesh topology; open observed surfaces can be legitimate. Counts are not a standalone quality score and depend on tessellation.",
            "The fixture has synthetic diffuse materials, 2 mm Gaussian depth noise, 1.5 percent pixel dropout, and no pose perturbation. This is a single local development diagnostic, not a real-room or held-out claim.",
        ],
        "sensor_parameters": metadata["sensor_parameters"],
        "source_area_sampling": {"method": "triangle selected proportional to area, uniform barycentric point",
                                 "sample_count_per_nonempty_stool_mesh": SAMPLE_COUNT, "seed": SAMPLE_SEED},
        "variants": [],
    }
    for name, path in variants:
        report["variants"].append(evaluate_variant(name, path, physical_scene, points, point_parts, parts))
        print(f"evaluated {name}", flush=True)
    write_new_report(args.output, report)
    print(str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
