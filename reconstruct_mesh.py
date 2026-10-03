"""CPU TSDF integration of prepared ScannerApp RGB-D observations. MIT."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import open3d as o3d

from scan_pipeline import assign_owners, load_frame


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_triangles(vertices, triangles, objects, structures, ownership):
    """A face is editable only when all three vertices have the same owner.

    Cross-boundary and ambiguous faces stay in the environment; the manifest
    reports them rather than pretending a box is a complete instance mask.
    """
    owner, candidates = assign_owners(
        vertices, objects, structures,
        ownership["box_margin_m"], ownership["structure_tolerance_m"],
    )
    corners = owner[triangles]
    face_owner = np.full(len(triangles), -1, dtype=np.int32)
    same = (corners[:, 0] >= 0) & np.all(corners == corners[:, :1], axis=1)
    face_owner[same] = corners[same, 0]
    return face_owner, owner, candidates


def reconstruct(scene_path, output_path, voxel_size=0.035, truncation=0.105):
    started = time.perf_counter()
    scene_path = Path(scene_path).resolve()
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    if scene.get("schema") != "spatial-scene-lab.observations.v1" or scene.get("units") != "m" or scene.get("up_axis") != "Y":
        raise ValueError("input must be metre / Y-up observations")
    if not np.isfinite([voxel_size, truncation]).all() or voxel_size <= 0 or truncation < voxel_size:
        raise ValueError("positive voxel size and truncation >= voxel size required")
    source = Path(scene["source"]).resolve()
    reference = o3d.io.read_point_cloud(str(source / "pointcloud.pcd"))
    if not len(reference.points) or not np.isfinite(np.asarray(reference.points)).all():
        raise ValueError("reference PCD must contain finite points")
    if not scene.get("input_files"):
        raise ValueError("input scan requires a frozen file-hash manifest")
    for name, expected in scene["input_files"].items():
        member = (source / name).resolve()
        if source not in member.parents or sha256(member) != expected:
            raise ValueError(f"input file changed or escaped scan directory: {name}")
    output = Path(output_path).resolve()
    if source == output or source in output.parents:
        raise ValueError("output must be outside input scan directory")
    output.mkdir(parents=True, exist_ok=True)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size, sdf_trunc=truncation,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    frame_records = []
    params = scene["parameters"]
    for metadata in scene["frames"]:
        frame_path = (source / metadata["source"]).resolve()
        if source not in frame_path.parents:
            raise ValueError("frame path must stay inside the input scan")
        frame = load_frame(frame_path)
        depth = frame["depth_mm"].copy()
        valid = ((depth > 0) & (depth != 65535)
                 & (depth <= params["max_depth_m"] * 1000)
                 & (frame["confidence"] >= params["min_confidence"]))
        depth[~valid] = 0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(frame["color"])),
            o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.uint16))),
            depth_scale=1000, depth_trunc=params["max_depth_m"], convert_rgb_to_intensity=False,
        )
        k = frame["intrinsics_depth"]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            depth.shape[1], depth.shape[0], k[0, 0], k[1, 1], k[0, 2], k[1, 2],
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(frame["camera_to_world_cv"]))
        frame_records.append({"source": metadata["source"], "valid_depth_pixels": int(valid.sum())})
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if not len(triangles) or not np.isfinite(vertices).all():
        raise ValueError("TSDF produced no finite surface")
    if not o3d.io.write_triangle_mesh(str(output / "surface.ply"), mesh, write_ascii=False):
        raise OSError("could not write fused surface")
    face_owner, owner, candidates = split_triangles(
        vertices, triangles, scene["objects"], scene["structures"], scene["ownership"],
    )
    parts = []
    colors = np.asarray(mesh.vertex_colors)
    for index in [-1, *range(len(scene["objects"]))]:
        faces = np.flatnonzero(face_owner == index)
        if not len(faces):
            continue
        record = scene["objects"][index] if index >= 0 else None
        identity = record["uuid"] if record else "environment"
        if not identity or any(char in identity for char in '/\\:') or identity in {'.', '..'}:
            raise ValueError("object identity must be safe for a surface filename")
        part = o3d.geometry.TriangleMesh()
        part.vertices = o3d.utility.Vector3dVector(vertices.copy())
        part.triangles = o3d.utility.Vector3iVector(triangles[faces].copy())
        part.vertex_colors = o3d.utility.Vector3dVector(colors.copy())
        part.remove_unreferenced_vertices()
        part.compute_vertex_normals()
        name = f"{identity}.ply"
        if not o3d.io.write_triangle_mesh(str(output / name), part, write_ascii=False):
            raise OSError(f"could not write surface part: {name}")
        boundary_faces = 0
        if index >= 0:
            boundary_faces = int(np.count_nonzero(
                (face_owner == -1) & np.any(candidates[triangles, index], axis=1)))
        parts.append({"id": identity, "name": record["name"] if record else "Environment",
                      "category": record["category"] if record else "Environment",
                      "path": name, "role": "observed_object_region" if record else "environment",
                      "vertices": len(part.vertices), "triangles": len(part.triangles),
                      "unmoved_candidate_boundary_triangles": boundary_faces,
                      "sha256": sha256(output / name)})
    # The scanner's PCD is a coordinate comparison, not independent ground truth.
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
    distances = np.asarray(cloud.compute_point_cloud_distance(reference))
    eligible = [part for part in parts if part["role"] == "observed_object_region"]
    if not eligible:
        raise ValueError("surface has no uniquely owned editable object region")
    preferred = max(eligible, key=lambda part: part["triangles"])
    result = {"schema": "spatial-scene-lab.surface.v1", "status": "pass",
              "units": "m", "up_axis": "Y", "method": "Open3D CPU ScalableTSDFVolume",
              "input_scene_sha256": sha256(scene_path), "frames": len(frame_records),
              "parameters": {"voxel_size_m": voxel_size, "sdf_truncation_m": truncation,
                             "depth_scale": 1000, "max_depth_m": params["max_depth_m"],
                             "min_confidence": params["min_confidence"]},
              "vertices": len(vertices), "triangles": len(triangles), "parts": parts,
              "preferred_editable_object": preferred["id"],
              "cameras": scene["frames"], "structures": scene["structures"],
              "reference_pcd": {"points": len(reference.points), "independent_ground_truth": False,
                                "mesh_vertex_distance_median_m": float(np.median(distances)),
                                "mesh_vertex_distance_p95_m": float(np.quantile(distances, .95))},
              "scope": "observed surface regions; instance completeness and physical accuracy unverified",
              "partition": {"all_triangles_accounted_for": sum(p["triangles"] for p in parts) == len(triangles),
                            "environment_triangles": int(np.count_nonzero(face_owner == -1)),
                            "ambiguous_vertices": int(np.count_nonzero(owner == -2))},
              "frame_records": frame_records, "elapsed_seconds": time.perf_counter() - started}
    save_json(output / "surface-scene.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--voxel-size", type=float, default=.035)
    parser.add_argument("--truncation", type=float, default=.105)
    args = parser.parse_args(argv)
    result = reconstruct(args.scene, args.output, args.voxel_size, args.truncation)
    print(json.dumps({key: result[key] for key in ["status", "frames", "vertices", "triangles",
                                                   "preferred_editable_object", "elapsed_seconds"]}))


if __name__ == "__main__":
    main()
