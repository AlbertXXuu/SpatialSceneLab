"""Raycast the frozen stool transfer cases; reference labels never enter scan/.

Physical parts use the original fixture's triangle primitives and sensor
conventions. This wrapper does not import the fitter or read a fitted asset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures import make_complex_fixture as base

CONFIG = Path(__file__).with_name("stool-transfer-cases.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stool_parts(design):
    """Independent authored physical primitives, before box priors or rays."""
    count = design["legs"]
    bottom = design["seat_bottom_m"]
    thickness = design["seat_thickness_m"]
    vertices, triangles = base._cylinder([0, bottom, 0], [0, bottom + thickness, 0],
                                        design["seat_radius_m"])
    axes = design.get("seat_axes_scale", [1., 1.])
    vertices[:, 0] *= axes[0]
    vertices[:, 2] *= axes[1]
    parts = [("seat", vertices, triangles)]
    angles = np.arange(count) * 2 * np.pi / count
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    feet = directions * design["foot_radius_m"]
    tops = directions * design["top_radius_m"]
    # Sloped end caps are tangent to y=0 rather than penetrating the floor.
    radius = design["leg_radius_m"]
    horizontal = abs(design["foot_radius_m"] - design["top_radius_m"])
    foot_height = radius * horizontal / np.hypot(bottom, horizontal)
    for index, (foot, top) in enumerate(zip(feet, tops)):
        parts.append((f"leg_{index + 1}", *base._cylinder(
            [foot[0], foot_height, foot[1]], [top[0], bottom, top[1]], radius)))
    elevation = design["brace_height_m"]
    fraction = (elevation - foot_height) / (bottom - foot_height)
    brace_centres = feet + fraction * (tops - feet)
    for index in range(count):
        other = (index + 1) % count
        a, b = brace_centres[index], brace_centres[other]
        parts.append((f"brace_{index + 1}_{other + 1}", *base._cylinder(
            [a[0], elevation, a[1]], [b[0], elevation, b[1]], design["brace_radius_m"])))
    return parts


def physical_scene(config, case):
    parts = stool_parts(case["design"])
    transform = np.eye(4)
    transform[:3, :3] = base._yaw(case["yaw_degrees"])
    transform[:3, 3] = case["position_m"]
    points, faces, primitive_metadata = [], [], []
    vertex_offset = triangle_offset = 0
    for name, vertices, triangles in parts:
        points.append(vertices @ transform[:3, :3].T + transform[:3, 3])
        faces.append(triangles + vertex_offset)
        primitive_metadata.append({"id": name, "triangle_start": triangle_offset,
                                   "triangle_count": len(triangles)})
        vertex_offset += len(vertices)
        triangle_offset += len(triangles)
    local = np.concatenate([part[1] for part in parts])
    spec = {"id": "same-stool", "kind": "stool", "category": "Chair",
            "position_m": case["position_m"], "yaw_degrees": case["yaw_degrees"],
            "scale": [1., 1., 1.], "rgb": config["defaults"]["rgb"],
            "design": case["design"], "prior": config["defaults"]["prior"]}
    objects = [{"id": "same-stool", "kind": "stool", "category": "Chair",
                "transform_world": transform.tolist(),
                "local_bounds_m": [local.min(0).tolist(), local.max(0).tolist()],
                "physical_parameters": {k: v for k, v in spec.items() if k != "prior"}}]
    room_vertices, room_triangles, room_owners, room_colours, _, structures = base._build_scene(config, {"objects": []})
    vertices = np.concatenate([*points, room_vertices]).astype(np.float32)
    triangles = np.concatenate([*faces, room_triangles + vertex_offset]).astype(np.int32)
    owners = np.concatenate([np.zeros(triangle_offset, np.int32), room_owners])
    colours = np.concatenate([np.tile(spec["rgb"], (triangle_offset, 1)), room_colours]).astype(np.uint8)
    return vertices, triangles, owners, colours, objects, structures, primitive_metadata, spec


def generate_case(root, config, case, open3d, provenance):
    directory = root / case["id"]
    scan, reference = directory / "scan", directory / "reference"
    scan.mkdir(parents=True)
    reference.mkdir()
    frame_reference = reference / "frame_labels"
    frame_reference.mkdir()
    vertices, triangles, owners, colours, objects, structures, primitives, spec = physical_scene(config, case)
    base._write_ply(reference / "mesh.ply", vertices, triangles)
    np.savez_compressed(reference / "labels.npz", vertices=vertices, triangles=triangles,
                        triangle_owner=owners, object_ids=np.asarray(["same-stool"]))
    priors = base._measured_priors({"scene_id": case["id"], "objects": [spec]}, objects, structures)
    base._write_roomplan(scan / "room.usdz", priors)
    scene = open3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(open3d.core.Tensor(vertices), open3d.core.Tensor(triangles.astype(np.uint32)))
    width, height = config["defaults"]["resolution"]
    fx = fy = width * .85
    cx, cy = (width - 1) / 2, (height - 1) / 2
    yy, xx = np.mgrid[:height, :width]
    local_directions = np.stack([(xx - cx) / fx, -(yy - cy) / fy, -np.ones_like(xx)], axis=-1)
    trajectory = {**config["defaults"]["trajectory"], **case.get("trajectory", {})}
    trajectory["focus_m"] = [case["position_m"][0],
                               (case["design"]["seat_bottom_m"] + case["design"]["seat_thickness_m"]) / 2,
                               case["position_m"][2]]
    poses = base._trajectory({"trajectory": trajectory})
    sensor = {**config["defaults"]["sensor"], **case.get("sensor", {})}
    rng = np.random.default_rng(case["seed"])
    max_depth = config["defaults"]["maximum_depth_m"]
    normals = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                       vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    areas = np.linalg.norm(normals, axis=1) / 2
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.asarray([.3, .8, .4])
    lit_colours = np.rint(colours * (.72 + .28 * abs(normals @ (light / np.linalg.norm(light))))[:, None]).astype(np.uint8)
    hit_count = np.zeros(len(triangles), np.int64)
    frame_stats, measured_points = [], []
    for index, truth_pose in enumerate(poses):
        directions = local_directions @ truth_pose[:3, :3].T
        origins = np.broadcast_to(truth_pose[:3, 3], directions.shape)
        hits = scene.cast_rays(open3d.core.Tensor(np.concatenate([origins, directions], axis=-1).astype(np.float32)), nthreads=1)
        exact = hits["t_hit"].numpy().astype(float)
        ids = hits["primitive_ids"].numpy().astype(np.int64)
        geometric_valid = np.isfinite(exact) & (exact >= .15) & (exact <= max_depth)
        rgb = np.full((height, width, 3), [100, 120, 140], np.uint8)
        rgb[geometric_valid] = lit_colours[ids[geometric_valid]]
        labels = np.full((height, width), -3, np.int32)
        labels[geometric_valid] = owners[ids[geometric_valid]]
        depth = np.where(geometric_valid, exact, 0) + rng.normal(0, sensor["depth_noise_sigma_m"], (height, width))
        dropout = rng.random((height, width)) < sensor["dropout_probability"]
        if sensor["stripe_dropout_period_px"]:
            dropout |= ((xx + index * 3) % sensor["stripe_dropout_period_px"]) < 2
        if "keep_depth_rows" in sensor:
            start, stop = sensor["keep_depth_rows"]
            dropout |= (yy < start) | (yy >= stop)
        valid = geometric_valid & ~dropout & (depth >= .15) & (depth <= max_depth)
        depth_mm = np.where(valid, np.rint(depth * 1000), 0).astype(np.uint16)
        confidence = np.where(valid, 2, 0).astype(np.uint8)
        depth_mm[0, 0], depth_mm[0, 1], confidence[0, :2] = 0, 65535, 0
        valid[0, :2] = False
        np.add.at(hit_count, ids[valid], 1)
        pose = truth_pose.copy()
        pose[:3, 3] += rng.normal(0, sensor["pose_translation_sigma_m"], 3)
        yaw_error = rng.normal(0, sensor["pose_yaw_sigma_degrees"])
        pose[:3, :3] = base._yaw(yaw_error) @ truth_pose[:3, :3]
        suffix = f"{index:05d}"
        Image.fromarray(depth_mm).save(scan / f"depth_{suffix}.png")
        Image.fromarray(confidence).save(scan / f"conf_{suffix}.png")
        Image.fromarray(rgb).resize((width * 2, height * 2), Image.Resampling.NEAREST).save(scan / f"frame_{suffix}.jpg", quality=95, subsampling=0)
        base._json(scan / f"frame_{suffix}.json", {"frame_index": index,
                   "cameraPoseARFrame": pose.ravel().tolist(),
                   "intrinsics": [fx * 2, 0, cx * 2, 0, fy * 2, cy * 2, 0, 0, 1]})
        np.savez_compressed(frame_reference / f"frame_{suffix}.npz", instance_owner=labels,
                            triangle_index=np.where(geometric_valid, ids, -1).astype(np.int32),
                            exact_depth_m=np.where(geometric_valid, exact, 0).astype(np.float32), measurement_valid=valid)
        camera_points = local_directions[valid] * (depth_mm[valid, None].astype(float) / 1000)
        measured_points.append(camera_points @ pose[:3, :3].T + pose[:3, 3])
        frame_stats.append({"frame_index": index, "geometric_hit_pixels": int(geometric_valid.sum()),
                            "valid_measurement_pixels": int(valid.sum()),
                            "target_hit_pixels": int(np.count_nonzero(labels == 0)),
                            "target_valid_pixels": int(np.count_nonzero((labels == 0) & valid))})
    points = np.concatenate(measured_points)
    with (scan / "pointcloud.pcd").open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# .PCD v0.7 - measured RGB-D with measured poses\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                     f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA ascii\n")
        np.savetxt(stream, points, fmt="%.9g")
    np.savez_compressed(reference / "visibility.npz", triangle_hit_count=hit_count,
                        triangle_seen=hit_count > 0, triangle_area_m2=areas)
    for primitive in primitives:
        start, size = primitive["triangle_start"], primitive["triangle_count"]
        primitive["valid_ray_count"] = int(hit_count[start:start + size].sum())
    base._json(reference / "physical-scene.json", {
        "schema": "spatial-scene-lab.physical-reference.v1", "generator_version": config["version"],
        "scene_id": case["id"], "seed": case["seed"], "units": "m", "up_axis": "Y",
        "licence": config["licence"], **provenance, "objects": objects, "structures": structures,
        "true_camera_poses_arkit": [pose.tolist() for pose in poses], "depth_resolution": [width, height],
        "intrinsics_depth": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], "sensor_parameters": sensor,
        "trajectory": trajectory, "prior_perturbations": {"same-stool": spec["prior"]},
        "frame_statistics": frame_stats, "target_primitives": primitives,
        "reference_construction": "Independent authored triangle primitives, first-hit CPU rays. Box priors are derived after physical geometry. No fitter or fitted asset is read.",
        "visibility_rule": "Valid ray counts include repeated observations; hit-triangle area is not visible surface area.",
        "known_limitations": ["Authored synthetic geometry with simple diffuse shading; no real sensor calibration.",
                              "Operator-selected topology and near-correct box priors; this is not automatic family discovery.",
                              "Primitive overlap and hidden surfaces remain in the reference mesh."]})
    return {"scene_id": case["id"], "scan": f"{case['id']}/scan", "reference_dir": f"{case['id']}/reference",
            "target_id": "same-stool", "expected_class": case["class"], "expected": case["expected"],
            "frames": len(poses), "target_valid_rays": sum(item["target_valid_pixels"] for item in frame_stats),
            "primitive_valid_rays": {p["id"]: p["valid_ray_count"] for p in primitives}}


def generate(output):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    root = Path(output).expanduser().absolute()
    for path in [*root.parents, root]:
        if base._is_link_or_junction(path):
            raise ValueError(f"Output must not traverse a link or junction: {path}")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError("Output must be new or empty; previous generated cases are never overwritten")
    import open3d
    provenance = {"config_sha256": sha256(CONFIG), "generator_sha256": sha256(__file__),
                  "shared_geometry_sensor_helper_sha256": sha256(base.__file__)}
    root.mkdir(parents=True, exist_ok=True)
    (root / "frozen-config.json").write_bytes(CONFIG.read_bytes())
    manifest = {"schema": "spatial-scene-lab.stool-transfer-suite.v1", **provenance,
                "units": "m", "up_axis": "Y", "depth_resolution": config["defaults"]["resolution"],
                "split": config["split"], "parameter_basis": config["parameter_basis"],
                "dependencies": {"numpy": np.__version__, "open3d": open3d.__version__},
                "scenes": [generate_case(root, config, case, open3d, provenance) for case in config["cases"]]}
    base._json(root / "suite-manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.output), indent=2))
