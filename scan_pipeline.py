"""Offline ScannerApp RGB-D observations, RoomPlan provenance and point edits.

This is a measured-observation baseline. RoomPlan boxes provide a geometric
selection prior, not a learned instance mask or a complete furniture model.
The USDA reader deliberately accepts only the parametric ScannerApp profile;
general USD assets belong in Blender's native importer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile

import numpy as np
from PIL import Image


SCHEMA = "spatial-scene-lab.observations.v1"
STRUCTURE_CATEGORIES = {"Wall", "Floor", "Door", "Window", "Opening"}
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _finite_array(value, shape, label):
    result = np.asarray(value, dtype=np.float64)
    if result.size != int(np.prod(shape)):
        raise ValueError(f"{label} must contain {int(np.prod(shape))} numbers")
    result = result.reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains non-finite values")
    return result


def camera_from_record(record, color_size, depth_size):
    """Return depth-resolution K and OpenCV-camera-to-ARKit-world transform.

    The upstream load_frame_info reads both arrays row-major. ARKit's camera
    looks down -Z with +Y up; RGB-D rays use +Z forward with +Y down. Only the
    camera's Y/Z columns are flipped. The world stays Y-up, in metres.
    """
    pose = _finite_array(record["cameraPoseARFrame"], (4, 4), "cameraPoseARFrame").copy()
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("camera pose has an invalid homogeneous row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5) or not np.isclose(
        np.linalg.det(rotation), 1, atol=2e-5
    ):
        raise ValueError("camera pose rotation is not a proper rigid transform")
    pose[:3, 1:3] *= -1
    k = _finite_array(record["intrinsics"], (3, 3), "intrinsics").copy()
    if (k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])
            or abs(np.linalg.det(k)) < 1e-12):
        raise ValueError("invalid pinhole intrinsics")
    if min(*color_size, *depth_size) <= 0:
        raise ValueError("image dimensions must be positive")
    k[0] *= depth_size[0] / color_size[0]
    k[1] *= depth_size[1] / color_size[1]
    return k, pose


def load_frame(frame_path):
    """Load a ScannerApp frame, retaining raw uint16 mm depth for TSDF users.

    Returns a dict with depth_mm, color (depth-sized uint8 RGB), confidence,
    intrinsics_depth, camera_to_world_cv, record and color_size. No downloads or
    files are written by this loader.
    """
    frame_path = Path(frame_path)
    source = frame_path.parent
    suffix = frame_path.stem.removeprefix("frame_")
    record = json.loads(frame_path.read_text(encoding="utf-8"))
    with Image.open(source / f"depth_{suffix}.png") as image:
        depth = np.asarray(image).copy()
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError(f"expected uint16 millimetre depth: depth_{suffix}.png")
    with Image.open(source / f"conf_{suffix}.png") as image:
        confidence = np.asarray(image).copy()
    if confidence.shape != depth.shape:
        raise ValueError(f"confidence dimensions differ from depth: conf_{suffix}.png")
    with Image.open(source / f"frame_{suffix}.jpg") as image:
        color_size = image.size
        color = np.asarray(image.convert("RGB").resize((depth.shape[1], depth.shape[0]), Image.Resampling.BILINEAR))
    k, pose = camera_from_record(record, color_size, (depth.shape[1], depth.shape[0]))
    return {"depth_mm": depth, "color": color, "confidence": confidence,
            "intrinsics_depth": k, "camera_to_world_cv": pose,
            "record": record, "color_size": color_size}


def backproject_rgbd(depth_mm, color, k, camera_to_world, confidence=None,
                     min_confidence=1, max_depth_m=6.0, pixel_step=1):
    """Backproject depth without interpolating millimetres or inventing holes.

    Color must already match the depth resolution. Returned UV coordinates
    preserve integer depth pixel positions; invalid/confidence-filtered rays
    never appear in the cloud. uint16 65535 is a sensor sentinel, not 65.535m.
    """
    depth = np.asarray(depth_mm)
    color = np.asarray(color)
    if depth.ndim != 2 or color.shape != (*depth.shape, 3):
        raise ValueError("depth must be HxW and color HxWx3 at the same resolution")
    if pixel_step < 1 or max_depth_m <= 0 or not np.isfinite(max_depth_m):
        raise ValueError("pixel_step and max_depth_m must be positive")
    k = _finite_array(k, (3, 3), "intrinsics")
    pose = _finite_array(camera_to_world, (4, 4), "camera_to_world")
    if confidence is not None and np.asarray(confidence).shape != depth.shape:
        raise ValueError("confidence shape does not match depth")
    y, x = np.mgrid[0:depth.shape[0]:pixel_step, 0:depth.shape[1]:pixel_step]
    sampled = depth[y, x].astype(np.float64)
    z = sampled / 1000.0
    valid = np.isfinite(z) & (z > 0) & (z <= max_depth_m) & (sampled != 65535)
    if confidence is not None:
        valid &= np.asarray(confidence)[y, x] >= min_confidence
    uv = np.column_stack([x[valid], y[valid]]).astype(np.int32)
    rays = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(k).T
    camera_points = rays * z[valid, None]
    points = camera_points @ pose[:3, :3].T + pose[:3, 3]
    colors = color[y[valid], x[valid]].astype(np.uint8)
    return points, colors, uv


def _tuple_numbers(text, label):
    return [float(x) for x in re.findall(NUMBER, text)]


def parse_roomplan(path):
    """Read this ScannerApp's parametric ASCII USDZ, with strict profile checks.

    Each directly referenced component contains one Cube or Floor Mesh and one
    authored matrix. Unsupported nesting/animation is rejected rather than
    silently treated as a world transform. USD matrices are row-vector, so
    transpose before using the column-vector convention in scene.json.
    """
    objects, structures = [], []
    with zipfile.ZipFile(path) as archive:
        usda = [x for x in archive.namelist() if x.lower().endswith(".usda")]
        roots = [x for x in usda if "/" not in x]
        if len(roots) != 1:
            raise ValueError("RoomPlan profile requires one root ASCII USDA layer")
        root = archive.read(roots[0]).decode("utf-8")
        if not re.search(r'upAxis\s*=\s*"Y"', root) or not re.search(
            r"metersPerUnit\s*=\s*1(?:\s|$)", root
        ):
            raise ValueError("RoomPlan must author Y-up and metresPerUnit=1")
        # The ScannerApp root has unrelated Section transforms, but the
        # Parametric_grp (which references these assets) must have none.
        parametric = root.split('def Xform "Parametric_grp"', 1)
        if len(parametric) != 2 or "xformOp:" in parametric[1]:
            raise ValueError("unsupported transformed or missing Parametric_grp")
        references = set(re.findall(r"@\./([^@]+\.usda)@", parametric[1]))
        if not references:
            raise ValueError("RoomPlan contains no parametric component references")
        for member in sorted(references):
            if member not in usda:
                raise ValueError(f"missing RoomPlan component: {member}")
            text = archive.read(member).decode("utf-8")
            name_match = re.search(r'def Xform\s+"([^\"]+)"', text)
            category_match = re.search(r'string Category\s*=\s*"([^\"]+)"', text)
            uuid_match = re.search(r'string UUID\s*=\s*"([^\"]+)"', text)
            matrices = re.findall(r"matrix4d xformOp:transform\s*=\s*([^\n]+)", text)
            if not name_match or not category_match or not uuid_match or len(matrices) != 1:
                raise ValueError(f"unsupported RoomPlan component metadata/transform: {member}")
            matrix = _finite_array(_tuple_numbers(matrices[0], member), (4, 4), member).T
            if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
                raise ValueError(f"invalid USD homogeneous transform: {member}")
            if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=2e-5):
                raise ValueError(f"unsupported non-rigid USD transform: {member}")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", uuid_match[1]):
                raise ValueError(f"unsafe RoomPlan UUID for local asset names: {member}")
            record = {"name": name_match[1], "uuid": uuid_match[1],
                      "category": category_match[1].split("(", 1)[0],
                      "category_label": category_match[1], "source_member": member,
                      "source_kind": "RoomPlan parametric measurement",
                      "transform_world": matrix.tolist()}
            cube = re.search(r'def Cube\s+"([^\"]+)"', text)
            if cube:
                scale_match = re.search(r"double3 xformOp:scale\s*=\s*\(([^)]+)\)", text)
                size_match = re.search(r"double size\s*=\s*(" + NUMBER + ")", text)
                order_match = re.search(r"xformOpOrder\s*=\s*\[([^\]]+)\]", text)
                if not scale_match or not size_match or not order_match or re.findall(
                    r'"([^\"]+)"', order_match[1]
                ) != ["xformOp:transform", "xformOp:scale"]:
                    raise ValueError(f"unsupported Cube transform order: {member}")
                dimensions = np.asarray(_tuple_numbers(scale_match[1], member)) * float(size_match[1])
                if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
                    raise ValueError(f"invalid Cube dimensions: {member}")
                record.update(geometry_kind="box", dimensions_m=dimensions.tolist())
            else:
                points_match = re.search(r"point3f\[\] points\s*=\s*\[([^\]]+)\]", text)
                indices_match = re.search(r"int\[\] faceVertexIndices\s*=\s*\[([^\]]+)\]", text)
                counts_match = re.search(r"int\[\] faceVertexCounts\s*=\s*\[([^\]]+)\]", text)
                if not points_match or not indices_match or not counts_match:
                    raise ValueError(f"unsupported RoomPlan geometry: {member}")
                points = np.asarray(_tuple_numbers(points_match[1], member)).reshape(-1, 3)
                indices = [int(x) for x in re.findall(r"[-+]?\d+", indices_match[1])]
                counts = [int(x) for x in re.findall(r"[-+]?\d+", counts_match[1])]
                if not counts or any(x != 3 for x in counts) or len(indices) != 3 * len(counts):
                    raise ValueError(f"only triangular RoomPlan Floor meshes are supported: {member}")
                if not np.isfinite(points).all() or not indices or min(indices) < 0 or max(indices) >= len(points):
                    raise ValueError(f"invalid RoomPlan mesh: {member}")
                record.update(geometry_kind="mesh", vertices_local_m=points.tolist(),
                              triangles=np.asarray(indices).reshape(-1, 3).tolist())
            (structures if record["category"] in STRUCTURE_CATEGORIES else objects).append(record)
    all_uuids = [x["uuid"] for x in objects + structures]
    if len(set(all_uuids)) != len(all_uuids):
        raise ValueError("RoomPlan UUIDs must be unique")
    return objects, structures


def _local_points(points, record):
    matrix = _finite_array(record["transform_world"], (4, 4), "RoomPlan transform")
    return (points - matrix[:3, 3]) @ np.linalg.inv(matrix[:3, :3]).T


def box_candidates(points, objects, margin_m=0.025):
    if margin_m < 0 or not np.isfinite(margin_m):
        raise ValueError("box margin must be finite and nonnegative")
    candidates = np.zeros((len(points), len(objects)), dtype=bool)
    for i, obj in enumerate(objects):
        local = _local_points(points, obj)
        half = np.asarray(obj["dimensions_m"], dtype=float) / 2
        candidates[:, i] = np.all(np.abs(local) <= half + margin_m, axis=1)
    return candidates


def structure_mask(points, structures, tolerance_m=0.025):
    """Exclude points on the finite authored floor/wall surfaces from objects."""
    if tolerance_m < 0 or not np.isfinite(tolerance_m):
        raise ValueError("structure tolerance must be finite and nonnegative")
    mask = np.zeros(len(points), dtype=bool)
    for record in structures:
        local = _local_points(points, record)
        if record["geometry_kind"] == "box":
            half = np.asarray(record["dimensions_m"], dtype=float) / 2
            thin_axis = int(np.argmin(half))
            within = np.all(np.abs(local) <= half + tolerance_m, axis=1)
            mask |= within & (np.abs(local[:, thin_axis]) <= half[thin_axis] + tolerance_m)
        elif record["category"] == "Floor":
            vertices = np.asarray(record["vertices_local_m"])
            triangles = np.asarray(record["triangles"])
            # Floor is an extruded planar polygon in its local XY plane.
            z = np.max(vertices[:, 2])
            near = np.abs(local[:, 2] - z) <= tolerance_m
            for triangle in triangles:
                a, b, c = vertices[triangle]
                if not np.allclose([a[2], b[2], c[2]], z, atol=1e-6):
                    continue
                v0, v1 = b[:2] - a[:2], c[:2] - a[:2]
                determinant = v0[0] * v1[1] - v0[1] * v1[0]
                if abs(determinant) < 1e-12:
                    continue
                v2 = local[:, :2] - a[:2]
                u = (v2[:, 0] * v1[1] - v2[:, 1] * v1[0]) / determinant
                v = (v0[0] * v2[:, 1] - v0[1] * v2[:, 0]) / determinant
                mask |= near & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
    return mask


def assign_owners(points, objects, structures=(), box_margin_m=0.025,
                  structure_tolerance_m=0.025):
    """Return owner labels and candidates; overlapping priors stay ambiguous.

    owner -1: environment, -2: unresolved overlapping object boxes, >=0: the
    object's index. Structure exclusion is an explicit parametric prior. It
    cannot establish semantic accuracy for an unlabelled real scan.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite Nx3 array")
    candidates = box_candidates(points, objects, box_margin_m)
    candidates[structure_mask(points, structures, structure_tolerance_m)] = False
    count = candidates.sum(axis=1).astype(np.int16)
    owner = np.full(len(points), -1, dtype=np.int32)
    owner[count > 1] = -2
    if objects:
        unique = count == 1
        owner[unique] = candidates[unique].argmax(axis=1)
    return owner, candidates


def voxel_select(points, voxel_size_m):
    """Keep one real sample per voxel, never average across ownership layers."""
    if voxel_size_m < 0 or not np.isfinite(voxel_size_m):
        raise ValueError("voxel_size_m must be finite and nonnegative")
    if voxel_size_m == 0 or not len(points):
        return np.arange(len(points))
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return np.sort(indices)


def write_ply(path, points, colors):
    """Write standard binary little-endian coloured point PLY."""
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors)
    if points.shape != (len(points), 3) or colors.shape != points.shape or not np.isfinite(points).all():
        raise ValueError("PLY requires finite Nx3 points and Nx3 colours")
    if np.any(np.abs(points) > np.finfo(np.float32).max):
        raise ValueError("PLY coordinates exceed float32 range")
    data = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, field in enumerate(["x", "y", "z"]):
        data[field] = points[:, i]
    for i, field in enumerate(["red", "green", "blue"]):
        data[field] = colors[:, i]
    header = ("ply\nformat binary_little_endian 1.0\ncomment SpatialSceneLab observed RGB-D samples\n"
              f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(data.tobytes())


def _hash_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _write_json(path, document):
    Path(path).write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_layers(output, scene, arrays, voxel_size_m=0.015):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    points, colors, owner = arrays["points"], arrays["colors"], arrays["owner"]
    selected = []
    layers = [{"owner": -1, "path": "environment.ply"},
              {"owner": -2, "path": "ambiguous.ply"}]
    layers += [{"owner": i, "path": f"objects/{obj['uuid']}.ply", "uuid": obj["uuid"]}
               for i, obj in enumerate(scene["objects"])]
    manifest = output / "scene.json"
    if manifest.is_file():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("schema") == SCHEMA:
            wanted = {layer["path"] for layer in layers}
            for old in previous.get("layers", []):
                relative = old.get("path", "")
                if relative in wanted or not re.fullmatch(r"objects/[A-Za-z0-9_-]+\.ply", relative):
                    continue
                obsolete = (output / relative).resolve()
                if output.resolve() in obsolete.parents and obsolete.is_file() and not obsolete.is_symlink():
                    obsolete.unlink()
    for layer in layers:
        indices = np.flatnonzero(owner == layer["owner"])
        keep = indices[voxel_select(points[indices], voxel_size_m)]
        selected.extend(keep.tolist())
        write_ply(output / layer["path"], points[keep], colors[keep])
        layer.update(observation_count=len(indices), exported_point_count=len(keep),
                     positions_sha256=_hash_array(points[indices]), colors_sha256=_hash_array(colors[indices]))
    all_keep = np.sort(np.asarray(selected, dtype=np.int64))
    write_ply(output / "observed.ply", points[all_keep], colors[all_keep])
    np.savez_compressed(output / "observations.npz", **arrays)
    scene["layers"] = layers
    scene["observations"] = "observations.npz"
    scene["observed_ply"] = "observed.ply"
    scene["export_voxel_size_m"] = voxel_size_m
    _write_json(output / "scene.json", scene)


def reconstruct(input_path, output_path, frame_step=1, pixel_step=2,
                min_confidence=1, max_depth_m=6, voxel_size_m=0.015,
                box_margin_m=0.025, structure_tolerance_m=0.025):
    started = time.perf_counter()
    source = Path(input_path).resolve()
    output = Path(output_path).resolve()
    if source == output or source in output.parents:
        raise ValueError("output must be outside the input scan directory")
    if frame_step < 1 or pixel_step < 1 or min_confidence not in (0, 1, 2):
        raise ValueError("positive frame/pixel steps and confidence 0/1/2 required")
    frame_paths = sorted(source.glob("frame_*.json"))[::frame_step]
    if not frame_paths:
        raise ValueError("input contains no frame_*.json")
    objects, structures = parse_roomplan(source / "room.usdz")
    point_sets, color_sets, uv_sets, index_sets, frames = [], [], [], [], []
    max_pixel_error, max_depth_error = 0.0, 0.0
    for frame_path in frame_paths:
        suffix = frame_path.stem.removeprefix("frame_")
        frame = load_frame(frame_path)
        record, depth, color, confidence = (frame[key] for key in ("record", "depth_mm", "color", "confidence"))
        color_size, k, pose = (frame[key] for key in ("color_size", "intrinsics_depth", "camera_to_world_cv"))
        points, colors, uv = backproject_rgbd(depth, color, k, pose, confidence,
                                             min_confidence, max_depth_m, pixel_step)
        # This inverse check detects code/calibration inconsistency; it is not
        # an independent measurement of real-world accuracy.
        if len(points):
            recovered = (points - pose[:3, 3]) @ np.linalg.inv(pose[:3, :3]).T
            projected = recovered @ k.T
            pixels = projected[:, :2] / projected[:, 2, None]
            max_pixel_error = max(max_pixel_error, float(np.max(np.abs(pixels - uv))))
            max_depth_error = max(max_depth_error, float(np.max(np.abs(recovered[:, 2] - depth[uv[:, 1], uv[:, 0]] / 1000.0))))
        frame_index = int(record.get("frame_index", suffix))
        point_sets.append(points)
        color_sets.append(colors)
        uv_sets.append(uv)
        index_sets.append(np.full(len(points), frame_index, dtype=np.int32))
        frames.append({"frame_index": frame_index, "source": frame_path.name,
                       "color_size": list(color_size), "depth_size": [depth.shape[1], depth.shape[0]],
                       "intrinsics_depth": k.tolist(), "camera_to_world_cv": pose.tolist(),
                       "retained_observations": len(points)})
    points, colors = np.concatenate(point_sets), np.concatenate(color_sets)
    if not len(points):
        raise ValueError("no valid RGB-D observations remain after depth/confidence filtering")
    owner, candidates = assign_owners(points, objects, structures, box_margin_m, structure_tolerance_m)
    arrays = {"points": points, "colors": colors, "pixel_uv": np.concatenate(uv_sets),
              "frame_index": np.concatenate(index_sets), "owner": owner,
              "candidates": candidates}
    input_members = {"room.usdz", "pointcloud.pcd"}
    for frame_path in frame_paths:
        suffix = frame_path.stem.removeprefix("frame_")
        input_members.update([frame_path.name, f"frame_{suffix}.jpg", f"depth_{suffix}.png", f"conf_{suffix}.png"])
    input_hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                    for name in sorted(input_members)}
    scene = {"schema": SCHEMA, "units": "m", "up_axis": "Y", "source": str(source),
             "input_files": input_hashes,
             "objects": objects, "structures": structures, "frames": frames,
             "ownership": {"method": "unique RoomPlan OBB with finite structure exclusion",
                           "box_margin_m": box_margin_m, "structure_tolerance_m": structure_tolerance_m,
                           "ambiguous_owner": -2, "environment_owner": -1,
                           "semantic_accuracy": "not measured; no pixel instance ground truth"},
             "parameters": {"frame_step": frame_step, "pixel_step": pixel_step,
                            "min_confidence": min_confidence, "max_depth_m": max_depth_m,
                            "depth_scale": 1000, "color_resampling": "Pillow bilinear to depth size"},
             "rights": {"raw_and_derived_scan_assets": "Consult the input license before redistribution."},
             "edits": []}
    _write_layers(output, scene, arrays, voxel_size_m)
    result = {"status": "pass", "scene": str(output / "scene.json"), "frames": len(frames),
              "observation_count": len(points), "object_count": len(objects),
              "structure_count": len(structures), "ambiguous_observations": int(np.count_nonzero(owner == -2)),
              "inverse_projection_check": {"max_pixel_error": max_pixel_error, "max_depth_error_m": max_depth_error,
                                           "scope": "numerical consistency, not physical accuracy"},
              "elapsed_seconds": time.perf_counter() - started}
    _write_json(output / "reconstruction-result.json", result)
    return result


def load_observations(scene_path):
    path = Path(scene_path).resolve()
    scene = json.loads(path.read_text(encoding="utf-8"))
    if scene.get("schema") != SCHEMA or scene.get("units") != "m" or scene.get("up_axis") != "Y":
        raise ValueError("unsupported observation scene schema or coordinates")
    with np.load(path.parent / scene["observations"], allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return scene, arrays


def verify_edit(before_scene, before, after_scene, after, target_index, translation, tolerance_m=1e-9):
    translation = _finite_array(translation, (3,), "translation")
    if before["points"].shape != after["points"].shape:
        return {"status": "fail", "checks": {"observation_shape_preserved": False},
                "complete_geometric_selection": False}
    expected = before["points"].copy()
    selected = before["owner"] == target_index
    expected[selected] += translation
    unchanged = ~selected
    maximum_error = float(np.max(np.abs(after["points"] - expected))) if len(expected) else 0.0
    after_objects = after_scene.get("objects", [])
    same_object_count = len(before_scene["objects"]) == len(after_objects)
    metadata_ok = same_object_count and all(before_scene["objects"][i] == after_objects[i]
                                           for i in range(len(after_objects)) if i != target_index)
    expected_transform = np.asarray(before_scene["objects"][target_index]["transform_world"]).copy()
    expected_transform[:3, 3] += translation
    expected_target = copy.deepcopy(before_scene["objects"][target_index])
    expected_target["transform_world"] = expected_transform.tolist()
    target_metadata_ok = same_object_count and expected_target == after_objects[target_index]
    # Layer hashes/counts are regenerated after a point translation, and the
    # edit log records that action. All other scene metadata must survive it.
    artifact_metadata = {"objects", "layers", "edits"}
    scene_metadata_ok = ({key: value for key, value in before_scene.items() if key not in artifact_metadata}
                         == {key: value for key, value in after_scene.items() if key not in artifact_metadata})
    source_arrays_ok = all(np.array_equal(before[key], after[key]) for key in before if key != "points")
    environment_indices = np.flatnonzero(after["owner"] < 0)
    original_target_indices = np.flatnonzero(selected)
    # Index/provenance identity proves no selected sample is also retained in
    # the environment. It does not claim that an OBB equals the real object.
    duplicate_target_indices = np.intersect1d(environment_indices, original_target_indices)
    unresolved = (before["owner"] == -2) & before["candidates"][:, target_index]
    checks = {"selected_observations": int(selected.sum()), "other_positions_identical": bool(
        np.array_equal(before["points"][unchanged], after["points"][unchanged])),
        "other_object_metadata_identical": metadata_ok, "colors_labels_provenance_identical": source_arrays_ok,
        "target_parametric_transform_matches": bool(target_metadata_ok),
        "target_identity_dimensions_metadata_preserved": bool(target_metadata_ok),
        "scene_metadata_preserved": scene_metadata_ok,
        "max_translation_error_m": maximum_error,
        "selected_sample_ids_remaining_in_environment": len(duplicate_target_indices),
        "unresolved_target_candidate_observations": int(unresolved.sum()),
        "all_target_candidates_accounted_for": not bool(unresolved.any())}
    passed = (checks["selected_observations"] > 0 and maximum_error <= tolerance_m
              and checks["other_positions_identical"] and metadata_ok and source_arrays_ok
              and target_metadata_ok and scene_metadata_ok
              and len(duplicate_target_indices) == 0)
    return {"status": "pass" if passed else "fail", "checks": checks,
            "scope": "selected observations only; semantic completeness unverified",
            "complete_geometric_selection": passed and not bool(unresolved.any())}


def edit_observations(scene_path, object_id, translation, output_path):
    source = Path(scene_path).resolve()
    output = Path(output_path).resolve()
    if output == source.parent:
        raise ValueError("edited scene must use a different output directory")
    before_scene, before = load_observations(source)
    matches = [i for i, obj in enumerate(before_scene["objects"]) if object_id in (obj["uuid"], obj["name"])]
    if len(matches) != 1:
        raise ValueError("object UUID/name must identify exactly one RoomPlan object")
    target = matches[0]
    translation = _finite_array(translation, (3,), "translation")
    if not np.any(translation):
        raise ValueError("translation must move the object")
    if not np.any(before["owner"] == target):
        raise ValueError("selected object has no uniquely owned observations")
    after_scene, after = copy.deepcopy(before_scene), {k: v.copy() for k, v in before.items()}
    after["points"][before["owner"] == target] += translation
    matrix = np.asarray(after_scene["objects"][target]["transform_world"])
    matrix[:3, 3] += translation
    after_scene["objects"][target]["transform_world"] = matrix.tolist()
    after_scene["edits"].append({"object_uuid": before_scene["objects"][target]["uuid"],
                                  "translation_m": translation.tolist(), "source_scene": str(source),
                                  "selection": "unique geometrical prior; ambiguous samples preserved"})
    _write_layers(output, after_scene, after, before_scene["export_voxel_size_m"])
    reopened_scene, reopened = load_observations(output / "scene.json")
    result = verify_edit(before_scene, before, reopened_scene, reopened, target, translation)
    result.update(object_uuid=before_scene["objects"][target]["uuid"], scene=str(output / "scene.json"))
    _write_json(output / "edit-result.json", result)
    if result["status"] != "pass":
        raise ValueError("edit did not preserve required source invariants")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reconstruct_parser = commands.add_parser("reconstruct", help="RGB-D to local coloured observation scene")
    reconstruct_parser.add_argument("--input", required=True, type=Path)
    reconstruct_parser.add_argument("--output", required=True, type=Path)
    reconstruct_parser.add_argument("--frame-step", type=int, default=1)
    reconstruct_parser.add_argument("--pixel-step", type=int, default=2)
    reconstruct_parser.add_argument("--min-confidence", type=int, choices=[0, 1, 2], default=1)
    reconstruct_parser.add_argument("--max-depth", type=float, default=6)
    reconstruct_parser.add_argument("--voxel-size", type=float, default=0.015)
    reconstruct_parser.add_argument("--box-margin", type=float, default=0.025)
    reconstruct_parser.add_argument("--structure-tolerance", type=float, default=0.025)
    edit_parser = commands.add_parser("edit", help="translate one object's uniquely selected real observations")
    edit_parser.add_argument("--scene", required=True, type=Path)
    edit_parser.add_argument("--object", required=True)
    edit_parser.add_argument("--translate", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    edit_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "reconstruct":
            result = reconstruct(args.input, args.output, args.frame_step, args.pixel_step,
                                 args.min_confidence, args.max_depth, args.voxel_size,
                                 args.box_margin, args.structure_tolerance)
        else:
            result = edit_observations(args.scene, args.object, args.translate, args.output)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(2, f"scan_pipeline: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
