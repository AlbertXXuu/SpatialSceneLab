"""Generate six original multi-part indoor RGB-D development scenes on CPU.

Physical geometry is authored before the independently perturbed RoomPlan box
priors. Ray labels and exact cameras live only in reference/, while scan/ uses
measured depth and measured poses. Parameters are declared stress hypotheses,
not a calibration of a real camera. No external assets or downloads are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import stat
import struct
import zipfile

import numpy as np
from PIL import Image


CONFIG_PATH = Path(__file__).with_name("complex-development-scenes.json")
VERSION = "2026-10-04.1"


def _is_link_or_junction(path):
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
        return True
    # Python 3.11 has no Path.is_junction; Windows lstat still exposes reparse
    # attributes. Reject other reparse entries too, rather than following them.
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _checked_output(output):
    """Inspect lexical ancestors and every existing entry before any writes."""
    root = Path(output).expanduser().absolute()
    for path in [*reversed(root.parents), root]:
        if _is_link_or_junction(path):
            raise ValueError(f"refusing output through link or junction: {path}")
    if root.exists():
        if not root.is_dir():
            raise FileExistsError(f"fixture output is an existing file: {root}")
        pending = [root]
        nonempty = False
        while pending:
            directory = pending.pop()
            for path in directory.iterdir():
                nonempty = True
                if _is_link_or_junction(path):
                    raise ValueError(f"refusing output tree containing link or junction: {path}")
                if path.is_dir():
                    pending.append(path)
        if nonempty:
            manifest = root / "suite-manifest.json"
            try:
                existing = json.loads(manifest.read_text(encoding="utf-8"))
            except (FileNotFoundError, IsADirectoryError, UnicodeError, json.JSONDecodeError) as error:
                raise ValueError("nonempty fixture output requires its existing suite-manifest.json") from error
            if not isinstance(existing, dict) or existing.get("schema") != "spatial-scene-lab.fixture-suite.v1":
                raise ValueError("refusing nonempty output without this generator's fixture-suite schema")
    return root


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _yaw(degrees):
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _box(center, dimensions):
    vertices = np.asarray([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                           [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    vertices = vertices * np.asarray(dimensions) / 2 + center
    triangles = np.asarray([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                            [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
                            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]], dtype=np.int32)
    return vertices, triangles


def _cylinder(start, end, radius, segments=16, end_radius=None):
    """Original capped circular frustum, including leaning thin furniture legs."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    axis = end - start
    axis /= np.linalg.norm(axis)
    helper = np.asarray([1., 0, 0]) if abs(axis[0]) < .8 else np.asarray([0., 1, 0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angles = np.arange(segments) * 2 * np.pi / segments
    ring = np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
    other_radius = radius if end_radius is None else end_radius
    vertices = np.vstack([start + radius * ring, end + other_radius * ring, start, end])
    triangles = []
    for i in range(segments):
        j = (i + 1) % segments
        triangles += [[i, j, segments + j], [i, segments + j, segments + i],
                      [2 * segments, j, i], [2 * segments + 1, segments + i, segments + j]]
    return vertices, np.asarray(triangles, dtype=np.int32)


def _ellipsoid(center, radii, segments=24, rings=10):
    """Watertight smooth cushion approximation with unique pole vertices."""
    vertices = [[0, 1, 0]]
    for latitude in range(1, rings):
        phi = np.pi * latitude / rings
        for longitude in range(segments):
            theta = 2 * np.pi * longitude / segments
            vertices.append([np.sin(phi) * np.cos(theta), np.cos(phi), np.sin(phi) * np.sin(theta)])
    vertices.append([0, -1, 0])
    triangles = []
    for i in range(segments):
        triangles.append([0, 1 + (i + 1) % segments, 1 + i])
    for row in range(rings - 2):
        for i in range(segments):
            a, b = 1 + row * segments + i, 1 + row * segments + (i + 1) % segments
            triangles += [[a, b, b + segments], [a, b + segments, a + segments]]
    pole = len(vertices) - 1
    for i in range(segments):
        a = 1 + (rings - 2) * segments + i
        b = 1 + (rings - 2) * segments + (i + 1) % segments
        triangles.append([a, b, pole])
    return np.asarray(vertices) * radii + center, np.asarray(triangles, dtype=np.int32)


def _furniture(spec):
    """Return local physical mesh parts; this function never reads spec['prior']."""
    parts = []
    colour = np.asarray(spec["rgb"], np.uint8)
    accent = np.asarray(spec.get("accent_rgb", spec["rgb"]), np.uint8)
    design = spec.get("design", {})

    def box(center, dimensions, rgb=None):
        parts.append((*_box(center, dimensions), colour if rgb is None else rgb))

    def rod(start, end, radius, rgb=None, end_radius=None):
        parts.append((*_cylinder(start, end, radius, end_radius=end_radius), colour if rgb is None else rgb))

    def cushion(center, radii, rgb=None):
        parts.append((*_ellipsoid(center, radii), colour if rgb is None else rgb))

    kind = spec["kind"]
    if kind == "chair":
        width, depth = design.get("seat_width_m", .58), design.get("seat_depth_m", .54)
        slats = design.get("back_slats", 3)
        box([0, .47, 0], [width, .06, depth])
        for x in [-width * .405, width * .405]:
            for z in [-depth * .398, depth * .398]:
                rod([x, 0, z], [x, .45, z], design.get("leg_radius_m", .018))
            rod([x, .45, -depth * .407], [x, 1.01, -depth * .481], .019)
        for y in np.linspace(.65, .95, slats):
            box([0, y, -depth * .463], [width * .81, .048, .032])
        cushion([0, .51, 0], [width * .465, .045, depth * .454], accent)
        cushion([0, .825, -depth / 2], [width * .414, .155, .035], accent)
    elif kind == "table":
        width, depth, height = design.get("width_m", 1.4), design.get("depth_m", .85), design.get("height_m", .79)
        box([0, height - .035, 0], [width, .07, depth])
        for x in [-(width / 2 - .11), width / 2 - .11]:
            for z in [-(depth / 2 - .11), depth / 2 - .11]:
                rod([x, 0, z], [x, height - .07, z], .033, end_radius=.025)
            box([x, .24, 0], [.035, .045, depth - .19])
        box([0, .24, 0], [width - .22, .035, .035])
    elif kind == "bookshelf":
        width, height, depth = design.get("width_m", 1.04), design.get("height_m", 1.86), design.get("depth_m", .48)
        levels = np.linspace(.035, height - .025, design.get("shelf_levels", 5))
        for x in [-width / 2, width / 2]:
            box([x, height / 2, 0], [.045, height, depth])
        box([0, height / 2, -depth / 2 + .01], [width - .04, height, .02])
        for y in levels:
            box([0, y, 0], [width - .04, .035, depth])
        for level, y in enumerate(levels[1:-1]):
            for index in range(4 - level):
                h = .24 + .027 * ((index + level) % 3)
                book_rgb = (accent.astype(int) + [index * 7, level * 9, -index * 3]).clip(0, 255).astype(np.uint8)
                box([-.35 + index * .12, y + .0175 + h / 2, -.035], [.068, h, .23], book_rgb)
    elif kind == "sofa":
        box([0, .235, 0], [1.5, .25, .83])
        for x in [-.63, .63]:
            for z in [-.33, .33]:
                rod([x, 0, z], [x, .13, z], .037)
        count = design.get("cushions", 2)
        cushion_width = .72 / count
        for x in np.linspace(-.72 + cushion_width, .72 - cushion_width, count):
            cushion([x, .435, .03], [cushion_width + .005, .12, .39])
            cushion([x, .73, -.335], [cushion_width, .295, .145], accent)
        for x in [-.805, .805]:
            cushion([x, .52, -.005], [.135, .285, .47])
    elif kind == "ottoman":
        box([0, .185, 0], [.65, .25, .57])
        cushion([0, .345, 0], [.39, .105, .34], accent)
        for x in [-.245, .245]:
            for z in [-.205, .205]:
                rod([x, 0, z], [x, .08, z], .027)
    elif kind == "stool":
        count = design.get("legs", 3)
        radius = design.get("seat_radius_m", .29)
        angles = np.arange(count) * 360 / count
        rod([0, .465, 0], [0, .515, 0], radius)
        for angle in angles:
            a = math.radians(angle)
            x, z = math.cos(a), math.sin(a)
            rod([.24 * x, 0, .24 * z], [.18 * x, .465, .18 * z], .024)
        for angle in angles:
            a, b = math.radians(angle), math.radians(angle + 360 / count)
            rod([.214 * math.cos(a), .19, .214 * math.sin(a)],
                [.214 * math.cos(b), .19, .214 * math.sin(b)], .012)
    elif kind == "roundtable":
        rod([0, .62, 0], [0, .67, 0], design.get("radius_m", .43))
        rod([0, .06, 0], [0, .62, 0], .052)
        rod([0, 0, 0], [0, .06, 0], .29)
    elif kind == "lamp":
        rod([0, 0, 0], [0, .04, 0], .22)
        height = design.get("height_m", 1.57)
        rod([0, .04, 0], [0, height - .04, 0], .013)
        rod([0, height - .29, 0], [0, height, 0], design.get("shade_radius_m", .23), accent, end_radius=.105)
    elif kind == "rack":
        for x in [-.51, .51]:
            for z in [-.235, .235]:
                rod([x, 0, z], [x, 1.48, z], .008)
        for y in [.18, .63, 1.08, 1.46]:
            for z in [-.235, .235]:
                rod([-.51, y, z], [.51, y, z], .009)
            for x in np.linspace(-.48, .48, 9):
                box([x, y, 0], [.018, .018, .47])
        rod([-.51, .18, -.235], [.51, 1.46, -.235], .008)
    elif kind == "screen":
        for x, yaw in [(-.52, -18), (0, 0), (.52, 18)]:
            vertices, triangles = _box([0, .78, 0], [.52, 1.46, .025])
            vertices = vertices @ _yaw(yaw).T + [x, 0, .08 if x else 0]
            parts.append((vertices, triangles, colour))
            for edge in [-.245, .245]:
                rod([x + edge, 0, .08 if x else 0], [x + edge, 1.52, .08 if x else 0], .015)
    elif kind == "desk":
        box([-.16, .765, 0], [1.48, .055, .68])
        box([.54, .765, .51], [.55, .055, .45])
        for x, z in [(-.81, -.25), (-.81, .25), (.57, -.25), (.73, .68)]:
            box([x, .37, z], [.045, .74, .045])
        box([.37, .405, .03], [.4, .65, .52])
        for y in [.24, .43, .62]:
            box([.37, y, .302], [.37, .165, .022], accent)
            rod([.29, y, .325], [.45, y, .325], .009)
    else:
        raise ValueError(f"unknown physical furniture kind: {kind}")
    return parts


def _build_scene(config, scene):
    vertices, triangles, owners, colours = [], [], [], []
    objects = []
    offset = 0

    def append(parts, owner, transform=None, scale=None):
        nonlocal offset
        local_vertices = []
        for points, faces, colour in parts:
            if scale is not None:
                points = points * scale
            local_vertices.append(points)
            world = points if transform is None else points @ transform[:3, :3].T + transform[:3, 3]
            vertices.append(world)
            triangles.append(faces + offset)
            owners.append(np.full(len(faces), owner, dtype=np.int32))
            colours.append(np.tile(colour, (len(faces), 1)))
            offset += len(points)
        return np.concatenate(local_vertices)

    for index, spec in enumerate(scene["objects"]):
        transform = np.eye(4)
        transform[:3, :3] = _yaw(spec["yaw_degrees"])
        transform[:3, 3] = spec["position_m"]
        points = append(_furniture(spec), index, transform, spec["scale"])
        objects.append({"id": spec["id"], "kind": spec["kind"], "category": spec["category"],
                        "transform_world": transform.tolist(), "local_bounds_m": [points.min(0).tolist(), points.max(0).tolist()],
                        "physical_parameters": {k: v for k, v in spec.items() if k != "prior"}})
    room = config["room"]
    x0, x1 = room["bounds_x_m"]
    z0, z1 = room["bounds_z_m"]
    height, floor_y = room["wall_height_m"], room["floor_y"]
    structures = [
        {"id": "floor", "category": "Floor", "center": [(x0 + x1) / 2, floor_y - .025, (z0 + z1) / 2], "dimensions": [x1 - x0, .05, z1 - z0], "rgb": [158, 152, 139]},
        {"id": "back-wall", "category": "Wall", "center": [(x0 + x1) / 2, floor_y + height / 2, z0 - .025], "dimensions": [x1 - x0, height, .05], "rgb": [187, 188, 183]},
        {"id": "left-wall", "category": "Wall", "center": [x0 - .025, floor_y + height / 2, (z0 + z1) / 2], "dimensions": [.05, height, z1 - z0], "rgb": [179, 184, 185]}
    ]
    for spec in structures:
        append([(*_box(spec["center"], spec["dimensions"]), spec["rgb"])], -1)
    return (np.concatenate(vertices).astype(np.float32), np.concatenate(triangles).astype(np.int32),
            np.concatenate(owners), np.concatenate(colours).astype(np.uint8), objects, structures)


def _measured_priors(scene, physical_objects, structures):
    records = []
    for index, (spec, physical) in enumerate(zip(scene["objects"], physical_objects)):
        lower, upper = np.asarray(physical["local_bounds_m"])
        transform = np.asarray(physical["transform_world"])
        local_center = (lower + upper) / 2
        prior = spec["prior"]
        center = transform[:3, :3] @ local_center + transform[:3, 3] + prior.get("offset_m", [0, 0, 0])
        rotation = _yaw(spec["yaw_degrees"] + prior.get("yaw_error_degrees", 0))
        measured = np.eye(4)
        measured[:3, :3], measured[:3, 3] = rotation, center
        records.append({"id": spec["id"], "name": f"Object{index:02d}", "category": spec["category"],
                        "dimensions": ((upper - lower) * prior.get("dimension_scale", [1, 1, 1])).tolist(),
                        "transform": measured.tolist(), "source_member": f"assets/{index:02d}_{spec['id']}.usda"})
    for index, physical in enumerate(structures):
        matrix = np.eye(4)
        matrix[:3, 3] = physical["center"]
        records.append({"id": f"{scene['scene_id']}-{physical['id']}", "name": f"Structure{index:02d}",
                        "category": physical["category"], "dimensions": physical["dimensions"],
                        "transform": matrix.tolist(), "source_member": f"assets/90_{physical['id']}.usda"})
    return records


def _write_roomplan(path, records):
    root = '#usda 1.0\n( metersPerUnit = 1\n upAxis = "Y" )\n'
    root += 'def Xform "room" {\n def Xform "Parametric_grp" {\n'
    components = []
    for record in records:
        name = record["name"]
        root += f' def Xform "{name}_grp" (prepend references = @./{record["source_member"]}@) {{ }}\n'
        matrix = np.asarray(record["transform"]).T
        matrix_text = "(" + ", ".join("(" + ", ".join(f"{x:.12g}" for x in row) + ")" for row in matrix) + ")"
        scale = ", ".join(f"{x:.12g}" for x in record["dimensions"])
        content = f'''#usda 1.0
( metersPerUnit = 1
  upAxis = "Y" )
def Xform "{name}" (customData = {{
    string Category = "{record['category']}"
    string UUID = "{record['id']}"
}}) {{
    def Cube "{name}" {{
        double size = 1
        double3 xformOp:scale = ({scale})
        matrix4d xformOp:transform = {matrix_text}
        uniform token[] xformOpOrder = ["xformOp:transform", "xformOp:scale"]
    }}
}}
'''
        components.append((record["source_member"], content))
    root += "}\n}\n"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for member, content in [("room.usda", root)] + components:
            info = zipfile.ZipInfo(member, date_time=(2026, 10, 4, 0, 0, 0))
            padding = (-(archive.fp.tell() + 30 + len(member.encode("utf-8")))) % 64
            if padding:
                if padding < 4:
                    padding += 64
                info.extra = struct.pack("<HH", 0x1986, padding - 4) + bytes(padding - 4)
            archive.writestr(info, content)


def _write_ply(path, vertices, triangles):
    header = ("ply\nformat binary_little_endian 1.0\ncomment original AlvenX physical reference\n"
              f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
              f"element face {len(triangles)}\nproperty list uchar int vertex_indices\nend_header\n")
    faces = np.empty(len(triangles), dtype=[("count", "u1"), ("indices", "<i4", (3,))])
    faces["count"], faces["indices"] = 3, triangles
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(np.asarray(vertices, dtype="<f4").tobytes())
        stream.write(faces.tobytes())


def _camera_pose(position, focus):
    z = np.asarray(position) - focus  # ARKit camera's backward +Z axis
    z /= np.linalg.norm(z)
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([x, y, z])
    pose[:3, 3] = position
    return pose


def _trajectory(scene):
    trajectory = scene["trajectory"]
    focus = np.asarray(trajectory["focus_m"], float)
    poses = []
    for i, angle in enumerate(np.linspace(-trajectory["arc_degrees"] / 2, trajectory["arc_degrees"] / 2, trajectory["frames"])):
        a = math.radians(angle)
        position = focus + [trajectory["radius_m"] * math.sin(a), 0, trajectory["radius_m"] * math.cos(a)]
        position[1] = trajectory["height_m"] + trajectory["height_amplitude_m"] * math.sin(2 * np.pi * i / trajectory["frames"])
        poses.append(_camera_pose(position, focus))
    return poses


def _tasks(scene, physical_objects):
    target = next(obj for obj in physical_objects if obj["id"] == scene["target_id"])
    pivot = np.asarray(target["transform_world"])[:3, 3].tolist()
    secondary = physical_objects[1]["id"]
    return [
        {"operation": "translate", "object_id": scene["target_id"], "translation_m": [.28, 0.0, .12]},
        {"operation": "rotate_y", "object_id": scene["target_id"], "rotation_degrees": 18.0, "pivot_world_m": pivot},
        {"operation": "translate_then_rotate_y", "object_id": secondary, "translation_m": [-.15, 0.0, .1], "rotation_degrees": -12.0, "pivot_world_m": np.asarray(physical_objects[1]["transform_world"])[:3, 3].tolist()},
        {"operation": "reopen_undo", "object_id": scene["target_id"], "expected": "restore exact pre-edit mesh, object IDs, ownership and world poses"}
    ]


def _generate_scene(root, config, scene, resolution, open3d, generator_sha256):
    directory = root / scene["scene_id"]
    scan, reference = directory / "scan", directory / "reference"
    for path in [directory, scan, reference]:
        if _is_link_or_junction(path):
            raise ValueError(f"refusing generated output through directory symlink: {path}")
        path.mkdir(parents=True, exist_ok=True)
    frame_reference = reference / "frame_labels"
    frame_reference.mkdir(exist_ok=True)
    vertices, triangles, owner, colours, physical_objects, structures = _build_scene(config, scene)
    # Neither the ray tracer nor the reference mesh receives measured box priors.
    ray_scene = open3d.t.geometry.RaycastingScene(nthreads=1)
    ray_scene.add_triangles(open3d.core.Tensor(vertices), open3d.core.Tensor(triangles.astype(np.uint32)))
    _write_ply(reference / "mesh.ply", vertices, triangles)
    np.savez_compressed(reference / "labels.npz", vertices=vertices, triangles=triangles,
                        triangle_owner=owner, object_ids=np.asarray([obj["id"] for obj in physical_objects]))
    priors = _measured_priors(scene, physical_objects, structures)
    _write_roomplan(scan / "room.usdz", priors)
    width, height = resolution
    fx = fy = width * .85
    cx, cy = (width - 1) / 2, (height - 1) / 2
    yy, xx = np.mgrid[:height, :width]
    local_directions = np.stack([(xx - cx) / fx, -(yy - cy) / fy, -np.ones_like(xx)], axis=-1)
    sensor = scene["sensor"]
    rng = np.random.default_rng(scene["seed"])
    camera_poses = _trajectory(scene)
    target_owner = next(i for i, spec in enumerate(scene["objects"]) if spec["id"] == scene["target_id"])
    hit_count = np.zeros(len(triangles), dtype=np.int64)
    normals = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]], vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    areas = np.linalg.norm(normals, axis=1) / 2
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    lighting = .72 + .28 * np.abs(normals @ (np.asarray([.3, .8, .4]) / np.linalg.norm([.3, .8, .4])))
    lit_colours = np.rint(colours * lighting[:, None]).clip(0, 255).astype(np.uint8)
    frame_stats, measured_points = [], []
    for index, truth_pose in enumerate(camera_poses):
        directions = local_directions @ truth_pose[:3, :3].T
        origins = np.broadcast_to(truth_pose[:3, 3], directions.shape)
        rays = np.concatenate([origins, directions], axis=-1).astype(np.float32)
        hits = ray_scene.cast_rays(open3d.core.Tensor(rays), nthreads=1)
        exact_depth = hits["t_hit"].numpy().astype(float)
        triangle_ids = hits["primitive_ids"].numpy().astype(np.int64)
        geometric_valid = np.isfinite(exact_depth) & (exact_depth >= .15) & (exact_depth <= config["defaults"]["maximum_depth_m"])
        valid_ids = triangle_ids[geometric_valid]
        rgb = np.full((height, width, 3), [100, 120, 140], dtype=np.uint8)
        rgb[geometric_valid] = lit_colours[valid_ids]
        labels = np.full((height, width), -3, dtype=np.int32)
        labels[geometric_valid] = owner[valid_ids]
        # Depth stays optical-axis metres: directions are deliberately unnormalised.
        noise = rng.normal(0, sensor["depth_noise_sigma_m"], (height, width))
        depth = np.where(geometric_valid, exact_depth, 0) + noise
        dropout = rng.random((height, width)) < sensor["dropout_probability"]
        period = sensor["stripe_dropout_period_px"]
        if period:
            dropout |= ((xx + index * 3) % period) < 2
        valid = geometric_valid & ~dropout & (depth >= .15) & (depth <= config["defaults"]["maximum_depth_m"])
        depth_mm = np.where(valid, np.rint(depth * 1000), 0).astype(np.uint16)
        confidence = np.where(valid, 2, 0).astype(np.uint8)
        # Guaranteed invalid sentinels exercise ScannerApp filtering in every scene.
        depth_mm[0, 0], depth_mm[0, 1], confidence[0, :2] = 0, 65535, 0
        valid[0, :2] = False
        np.add.at(hit_count, triangle_ids[valid], 1)
        measured_pose = truth_pose.copy()
        measured_pose[:3, 3] += rng.normal(0, sensor["pose_translation_sigma_m"], 3)
        yaw_error = rng.normal(0, sensor["pose_yaw_sigma_degrees"])
        measured_pose[:3, :3] = _yaw(yaw_error) @ truth_pose[:3, :3]
        suffix = f"{index:05d}"
        Image.fromarray(depth_mm).save(scan / f"depth_{suffix}.png")
        Image.fromarray(confidence).save(scan / f"conf_{suffix}.png")
        Image.fromarray(rgb).resize((width * 2, height * 2), Image.Resampling.NEAREST).save(
            scan / f"frame_{suffix}.jpg", quality=95, subsampling=0)
        record = {"frame_index": index, "cameraPoseARFrame": measured_pose.ravel().tolist(),
                  "intrinsics": [fx * 2, 0, cx * 2, 0, fy * 2, cy * 2, 0, 0, 1]}
        _json(scan / f"frame_{suffix}.json", record)
        stored_triangles = np.where(geometric_valid, triangle_ids, -1).astype(np.int32)
        np.savez_compressed(frame_reference / f"frame_{suffix}.npz", instance_owner=labels,
                            triangle_index=stored_triangles, exact_depth_m=np.where(geometric_valid, exact_depth, 0).astype(np.float32),
                            measurement_valid=valid)
        camera_points = local_directions[valid] * (depth_mm[valid, None].astype(float) / 1000)
        measured_points.append(camera_points @ measured_pose[:3, :3].T + measured_pose[:3, 3])
        frame_stats.append({"frame_index": index, "geometric_hit_pixels": int(geometric_valid.sum()),
                            "valid_measurement_pixels": int(valid.sum()), "target_hit_pixels": int(np.count_nonzero(labels == target_owner)),
                            "target_valid_pixels": int(np.count_nonzero((labels == target_owner) & valid)),
                            "actual_pose_offset_m": (measured_pose[:3, 3] - truth_pose[:3, 3]).tolist(), "actual_yaw_error_degrees": float(yaw_error)})
    points = np.concatenate(measured_points)
    if not len(points) or not any(item["target_valid_pixels"] for item in frame_stats):
        raise RuntimeError(f"scene {scene['scene_id']} has no usable observations or no visible target")
    with (scan / "pointcloud.pcd").open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# .PCD v0.7 - measured RGB-D with measured poses\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                     f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA ascii\n")
        np.savetxt(stream, points, fmt="%.9g")
    np.savez_compressed(reference / "visibility.npz", triangle_hit_count=hit_count,
                        triangle_seen=hit_count > 0, triangle_area_m2=areas)
    object_stats = []
    for index, physical in enumerate(physical_objects):
        selected = owner == index
        object_stats.append({"object_id": physical["id"], "triangle_count": int(selected.sum()),
                             "physical_triangle_area_m2": float(areas[selected].sum()),
                             "ray_hit_triangle_area_m2": float(areas[selected & (hit_count > 0)].sum()),
                             "valid_ray_count": int(hit_count[selected].sum())})
    tasks = _tasks(scene, physical_objects)
    _json(reference / "physical-scene.json", {"schema": "spatial-scene-lab.physical-reference.v1", "generator_version": VERSION,
          "scene_id": scene["scene_id"], "seed": scene["seed"], "units": "m", "up_axis": "Y", "licence": config["licence"],
          "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(), "generator_sha256": generator_sha256,
          "objects": physical_objects, "structures": structures,
          "true_camera_poses_arkit": [pose.tolist() for pose in camera_poses], "depth_resolution": list(resolution),
          "intrinsics_depth": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], "sensor_parameters": sensor,
          "prior_perturbations": {spec["id"]: spec["prior"] for spec in scene["objects"]}, "frame_statistics": frame_stats,
          "object_statistics": object_stats, "edit_tasks": tasks,
          "reference_construction": "Original multi-part physical triangle geometry; first-hit CPU ray labels. Boxes are measured priors made afterwards, never reference labels.",
          "visibility_rule": "triangle_seen means at least one valid measured ray hit the physical triangle; triangle area sums are a coarse visibility proxy, not visible surface area or an evaluation denominator.",
          "known_limitations": ["Synthetic constant diffuse materials and simple directional shading; no photorealistic sensor simulation.", "Overlapping primitive parts may include hidden internal triangles; physical full-mesh area is not exposed outer surface area.", "Development stress parameters are hypotheses; there are no held-out test groups or real-room labels in this suite.", "Edit tasks are definitions and are not claimed as completed Blender interactions."]})
    return {"scene_id": scene["scene_id"], "scan": f"{scene['scene_id']}/scan", "reference_mesh": f"{scene['scene_id']}/reference/mesh.ply",
            "reference_labels": f"{scene['scene_id']}/reference/labels.npz", "reference_metadata": f"{scene['scene_id']}/reference/physical-scene.json",
            "reference_visibility": f"{scene['scene_id']}/reference/visibility.npz", "target_id": scene["target_id"],
            "seed": scene["seed"], "frames": len(camera_poses), "factors": scene["factors"], "edit_tasks": tasks}


def generate_suite(output, resolution=(160, 120), scene_ids=None):
    """Create a replayable fixture suite and return its completed manifest.

    Existing generator files are overwritten deterministically; errors propagate.
    Small resolutions are supported for tests; the declared experiment uses
    160x120 or 192x128. Open3D is imported only for actual CPU ray generation.
    """
    if (not isinstance(resolution, (tuple, list)) or len(resolution) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in resolution)
            or not 16 <= resolution[0] <= 640 or not 12 <= resolution[1] <= 480):
        raise ValueError("resolution must be (integer width 16..640, integer height 12..480)")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["generator_version"] != VERSION:
        raise ValueError("scene configuration and generator versions differ")
    available = {scene["scene_id"]: scene for scene in config["scenes"]}
    if len(available) != 6:
        raise ValueError("configuration must contain six unique development scenes")
    if scene_ids is None:
        selected = list(available)
    elif isinstance(scene_ids, (list, tuple)) and scene_ids and all(isinstance(s, str) for s in scene_ids):
        selected = list(scene_ids)
        if len(set(selected)) != len(selected):
            raise ValueError("scene_ids must not contain duplicates")
        unknown = set(selected) - set(available)
        if unknown:
            raise ValueError(f"unknown scene IDs: {sorted(unknown)}")
    else:
        raise ValueError("scene_ids must be a nonempty sequence of scene ID strings or None")
    root = _checked_output(output)
    import open3d
    generator_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "spatial-scene-lab.fixture-suite.v1", "generator_version": VERSION,
                "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(), "generator_sha256": generator_sha256,
                "licence": config["licence"],
                "split": "development", "units": "m", "up_axis": "Y", "depth_resolution": list(resolution),
                "parameter_basis": config["parameter_basis"], "evidence": config["evidence"], "hypotheses": config["hypotheses"],
                "dependencies": {"numpy": np.__version__, "open3d": open3d.__version__},
                "scenes": [_generate_scene(root, config, available[scene_id], resolution, open3d, generator_sha256) for scene_id in selected]}
    _json(root / "suite-manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--scene", action="append", dest="scene_ids")
    args = parser.parse_args()
    print(json.dumps(generate_suite(args.output, (args.width, args.height), args.scene_ids), indent=2))
