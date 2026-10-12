"""Independent dining geometry acceptance and read-only reference diagnostics.

Fit inputs and candidate meshes are explicit paths. Embedded machine paths are
provenance only. References are never imported by the fitter; this module does
not fit, repair, substitute geometry, or change the predeclared thresholds.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np

import evaluate_stool_quality as q
from evaluate_stool_transfer import convex_planes
from geometry_compare import compare_triangle_geometry


TARGETS = ("same-table", "same-chair-west", "same-chair-east")
DISTANCE_GATE_M = .010
FLOOR_GATE_M = .003
CONTACT_MARGIN_M = 1e-7
REFERENCE_PRECISION_M = .0002
SAMPLE_COUNT = 30_000
SAMPLE_SEED = 20261013
_UNSET = object()


def record(path, base):
    path, base = Path(path).resolve(), Path(base).resolve()
    import os
    return {"path": Path(os.path.relpath(path, base)).as_posix(), "sha256": q.sha256(path)}


def deterministic_npz(path, **arrays):
    """Stable ZIP ordering, timestamps and NumPy headers for a portable pack."""
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, value in sorted(arrays.items()):
            stream = io.BytesIO()
            np.save(stream, np.asarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, stream.getvalue())


def compact_mesh(vertices, faces):
    used, inverse = np.unique(faces, return_inverse=True)
    return q.arrays_mesh(vertices[used], inverse.reshape((-1, 3)))


def sample_area(mesh, count=SAMPLE_COUNT, seed=SAMPLE_SEED):
    if mesh is None or not len(mesh.triangles):
        return np.empty((0, 3))
    corners = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
    area = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1) / 2
    if not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError("mesh requires positive finite area")
    rng = np.random.default_rng(seed)
    triangles = corners[rng.choice(len(corners), count, p=area / area.sum())]
    root, v = np.sqrt(rng.random(count)), rng.random(count)
    return triangles[:, 0] * (1 - root[:, None]) + triangles[:, 1] * (root * (1 - v))[:, None] + triangles[:, 2] * (root * v)[:, None]


def pack_reference(source, output):
    """Extract only the three target solids, primitive labels and valid rays."""
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("reference pack output must be new or empty")
    metadata = json.loads((source / "physical-scene.json").read_text(encoding="utf-8"))
    if metadata.get("units") != "m" or metadata.get("up_axis") != "Y":
        raise ValueError("reference requires metres and Y up")
    with np.load(source / "labels.npz", allow_pickle=False) as data:
        vertices, faces = data["vertices"], data["triangles"]
        owners, ids = data["triangle_owner"], data["object_ids"].tolist()
    q.arrays_mesh(vertices, faces)
    if len(set(ids)) != len(ids) or any(ids.count(uuid) != 1 for uuid in TARGETS):
        raise ValueError("reference target identities differ")
    indices = np.flatnonzero(np.isin(owners, [ids.index(uuid) for uuid in TARGETS]))
    old_to_new = np.full(len(faces), -1, dtype=int)
    old_to_new[indices] = np.arange(len(indices))
    selected_faces = faces[indices]
    used, inverse = np.unique(selected_faces, return_inverse=True)
    selected_vertices, selected_faces = vertices[used], inverse.reshape((-1, 3)).astype(np.int32)
    selected_owner = np.asarray([TARGETS.index(ids[index]) for index in owners[indices]], dtype=np.int32)
    part_labels = np.full(len(selected_faces), -1, dtype=np.int32)
    parts, objects = [], []
    for owner, uuid in enumerate(TARGETS):
        target_indices = np.flatnonzero(selected_owner == owner)
        target = q.arrays_mesh(selected_vertices, selected_faces[target_indices])
        components = np.asarray(target.cluster_connected_triangles()[0])
        order = sorted(np.unique(components), key=lambda component: int(np.flatnonzero(components == component)[0]))
        object_parts = []
        for index, component in enumerate(order):
            face_indices = target_indices[components == component]
            part_index = len(parts)
            part_labels[face_indices] = part_index
            name = f"{uuid}/physical_component_{index:02d}"
            parts.append({"id": name, "owner_index": owner, "part_index": part_index,
                          "triangle_count": len(face_indices)})
            object_parts.append(name)
        expected = next(item["valid_ray_count"] for item in metadata["object_statistics"] if item["object_id"] == uuid)
        objects.append({"uuid": uuid, "owner_index": owner, "parts": object_parts, "valid_ray_count": expected})
    width, height = metadata["depth_resolution"]
    intrinsic = np.asarray(metadata["intrinsics_depth"], dtype=float)
    poses = np.asarray(metadata["true_camera_poses_arkit"], dtype=float)
    ray_values = {key: [] for key in ("frame_index", "pixel_uv", "exact_depth_m", "triangle_index")}
    source_files = [record(source / name, source) for name in ("physical-scene.json", "labels.npz")]
    files = sorted((source / "frame_labels").glob("frame_*.npz"))
    if len(files) != len(poses) or not files:
        raise ValueError("reference frame count differs from true poses")
    for index, path in enumerate(files):
        if path.name != f"frame_{index:05d}.npz":
            raise ValueError("reference frame labels must be contiguous")
        with np.load(path, allow_pickle=False) as frame:
            mask = frame["measurement_valid"].astype(bool) & np.isin(frame["instance_owner"], [ids.index(uuid) for uuid in TARGETS])
            if mask.shape != (height, width):
                raise ValueError("reference resolution mismatch")
            hits = frame["triangle_index"][mask]
            if np.any(hits < 0) or np.any(hits >= len(faces)) or np.any(old_to_new[hits] < 0):
                raise ValueError("reference ray has mismatched triangle owner")
            if np.any(owners[hits] != frame["instance_owner"][mask]):
                raise ValueError("reference ray ownership inconsistent")
            yy, xx = np.nonzero(mask)
            ray_values["frame_index"].append(np.full(len(hits), index, dtype=np.int32))
            ray_values["pixel_uv"].append(np.column_stack((xx, yy)).astype(np.int32))
            ray_values["exact_depth_m"].append(frame["exact_depth_m"][mask])
            ray_values["triangle_index"].append(old_to_new[hits].astype(np.int32))
        source_files.append(record(path, source))
    output.mkdir(parents=True, exist_ok=True)
    deterministic_npz(output / "labels.npz", vertices=selected_vertices, triangles=selected_faces,
                      triangle_owner=selected_owner, triangle_part=part_labels, object_ids=np.asarray(TARGETS))
    deterministic_npz(output / "rays.npz", **{key: np.concatenate(value) for key, value in ray_values.items()})
    packed = {"schema": "alvenx.spatial.dining-reference.v1", "units": "m", "up_axis": "Y",
              "scope": "Evaluation-only extraction of original target geometry and all measurement-valid captured rays; no construction parameters.",
              "depth_resolution": [width, height], "intrinsics_depth": intrinsic.tolist(),
              "ray_reconstruction_precision_bound_m": REFERENCE_PRECISION_M,
              "true_camera_poses_arkit": poses.tolist(), "objects": objects, "parts": parts,
              "source_files": source_files, "files": [record(output / name, output) for name in ("labels.npz", "rays.npz")]}
    (output / "reference.json").write_text(json.dumps(packed, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    load_reference(output)
    return packed


def load_reference(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "reference.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != "alvenx.spatial.dining-reference.v1" or metadata.get("units") != "m" or metadata.get("up_axis") != "Y":
        raise ValueError("incompatible dining reference schema or coordinates")
    for item in metadata["files"]:
        if item["path"] not in ("labels.npz", "rays.npz") or q.sha256(directory / item["path"]) != item["sha256"]:
            raise ValueError("reference pack hash changed")
    with np.load(directory / "labels.npz", allow_pickle=False) as data:
        vertices, faces = data["vertices"], data["triangles"]
        owners, part_labels, ids = data["triangle_owner"], data["triangle_part"], data["object_ids"].tolist()
    q.arrays_mesh(vertices, faces)
    if (ids != list(TARGETS) or owners.shape != (len(faces),) or part_labels.shape != (len(faces),) or
            owners.dtype.kind not in "iu" or part_labels.dtype.kind not in "iu" or
            [item.get("uuid") for item in metadata["objects"]] != list(TARGETS) or
            [item.get("part_index") for item in metadata["parts"]] != list(range(len(metadata["parts"]))) or
            len({item.get("id") for item in metadata["parts"]}) != len(metadata["parts"])):
        raise ValueError("reference IDs or geometry labels mismatch")
    if np.any(owners < 0) or np.any(owners >= len(ids)) or np.any(part_labels < 0) or np.any(part_labels >= len(metadata["parts"])):
        raise ValueError("reference owner or part outside label range")
    with np.load(directory / "rays.npz", allow_pickle=False) as data:
        frames, uv, depth, hit_faces = data["frame_index"], data["pixel_uv"], data["exact_depth_m"], data["triangle_index"]
    poses, intrinsic = np.asarray(metadata["true_camera_poses_arkit"]), np.asarray(metadata["intrinsics_depth"])
    if (frames.shape != depth.shape or hit_faces.shape != depth.shape or uv.shape != (len(depth), 2) or
            frames.dtype.kind not in "iu" or hit_faces.dtype.kind not in "iu" or uv.dtype.kind not in "iu" or
            np.any(frames < 0) or np.any(frames >= len(poses)) or np.any(hit_faces < 0) or np.any(hit_faces >= len(faces)) or
            poses.shape != (len(poses), 4, 4) or intrinsic.shape != (3, 3) or
            not np.isfinite(poses).all() or not np.isfinite(intrinsic).all() or
            min(intrinsic[0, 0], intrinsic[1, 1]) <= 0 or not np.isfinite(depth).all() or np.any(depth <= 0)):
        raise ValueError("invalid packed reference rays or cameras")
    width, height = metadata["depth_resolution"]
    if np.any(uv < 0) or np.any(uv[:, 0] >= width) or np.any(uv[:, 1] >= height):
        raise ValueError("packed rays outside camera resolution")
    directions = np.column_stack(((uv[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0],
                                   -(uv[:, 1] - intrinsic[1, 2]) / intrinsic[1, 1], -np.ones(len(uv))))
    points = np.einsum("nij,nj->ni", poses[frames, :3, :3], directions * depth[:, None]) + poses[frames, :3, 3]
    results = {}
    for owner, uuid in enumerate(ids):
        mesh = compact_mesh(vertices, faces[owners == owner])
        scene = q.ray_scene(mesh)
        mask = owners[hit_faces] == owner
        expected = next(item["valid_ray_count"] for item in metadata["objects"] if item["uuid"] == uuid)
        if int(mask.sum()) != expected:
            raise ValueError("reference valid-ray denominator mismatch")
        consistency = q.distances(scene, points[mask])
        if len(consistency) and consistency.max() > REFERENCE_PRECISION_M:
            raise ValueError("reference rays disagree with physical mesh")
        parts = []
        for part in (item for item in metadata["parts"] if item["owner_index"] == owner):
            part_mask = part_labels == part["part_index"]
            if np.any(owners[part_mask] != owner) or int(part_mask.sum()) != part["triangle_count"]:
                raise ValueError("reference component ownership or count mismatch")
            parts.append({**part, "mesh": compact_mesh(vertices, faces[part_mask])})
        results[uuid] = {"mesh": mesh, "scene": scene, "points": points[mask], "ray_parts": part_labels[hit_faces[mask]],
                         "parts": parts, "ray_consistency": q.distance_summary(consistency)}
    return results, metadata


def load_inputs(scene_path, observations_path):
    scene = json.loads(Path(scene_path).read_text(encoding="utf-8"))
    ids = [item["uuid"] for item in scene["objects"]]
    if scene.get("units") != "m" or scene.get("up_axis") != "Y" or len(set(ids)) != len(ids) or any(ids.count(uuid) != 1 for uuid in TARGETS):
        raise ValueError("input target identities or coordinate convention mismatch")
    floors = [item for item in scene.get("structures", []) if item.get("category") == "Floor"]
    if len(floors) != 1:
        raise ValueError("exactly one measured floor required")
    floor = floors[0]
    transform, dimensions = np.asarray(floor["transform_world"]), np.asarray(floor["dimensions_m"])
    if transform.shape != (4, 4) or dimensions.shape != (3,) or not np.isfinite(transform).all() or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError("malformed floor prior")
    if not np.allclose(transform[:3, 1], [0, 1, 0], atol=1e-8):
        raise ValueError("floor prior must be horizontal in world Y up")
    floor_y = float(transform[1, 3] + dimensions[1] / 2)
    with np.load(observations_path, allow_pickle=False) as data:
        points, frames, owners, candidates = data["points"], data["frame_index"], data["owner"], data["candidates"]
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or
            frames.shape != (len(points),) or owners.shape != (len(points),) or
            frames.dtype.kind not in "iu" or owners.dtype.kind not in "iu" or np.any(frames < 0) or
            np.any(owners < -2) or np.any(owners >= len(ids)) or
            candidates.shape != (len(points), len(ids)) or candidates.dtype.kind != "b"):
        raise ValueError("malformed observation arrays")
    return scene, ids, floor_y, points, frames, owners, candidates


def parameter_consistency(mesh, component):
    """Verify dimensions/endpoints against exported triangles, independently."""
    if mesh is None or not len(mesh.triangles):
        return {"passed": False, "reason": "missing component triangles"}
    parameters = component.get("parameters", {})
    kind = component.get("primitive_type", parameters.get("kind", parameters.get("type")))
    vertices = np.asarray(mesh.vertices)
    try:
        if kind in ("box", "domed_box"):
            center = np.asarray(parameters["center_m"], dtype=float)
            dimensions = np.asarray(parameters["dimensions_m"], dtype=float)
            rotation = np.asarray(parameters["rotation_world_3x3"], dtype=float)
            if (center.shape != (3,) or dimensions.shape != (3,) or rotation.shape != (3, 3) or
                    not np.isfinite(center).all() or not np.isfinite(dimensions).all() or not np.isfinite(rotation).all() or
                    np.any(dimensions <= 0) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or
                    not np.isclose(np.linalg.det(rotation), 1, atol=1e-7)):
                raise ValueError("invalid box parameters")
            local = (vertices - center) @ rotation
            if kind == "box":
                error = float(np.max(np.abs(np.abs(local) - dimensions / 2)))
                corners = np.unique(np.round(local / (dimensions / 2), 5), axis=0)
                passed = error <= 2e-6 and len(corners) == 8
            else:
                axis, sign = parameters["cap_axis"], parameters["cap_sign"]
                height, power = float(parameters["cap_height_m"]), float(parameters["profile_power"])
                if axis not in (0, 1, 2) or sign not in (-1, 1) or not np.isfinite(height) or height < 0 or not np.isfinite(power) or power <= 0:
                    raise ValueError("invalid domed box cap parameters")
                tangent = [index for index in range(3) if index != axis]
                uv = local[:, tangent] / (dimensions[tangent] / 2)
                outward = sign * local[:, axis]
                cap = dimensions[axis] / 2 + height * np.prod(np.maximum(0, 1 - uv ** 2) ** power, axis=1)
                bottom = -dimensions[axis] / 2
                top_error, bottom_error = abs(outward - cap), abs(outward - bottom)
                side_error = np.min(abs(abs(local[:, tangent]) - dimensions[tangent] / 2), axis=1)
                wall_error = np.maximum.reduce((side_error, np.maximum(bottom - outward, 0), np.maximum(outward - cap, 0)))
                surface_error = np.minimum.reduce((top_error, bottom_error, wall_error))
                tangent_error = float(np.maximum(abs(uv) - 1, 0).max() * dimensions[tangent].max() / 2)
                crown_error = abs(float(outward.max()) - (dimensions[axis] / 2 + height))
                error = max(float(surface_error.max()), tangent_error, crown_error, abs(float(outward.min()) - bottom))
                passed = error <= 2e-6
        elif kind == "half_ellipsoid":
            center = np.asarray(parameters["center_m"], dtype=float)
            radii = np.asarray(parameters["radii_m"], dtype=float)
            rotation = np.asarray(parameters["rotation_world_3x3"], dtype=float)
            axis, sign = parameters["cap_axis"], parameters["cap_sign"]
            if (center.shape != (3,) or radii.shape != (3,) or rotation.shape != (3, 3) or
                    not np.isfinite(center).all() or not np.isfinite(radii).all() or not np.isfinite(rotation).all() or
                    np.any(radii <= 0) or axis not in (0, 1, 2) or sign not in (-1, 1) or
                    not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-7)):
                raise ValueError("invalid half ellipsoid parameters")
            local = (vertices - center) @ rotation
            norm = np.linalg.norm(local / radii, axis=1)
            axial = sign * local[:, axis]
            curve_error = abs(norm - 1) * radii.max()
            plane_error = np.maximum(abs(axial), np.maximum(norm - 1, 0) * radii.max())
            error = max(float(np.minimum(curve_error, plane_error).max()),
                        float(np.maximum(-axial, 0).max()), abs(float(axial.max()) - radii[axis]))
            passed = error <= 2e-6
        elif kind == "capped_cylinder":
            start, end = np.asarray(parameters["start_m"], dtype=float), np.asarray(parameters["end_m"], dtype=float)
            radius = float(parameters["radius_m"])
            if start.shape != (3,) or end.shape != (3,) or not np.isfinite(start).all() or not np.isfinite(end).all() or not np.isfinite(radius) or radius <= 0:
                raise ValueError("invalid cylinder parameters")
            length = float(np.linalg.norm(end - start))
            if length <= 0:
                raise ValueError("cylinder endpoints coincide")
            axis = (end - start) / length
            local = vertices - start
            height = local @ axis
            radii = np.linalg.norm(local - height[:, None] * axis, axis=1)
            end_error = np.minimum(abs(height), abs(height - length))
            radius_error = np.minimum(radii, abs(radii - radius))
            endpoint_error = max(float(np.linalg.norm(vertices - point, axis=1).min()) for point in (start, end))
            error = max(float(end_error.max()), float(radius_error.max()), endpoint_error)
            passed = error <= 2e-6 and np.count_nonzero(radii > radius / 2) >= 12
        else:
            return {"passed": False, "reason": f"unsupported primitive_type {kind!r}"}
        return {"passed": bool(passed), "primitive_type": kind, "maximum_vertex_parameter_error_m": error}
    except (KeyError, TypeError, ValueError) as error:
        return {"passed": False, "reason": str(error)}


def solid_contact(first, second, first_planes=_UNSET, second_planes=_UNSET):
    planes1 = convex_planes(first) if first_planes is _UNSET else first_planes
    planes2 = convex_planes(second) if second_planes is _UNSET else second_planes
    if planes1 is None or planes2 is None:
        # Curved caps can have a rectangular base and a nonconvex cap boundary.
        # For those closed positive-volume solids prove a contact with a point
        # strictly inside both actual triangle meshes, rather than their hulls.
        for mesh in (first, second):
            if mesh is None or not len(mesh.triangles) or not mesh.is_watertight():
                return {"passed": False, "interior_margin_m": None, "reason": "contact requires closed actual triangle solids"}
            corners = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
            volume = float(np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])).sum() / 6)
            if volume <= 1e-12:
                return {"passed": False, "interior_margin_m": None, "reason": "contact requires outward positive-volume solids"}
        vertices1, vertices2 = np.asarray(first.vertices), np.asarray(second.vertices)
        center1, center2 = vertices1.mean(0), vertices2.mean(0)
        candidates = [(center1 + center2)[None, :] / 2, center1[None, :], center2[None, :]]
        for mesh, center in ((first, center1), (second, center2)):
            vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            surface = np.vstack((vertices, vertices[faces].mean(1)))
            for fraction in (.0001, .001, .01, .05, .2):
                candidates.append(surface * (1 - fraction) + center * fraction)
        candidates = np.vstack(candidates).astype(np.float32)
        signed1 = q.ray_scene(first).compute_signed_distance(q.o3d.core.Tensor(candidates), nthreads=1, nsamples=3).numpy()
        signed2 = q.ray_scene(second).compute_signed_distance(q.o3d.core.Tensor(candidates), nthreads=1, nsamples=3).numpy()
        margins = np.minimum(-signed1, -signed2)
        best = int(np.argmax(margins))
        margin = float(margins[best])
        passed = bool(margin > max(CONTACT_MARGIN_M, 2e-6))
        return {"passed": passed, "interior_margin_m": margin, "interior_witness_m": candidates[best].astype(float).tolist() if passed else None,
                "method": "verified common point with negative Open3D signed triangle distance to each closed positive-volume solid (three parity rays); no convex-hull substitution"}
    planes = np.concatenate((planes1, planes2))
    # A verified witness strictly inside all actual unit triangle halfspaces
    # proves positive solid overlap; no optimiser's success flag is accepted.
    # Projection into margin-eroded halfspaces uses NumPy and keeps this script
    # runnable in the project's existing Open3D runtime without new dependencies.
    vertices1, vertices2 = np.asarray(first.vertices), np.asarray(second.vertices)
    center1, center2 = vertices1.mean(0), vertices2.mean(0)
    candidates = np.vstack((center1, center2, (center1 + center2) / 2,
                            (vertices1 + center2) / 2, (vertices2 + center1) / 2))
    margins = -(candidates @ planes[:, :3].T + planes[:, 3]).max(1)
    point = candidates[int(np.argmax(margins))].copy()
    for _ in range(4000):
        signed = planes[:, :3] @ point + planes[:, 3]
        worst = int(np.argmax(signed))
        if signed[worst] < -CONTACT_MARGIN_M:
            break
        point -= (signed[worst] + CONTACT_MARGIN_M * 4) * planes[worst, :3]
    margin = float(-(planes[:, :3] @ point + planes[:, 3]).max())
    passed = bool(margin > CONTACT_MARGIN_M)
    return {"passed": passed, "interior_margin_m": margin, "interior_witness_m": point.tolist() if passed else None,
            "method": "verified common interior point in actual unit triangle halfspaces; deterministic eroded-halfspace projection"}


def relative_mesh_path(fit_dir, filename):
    if not isinstance(filename, str) or not filename or Path(filename).is_absolute():
        raise ValueError("mesh filename must be relative to explicit fit directory")
    path = (Path(fit_dir) / filename).resolve()
    if not path.is_relative_to(Path(fit_dir).resolve()):
        raise ValueError("mesh filename escapes fit directory")
    return path


def check_object(fit_dir, obj, observed, frames, floor_y, training_support_points=None):
    components = obj.get("components", [])
    names = [item.get("name") for item in components]
    if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("component names must be unique and nonempty")
    assembly_path = relative_mesh_path(fit_dir, obj.get("mesh"))
    hashes = {"assembly": q.sha256(assembly_path)} if assembly_path.is_file() else {}
    assembly, status = q.read_mesh(assembly_path)
    meshes, planes, component_rows, all_vertices, all_faces, offset = {}, {}, [], [], [], 0
    train = observed[frames % 2 == 0] if training_support_points is None else training_support_points
    for component in components:
        name = component["name"]
        path = relative_mesh_path(fit_dir, component.get("mesh"))
        if path.is_file():
            hashes[name] = q.sha256(path)
        mesh, mesh_status = q.read_mesh(path)
        meshes[name] = mesh
        planes[name] = convex_planes(mesh)
        volume = None
        winding = False
        closed = bool(mesh is not None and len(mesh.triangles) and mesh.is_watertight())
        if closed:
            v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            c = v[f]
            volume = float(np.einsum("ij,ij->i", c[:, 0], np.cross(c[:, 1], c[:, 2])).sum() / 6)
            directed = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
            unique, counts = np.unique(directed, axis=0, return_counts=True)
            winding = bool(np.all(counts == 1) and np.array_equal(unique, np.unique(directed[:, ::-1], axis=0)))
        if mesh is not None and len(mesh.triangles):
            all_vertices.append(np.asarray(mesh.vertices))
            all_faces.append(np.asarray(mesh.triangles) + offset)
            offset += len(mesh.vertices)
        component_rows.append({"name": name, "role": component.get("role"), "mesh": component["mesh"],
                               "sha256": hashes.get(name), "status": mesh_status,
                               "watertight": closed, "signed_volume_m3": volume,
                               "consistent_edge_winding": winding,
                               "outward_convex": planes[name] is not None,
                               "positive_volume": bool(volume is not None and volume > 1e-12),
                               "parameter_consistency": parameter_consistency(mesh, component), "topology": q.topology(mesh)})
    matrix = np.asarray([q.distances(q.ray_scene(meshes[name]), train) for name in names])
    assigned = np.argmin(matrix, axis=0) if len(train) else np.empty(0, dtype=int)
    for index, row in enumerate(component_rows):
        selected = assigned == index
        supported = selected & (matrix[index] <= DISTANCE_GATE_M)
        support_points = train[supported]
        row["training_support"] = {"assigned_point_count": int(selected.sum()),
                                   "within_10mm_count": int(supported.sum()),
                                   "distinct_supported_xyz_count": len(np.unique(support_points, axis=0)),
                                   "supported_xyz_bounds_m": [support_points.min(0).tolist(), support_points.max(0).tolist()] if len(support_points) else None,
                                   "sparse_support_note": "fewer than ten candidate-even samples; positive support does not establish well-constrained parameters" if len(support_points) < 10 else None,
                                   "distance": q.distance_summary(matrix[index, selected]),
                                   "method": "all target candidate-even points; closest actual component, stable name-order tie break; support requires actual triangle distance <=10mm"}
        row["training_support"]["passed"] = row["training_support"]["within_10mm_count"] > 0
    assembly_consistency = {"status": "fail", "reason": "missing assembly or component triangles"}
    if assembly is not None and len(all_vertices) == len(components):
        assembly_consistency = compare_triangle_geometry(np.concatenate(all_vertices), np.concatenate(all_faces),
                                                         np.asarray(assembly.vertices), np.asarray(assembly.triangles), tolerance=2e-6)
    declaration = obj.get("contact_floor", {})
    contact_rows, graph = [], {name: set() for name in names}
    for joint in declaration.get("joints", []):
        pair = joint.get("parts", [])
        if len(pair) != 2 or pair[0] == pair[1] or any(name not in graph for name in pair):
            raise ValueError("contact references invalid component identities")
        contact = solid_contact(meshes[pair[0]], meshes[pair[1]], planes[pair[0]], planes[pair[1]])
        contact_rows.append({"parts": pair, **contact})
        if contact["passed"]:
            graph[pair[0]].add(pair[1])
            graph[pair[1]].add(pair[0])
    seen, pending = set(), [names[0]]
    while pending:
        node = pending.pop()
        if node not in seen:
            seen.add(node)
            pending.extend(graph[node] - seen)
    feet = declaration.get("feet", [])
    foot_names = [item.get("name") for item in feet]
    support_names = [item["name"] for item in components if item.get("role") == "floor_support"]
    roles = [item.get("role") for item in components]
    primary_parts = roles.count("top") == 1 if obj["category"] == "Table" else roles.count("seat") == 1 and roles.count("back") == 1
    foot_rows = []
    for name in foot_names:
        mesh = meshes.get(name)
        lowest = float(np.asarray(mesh.vertices)[:, 1].min()) if mesh is not None and len(mesh.vertices) else None
        gap = lowest - floor_y if lowest is not None else None
        foot_rows.append({"name": name, "lowest_vertex_y_m": lowest, "gap_m": gap,
                          "passed": bool(gap is not None and abs(gap) <= FLOOR_GATE_M)})
    all_min = min((float(np.asarray(mesh.vertices)[:, 1].min()) for mesh in meshes.values() if mesh is not None and len(mesh.vertices)), default=None)
    distances = q.distances(q.ray_scene(assembly), observed)
    validation = q.distance_summary(distances[frames % 2 == 1])
    within = next(item for item in validation["within"] if item["distance_m"] == DISTANCE_GATE_M)
    observation_checks = {"all_original_odd_frame_owner_observations_retained": validation["sample_count"] == int(np.count_nonzero(frames % 2 == 1)),
                          "nonempty_validation": validation["sample_count"] > 0,
                          "triangle_p95_at_most_10mm": bool(validation["p95_distance_m"] is not None and validation["p95_distance_m"] <= DISTANCE_GATE_M),
                          "at_least_95_percent_within_10mm": bool(within["fraction"] is not None and within["fraction"] >= .95)}
    structure_checks = {"every_part_has_measured_training_support": all(row["training_support"]["passed"] for row in component_rows),
                        "selected_furniture_family_has_required_primary_parts": primary_parts,
                        "all_parts_watertight_outward_positive_volume": all(row["watertight"] and row["positive_volume"] and row["consistent_edge_winding"] and row["topology"]["connected_triangle_components"] == 1 for row in component_rows),
                        "parameters_match_actual_triangles": all(row["parameter_consistency"]["passed"] for row in component_rows),
                        "assembly_equals_components": assembly_consistency["status"] == "pass",
                        "exactly_four_distinct_feet": len(foot_names) == 4 and len(set(foot_names)) == 4 and set(foot_names) == set(support_names),
                        "feet_within_3mm_of_measured_floor": bool(foot_rows and all(item["passed"] for item in foot_rows)),
                        "no_geometry_below_floor_beyond_3mm": bool(all_min is not None and all_min >= floor_y - FLOOR_GATE_M),
                        "all_declared_contacts_have_positive_overlap": bool(contact_rows and all(item["passed"] for item in contact_rows)),
                        "intended_contact_graph_connects_every_component": len(seen) == len(names),
                        "floor_declaration_matches_input": bool(np.isclose(declaration.get("floor_y_m", np.nan), floor_y, atol=1e-8))}
    row = {"uuid": obj["uuid"], "assembly_mesh": obj["mesh"], "assembly_sha256": hashes.get("assembly"), "status": status,
           "observation_gate": {"passed": all(observation_checks.values()), "checks": observation_checks},
           "structural_gate": {"passed": all(structure_checks.values()), "checks": structure_checks},
           "observed_point_mesh_distance": {"training_even": q.distance_summary(distances[frames % 2 == 0]), "validation_odd": validation},
           "exclusions": {"count_from_validation_gate": 0, "declared_by_fitter": obj.get("exclusions", []), "policy": "all original owner-assigned points remain in reported distances and gates"},
           "components": component_rows, "assembly_consistency": assembly_consistency,
           "training_support_pool": {"candidate_even_point_count": len(train),
                                     "original_owner_even_point_count": int(np.count_nonzero(frames % 2 == 0)),
                                     "method": "original observations.npz target candidates column and frame parity; no fitter support flags"},
           "contacts": contact_rows, "floor": {"measured_y_m": floor_y, "feet": foot_rows}}
    return row, assembly, meshes, hashes


def reference_metrics(assembly, component_meshes, reference):
    scene = q.ray_scene(assembly)
    ray_distances = q.distances(scene, reference["points"])
    return {"valid_reference_ray_hit_recall": q.distance_summary(ray_distances),
            "reverse_fitted_area_to_reference": q.distance_summary(q.distances(reference["scene"], sample_area(assembly))),
            "reference_ray_to_physical_mesh_consistency": reference["ray_consistency"],
            "reference_ray_reconstruction_precision_bound_m": REFERENCE_PRECISION_M,
            "reference_ray_precision_reason": "original Float32 depth and triangle intersections; tolerance bounds reference ray reconstruction only, while candidate observation gate remains10mm",
            "per_reference_component": [{"id": part["id"], "triangle_count": part["triangle_count"],
                "world_bounds_m": [np.asarray(part["mesh"].vertices).min(0).tolist(), np.asarray(part["mesh"].vertices).max(0).tolist()],
                "valid_reference_ray_hit_recall": q.distance_summary(ray_distances[reference["ray_parts"] == part["part_index"]]),
                "reference_area_to_fitted_mesh": q.distance_summary(q.distances(scene, sample_area(part["mesh"]))) }
                for part in reference["parts"]],
            "per_fitted_component_reverse_area": [{"name": name,
                "reverse_fitted_area_to_reference_object": q.distance_summary(q.distances(reference["scene"], sample_area(mesh)))}
                for name, mesh in component_meshes.items()]}


def evaluate(scene_path, observations_path, fit_dir, reference_dir=None, baseline_dir=None, *, report_base=None):
    scene_path, observations_path, fit_dir = Path(scene_path), Path(observations_path), Path(fit_dir)
    base = Path(report_base) if report_base else fit_dir
    inputs = {name: record(path, base) for name, path in (("scene", scene_path), ("observations", observations_path))}
    scene, ids, floor_y, points, frames, owners, candidates = load_inputs(scene_path, observations_path)
    fit_path = fit_dir / "fit.json"
    fit_hash = q.sha256(fit_path)
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    objects = fit.get("objects", [])
    if (fit.get("schema") != "alvenx.spatial.dining-fit.v1" or fit.get("units") != "m" or fit.get("up_axis") != "Y" or
            len(objects) != len(TARGETS) or set(obj.get("uuid") for obj in objects) != set(TARGETS)):
        raise ValueError("fit target identities or schema mismatch")
    input_match = all(fit.get("input_files", {}).get(name, {}).get("sha256") == item["sha256"] for name, item in inputs.items())
    references, reference_provenance = load_reference(reference_dir) if reference_dir else ({}, None)
    baseline_provenance = None
    if baseline_dir:
        baseline_manifest = Path(baseline_dir) / "baseline.json"
        if baseline_manifest.is_file():
            baseline_provenance = json.loads(baseline_manifest.read_text(encoding="utf-8"))
            if (baseline_provenance.get("schema") != "alvenx.spatial.dining-baseline.v1" or
                    baseline_provenance.get("original_input_scene_sha256") != inputs["scene"]["sha256"] or
                    baseline_provenance.get("frozen_observation_inputs") != {name: item["sha256"] for name, item in inputs.items()}):
                raise ValueError("baseline frozen inputs differ from candidate observations")
            for item in baseline_provenance["files"]:
                if item["path"] not in [uuid + ".ply" for uuid in TARGETS] or q.sha256(Path(baseline_dir) / item["path"]) != item["sha256"]:
                    raise ValueError("baseline byte hashes differ from archived original")
    rows, frozen, component_hashes = [], {fit_path: fit_hash, scene_path: inputs["scene"]["sha256"], observations_path: inputs["observations"]["sha256"]}, {}
    if references:
        for name in ("reference.json", "labels.npz", "rays.npz"):
            path = Path(reference_dir) / name
            frozen[path] = q.sha256(path)
    for uuid in TARGETS:
        obj = next(item for item in objects if item["uuid"] == uuid)
        source = next(item for item in scene["objects"] if item["uuid"] == uuid)
        if obj.get("source_owner_index") != ids.index(uuid) or obj.get("category") != source["category"]:
            raise ValueError("fit source owner or category mismatch")
        selected = owners == ids.index(uuid)
        support_points = points[candidates[:, ids.index(uuid)] & (frames % 2 == 0)]
        row, assembly, components, hashes = check_object(fit_dir, obj, points[selected], frames[selected], floor_y, support_points)
        for item in obj["components"]:
            path = relative_mesh_path(fit_dir, item["mesh"])
            if path.is_file():
                frozen[path] = hashes[item["name"]]
                component_hashes[uuid + "/" + item["name"]] = hashes[item["name"]]
        assembly_path = relative_mesh_path(fit_dir, obj["mesh"])
        if assembly_path.is_file():
            frozen[assembly_path] = hashes["assembly"]
        if references:
            row["reference_evaluation"] = reference_metrics(assembly, components, references[uuid])
        if baseline_dir:
            baseline_path = Path(baseline_dir) / (uuid + ".ply")
            baseline_record = record(baseline_path, base) if baseline_path.is_file() else {"status": "missing"}
            if baseline_path.is_file():
                frozen[baseline_path] = baseline_record["sha256"]
            baseline, status = q.read_mesh(baseline_path)
            baseline_scene = q.ray_scene(baseline)
            row["original_b1_baseline"] = {"mesh": baseline_record,
                "same_frozen_scene_and_observations_verified": baseline_provenance is not None,
                "status": status, "topology": q.topology(baseline),
                "same_original_owner_validation": q.distance_summary(q.distances(baseline_scene, points[selected][frames[selected] % 2 == 1]))}
            if references:
                row["original_b1_baseline"]["reference_evaluation"] = reference_metrics(baseline, {}, references[uuid])
        rows.append(row)
    for path, digest in frozen.items():
        if q.sha256(path) != digest:
            raise ValueError(f"candidate or original input changed during evaluation: {path.name}")
    passed = bool(input_match and all(row["observation_gate"]["passed"] and row["structural_gate"]["passed"] for row in rows))
    return {"schema": "alvenx.spatial.dining-evaluation.v1", "status": "passed" if passed else "failed",
            "quality_gate": {"passed": passed, "input_content_matches_fit_record": input_match,
                             "criteria": "Per object all original odd-frame owner points triangle P95 <=10mm and >=95% within10mm; every component measured candidate-even support; closed positive-volume solids, connected intended contacts and four feet within3mm."},
            "fit_json_sha256": fit_hash, "component_input_sha256": component_hashes, "fit": record(fit_path, base),
            "inputs": inputs, "inputs_and_candidate_unchanged_during_evaluation": True,
            "evaluator": {"script_sha256": q.sha256(__file__), "triangle_distance_helper_sha256": q.sha256(q.__file__),
                          "open3d_version": q.o3d.__version__},
            "objects": rows, "reference_evaluation_present": bool(references),
            "reference": {"manifest": record(Path(reference_dir) / "reference.json", base),
                          "files": reference_provenance["files"], "source_files": reference_provenance["source_files"]} if references else None,
            "area_sampling": {"sample_count_per_mesh": SAMPLE_COUNT, "seed": SAMPLE_SEED, "method": "uniform barycentric, triangle probability proportional to area"},
            "limitations": ["Odd frames are development validation, not an untouched final test set.",
                "Captured valid-ray recall counts repeated rays; it does not estimate full or visible surface area.",
                "Reverse distances sample fitted component shells including internal overlap faces; they are not exposed union area.",
                "Reference parts are original disconnected triangle primitives; fitted parts may use different structural grouping.",
                "Connected solids retain independent editable shells; acceptance does not claim a welded union.",
                "Fitter read scope is recorded separately; this evaluator does not establish which files an upstream process opened."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--fit-dir", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pack-reference", type=Path)
    parser.add_argument("--pack-output", type=Path)
    args = parser.parse_args()
    if args.pack_reference:
        if args.pack_output is None:
            parser.error("--pack-output required with --pack-reference")
        metadata = pack_reference(args.pack_reference, args.pack_output)
        print(json.dumps({"pack": str(args.pack_output), "objects": metadata["objects"]}, indent=2))
        return
    if any(value is None for value in (args.scene, args.observations, args.fit_dir, args.output)):
        parser.error("--scene --observations --fit-dir --output required")
    output = q.new_report_path(args.output)
    report = evaluate(args.scene, args.observations, args.fit_dir, args.reference, args.baseline_dir, report_base=output.parent)
    q.write_new_report(output, report)
    print(json.dumps({"output": str(output), "status": report["status"], "quality_gate": report["quality_gate"],
                      "objects": [{"uuid": row["uuid"], "observation_gate": row["observation_gate"],
                                   "structural_gate": row["structural_gate"]} for row in report["objects"]]}, indent=2))
    raise SystemExit(0 if report["status"] == "passed" and report["quality_gate"].get("passed") is True else 2)


if __name__ == "__main__":
    main()
