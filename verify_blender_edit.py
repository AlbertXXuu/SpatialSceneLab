"""Measured native Blender editing regression, not a user study. MIT."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_edit as edit
import blender_scene
from geometry_compare import compare_triangle_geometry

POSITION_TOLERANCE = 5e-5
COLOR_TOLERANCE = 1 / 255 + 1e-6


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def material_values(material):
    shader = next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
    if shader is None or not shader.inputs["Base Color"].is_linked:
        raise ValueError("requires observed vertex-color material")
    return [shader.inputs["Roughness"].default_value,
            shader.inputs["Metallic"].default_value,
            float(any(node.type in {"VERTEX_COLOR", "ATTRIBUTE"} for node in material.node_tree.nodes))]


def mesh_arrays(obj):
    mesh = obj.data
    if any(len(face.vertices) != 3 for face in mesh.polygons):
        raise ValueError("editing evidence requires triangular observed faces")
    xyz = np.empty((len(mesh.vertices), 3), np.float32)
    mesh.vertices.foreach_get("co", xyz.ravel())
    indices = np.empty(len(mesh.loops), np.int32)
    mesh.loops.foreach_get("vertex_index", indices)
    indices = indices.reshape(-1, 3)
    matrix = np.asarray(obj.matrix_world, np.float64)
    world = xyz @ matrix[:3, :3].T + matrix[:3, 3]
    layer = mesh.color_attributes[0]
    rgba = np.empty((len(layer.data), 4), np.float32)
    layer.data.foreach_get("color", rgba.ravel())
    colors = rgba[indices] if layer.domain == "POINT" else rgba.reshape(-1, 3, 4)
    material_ids = np.empty(len(mesh.polygons), np.int32)
    mesh.polygons.foreach_get("material_index", material_ids)
    materials = np.asarray([material_values(mat) for mat in mesh.materials])[material_ids]
    return world, indices, colors, materials


def snapshot():
    edit.source_inventory()
    values = {key: [] for key in ("ids", "owners", "corners", "colors", "materials", "reviewed")}
    for obj in edit.managed_meshes():
        world, faces, colors, materials = mesh_arrays(obj)
        ids = np.empty(len(faces), np.int32)
        obj.data.attributes[edit.SOURCE_FACE].data.foreach_get("value", ids)
        reviewed = np.empty(len(faces), np.int32)
        obj.data.attributes[edit.REVIEWED].data.foreach_get("value", reviewed)
        values["ids"].append(ids)
        values["owners"].append(np.full(len(faces), obj["alvenx_id"]))
        values["corners"].append(world[faces])
        values["colors"].append(colors)
        values["materials"].append(materials)
        values["reviewed"].append(reviewed)
    merged = {key: np.concatenate(value) for key, value in values.items()}
    order = np.argsort(merged["ids"])
    return {key: value[order] for key, value in merged.items()}


def check_state(expected, actual):
    if not np.array_equal(expected["ids"], actual["ids"]):
        return {"status": "fail", "reason": "source-face inventory differs"}
    position = float(np.max(np.linalg.norm(expected["corners"] - actual["corners"], axis=-1)))
    color = float(np.max(np.abs(expected["colors"] - actual["colors"])))
    material = float(np.max(np.abs(expected["materials"] - actual["materials"])))
    owner = bool(np.array_equal(expected["owners"], actual["owners"]))
    reviewed = bool(np.array_equal(expected["reviewed"], actual["reviewed"]))
    return {"status": "pass" if position <= POSITION_TOLERANCE and color <= 1e-6
            and material <= 1e-6 and owner and reviewed else "fail",
            "maximum_corner_distance_m": position, "maximum_linear_rgba_difference": color,
            "maximum_material_value_difference": material, "owners_match": owner,
            "reviewed_flags_match": reviewed, "source_face_instances": len(actual["ids"])}


def require(check):
    if check["status"] != "pass":
        raise AssertionError(json.dumps(check))
    return check


def appearance_snapshot():
    return {obj["alvenx_id"]: dict(zip(("vertices", "faces", "colors", "materials"), mesh_arrays(obj)))
            for obj in edit.managed_meshes()}


def check_appearance(expected, actual):
    if set(expected) != set(actual):
        return {"status": "fail", "reason": "appearance object identities differ"}
    reports = []
    for identity, source in expected.items():
        returned = actual[identity]
        pairs = compare_triangle_geometry(source["vertices"], source["faces"], returned["vertices"],
                                          returned["faces"], return_correspondence=True)
        if pairs["status"] != "pass":
            reports.append({"id": identity, "status": "fail", "geometry": pairs})
            continue
        face_indices = np.asarray(pairs.pop("matched_actual_face_indices"))
        permutations = np.asarray(pairs.pop("matched_actual_corner_permutations"))
        colors = returned["colors"][face_indices][np.arange(len(face_indices))[:, None], permutations]
        error = float(np.max(np.abs(source["colors"] - colors)))
        material = float(np.max(np.abs(source["materials"] - returned["materials"][face_indices])))
        reports.append({"id": identity, "status": "pass" if error <= COLOR_TOLERANCE and material <= 1e-5 else "fail",
                        "maximum_linear_rgba_difference": error, "maximum_material_value_difference": material})
    return {"status": "pass" if all(item["status"] == "pass" for item in reports) else "fail",
            "linear_rgba_tolerance": COLOR_TOLERANCE, "objects": reports}


def activate(obj):
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for other in bpy.context.selected_objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def undo_redo(previous, current):
    undo = bpy.ops.ed.undo()
    first = require(check_state(previous, snapshot()))
    redo = bpy.ops.ed.redo()
    second = require(check_state(current, snapshot()))
    return {"status": "pass", "native_undo_return": sorted(undo), "native_redo_return": sorted(redo),
            "undo": first, "redo": second,
            "context": "background Blender native undo stack, explicit checkpoints"}


def transfer(target_id, count, results):
    previous = snapshot()
    target = edit.object_by_id(target_id)
    source = edit.object_by_id("environment")
    activate(source)
    center = np.asarray(edit.geometry_center(target))
    world, faces, _, _ = mesh_arrays(source)
    indices = np.argsort(np.linalg.norm(world[faces].mean(axis=1) - center, axis=1))[:count]
    for face in source.data.polygons:
        face.select = False
    ids = []
    for index in indices:
        source.data.polygons[int(index)].select = True
        ids.append(source.data.attributes[edit.SOURCE_FACE].data[int(index)].value)
    # Test the actual operator's Edit Mode path, including its native separate/join.
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.context.tool_settings.mesh_select_mode = (False, False, True)
    bm = bmesh.from_edit_mesh(source.data)
    bm.faces.ensure_lookup_table()
    for face in bm.faces:
        face.select_set(False)
    for edge in bm.edges:
        edge.select_set(False)
    for vertex in bm.verts:
        vertex.select_set(False)
    for index in indices:
        bm.faces[int(index)].select_set(True)
    bmesh.update_edit_mesh(source.data, loop_triangles=False, destructive=False)
    started = time.perf_counter()
    operator = bpy.ops.alvenx.reassign_faces(target_id=target_id)
    elapsed = time.perf_counter() - started
    if operator != {"FINISHED"}:
        raise RuntimeError("reassignment operator did not finish")
    expected = {key: value.copy() for key, value in previous.items()}
    selected = np.isin(expected["ids"], ids)
    # Fixed-width string arrays must accommodate every identity in this scene.
    expected["owners"] = expected["owners"].astype(f'<U{max(len(target_id), expected["owners"].dtype.itemsize // 4)}')
    expected["owners"][selected] = target_id
    expected["reviewed"][selected] = 1
    current = snapshot()
    check = require(check_state(expected, current))
    bpy.ops.ed.undo_push(message="assigned observed faces")
    recovery = undo_redo(previous, current)
    results.append({"kind": "face_reassignment", "target_id": target_id, "selected_faces": len(ids),
                    "source_face_ids": sorted(ids), "operator_seconds": elapsed, "state": check,
                    "undo_redo": recovery,
                    "selection_policy": "scripted nearest centroids to target bounds centre; no reference labels"})


def transform(target_id, translation, degrees, results):
    previous = snapshot()
    obj = edit.object_by_id(target_id)
    activate(obj)
    pivot = edit.geometry_center(obj)
    delta = (Matrix.Translation(Vector(translation)) @ Matrix.Translation(Vector(pivot))
             @ Matrix.Rotation(np.deg2rad(degrees), 4, "Z") @ Matrix.Translation(-Vector(pivot)))
    bpy.context.scene.ax_translation = translation
    bpy.context.scene.ax_rotation = degrees
    started = time.perf_counter()
    operator = bpy.ops.alvenx.rigid_edit()
    elapsed = time.perf_counter() - started
    if operator != {"FINISHED"}:
        raise RuntimeError("rigid edit operator did not finish")
    expected = {key: value.copy() for key, value in previous.items()}
    selected = expected["owners"] == target_id
    matrix = np.asarray(delta)
    expected["corners"][selected] = expected["corners"][selected] @ matrix[:3, :3].T + matrix[:3, 3]
    current = snapshot()
    check = require(check_state(expected, current))
    bpy.ops.ed.undo_push(message="edited observed region")
    recovery = undo_redo(previous, current)
    results.append({"kind": "rigid_edit", "target_id": target_id, "operator_seconds": elapsed,
                    "translation_blender_z_up_m": list(translation), "rotation_world_z_degrees": degrees,
                    "pivot_world_z_up_m": pivot, "state": check, "undo_redo": recovery})


def render(path):
    scene = bpy.context.scene
    scene.cycles.samples = 8
    scene.cycles.seed = 0
    scene.cycles.use_adaptive_sampling = False
    scene.render.resolution_x = 1000
    scene.render.resolution_y = 700
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    # Render Result pixel buffers can be empty after write_still; use the saved file.
    image = bpy.data.images.load(str(path), check_existing=False)
    pixels = np.empty(len(image.pixels), np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    return pixels


def view_record():
    camera = bpy.context.scene.camera
    return {"camera": [list(row) for row in camera.matrix_world], "ortho_scale": camera.data.ortho_scale,
            "lights": [{"name": obj.name, "matrix": [list(row) for row in obj.matrix_world],
                        "energy": obj.data.energy, "size": obj.data.size}
                       for obj in bpy.context.scene.objects if obj.type == "LIGHT"]}


def restore_view(record):
    blender_scene.setup_view(edit.managed_meshes())
    scene = bpy.context.scene
    scene.camera.matrix_world = Matrix(record["camera"])
    scene.camera.data.ortho_scale = record["ortho_scale"]
    for light in record["lights"]:
        obj = bpy.data.objects[light["name"]]
        obj.matrix_world = Matrix(light["matrix"])
        obj.data.energy = light["energy"]
        obj.data.size = light["size"]
    bpy.context.view_layer.update()


def edit_run(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("retain prior results; output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    result = {"schema": "spatial-scene-lab.native-edit-task.v1", "status": "running",
              "blender": bpy.app.version_string, "source_scene_sha256": edit.digest(args.scene),
              "primary": args.primary, "secondary": args.secondary,
              "execution": "scripted native Blender operators; not interactive user timing",
              "operations": []}
    started = time.perf_counter()
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.context.preferences.edit.use_global_undo = True
        edit.import_surface(args.scene)
        initial = snapshot()
        result["initial_source_faces"] = len(initial["ids"])
        source_registry = bpy.context.scene["ax_sources"]
        blender_scene.setup_view(edit.managed_meshes())
        if args.render:
            render(output / "before.png")
        bpy.ops.wm.save_as_mainfile(filepath=str(output / "before.blend"))
        bpy.ops.ed.undo_push(message="initial observed scene")
        transfer(args.primary, 12, result["operations"])
        transform(args.primary, (.25, -.1, 0), 0, result["operations"])
        transform(args.secondary, (0, 0, 0), 32, result["operations"])
        # Reassignment into an already transformed region checks coordinate conversion.
        transfer(args.secondary, 8, result["operations"])
        final = snapshot()
        np.savez_compressed(output / "expected-state.npz", **final)
        expected_metadata = {"source_registry": source_registry,
                             "journal": bpy.context.scene["ax_journal"],
                             "input_manifest_sha256": bpy.context.scene["ax_source_manifest_sha256"]}
        save_json(output / "expected-metadata.json", expected_metadata)
        if len(json.loads(expected_metadata["journal"])) != 4:
            raise AssertionError("undo/redo did not preserve four committed edit records")
        bpy.ops.wm.save_as_mainfile(filepath=str(output / "after.blend"))
        expected_geometry = blender_scene.world_geometry()
        expected_appearance = appearance_snapshot()
        sidecar = edit.export_package(output / "after.glb")
        identity = json.loads(sidecar.read_text(encoding="utf-8"))
        result["identity_package"] = {"status": "pass" if identity["asset_sha256"] == edit.digest(output / "after.glb")
                                      and identity["journal"] == json.loads(expected_metadata["journal"])
                                      and identity["sources"] == json.loads(source_registry) else "fail",
                                      "scope": identity["face_identity_scope"]}
        require(result["identity_package"])
        view = view_record()
        pixels = render(output / "after.png") if args.render else None
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=str(output / "after.glb"))
        result["glb_geometry"] = require(blender_scene.compare_geometry(expected_geometry, blender_scene.world_geometry()))
        actual_appearance = appearance_snapshot()
        result["glb_appearance"] = require(check_appearance(expected_appearance, actual_appearance))
        # Two appearance faults should be rejected even when geometry is unchanged.
        fault = copy.deepcopy(actual_appearance)
        first = next(iter(fault))
        fault[first]["colors"][0, :, 3] = 0
        result["changed_color_rejected"] = check_appearance(expected_appearance, fault)["status"] == "fail"
        fault = copy.deepcopy(actual_appearance)
        fault[first]["materials"][0, 0] += .2
        result["changed_material_rejected"] = check_appearance(expected_appearance, fault)["status"] == "fail"
        if not result["changed_color_rejected"] or not result["changed_material_rejected"]:
            raise AssertionError("appearance checker accepted a deliberate fault")
        if args.render:
            restore_view(view)
            returned_pixels = render(output / "after-reimport.png")
            difference = np.abs(pixels.reshape(-1, 4)[:, :3] - returned_pixels.reshape(-1, 4)[:, :3])
            image_check = {"status": "pass" if float(difference.mean()) <= .01 else "fail",
                           "rgb_mean_absolute_difference": float(difference.mean()),
                           "rgb_maximum_difference": float(difference.max()), "mean_difference_tolerance": .01,
                           "scope": "same-state fixed-view CPU Cycles render; one authored development case"}
            result["render_roundtrip"] = require(image_check)
        result["status"] = "pass"
    except Exception as error:
        result.update(status="fail", error=str(error))
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        save_json(output / "task-results.json", result)
        print("ALVENX_EDIT_TASK " + json.dumps({"status": result["status"], "primary": args.primary}))


def reopen_run(args):
    output = Path(args.output).resolve()
    with np.load(output / "expected-state.npz", allow_pickle=False) as archive:
        expected = {key: archive[key] for key in archive.files}
    metadata = json.loads((output / "expected-metadata.json").read_text())
    bpy.ops.wm.open_mainfile(filepath=str(output / "after.blend"), load_ui=False)
    result = require(check_state(expected, snapshot()))
    result["new_process"] = True
    result["metadata_match"] = (bpy.context.scene["ax_sources"] == metadata["source_registry"]
                                and bpy.context.scene["ax_journal"] == metadata["journal"]
                                and bpy.context.scene["ax_source_manifest_sha256"] == metadata["input_manifest_sha256"])
    if not result["metadata_match"] or bpy.context.scene.unit_settings.scale_length != 1:
        result["status"] = "fail"
    save_json(output / "reopen-results.json", result)
    require(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("edit", "reopen"), default="edit")
    parser.add_argument("--scene")
    parser.add_argument("--output", required=True)
    parser.add_argument("--primary")
    parser.add_argument("--secondary")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    edit.register()
    edit_run(args) if args.mode == "edit" else reopen_run(args)


if __name__ == "__main__":
    main()
