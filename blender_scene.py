"""Native Blender edit, save, GLB export and world-geometry round trip. MIT.

blender --background --python blender_scene.py -- --scene surface-scene.json
        --output output-directory [--object UUID --translate .25 0 0]
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree


Y_UP_TO_Z_UP = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def world_geometry():
    result = {}
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        if "alvenx_id" not in obj:
            raise ValueError(f"mesh has no provenance identity: {obj.name}")
        identity = obj["alvenx_id"]
        if identity in result:
            raise ValueError(f"duplicate mesh identity: {identity}")
        obj.data.calc_loop_triangles()
        result[identity] = {"vertices": [list(obj.matrix_world @ vertex.co) for vertex in obj.data.vertices],
                            "triangles": sum(len(face.vertices) - 2 for face in obj.data.polygons),
                            "faces": [list(face.vertices) for face in obj.data.loop_triangles],
                            "role": obj["alvenx_role"],
                            "source_sha256": obj.get("source_sha256"),
                            "vertex_colors_present": bool(obj.data.color_attributes),
                            "materials_present": bool(obj.data.materials)}
    return result


def directed_distance(source, destination):
    if not source or not destination:
        raise ValueError("round trip contains empty geometry")
    tree = KDTree(len(destination))
    for index, point in enumerate(destination):
        tree.insert(Vector(point), index)
    tree.balance()
    return max(tree.find(Vector(point))[2] for point in source)


def compare_geometry(expected, actual, tolerance=5e-5):
    if set(expected) != set(actual):
        return {"status": "fail", "reason": "object identities differ",
                "missing": sorted(set(expected) - set(actual)), "extra": sorted(set(actual) - set(expected))}
    objects = []
    for identity, original in expected.items():
        returned = actual[identity]
        error = max(directed_distance(original["vertices"], returned["vertices"]),
                    directed_distance(returned["vertices"], original["vertices"]))
        triangle_match = original["triangles"] == returned["triangles"]
        # Exporters may split color/normal seams and reorder vertices. Map
        # both meshes into one positional reference before comparing faces.
        reference = KDTree(len(original["vertices"]))
        for index, point in enumerate(original["vertices"]):
            reference.insert(Vector(point), index)
        reference.balance()
        def connectivity(record):
            mapping = [reference.find(Vector(point))[1] for point in record["vertices"]]
            return Counter(tuple(sorted(mapping[index] for index in face)) for face in record["faces"])
        connectivity_match = connectivity(original) == connectivity(returned)
        role_match = original["role"] == returned["role"]
        provenance_match = original["source_sha256"] == returned["source_sha256"]
        appearance_present = (original["vertex_colors_present"] and returned["vertex_colors_present"]
                              and original["materials_present"] and returned["materials_present"])
        objects.append({"id": identity, "maximum_world_vertex_distance_m": error,
                        "triangles_match": triangle_match, "connectivity_match": connectivity_match, "role_match": role_match,
                        "source_provenance_match": provenance_match, "appearance_present": appearance_present,
                        "status": "pass" if error <= tolerance and triangle_match and connectivity_match and role_match
                        and provenance_match and appearance_present else "fail"})
    return {"status": "pass" if all(item["status"] == "pass" for item in objects) else "fail",
            "tolerance_m": tolerance, "objects": objects,
            "maximum_world_vertex_distance_m": max(item["maximum_world_vertex_distance_m"] for item in objects)}


def verifier_negative_controls(geometry):
    """Faults in the returned geometry must be detected, not merely accepted."""
    identity = next(iter(geometry))
    faults = {}
    missing = copy.deepcopy(geometry)
    del missing[identity]
    faults["missing_object"] = missing
    moved = copy.deepcopy(geometry)
    moved[identity]["vertices"][0][0] += .02
    faults["unexpected_other_vertex_change"] = moved
    provenance = copy.deepcopy(geometry)
    provenance[identity]["source_sha256"] = "wrong-source"
    faults["changed_source_hash"] = provenance
    appearance = copy.deepcopy(geometry)
    appearance[identity]["vertex_colors_present"] = False
    faults["missing_vertex_colors"] = appearance
    topology = copy.deepcopy(geometry)
    old = topology[identity]["faces"][0][0]
    topology[identity]["faces"][0][0] = (old + 7) % len(topology[identity]["vertices"])
    faults["changed_triangle_connectivity"] = topology
    results = {name: compare_geometry(geometry, returned)["status"] == "fail"
               for name, returned in faults.items()}
    if not all(results.values()):
        raise RuntimeError("geometry verifier accepted an injected fault")
    return {"status": "pass", "injected_faults_detected": results}


def add_material(obj):
    material = bpy.data.materials.new(f"Observed color: {obj.name}")
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Roughness"].default_value = .75
    if obj.data.color_attributes:
        attribute = material.node_tree.nodes.new("ShaderNodeVertexColor")
        attribute.layer_name = obj.data.color_attributes[0].name
        material.node_tree.links.new(attribute.outputs["Color"], shader.inputs["Base Color"])
    obj.data.materials.append(material)


def setup_view(meshes):
    points = [obj.matrix_world @ vertex.co for obj in meshes for vertex in obj.data.vertices]
    lower = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    upper = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    center = (lower + upper) / 2
    span = max(upper - lower)
    camera_data = bpy.data.cameras.new("Review camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = span * 1.65
    camera = bpy.data.objects.new("Review camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = center + Vector((span, -span, span * .9))
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera
    for name, offset, power in [("Key", (1, -1, 2), 2200), ("Fill", (-1, .5, 1.5), 1500)]:
        data = bpy.data.lights.new(name, "AREA")
        data.energy = power
        data.shape = "DISK"
        data.size = span * 1.5
        light = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(light)
        light.location = center + Vector(offset) * span
        light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()
    scene = bpy.context.scene
    scene.world = bpy.data.worlds.new("Local review background")
    scene.world.color = (.8, .86, .94)
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 800
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1
    scene.view_settings.view_transform = "AgX"


def export_glb(path):
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB",
                              export_extras=True, export_cameras=True)


def reopen_glb(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path))
    return world_geometry()


def run(scene_path, output_path, object_id, translation, render=True):
    scene_path = Path(scene_path).resolve()
    document = json.loads(scene_path.read_text(encoding="utf-8"))
    if document.get("schema") != "spatial-scene-lab.surface.v1" or document.get("units") != "m" or document.get("up_axis") != "Y":
        raise ValueError("requires prepared metre / Y-up surface bundle")
    output = Path(output_path).resolve()
    if output == scene_path.parent:
        raise ValueError("output must differ from the source surface directory")
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    meshes = []
    for part in document["parts"]:
        source = (scene_path.parent / part["path"]).resolve()
        if scene_path.parent not in source.parents or sha256(source) != part["sha256"]:
            raise ValueError("surface path or SHA256 mismatch")
        existing = set(bpy.data.objects)
        bpy.ops.wm.ply_import(filepath=str(source))
        imported = [obj for obj in bpy.data.objects if obj not in existing and obj.type == "MESH"]
        if len(imported) != 1:
            raise ValueError("one mesh per surface part required")
        obj = imported[0]
        obj.name = part["name"]
        obj["alvenx_id"] = part["id"]
        obj["alvenx_role"] = part["role"]
        obj["source_sha256"] = part["sha256"]
        obj["unmoved_candidate_boundary_triangles"] = part["unmoved_candidate_boundary_triangles"]
        obj.matrix_world = Y_UP_TO_Z_UP
        add_material(obj)
        meshes.append(obj)
    # Keep capture cameras as inspectable objects; their CV axes are converted
    # into Blender camera axes independently of the world Y-up conversion.
    cv_to_camera = Matrix.Diagonal(Vector((1, -1, -1, 1)))
    for frame in document.get("cameras", []):
        camera_data = bpy.data.cameras.new(f"Capture {frame['frame_index']}")
        camera = bpy.data.objects.new(camera_data.name, camera_data)
        bpy.context.collection.objects.link(camera)
        camera.matrix_world = Y_UP_TO_Z_UP @ Matrix(frame["camera_to_world_cv"]) @ cv_to_camera
        camera["source_frame"] = frame["source"]
        camera["intrinsics_depth"] = json.dumps(frame["intrinsics_depth"])
        camera["depth_size"] = frame["depth_size"]
        camera["calibration_scope"] = "capture pose and intrinsic metadata; viewport lens is illustrative"
        camera.hide_render = True
        camera.hide_viewport = True
    target_id = object_id or document["preferred_editable_object"]
    targets = [obj for obj in meshes if obj["alvenx_id"] == target_id]
    if len(targets) != 1 or targets[0]["alvenx_role"] != "observed_object_region":
        raise ValueError("select one observed object region, not the environment")
    setup_view(meshes)
    before = world_geometry()
    negative_controls = verifier_negative_controls(before)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "before.blend"))
    export_glb(output / "before.glb")
    if render:
        bpy.context.scene.render.filepath = str(output / "before.png")
        bpy.ops.render.render(write_still=True)
    delta = Y_UP_TO_Z_UP.to_3x3() @ Vector(translation)
    targets[0].location += delta
    bpy.context.view_layer.update()
    after = world_geometry()
    expected = {identity: {**record, "vertices": [list(Vector(point) + delta) for point in record["vertices"]]}
                if identity == target_id else record for identity, record in before.items()}
    edit_check = compare_geometry(expected, after)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "after.blend"))
    export_glb(output / "after.glb")
    if render:
        bpy.context.scene.render.filepath = str(output / "after.png")
        bpy.ops.render.render(write_still=True)
    before_check = compare_geometry(before, reopen_glb(output / "before.glb"))
    after_check = compare_geometry(after, reopen_glb(output / "after.glb"))
    bpy.ops.wm.open_mainfile(filepath=str(output / "after.blend"))
    blend_check = compare_geometry(after, world_geometry())
    result = {"status": "pass" if all(check["status"] == "pass" for check in
                                       [edit_check, before_check, after_check, blend_check]) else "fail",
              "blender": bpy.app.version_string, "input_scene_sha256": sha256(scene_path),
              "target_id": target_id, "translation_source_y_up_m": translation,
              "translation_blender_z_up_m": list(delta), "render_device": "CPU",
              "object_count": len(before), "edit": edit_check, "before_glb": before_check,
              "after_glb": after_check, "after_blend": blend_check,
              "negative_controls": negative_controls,
              "scope": "selected observed surface region; candidate boundary faces remain in environment",
              "outputs": {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
                          for name in ["before.blend", "after.blend", "before.glb", "after.glb"]}}
    save_json(output / "blender-verification.json", result)
    print("ALVENX_VERIFICATION " + json.dumps({"status": result["status"], "objects": len(before),
                                               "target": target_id, "blender": bpy.app.version_string}))
    if result["status"] != "pass":
        raise RuntimeError("Blender round-trip verification failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--object")
    parser.add_argument("--translate", type=float, nargs=3, default=[.25, 0, 0])
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if not all(abs(value) < float("inf") for value in args.translate):
        raise ValueError("translation must be finite")
    if not any(args.translate):
        raise ValueError("select a nonzero translation to verify an actual edit")
    run(args.scene, args.output, args.object, args.translate, not args.no_render)


if __name__ == "__main__":
    main()
