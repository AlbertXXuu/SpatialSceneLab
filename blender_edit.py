"""Local Blender panel for observed-surface ownership and rigid edits. MIT.

Install this file as a Blender addon, or run it in Blender's Text Editor.
Preparation from a source checkout:
blender --background --python blender_edit.py -- --scene surface-scene.json
        --output review.blend
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import struct
from pathlib import Path
import sys

import bmesh
import bpy
from bpy.props import EnumProperty, FloatProperty, FloatVectorProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Matrix, Vector

bl_info = {"name": "AlvenX SpatialSceneLab", "author": "AlvenX",
           "version": (0, 2, 0), "blender": (5, 1, 0),
           "location": "3D View > Sidebar > AlvenX", "category": "Mesh",
           "description": "Review observed surface regions, reassign faces and edit in metres"}

SOURCE_FACE = "ax_source_face"
REVIEWED = "ax_reviewed"
Y_UP_TO_Z_UP = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def managed_meshes():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and "alvenx_id" in obj]


def object_by_id(identity):
    matches = [obj for obj in managed_meshes() if obj["alvenx_id"] == identity]
    if len(matches) != 1:
        raise ValueError(f"requires one observed region with identity {identity!r}")
    return matches[0]


def append_journal(event):
    events = json.loads(bpy.context.scene.get("ax_journal", "[]"))
    events.append(event)
    bpy.context.scene["ax_journal"] = json.dumps(events, allow_nan=False, separators=(",", ":"))


def source_inventory():
    """Source-face identity is part SHA + initial face number, independent of owner."""
    if bpy.context.mode != "OBJECT":
        raise ValueError("inspect source inventory in Object Mode")
    result = {}
    meshes = managed_meshes()
    identities = [obj["alvenx_id"] for obj in meshes]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate observed object identity")
    for obj in meshes:
        layer = obj.data.attributes.get(SOURCE_FACE)
        if layer is None or layer.domain != "FACE" or layer.data_type != "INT":
            raise ValueError(f"source-face lineage is missing: {obj.name}")
        for value in layer.data:
            if value.value in result:
                raise ValueError(f"duplicate source-face identity: {value.value}")
            result[value.value] = obj["alvenx_id"]
    sources = json.loads(bpy.context.scene.get("ax_sources", "[]"))
    expected = {index for source in sources for index in range(source["offset"], source["offset"] + source["faces"])}
    if set(result) != expected:
        raise ValueError("source-face inventory lost or gained a face")
    return result


def import_surface(scene_path):
    """Validate the complete bundle before creating any editable data."""
    path = Path(scene_path).resolve()
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (doc.get("schema") != "spatial-scene-lab.surface.v1"
            or doc.get("units") != "m" or doc.get("up_axis") != "Y"):
        raise ValueError("requires a prepared metre / Y-up surface bundle")
    if managed_meshes() or bpy.context.scene.get("ax_sources"):
        raise ValueError("open an empty review document before importing another bundle")
    identities = []
    paths = []
    for part in doc.get("parts", []):
        identity = part.get("id")
        source = (path.parent / part["path"]).resolve()
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("surface identities must be nonempty and unique")
        if path.parent not in source.parents or not source.is_file() or digest(source) != part["sha256"]:
            raise ValueError("surface path or SHA256 mismatch")
        if part.get("role") not in {"environment", "observed_object_region"} or part.get("triangles", 0) <= 0:
            raise ValueError("requires nonempty observed regions or environment")
        identities.append(identity)
        paths.append(source)
    if not paths:
        raise ValueError("surface bundle has no parts")
    old_objects = set(bpy.data.objects)
    old_materials = set(bpy.data.materials)
    old_meshes = set(bpy.data.meshes)
    sources = []
    offset = 0
    try:
        for part, source in zip(doc["parts"], paths):
            prior = set(bpy.data.objects)
            bpy.ops.wm.ply_import(filepath=str(source))
            imported = [obj for obj in bpy.data.objects if obj not in prior and obj.type == "MESH"]
            if len(imported) != 1:
                raise ValueError("requires one mesh per part")
            obj = imported[0]
            if len(obj.data.polygons) != part["triangles"] or any(len(face.vertices) != 3 for face in obj.data.polygons):
                raise ValueError("part triangle count differs from its manifest")
            obj.name = part["name"] if part["role"] != "environment" else "Environment + unassigned"
            obj["alvenx_id"] = part["id"]
            obj["alvenx_role"] = part["role"]
            # This hash identifies the immutable input part, not its edited content.
            obj["source_sha256"] = part["sha256"]
            obj["source_hash_scope"] = "initial input part lineage; edited content has an export hash"
            obj.matrix_world = Y_UP_TO_Z_UP
            layer = obj.data.attributes.new(SOURCE_FACE, "INT", "FACE")
            reviewed = obj.data.attributes.new(REVIEWED, "INT", "FACE")
            for index, item in enumerate(layer.data):
                item.value = offset + index
                reviewed.data[index].value = 0
            sources.append({"id": part["id"], "sha256": part["sha256"],
                            "offset": offset, "faces": len(layer.data)})
            offset += len(layer.data)
            material = bpy.data.materials.new(f"Observed color: {obj.name}")
            material.use_nodes = True
            shader = material.node_tree.nodes.get("Principled BSDF")
            shader.inputs["Roughness"].default_value = .75
            if not obj.data.color_attributes:
                raise ValueError("observed surface requires vertex colours")
            node = material.node_tree.nodes.new("ShaderNodeVertexColor")
            node.layer_name = obj.data.color_attributes[0].name
            material.node_tree.links.new(node.outputs["Color"], shader.inputs["Base Color"])
            obj.data.materials.append(material)
        bpy.context.scene["ax_sources"] = json.dumps(sources, separators=(",", ":"))
        bpy.context.scene["ax_journal"] = "[]"
        bpy.context.scene["ax_source_manifest_sha256"] = digest(path)
        bpy.context.scene["ax_environment_scope"] = "environment includes unassigned/ambiguous boundary faces"
        bpy.context.scene.unit_settings.system = "METRIC"
        bpy.context.scene.unit_settings.scale_length = 1
        bpy.context.view_layer.update()
        source_inventory()
    except Exception:
        for obj in set(bpy.data.objects) - old_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in set(bpy.data.meshes) - old_meshes:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        for material in set(bpy.data.materials) - old_materials:
            if material.users == 0:
                bpy.data.materials.remove(material)
        for key in ("ax_sources", "ax_journal", "ax_source_manifest_sha256", "ax_environment_scope"):
            if key in bpy.context.scene:
                del bpy.context.scene[key]
        raise
    return doc


def reassign_selected(context, target_id):
    source = context.edit_object if context.mode == "EDIT_MESH" else context.active_object
    if source is None or source.type != "MESH" or "alvenx_id" not in source:
        raise ValueError("select an imported source mesh")
    target = object_by_id(target_id)
    if source == target:
        raise ValueError("source and destination must be different regions")
    if context.mode == "EDIT_MESH" and len(context.objects_in_mode) != 1:
        raise ValueError("review one source mesh at a time; leave multi-object Edit Mode")
    for obj in (source, target):
        if (obj.data.attributes.get(SOURCE_FACE) is None
                or obj.data.attributes.get(REVIEWED) is None):
            raise ValueError("source and destination require prepared face lineage")
    if context.mode == "EDIT_MESH":
        mesh = bmesh.from_edit_mesh(source.data)
        selected = [face for face in mesh.faces if face.select]
        count = len(selected)
        if not count or count == len(mesh.faces):
            raise ValueError("select a nonempty proper subset of source faces")
        layer = mesh.faces.layers.int.get(SOURCE_FACE)
        if layer is None:
            raise ValueError("source-face lineage is missing")
        ids = sorted(face[layer] for face in selected)
    else:
        selected = [face for face in source.data.polygons if face.select]
        count = len(selected)
        if not count or count == len(source.data.polygons):
            raise ValueError("select a nonempty proper subset of source faces")
        layer = source.data.attributes.get(SOURCE_FACE)
        if layer is None:
            raise ValueError("source-face lineage is missing")
        ids = sorted(layer.data[face.index].value for face in selected)
    source_id = source["alvenx_id"]
    # Native separate/join retain per-corner colours, material slots, attributes
    # and convert between source and destination world transforms.
    prior = set(bpy.data.objects)
    if context.mode != "EDIT_MESH":
        for obj in context.selected_objects:
            obj.select_set(False)
        source.select_set(True)
        context.view_layer.objects.active = source
        bpy.ops.object.mode_set(mode="EDIT")
        # Object Mode can retain selected edge/vertex flags from a previous
        # edit. Honour exactly the selected faces, rather than expanding them
        # while Blender flushes those stale vertex flags into Edit Mode.
        context.tool_settings.mesh_select_mode = (False, False, True)
        bm = bmesh.from_edit_mesh(source.data)
        face_ids = bm.faces.layers.int[SOURCE_FACE]
        wanted = set(ids)
        for face in bm.faces:
            face.select_set(False)
        for edge in bm.edges:
            edge.select_set(False)
        for vertex in bm.verts:
            vertex.select_set(False)
        for face in bm.faces:
            if face[face_ids] in wanted:
                face.select_set(True)
        bmesh.update_edit_mesh(source.data, loop_triangles=False, destructive=False)
    bpy.ops.mesh.separate(type="SELECTED")
    bpy.ops.object.mode_set(mode="OBJECT")
    moved = [obj for obj in bpy.data.objects if obj not in prior and obj.type == "MESH"]
    if len(moved) != 1:
        raise RuntimeError("native separation did not create one transfer mesh")
    transfer = moved[0]
    review = transfer.data.attributes.get(REVIEWED)
    for item in review.data:
        item.value = 1
    for obj in context.selected_objects:
        obj.select_set(False)
    target.select_set(True)
    transfer.select_set(True)
    context.view_layer.objects.active = target
    bpy.ops.object.join()
    context.view_layer.update()
    source_inventory()
    append_journal({"kind": "face_reassignment", "source": source_id,
                    "destination": target_id, "source_face_ids": ids,
                    "selection_scope": "explicitly selected observed faces"})
    return ids


def rigid_edit(obj, translation, rotation_degrees, pivot):
    values = [*translation, rotation_degrees, *pivot]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("transform values must be finite")
    if "alvenx_id" not in obj or obj.get("alvenx_role") != "observed_object_region":
        raise ValueError("select an observed object region")
    if not any(translation) and rotation_degrees == 0:
        raise ValueError("enter a nonzero translation or rotation")
    delta = (Matrix.Translation(Vector(translation)) @ Matrix.Translation(Vector(pivot))
             @ Matrix.Rotation(math.radians(rotation_degrees), 4, "Z")
             @ Matrix.Translation(-Vector(pivot)))
    obj.matrix_world = delta @ obj.matrix_world
    bpy.context.view_layer.update()
    append_journal({"kind": "rigid_edit", "id": obj["alvenx_id"],
                    "translation_blender_z_up_m": list(translation),
                    "rotation_world_z_degrees": rotation_degrees,
                    "pivot_world_z_up_m": list(pivot), "delta_world": [list(row) for row in delta]})
    return delta


def geometry_center(obj):
    points = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    if not points:
        raise ValueError("observed region is empty")
    return [(min(point[axis] for point in points) + max(point[axis] for point in points)) / 2 for axis in range(3)]


def export_package(path):
    if bpy.context.mode != "OBJECT":
        raise ValueError("export in Object Mode")
    source_inventory()
    path = Path(path).resolve()
    if path.suffix.lower() != ".glb":
        raise ValueError("export path must end in .glb")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.glb")
    sidecar = path.with_suffix(".identity.json")
    sidecar_temp = sidecar.with_name(sidecar.name + ".partial")
    if temporary.exists() or sidecar_temp.exists():
        raise ValueError("unfinished export already exists; inspect it before retrying")
    active = bpy.context.view_layer.objects.active
    selection = list(bpy.context.selected_objects)
    meshes = managed_meshes()
    if any(obj.name not in bpy.context.view_layer.objects or obj.hide_viewport for obj in meshes):
        raise ValueError("make all observed regions available in the active view layer before exporting")
    hidden = {obj: (obj.hide_get(), obj.hide_select) for obj in meshes}
    try:
        for obj in selection:
            obj.select_set(False)
        for obj in meshes:
            obj.hide_set(False)
            obj.hide_select = False
            if not obj.visible_get():
                raise ValueError("make observed collections visible in the active view layer before exporting")
            obj.select_set(True)
        bpy.ops.export_scene.gltf(filepath=str(temporary), export_format="GLB",
                                  export_extras=True, use_selection=True)
        with temporary.open("rb") as stream:
            magic, version, length = struct.unpack("<4sII", stream.read(12))
            json_length, chunk_type = struct.unpack("<II", stream.read(8))
            if magic != b"glTF" or version != 2 or length != temporary.stat().st_size or chunk_type != 0x4E4F534A:
                raise ValueError("native exporter did not write a valid GLB 2 JSON header")
            nodes = json.loads(stream.read(json_length))["nodes"]
        exported_ids = [node.get("extras", {}).get("alvenx_id") for node in nodes if "mesh" in node]
        expected_ids = [obj["alvenx_id"] for obj in meshes]
        if len(exported_ids) != len(expected_ids) or set(exported_ids) != set(expected_ids):
            raise ValueError("exported mesh identities differ from the observed scene; package withheld")
        identity = {"schema": "spatial-scene-lab.edit-identity.v1", "units": "m", "up_axis_glb": "Y",
                    "asset_sha256": digest(temporary),
                    "input_manifest_sha256": bpy.context.scene["ax_source_manifest_sha256"],
                    "sources": json.loads(bpy.context.scene["ax_sources"]),
                    "journal": json.loads(bpy.context.scene["ax_journal"]),
                    "native_face_attributes_in_glb": False,
                    "face_identity_scope": "source IDs in native blend face order; blend is authoritative for editing",
                    "objects": [{"id": obj["alvenx_id"], "role": obj["alvenx_role"],
                                 "source_face_ids": [item.value for item in obj.data.attributes[SOURCE_FACE].data],
                                 "reviewed": [item.value for item in obj.data.attributes[REVIEWED].data]}
                                for obj in managed_meshes()]}
        sidecar_temp.write_text(json.dumps(identity, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Keep the previous pair on ordinary write/rename errors. A hard process
        # termination between two replacements is detectable through the binding
        # hash but is not an atomic two-file transaction; native .blend is separate.
        backups = {member: member.with_name(member.name + ".rollback") for member in (path, sidecar)}
        if any(backup.exists() for backup in backups.values()):
            raise ValueError("prior rollback files exist; inspect them before exporting")
        existed = {member: member.exists() for member in backups}
        for member, backup in backups.items():
            if existed[member]:
                shutil.copyfile(member, backup)
        try:
            os.replace(temporary, path)
            os.replace(sidecar_temp, sidecar)
        except OSError:
            for member, backup in backups.items():
                if existed[member]:
                    os.replace(backup, member)
                elif member.exists():
                    member.unlink()
            raise
        else:
            for backup in backups.values():
                if backup.exists():
                    backup.unlink()
    finally:
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        for obj in selection:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = active
        for obj, (hide, hide_select) in hidden.items():
            obj.hide_set(hide)
            obj.hide_select = hide_select
        for temporary_path in (temporary, sidecar_temp):
            if temporary_path.exists():
                temporary_path.unlink()
    return sidecar


def target_items(_self, _context):
    return [(obj["alvenx_id"], f'{obj.name} · {obj["alvenx_id"]}', "Observed surface region")
            for obj in managed_meshes()]


class AX_OT_import(bpy.types.Operator, ImportHelper):
    bl_idname = "alvenx.import_surface"
    bl_label = "Import surface bundle"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, _context):
        try:
            import_surface(self.filepath)
        except (ValueError, OSError, KeyError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class AX_OT_reassign(bpy.types.Operator):
    bl_idname = "alvenx.reassign_faces"
    bl_label = "Assign selected faces"
    bl_description = "Transfer a proper subset of selected source faces to the destination; preserve observed appearance"
    bl_options = {"REGISTER", "UNDO"}
    target_id: StringProperty()

    def execute(self, context):
        try:
            ids = reassign_selected(context, self.target_id or context.scene.ax_target)
        except (ValueError, KeyError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Assigned {len(ids)} observed faces")
        return {"FINISHED"}


class AX_OT_transform(bpy.types.Operator):
    bl_idname = "alvenx.rigid_edit"
    bl_label = "Apply metre / degree edit"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if context.mode != "OBJECT" or context.active_object is None:
            self.report({"ERROR"}, "Select an observed region in Object Mode")
            return {"CANCELLED"}
        try:
            obj = context.active_object
            rigid_edit(obj, context.scene.ax_translation, context.scene.ax_rotation,
                       geometry_center(obj))
        except (ValueError, KeyError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class AX_OT_export(bpy.types.Operator, ExportHelper):
    bl_idname = "alvenx.export_package"
    bl_label = "Export GLB + identity record"
    filename_ext = ".glb"
    filter_glob: StringProperty(default="*.glb", options={"HIDDEN"})

    def execute(self, _context):
        try:
            export_package(self.filepath)
        except (ValueError, OSError, KeyError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class AX_PT_review(bpy.types.Panel):
    bl_label = "AlvenX · SpatialSceneLab"
    bl_idname = "AX_PT_review"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AlvenX"

    def draw(self, context):
        layout = self.layout
        layout.operator("alvenx.import_surface", icon="IMPORT")
        if not managed_meshes():
            layout.label(text="Import a prepared observed-surface bundle")
            return
        obj = context.edit_object if context.mode == "EDIT_MESH" else context.active_object
        if obj and "alvenx_id" in obj:
            layout.label(text=obj["alvenx_id"], icon="MESH_DATA")
            layout.label(text=f"{len(obj.data.polygons):,} observed faces")
        layout.label(text="Environment also contains unassigned faces")
        box = layout.box()
        box.label(text="Review selected faces")
        box.prop(context.scene, "ax_target", text="Destination")
        box.operator("alvenx.reassign_faces")
        box = layout.box()
        box.label(text="Rigid edit · world Z-up")
        box.prop(context.scene, "ax_translation", text="Metres")
        box.prop(context.scene, "ax_rotation", text="Z degrees")
        box.label(text="Pivot: current geometry bounds centre")
        box.operator("alvenx.rigid_edit")
        row = layout.row(align=True)
        row.operator("ed.undo", text="Undo")
        row.operator("ed.redo", text="Redo")
        layout.operator("wm.save_as_mainfile", text="Save editable .blend", icon="FILE_TICK")
        layout.operator("alvenx.export_package", icon="EXPORT")


CLASSES = (AX_OT_import, AX_OT_reassign, AX_OT_transform, AX_OT_export, AX_PT_review)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ax_target = EnumProperty(name="Destination", items=target_items)
    bpy.types.Scene.ax_translation = FloatVectorProperty(name="Translation", size=3, subtype="TRANSLATION", default=(.2, 0, 0))
    bpy.types.Scene.ax_rotation = FloatProperty(name="World Z rotation", subtype="NONE", default=15)


def unregister():
    for key in ("ax_target", "ax_translation", "ax_rotation"):
        delattr(bpy.types.Scene, key)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


def main():
    register()
    if "--" not in sys.argv:
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    output = Path(args.output).resolve()
    if output.exists() or output.suffix.lower() != ".blend":
        raise ValueError("output must be a new .blend file")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_surface(args.scene)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    print("ALVENX_EDIT_PREPARED " + json.dumps({"faces": len(source_inventory()), "output": str(output)}))


if __name__ == "__main__":
    main()
