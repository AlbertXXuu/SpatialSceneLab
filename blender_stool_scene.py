"""Insert the accepted fitted stool into the public B1 dining scene. MIT.

Run with Blender --background --factory-startup --python-exit-code 1 --python
blender_stool_scene.py -- --archive dining-b1-author-task.zip --fitted
editable-stool.blend --output NEW_DIRECTORY --scratch TEMP_DIRECTORY --mode build
Then run --mode reopen in a separate process with the same explicit paths.
The original observed stool remains hidden in a named native comparison layer.
All other original objects are retained; the visible GLB excludes that layer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import zipfile

try:
    import bpy
    import numpy as np
    from mathutils import Vector
except ImportError:
    bpy = None

ROOT_NAME = 'Fitted Stool (structural completion)'
COMPARISON = 'Original stool - observed comparison (hidden)'
FITTED = 'Repaired stool - 7 editable structural parts'
TARGET_ID = 'same-stool'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def validate_members(archive):
    """Check every entry, although only before.blend is ever extracted."""
    names, total = set(), 0
    require(len(archive.infolist()) <= 512, 'Unexpectedly many archive members')
    for item in archive.infolist():
        path = PurePosixPath(item.filename)
        require(item.orig_filename == item.filename and not path.is_absolute() and item.filename and '\\' not in item.filename
                and ':' not in item.filename and '..' not in path.parts
                and all(part.rstrip(' .') == part for part in path.parts), 'Unsafe archive path')
        require(item.filename.casefold() not in names, 'Duplicate archive path')
        names.add(item.filename.casefold())
        require(not stat.S_ISLNK(item.external_attr >> 16), 'Archive links are forbidden')
        require(not item.flag_bits & 1, 'Encrypted archives are unsupported')
        total += item.file_size
        require(item.file_size <= 64 * 1024 * 1024 and total <= 128 * 1024 * 1024,
                'Archive exceeds the bounded public task size')
    require('before.blend' in archive.namelist(), 'Archive must contain before.blend at its root')
    require(not archive.getinfo('before.blend').is_dir(), 'before.blend must be a file')


def check_paths(archive, fitted, output, scratch, mode):
    require(archive.is_file() and fitted.is_file(), 'Both explicit input files must exist')
    require(archive.suffix.lower() == '.zip' and fitted.suffix.lower() == '.blend', 'Expected ZIP and native Blender inputs')
    require(output not in (archive.parent, fitted.parent), 'Output must be separate from input directories')
    require(not archive.is_relative_to(output) and not fitted.is_relative_to(output), 'Output must not contain input files')
    require(not output.is_relative_to(scratch) and not scratch.is_relative_to(output), 'Scratch and output directories must be separate')
    require(not any('frozen' in p.lower() or p.lower() in {'input', 'inputs', 'fixtures'} for p in output.parts), 'Output may not target frozen/input directories')
    if mode == 'build':
        require(not output.exists() or (output.is_dir() and not any(output.iterdir())), 'Build output must be new or empty')
    else:
        require(output.is_dir(), 'Reopen output does not exist')


def plain(value):
    if hasattr(value, 'to_dict'):
        return {k: plain(v) for k, v in value.to_dict().items()}
    if hasattr(value, 'to_list'):
        return value.to_list()
    return value


def properties(block):
    return {key: plain(block[key]) for key in sorted(block.keys())}


def mesh_arrays(obj):
    mesh = obj.data
    require(all(len(p.vertices) == 3 for p in mesh.polygons), 'Expected triangular mesh')
    xyz = np.empty((len(mesh.vertices), 3), dtype='<f4')
    mesh.vertices.foreach_get('co', xyz.ravel())
    faces = np.empty(len(mesh.loops), dtype='<i4')
    mesh.loops.foreach_get('vertex_index', faces)
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    return xyz.astype(float) @ matrix[:3, :3].T + matrix[:3, 3], faces.reshape(-1, 3)


def object_snapshot(obj):
    result = {'type': obj.type, 'matrix_world': [list(row) for row in obj.matrix_world],
              'parent': obj.parent.name if obj.parent else None, 'properties': properties(obj),
              'hide_render': obj.hide_render, 'hide_viewport': obj.hide_viewport,
              'hidden_in_view_layer': obj.hide_get(), 'collections': sorted(c.name for c in obj.users_collection)}
    if obj.type == 'MESH':
        mesh = obj.data
        xyz = np.array([v.co[:] for v in mesh.vertices], dtype='<f4')
        faces = np.array([p.vertices[:] for p in mesh.polygons], dtype='<i4')
        result['mesh_sha256'] = hashlib.sha256(xyz.tobytes() + faces.tobytes()).hexdigest()
        result['vertices'], result['triangles'] = len(xyz), len(faces)
        result['attributes'] = {}
        for attr in mesh.attributes:
            if attr.data_type in {'INT', 'BOOLEAN', 'FLOAT'}:
                dtype = '<f4' if attr.data_type == 'FLOAT' else '<i4'
                data = np.asarray([x.value for x in attr.data], dtype=dtype)
            elif attr.data_type in {'FLOAT_COLOR', 'BYTE_COLOR'}:
                data = np.asarray([x.color[:] for x in attr.data], dtype='<f4')
            elif attr.data_type == 'FLOAT_VECTOR':
                data = np.asarray([x.vector[:] for x in attr.data], dtype='<f4')
            else:
                continue
            result['attributes'][attr.name] = {'domain': attr.domain, 'type': attr.data_type,
                                               'sha256': hashlib.sha256(data.tobytes()).hexdigest()}
        result['materials'] = []
        for mat in mesh.materials:
            nodes = []
            if mat and mat.use_nodes:
                for node in mat.node_tree.nodes:
                    values = {}
                    for socket in node.inputs:
                        if hasattr(socket, 'default_value'):
                            v = socket.default_value
                            values[socket.name] = list(v) if hasattr(v, '__len__') and not isinstance(v, str) else v
                    nodes.append({'name': node.name, 'type': node.type, 'inputs': values,
                                  'color_layer': getattr(node, 'layer_name', None)})
            result['materials'].append({'name': mat.name, 'diffuse_color': list(mat.diffuse_color),
                                        'nodes': nodes, 'links': sorted((link.from_node.name, link.from_socket.name,
                                          link.to_node.name, link.to_socket.name) for link in mat.node_tree.links) if mat.use_nodes else []})
    return result


def unchanged_source(expected, target_name):
    checks = {}
    for name, before in expected.items():
        obj = bpy.context.scene.objects.get(name)
        require(obj is not None, f'Source object missing: {name}')
        actual = object_snapshot(obj)
        if name == target_name:
            for key in ('hide_render', 'hidden_in_view_layer', 'collections'):
                actual[key] = before[key]
        # JSON normalization turns RNA tuples into serializable lists consistently.
        require(json.dumps(actual, sort_keys=True) == json.dumps(before, sort_keys=True), f'Source object changed: {name}')
        checks[name] = {'exact_mesh_transform_color_face_properties_preserved': True,
                        'target_hidden_comparison': name == target_name}
    return checks


def unit_state():
    unit = bpy.context.scene.unit_settings
    return {'system': unit.system, 'scale_length': unit.scale_length, 'length_unit': unit.length_unit}


def source_state():
    import verify_blender_edit as verification
    value = verification.snapshot()
    return {key: hashlib.sha256(array.tobytes()).hexdigest() for key, array in value.items()}


def viewport():
    target = Vector((0, 1.85, .52))
    eye = Vector((2.45, -2.8, 2.2))
    rotation = (target - eye).to_track_quat('-Z', 'Y')
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                space.overlay.show_overlays = False
                space.shading.type = 'MATERIAL'
                space.region_3d.view_location = target
                space.region_3d.view_rotation = rotation
                space.region_3d.view_distance = 4.4
                space.region_3d.view_perspective = 'PERSP'
    return {'eye_blender_m': list(eye), 'target_blender_m': list(target), 'lens_mm': 50,
            'resolution': [1400, 1100]}


def render_pair(output, target_obj, fit_objects, camera_record):
    scene = bpy.context.scene
    original_camera = scene.camera
    data = bpy.data.cameras.new('Temporary comparison camera')
    camera = bpy.data.objects.new('Temporary comparison camera', data)
    scene.collection.objects.link(camera)
    camera.location = camera_record['eye_blender_m']
    camera.rotation_euler = (Vector(camera_record['target_blender_m']) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    data.lens = camera_record['lens_mm']
    scene.camera = camera
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x, scene.render.resolution_y = camera_record['resolution']
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    comparison = bpy.data.collections[COMPARISON]
    for before in (True, False):
        comparison.hide_render = not before
        target_obj.hide_render = not before
        for obj in fit_objects:
            obj.hide_render = before
        scene.render.filepath = str(output / ('dining-before.png' if before else 'dining-repaired.png'))
        bpy.ops.render.render(write_still=True)
    for obj in fit_objects:
        obj.hide_render = False
    scene.camera = original_camera
    bpy.data.objects.remove(camera, do_unlink=True)
    bpy.data.cameras.remove(data)


def build(archive, fitted, output, scratch):
    input_hashes = {'archive': sha(archive), 'fitted': sha(fitted)}
    scratch.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        validate_members(zipped)
        with tempfile.TemporaryDirectory(prefix='scene-', dir=scratch) as temporary:
            source = Path(temporary) / 'before.blend'
            with source.open('xb') as stream:
                stream.write(zipped.read('before.blend'))
            before_sha = sha(source)
            bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=True, use_scripts=False)
    scene = bpy.context.scene
    require(unit_state()['system'] == 'METRIC' and unit_state()['scale_length'] == 1, 'Source scene must be metric')
    original = {obj.name: object_snapshot(obj) for obj in scene.objects}
    original_units, original_faces = unit_state(), source_state()
    targets = [obj for obj in scene.objects if obj.get('alvenx_id') == TARGET_ID]
    require(len(targets) == 1, 'Requires one source stool with the expected identity')
    target = targets[0]
    with bpy.data.libraries.load(str(fitted), link=False) as (source, destination):
        require(ROOT_NAME in source.objects and len(source.objects) == 8, 'Expected only the accepted root and seven fitted meshes')
        destination.objects = source.objects
    added = list(destination.objects)
    require(all(obj is not None for obj in added), 'Incomplete fitted asset')
    fit_collection = bpy.data.collections.new(FITTED)
    scene.collection.children.link(fit_collection)
    for obj in added:
        fit_collection.objects.link(obj)
    root = next(obj for obj in added if obj.name == ROOT_NAME)
    parts = [obj for obj in added if obj.type == 'MESH']
    require(len(parts) == 7 and all(obj.parent == root for obj in parts), 'Expected seven children of fitted root')
    require(root.get('provenance_role') == 'fitted/structural' and root.get('native_axis_units') == 'Z-up metres', 'Unexpected native fit provenance/units')
    require(root.get('component_count') == 7 and all(obj.get('source_fit_json_sha256') == root.get('source_fit_json_sha256') for obj in parts), 'Fit provenance differs across components')
    bpy.context.view_layer.update()
    observed_xyz, _ = mesh_arrays(target)
    fitted_xyz = np.concatenate([mesh_arrays(obj)[0] for obj in parts])
    bounds_distance = float(np.max(np.abs((observed_xyz.min(0) + observed_xyz.max(0)) / 2 - (fitted_xyz.min(0) + fitted_xyz.max(0)) / 2)))
    require(bounds_distance < .08, 'Fitted stool is not in the source stool world frame')
    comparison = bpy.data.collections.new(COMPARISON)
    scene.collection.children.link(comparison)
    for collection in list(target.users_collection):
        collection.objects.unlink(target)
    comparison.objects.link(target)
    target.hide_render = True
    target.hide_set(True)
    comparison.hide_render = True
    # Keep the collection available in the Outliner, while its mesh is hidden.
    for obj in scene.objects:
        obj.select_set(False)
    root.select_set(True)
    bpy.context.view_layer.objects.active = root
    camera_record = viewport()
    source_checks = unchanged_source(original, target.name)
    require(source_state() == original_faces and unit_state() == original_units, 'Source identities or units changed')
    record = {'schema': 'alvenx.spatial.stool-scene-integration.v1', 'status': 'pending_fresh_reopen',
              'build_process_id': os.getpid(), 'input_sha256': input_hashes, 'before_blend_sha256': before_sha,
              'source_objects': original, 'source_face_snapshot_sha256': original_faces, 'source_units': original_units,
              'original_stool_name': target.name, 'original_stool_id': TARGET_ID, 'comparison_collection': COMPARISON,
              'fit_root': ROOT_NAME, 'fitted_parts': {obj.name: object_snapshot(obj) for obj in parts},
              'root_snapshot': object_snapshot(root), 'source_preservation': source_checks,
              'placement': {'operation': 'append existing world-space asset; no additional registration',
                            'maximum_observed_vs_fitted_bounds_center_difference_m': bounds_distance},
              'fixed_camera': camera_record, 'scope': 'Only the accepted original stool is replaced in the visible scene; table, chairs and environment retain their observed incomplete geometry.'}
    check_paths(archive, fitted, output, scratch, 'build')
    output.mkdir(parents=True, exist_ok=True)
    render_pair(output, target, added, camera_record)
    require(source_state() == original_faces, 'Rendering changed source identities')
    unchanged_source(original, target.name)
    for obj in scene.objects:
        obj.select_set(False)
    root.select_set(True)
    bpy.context.view_layer.objects.active = root
    bpy.ops.wm.save_as_mainfile(filepath=str(output / 'repaired-dining.blend'), check_existing=False, compress=True)
    for obj in scene.objects:
        obj.select_set(obj == root or (obj.type == 'MESH' and obj != target and not obj.hide_render))
    bpy.ops.export_scene.gltf(filepath=str(output / 'repaired-dining.glb'), export_format='GLB', use_selection=True,
                              export_yup=True, export_extras=True, export_animations=False, export_cameras=False,
                              export_lights=False)
    require(input_hashes == {'archive': sha(archive), 'fitted': sha(fitted)}, 'Input files changed')
    record['artifacts'] = {name: {'sha256': sha(output / name), 'bytes': (output / name).stat().st_size}
                           for name in ('repaired-dining.blend', 'repaired-dining.glb', 'dining-before.png', 'dining-repaired.png')}
    save_json(output / 'scene-integration.json', record)
    print('BUILT', json.dumps(record['artifacts']))


def move_undo_redo(record):
    root = bpy.context.scene.objects[ROOT_NAME]
    initial = np.array(root.matrix_world)
    original_world = {o.name: mesh_arrays(o)[0] for o in root.children}
    bpy.context.preferences.edit.use_global_undo = True
    bpy.ops.ed.undo_push(message='Integrated dining scene before stool edit')
    root.location.x += .15
    root.location.y += .10
    bpy.context.view_layer.update()
    delta = np.array((.15, .10, 0))
    for name, xyz in original_world.items():
        require(np.max(np.abs(mesh_arrays(bpy.context.scene.objects[name])[0] - xyz - delta)) < 3e-7, 'Moving root did not move every part rigidly')
    require(source_state() == record['source_face_snapshot_sha256'], 'Moving fitted root changed source geometry')
    moved = np.array(root.matrix_world)
    bpy.ops.ed.undo_push(message='Integrated dining scene after stool edit')
    undo = bpy.ops.ed.undo()
    require(np.array_equal(np.array(bpy.context.scene.objects[ROOT_NAME].matrix_world), initial), 'Native undo did not restore root')
    redo = bpy.ops.ed.redo()
    require(np.array_equal(np.array(bpy.context.scene.objects[ROOT_NAME].matrix_world), moved), 'Native redo did not restore movement')
    bpy.ops.ed.undo()
    require(np.array_equal(np.array(bpy.context.scene.objects[ROOT_NAME].matrix_world), initial), 'Second undo failed')
    require(source_state() == record['source_face_snapshot_sha256'], 'Undo/redo changed source faces')
    unchanged_source(record['source_objects'], record['original_stool_name'])
    return {'status': 'passed', 'translation_m': list(delta), 'editable_parts_moved': 7,
            'native_undo': sorted(undo), 'native_redo': sorted(redo), 'saved_asset_unchanged': True,
            'scope': 'Background native Blender undo stack with explicit checkpoints; no claim of manual usability study.'}


def reopen(archive, fitted, output, scratch):
    from geometry_compare import compare_triangle_geometry
    import verify_blender_edit as verification
    record_path = output / 'scene-integration.json'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    require(os.getpid() != record['build_process_id'], 'Must reopen in a fresh Blender process')
    require(record['input_sha256'] == {'archive': sha(archive), 'fitted': sha(fitted)}, 'Inputs changed after integration')
    for name, value in record['artifacts'].items():
        require(sha(output / name) == value['sha256'], f'Artifact changed: {name}')
    bpy.ops.wm.open_mainfile(filepath=str(output / 'repaired-dining.blend'), load_ui=True, use_scripts=False)
    require(unit_state() == record['source_units'], 'Saved units differ')
    require(source_state() == record['source_face_snapshot_sha256'], 'Source face lineage, ownership, colors or coordinates differ')
    native_checks = unchanged_source(record['source_objects'], record['original_stool_name'])
    target = bpy.context.scene.objects[record['original_stool_name']]
    require(target.hide_get() and target.hide_render and COMPARISON in {c.name for c in target.users_collection}, 'Original stool must remain hidden in named comparison layer')
    for name, expected in record['fitted_parts'].items():
        require(json.dumps(object_snapshot(bpy.context.scene.objects[name]), sort_keys=True) == json.dumps(expected, sort_keys=True), f'Saved fitted component changed: {name}')
    require(object_snapshot(bpy.context.scene.objects[ROOT_NAME]) == record['root_snapshot'], 'Fitted root changed')
    edit_check = move_undo_redo(record)
    visible = {o.name: mesh_arrays(o) for o in bpy.context.scene.objects if o.type == 'MESH' and o.name != target.name}
    observed = verification.appearance_snapshot()
    observed.pop(TARGET_ID)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.import_scene.gltf(filepath=str(output / 'repaired-dining.glb'))
    returned = {o.name: o for o in bpy.context.scene.objects if o.type == 'MESH'}
    require(set(returned) == set(visible), 'GLB visible mesh inventory differs or contains hidden original stool')
    geometry_checks = {}
    for name, (xyz, faces) in visible.items():
        returned_xyz, returned_faces = mesh_arrays(returned[name])
        check = compare_triangle_geometry(xyz, faces, returned_xyz, returned_faces, tolerance=5e-6)
        require(check['status'] == 'pass', f'GLB world geometry changed: {name}')
        geometry_checks[name] = check
    appearance = verification.check_appearance(observed, verification.appearance_snapshot())
    require(appearance['status'] == 'pass', 'GLB observed colors/materials changed')
    for name in record['fitted_parts']:
        obj = returned[name]
        require(obj.parent and obj.parent.name == ROOT_NAME and obj.get('source_part') == record['fitted_parts'][name]['properties']['source_part'], 'GLB fitted hierarchy or identities changed')
    record.update({'status': 'passed', 'fresh_process_native_reopen': True,
                   'native_source_preservation': native_checks, 'native_edit_undo_redo': edit_check,
                   'glb_visible_mesh_count': len(returned), 'glb_hidden_original_excluded': True,
                   'glb_geometry': geometry_checks, 'glb_source_appearance': appearance,
                   'glb_scope': 'Visible oriented triangle geometry, colors and object metadata. Native custom per-face lineage remains in the .blend; glTF does not preserve arbitrary Blender mesh attributes.'})
    require(record['input_sha256'] == {'archive': sha(archive), 'fitted': sha(fitted)}, 'Inputs changed during verification')
    require(all(sha(output / name) == value['sha256'] for name, value in record['artifacts'].items()), 'Verification changed output artifacts')
    save_json(record_path, record)
    print('VERIFIED', json.dumps({'status': 'passed', 'source_objects': len(native_checks), 'visible_meshes': len(returned),
                                 'maximum_glb_corner_error_m': max(x['maximum_matched_corner_distance_m'] for x in geometry_checks.values())}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--fitted', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scratch', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('build', 'reopen'))
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None)
    require(bpy is not None, 'Run this entrypoint with Blender Python')
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    paths = [getattr(args, key).resolve() for key in ('archive', 'fitted', 'output', 'scratch')]
    check_paths(*paths, args.mode)
    (build if args.mode == 'build' else reopen)(*paths)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise
