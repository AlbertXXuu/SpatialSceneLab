"""Build and reopen a seven-part observation-constrained stool in Blender. MIT.

blender --background --factory-startup --python-exit-code 1 --python blender_stool.py --
    --fit fitted-directory --output native-directory --mode build
Run the same command with --mode reopen in a separate Blender process.
Optional --observations FILE verifies that explicit file against the recorded
input hash. Paths embedded in fit.json are provenance, never automatic reads.
Output must be new or empty, separate from inputs, and outside frozen directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry_compare import compare_triangle_geometry

ROOT_NAME = 'Fitted Stool (structural completion)'
DISPLAY = {'seat': 'Seat', 'leg_1': 'Leg 1', 'leg_2': 'Leg 2', 'leg_3': 'Leg 3',
           'brace_1_2': 'Brace 1-2', 'brace_2_3': 'Brace 2-3', 'brace_3_1': 'Brace 3-1'}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_json(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def input_bundle(directory):
    fit_path = directory / 'fit.json'
    fit = json.loads(fit_path.read_text(encoding='utf-8'))
    require(fit.get('schema') == 'alvenx.spatial.observation-constrained-stool-fit.v1', 'Unsupported fit schema')
    require(fit.get('units') == 'm' and fit.get('up_axis') == 'Y', 'Fit must use Y-up metres')
    require(fit.get('quality_gate', {}).get('passed') is True, 'Fit must pass its bounded prototype gate')
    parts = fit.get('editable_components', [])
    require(len(parts) == 7 and {p.get('name') for p in parts} == set(DISPLAY), 'Expected exactly seven named fitted components')
    require({p.get('part_id') for p in parts} == set(range(7)), 'Part identities must be unique 0..6')
    require(all(p.get('type') == 'capped_cylinder' and p.get('provenance_role') == 'fitted/structural' for p in parts), 'Unexpected component type or provenance')
    require(fit.get('contact_checks', {}).get('all_nine_intended_joints_in_solid_contact') is True, 'All nine intended joints must pass the fit contact checks')
    filenames = fit.get('outputs', {}).get('editable_parts', [])
    require(len(filenames) == 7 and len(set(filenames)) == 7, 'Expected seven unique PLY filenames')
    sources = {}
    for filename in filenames:
        relative = Path(filename)
        require(not relative.is_absolute() and len(relative.parts) == 1 and relative.suffix.lower() == '.ply', 'PLY filenames must be local basenames')
        require(relative.stem in DISPLAY, 'Unexpected PLY part filename')
        source = (directory / relative).resolve()
        require(source.parent == directory and source.is_file(), 'Part file must exist directly in the fit directory')
        sources[relative.stem] = source
    require(set(sources) == set(DISPLAY), 'Part filenames do not match component names')
    return fit, fit_path, sources


def observation_check(fit, explicit_path):
    expected = fit['input_files']['observations']['sha256']
    require(isinstance(expected, str) and len(expected) == 64, 'Invalid observation source hash')
    if explicit_path is None:
        return {'status': 'not_requested', 'expected_sha256': expected,
                'scope': 'No input metadata path was followed; use --observations to verify a raw observation file.'}
    actual = sha(explicit_path.resolve(strict=True))
    require(actual == expected, 'Explicit observations file does not match the fit input hash')
    return {'status': 'passed', 'expected_sha256': expected, 'actual_sha256': actual}


def check_output(fit_dir, output_dir, mode):
    require(output_dir != fit_dir and not output_dir.is_relative_to(fit_dir)
            and not fit_dir.is_relative_to(output_dir), 'Output must be separate from the fit input directory')
    require(not any('frozen' in p.lower() or p.lower() in {'input', 'inputs', 'fixtures'} for p in output_dir.parts), 'Output may not target a frozen or input directory')
    if mode == 'build':
        require(not output_dir.exists() or (output_dir.is_dir() and not any(output_dir.iterdir())), 'Build output must be new or empty; existing artifacts are never overwritten')
    else:
        require(output_dir.is_dir(), 'Reopen output directory does not exist')


def read_part(path, expected_part=None):
    lines = path.read_text(encoding='ascii').splitlines()
    end = lines.index('end_header')
    require(lines[:2] == ['ply', 'format ascii 1.0'], 'Expected fitted ASCII PLY')
    nv = int(next(line for line in lines[:end] if line.startswith('element vertex')).split()[2])
    nf = int(next(line for line in lines[:end] if line.startswith('element face')).split()[2])
    vertices = np.array([[float(v) for v in line.split()[:3]] for line in lines[end + 1:end + 1 + nv]])
    rows = [[int(v) for v in line.split()] for line in lines[end + 1 + nv:]]
    require(nv > 0 and nf > 0 and len(vertices) == nv and len(rows) == nf and all(len(row) == 5 and row[0] == 3 for row in rows), 'Expected complete triangular fitted part PLY')
    faces = np.array([row[1:4] for row in rows], dtype=np.int32)
    require(np.isfinite(vertices).all() and (faces >= 0).all() and (faces < nv).all(), 'Invalid mesh coordinates or face indices')
    if expected_part is not None:
        require(nv == expected_part['vertex_count'] and nf == expected_part['face_count'], 'Mesh counts differ from fit metadata')
        vertex_ids = [int(line.split()[6]) for line in lines[end + 1:end + 1 + nv]]
        require(all(i == expected_part['part_id'] for i in vertex_ids) and all(row[4] == expected_part['part_id'] for row in rows), 'PLY part identities differ from fit metadata')
    # A proper rigid rotation, determinant +1: input Y-up -> Blender Z-up.
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] *= -1
    return vertices, faces


def local_arrays(obj):
    vertices = np.array([v.co[:] for v in obj.data.vertices], dtype='<f4')
    faces = np.array([p.vertices[:] for p in obj.data.polygons], dtype='<i4')
    require(faces.shape[1] == 3, 'Native part must remain triangular')
    return vertices, faces


def world_arrays(obj):
    vertices, faces = local_arrays(obj)
    matrix = np.array(obj.matrix_world, dtype=float)
    return vertices.astype(float) @ matrix[:3, :3].T + matrix[:3, 3], faces


def snapshot(obj):
    vertices, faces = local_arrays(obj)
    world, _ = world_arrays(obj)
    return {'vertices': len(vertices), 'faces': len(faces),
            'local_mesh_bytes_sha256': hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest(),
            'world_matrix': [list(row) for row in obj.matrix_world],
            'bounds_min_blender_m': world.min(axis=0).tolist(), 'bounds_max_blender_m': world.max(axis=0).tolist(),
            'parent': obj.parent.name if obj.parent else None,
            'provenance_role': obj.get('provenance_role'), 'source_part': obj.get('source_part'),
            'source_ply_sha256': obj.get('source_ply_sha256')}


def clear():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.users == 0 or collection.name == 'Collection':
            bpy.data.collections.remove(collection)


def srgb_linear(channel):
    value = channel / 255
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def build(fit_dir, output_dir, observations):
    fit, fit_path, sources = input_bundle(fit_dir)
    fit_hash = sha(fit_path)
    checked_observations = observation_check(fit, observations)
    source_hashes = {name: sha(path) for name, path in sources.items()}
    source_meshes = {part['name']: read_part(sources[part['name']], part) for part in fit['editable_components']}
    for vertices, faces in source_meshes.values():
        check_closed(vertices, faces)
    clear()
    scene = bpy.context.scene
    scene.name = 'Fitted stool | observation-constrained completion'
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.0
    scene.unit_settings.length_unit = 'METERS'
    collection = bpy.data.collections.new('Fitted stool - 7 editable parts')
    scene.collection.children.link(collection)
    root = bpy.data.objects.new(ROOT_NAME, None)
    collection.objects.link(root)
    seat = next(p for p in fit['editable_components'] if p['name'] == 'seat')
    origin = np.array([seat['center_m'][0], -seat['center_m'][2], fit['contact_checks']['floor_y_m']])
    root.location = origin
    root.empty_display_type = 'PLAIN_AXES'
    root.empty_display_size = .06
    root['provenance_role'] = 'fitted/structural'
    root['description'] = 'Observation-constrained seven-cylinder model with structural continuation in unseen regions.'
    root['source_fit_json_sha256'] = fit_hash
    root['source_fit_json'] = 'fit.json'
    root['assumptions_json'] = json.dumps(fit['assumptions'], ensure_ascii=False)
    root['source_observations_sha256'] = fit['input_files']['observations']['sha256']
    root['source_scene_sha256'] = fit['input_files']['scene']['sha256']
    root['input_axis_units'] = 'Y-up metres'
    root['native_axis_units'] = 'Z-up metres'
    root['coordinate_conversion'] = '(x, y, z) -> (x, -z, y)'
    root['component_count'] = 7
    root['contact_model'] = 'Nine intended solid-overlap joints; seven individually closed editable shells, no destructive welding.'
    root['quality_scope'] = fit['quality_gate']['interpretation']
    for part in fit['editable_components']:
        source = sources[part['name']]
        vertices, faces = source_meshes[part['name']]
        part_origin = vertices.mean(axis=0)
        mesh = bpy.data.meshes.new(DISPLAY[part['name']] + ' - fitted mesh')
        mesh.from_pydata((vertices - part_origin).tolist(), [], faces.tolist())
        mesh.update()
        obj = bpy.data.objects.new(DISPLAY[part['name']], mesh)
        collection.objects.link(obj)
        obj.parent = root
        obj.location = part_origin - origin
        obj['provenance_role'] = 'fitted/structural'
        obj['source_part'] = part['name']
        obj['part_id'] = part['part_id']
        obj['source_ply_sha256'] = sha(source)
        obj['source_fit_json_sha256'] = fit_hash
        obj['component_parameters_json'] = json.dumps(part)
        mat = bpy.data.materials.new(DISPLAY[part['name']] + ' - measured training color')
        color = tuple(srgb_linear(c) for c in part['color_rgb']) + (1,)
        mat.diffuse_color = color
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        bsdf.inputs['Base Color'].default_value = color
        bsdf.inputs['Roughness'].default_value = .8
        obj.data.materials.append(mat)
        for poly in mesh.polygons:
            poly.use_smooth = False
    scene['asset_provenance'] = 'Fitted/structural completion, separate from observed source points.'
    scene['fit_json_sha256'] = fit_hash
    scene['source_observations_sha256'] = fit['input_files']['observations']['sha256']
    scene.frame_set(1)
    scene.frame_start = scene.frame_end = 1
    bpy.context.view_layer.update()
    all_vertices = np.concatenate([row[0] for row in source_meshes.values()])
    target = Vector((all_vertices.min(axis=0) + all_vertices.max(axis=0)) / 2)
    span = float(np.max(np.ptp(all_vertices, axis=0)))
    camera = target + Vector((2.7, -1.04, 1.25)) * span
    view_rotation = (target - camera).to_track_quat('-Z', 'Y')
    view_count = 0
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                space.shading.type = 'SOLID'
                space.shading.light = 'STUDIO'
                space.shading.color_type = 'MATERIAL'
                space.shading.show_shadows = True
                space.shading.show_cavity = False
                space.overlay.show_overlays = False
                space.show_gizmo = False
                space.region_3d.view_location = target
                space.region_3d.view_rotation = view_rotation
                space.region_3d.view_distance = span * 2.4
                space.region_3d.view_perspective = 'ORTHO'
                space.clip_start = .001
                space.clip_end = 100
                view_count += 1
    for obj in scene.objects:
        obj.select_set(False)
    root.select_set(True)
    bpy.context.view_layer.objects.active = root
    parts = sorted((o for o in scene.objects if o.type == 'MESH'), key=lambda o: o.name)
    record = {'schema': 'alvenx.spatial.editable-stool-native-verification.v1', 'status': 'pending_fresh_process_verification',
              'fit_json_sha256': fit_hash, 'fit_quality_gate': fit['quality_gate'],
              'native_units': 'metres', 'native_up_axis': 'Z', 'glb_units': 'metres', 'glb_up_axis': 'Y',
              'frame': 1, 'root_name': ROOT_NAME, 'saved_view_count': view_count,
              'observations_validation': checked_observations, 'build_process_id': os.getpid(),
              'reference_geometry_read': False,
              'part_source_files': {name: {'filename': path.name, 'sha256': source_hashes[name]} for name, path in sources.items()},
              'build_parts': {o.name: snapshot(o) for o in parts}}
    scene['build_snapshot_json'] = json.dumps(record['build_parts'])
    require(sha(fit_path) == fit_hash and all(sha(sources[name]) == value for name, value in source_hashes.items()), 'Fit inputs changed during build')
    check_output(fit_dir, output_dir, 'build')
    output_dir.mkdir(parents=True, exist_ok=True)
    blend, glb = output_dir / 'editable-stool.blend', output_dir / 'editable-stool.glb'
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), check_existing=False)
    for obj in scene.objects:
        obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(glb), export_format='GLB', use_selection=True,
                              export_yup=True, export_extras=True, export_animations=False,
                              export_cameras=False, export_lights=False)
    record['blend_sha256'] = sha(blend)
    record['glb_sha256'] = sha(glb)
    save_json(output_dir / 'native-verification.json', record)
    print('BUILT', str(blend), str(glb))


def check_closed(vertices, faces):
    edges = {}
    for face in faces:
        for a, b in zip(face, np.roll(face, -1)):
            key = tuple(sorted((int(a), int(b))))
            edges[key] = edges.get(key, 0) + 1
    require(all(count == 2 for count in edges.values()), 'Every fitted component must be closed')
    volume = float(np.einsum('ij,ij->i', vertices[faces[:, 0]], np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]])).sum() / 6)
    require(volume > 0, 'Fitted component must have outward winding and positive volume')
    return volume


def verify(fit_dir, output_dir, observations):
    fit, fit_path, sources = input_bundle(fit_dir)
    blend, glb = output_dir / 'editable-stool.blend', output_dir / 'editable-stool.glb'
    report = output_dir / 'native-verification.json'
    record = json.loads(report.read_text(encoding='utf-8'))
    require(os.getpid() != record['build_process_id'], 'Reopen verification must run in a separate Blender process')
    require(sha(fit_path) == record['fit_json_sha256'], 'Fitting source changed after asset creation')
    require(sha(blend) == record['blend_sha256'] and sha(glb) == record['glb_sha256'], 'Asset files changed after build')
    for name, path in sources.items():
        require(path.name == record['part_source_files'][name]['filename'] and sha(path) == record['part_source_files'][name]['sha256'], 'PLY source changed after asset creation')
    checked_observations = observation_check(fit, observations)
    bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=True)
    scene = bpy.context.scene
    mesh_objects = {o.name: o for o in scene.objects if o.type == 'MESH'}
    require(set(mesh_objects) == set(DISPLAY.values()) and len(mesh_objects) == 7, 'Expected seven named native meshes')
    root = scene.objects[ROOT_NAME]
    require(root.type == 'EMPTY' and root['source_fit_json_sha256'] == record['fit_json_sha256'], 'Root identity or fit provenance changed')
    root_metadata = {'provenance_role': 'fitted/structural', 'component_count': 7,
                     'assumptions_json': json.dumps(fit['assumptions'], ensure_ascii=False),
                     'source_observations_sha256': fit['input_files']['observations']['sha256'],
                     'source_scene_sha256': fit['input_files']['scene']['sha256']}
    require(all(root.get(key) == value for key, value in root_metadata.items()), 'Native root provenance or assumptions changed')
    require(scene.unit_settings.scale_length == 1 and scene.frame_current == 1, 'Units or saved frame changed')
    require(len(scene.objects) == 8, 'Only fitted root and seven editable meshes belong in saved native scene')
    native_checks, triangles = {}, {}
    for name, obj in mesh_objects.items():
        actual = snapshot(obj)
        expected = record['build_parts'][name]
        require(actual == expected, f'Native mesh bytes or metadata changed: {name}')
        require(obj.parent == root, f'Parent changed: {name}')
        vertices, faces = world_arrays(obj)
        source_vertices, source_faces = read_part(sources[obj['source_part']])
        require(np.array_equal(faces, source_faces), f'Source face identities changed: {name}')
        source_error = float(np.max(np.abs(vertices - source_vertices)))
        require(source_error <= 3e-7, f'Source vertex coordinates changed: {name}')
        volume = check_closed(vertices, faces)
        triangles[name] = (vertices, faces)
        native_checks[name] = {'exact_local_mesh_bytes_match': True, 'exact_source_face_indices_match': True,
                               'max_source_vertex_error_m': source_error, 'closed_two_faces_per_edge': True,
                               'positive_signed_volume_m3': volume, 'parent_root_and_metadata_match': True}
    viewport_checks = []
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                require(not space.overlay.show_overlays, 'Saved view overlays must be hidden')
                require(space.shading.type == 'SOLID' and space.shading.color_type == 'MATERIAL', 'Saved view must use solid material display')
                viewport_checks.append({'screen': screen.name, 'overlays': False, 'mode': 'SOLID',
                                        'view_distance_m': space.region_3d.view_distance,
                                        'view_location': list(space.region_3d.view_location)})
    require(viewport_checks, 'Saved native view is missing')
    clear()
    bpy.ops.import_scene.gltf(filepath=str(glb))
    imported = {o.name: o for o in bpy.context.scene.objects if o.type == 'MESH'}
    require(set(imported) == set(triangles), 'GLB component names differ')
    imported_root = bpy.context.scene.objects.get(ROOT_NAME)
    require(imported_root is not None and all(imported_root.get(key) == value for key, value in root_metadata.items()), 'GLB root provenance or assumptions changed')
    roundtrip = {}
    for name, obj in imported.items():
        vertices, faces = world_arrays(obj)
        expected_vertices, expected_faces = triangles[name]
        comparison = compare_triangle_geometry(expected_vertices, expected_faces, vertices, faces, tolerance=5e-7)
        require(comparison['status'] == 'pass' and comparison['winding_ambiguous_matched_faces'] == 0, f'GLB oriented shape/units changed: {name}')
        error = comparison['maximum_matched_corner_distance_m']
        require(obj['source_part'] == record['build_parts'][name]['source_part'] and obj['source_fit_json_sha256'] == record['fit_json_sha256'], 'GLB provenance changed')
        require(obj.get('provenance_role') == 'fitted/structural' and obj.get('source_ply_sha256') == record['build_parts'][name]['source_ply_sha256'], 'GLB part role or source hash changed')
        require(obj.parent and obj.parent.name == ROOT_NAME, 'GLB root relationship changed')
        roundtrip[name] = {'triangles': len(faces), 'max_triangle_coordinate_error_m': error,
                           'name_parent_provenance_and_fit_hash_preserved': True, 'oriented_geometry': comparison}
    # Verify actual binary glTF metadata states version 2.0; metres and Y-up are glTF standard.
    with glb.open('rb') as stream:
        header = np.frombuffer(stream.read(12), dtype='<u4')
    require(header[0] == 0x46546C67 and header[1] == 2 and header[2] == glb.stat().st_size, 'Invalid binary glTF header')
    record.update({'status': 'passed', 'fresh_background_native_reopen': True,
                   'native_mesh_count': 7, 'native_scene_object_count': 8,
                   'native_checks': native_checks, 'saved_view_checks': viewport_checks,
                   'glb_roundtrip': roundtrip, 'glb_header_version': 2,
                   'observations_validation': checked_observations,
                   'asset_files_unchanged_by_verification': sha(blend) == record['blend_sha256'] and sha(glb) == record['glb_sha256']})
    require(record['asset_files_unchanged_by_verification'], 'Asset changed during verification')
    save_json(report, record)
    print('VERIFIED', json.dumps({'status': record['status'], 'parts': len(native_checks),
                                'max_glb_error_m': max(row['max_triangle_coordinate_error_m'] for row in roundtrip.values())}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True, help='Directory containing fit.json and its seven PLY parts')
    parser.add_argument('--output', type=Path, required=True, help='Separate directory for native artifacts')
    parser.add_argument('--mode', choices=('build', 'reopen'), required=True)
    parser.add_argument('--observations', type=Path, help='Explicit raw observations NPZ to check against the recorded input hash')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
    try:
        fit_dir, output_dir = args.fit.resolve(strict=True), args.output.resolve()
        check_output(fit_dir, output_dir, args.mode)
        (build if args.mode == 'build' else verify)(fit_dir, output_dir, args.observations)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
