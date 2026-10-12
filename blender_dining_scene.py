"""Complete the three dining targets while preserving the accepted source scene. MIT.

Blender --background --factory-startup --python-exit-code 1 --python
blender_dining_scene.py -- --source repaired-dining.blend --fit FIT_DIRECTORY
--evaluation evaluation.json --layout passed-layout.json --output NEW_DIRECTORY --mode build
Run --mode reopen with the same paths in a fresh Blender process.

PLY input is world Y-up metres. The existing stool import helper applies the
proper rotation (x,y,z)->(x,-z,y) exactly once. Source meshes and the accepted
stool retain their exact source geometry in the native file; the three observed targets are
available in a named comparison collection. Only visible geometry goes to GLB.
The explicit independently accepted layout translates new furniture roots only;
the frozen observation-frame fitting evidence remains unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys

try:
    import bpy
    import numpy as np
    from mathutils import Vector
except ImportError:
    bpy = None

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_stool_scene import require, sha, save_json

TARGETS = {'same-table': ('Table', 'Completed Table'),
           'same-chair-west': ('Chair', 'Completed Chair West'),
           'same-chair-east': ('Chair', 'Completed Chair East')}
COMPARISON = 'Original dining observations + ambiguous fragments (hidden)'
COMPLETED = 'Completed dining - individually editable components'
STOOL_ROOT = 'Fitted Stool (structural completion)'
COMPONENT_TYPES = {'box', 'capped_cylinder', 'domed_box', 'half_ellipsoid'}


def check_paths(source, fit_dir, evaluation, layout, output, mode):
    require(source.is_file() and source.suffix.lower() == '.blend', 'Source must be an existing native Blender file')
    require(fit_dir.is_dir() and (fit_dir / 'fit.json').is_file(), 'Fit directory must contain fit.json')
    require(evaluation.is_file() and evaluation.suffix.lower() == '.json', 'Explicit evaluation JSON must exist')
    require(layout.is_file() and layout.suffix.lower() == '.json', 'Explicit passed layout JSON must exist')
    require(output != source.parent and not source.is_relative_to(output)
            and not evaluation.is_relative_to(output) and not layout.is_relative_to(output),
            'Output must be separate from source, evaluation and layout inputs')
    require(output != fit_dir and not output.is_relative_to(fit_dir)
            and not fit_dir.is_relative_to(output), 'Output must be separate from the fit input directory')
    require(not any('frozen' in p.lower() or p.lower() in {'input', 'inputs', 'fixtures'} for p in output.parts),
            'Output may not target frozen/input directories')
    if mode == 'build':
        require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
                'Build output must be new or empty; existing artifacts are never overwritten')
    else:
        require(output.is_dir(), 'Reopen output directory does not exist')


def component_path(directory, filename):
    require(isinstance(filename, str) and filename and '\\' not in filename and ':' not in filename,
            'Component mesh must be a safe relative PLY path')
    relative = PurePosixPath(filename)
    require(not relative.is_absolute() and '..' not in relative.parts and relative.suffix.lower() == '.ply'
            and all(part.rstrip(' .') == part for part in relative.parts), 'Unsafe component PLY path')
    source = (directory / Path(relative)).resolve()
    require(source.is_relative_to(directory) and source.is_file(), 'Component PLY must exist within the explicit fit directory')
    return source


def input_bundle(directory, evaluation_path):
    fit_path = directory / 'fit.json'
    fit = json.loads(fit_path.read_text(encoding='utf-8'))
    require(fit.get('schema') == 'alvenx.spatial.dining-fit.v1', 'Unsupported dining fit schema')
    require(fit.get('units') == 'm' and fit.get('up_axis') == 'Y', 'Fit must use world Y-up metres')
    require(fit.get('quality_gate', {}).get('passed') is True, 'Dining fit must pass its declared gate')
    objects = fit.get('objects', [])
    require(isinstance(objects, list) and len(objects) == len(TARGETS)
            and {o.get('uuid') for o in objects} == set(TARGETS), 'Fit must contain exactly the three dining targets')
    sources = {}
    for obj in objects:
        require(obj.get('category') == TARGETS[obj['uuid']][0], 'Unexpected target category')
        parts = obj.get('components', [])
        require(isinstance(parts, list) and len(parts) >= 5, 'Dining target needs independently editable structural parts')
        names = set()
        for part in parts:
            name = part.get('name')
            require(isinstance(name, str) and name and '/' not in name and '\\' not in name and name not in names,
                    'Component names must be nonempty and unique within the target')
            names.add(name)
            require(part.get('type') in COMPONENT_TYPES and part.get('editable') is True
                    and part.get('provenance_role') == 'fitted/structural'
                    and isinstance(part.get('parameters'), dict), 'Unexpected component type, editability or provenance')
            sources[obj['uuid'] + '/' + name] = component_path(directory, part.get('mesh'))
    require(len(set(sources.values())) == len(sources), 'Components must use distinct PLY files')
    hashes = {key: sha(path) for key, path in sources.items()}
    evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
    require(evaluation.get('schema') == 'alvenx.spatial.dining-evaluation.v1'
            and evaluation.get('status') == 'passed' and evaluation.get('quality_gate', {}).get('passed') is True,
            'Independent actual-triangle evaluation must pass before native integration')
    require(evaluation.get('fit_json_sha256') == sha(fit_path)
            and evaluation.get('component_input_sha256') == hashes, 'Evaluation must match the exact fit and component bytes')
    checks = evaluation.get('objects', [])
    require(isinstance(checks, list) and len(checks) == len(TARGETS)
            and {obj.get('uuid') for obj in checks} == set(TARGETS)
            and all(obj.get('observation_gate', {}).get('passed') is True
                    and obj.get('structural_gate', {}).get('passed') is True for obj in checks),
            'Every target must pass observation and structural gates')
    return fit, fit_path, sources, hashes, evaluation


def layout_transform_sha256(objects):
    translations = {row['uuid']: [float(value) for value in row['translation_world_m']] for row in objects}
    return hashlib.sha256(json.dumps(translations, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def layout_bundle(path, source, fit_hash, component_hashes):
    layout = json.loads(path.read_text(encoding='utf-8'))
    gate = layout.get('quality_gate', {})
    require(layout.get('schema') == 'alvenx.spatial.dining-layout.v1' and layout.get('status') == 'passed'
            and layout.get('units') == 'm' and layout.get('up_axis') == 'Y'
            and gate.get('passed') is True and gate.get('all_actual_component_pairs_certified') is True
            and gate.get('frozen_geometry_unchanged') is True
            and gate.get('required_cross_furniture_clearance_m', 0) >= .005,
            'Independent actual-solid layout gate must pass at >=5mm separation')
    require(layout.get('fit_json_sha256') == fit_hash and layout.get('component_input_sha256') == component_hashes
            and layout.get('stool_source_blend_sha256') == sha(source),
            'Layout must match the exact frozen fit, component and accepted stool source bytes')
    objects = layout.get('objects', [])
    require(isinstance(objects, list) and len(objects) == 4
            and {row.get('uuid') for row in objects} == {*TARGETS, 'same-stool'},
            'Layout must declare exactly four furniture translations')
    translations = {}
    for row in objects:
        delta = row.get('translation_world_m')
        require(isinstance(delta, list) and len(delta) == 3
                and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in delta)
                and delta[1] == 0, 'Layout translations must be finite world-horizontal metre vectors')
        translations[row['uuid']] = [float(x) for x in delta]
    require(translations['same-stool'] == [0., 0., 0.], 'Accepted stool must remain unchanged in this dining completion')
    require(layout.get('layout_transform_sha256') == layout_transform_sha256(objects),
            'Layout transform hash does not match the declared authored translations')
    stool_hashes = layout.get('stool_component_input_sha256', {})
    require(set(stool_hashes) == {'seat', 'leg_1', 'leg_2', 'leg_3', 'brace_1_2', 'brace_2_3', 'brace_3_1'},
            'Layout must bind all seven accepted stool component inputs')
    stool_sources = {}
    for name, digest in stool_hashes.items():
        matches = [row for row in layout.get('input_files', []) if row.get('sha256') == digest]
        require(len(matches) == 1, 'Layout must name each exact accepted stool PLY input')
        ply = (path.parent / matches[0]['path']).resolve()
        require(ply.is_file() and ply.suffix.lower() == '.ply' and sha(ply) == digest,
                'Layout stool component bytes changed after independent evaluation')
        stool_sources[name] = ply
    authored = layout.get('authored_layout', {})
    require(authored.get('passed') is True and authored.get('failed_pair_count') == 0
            and isinstance(authored.get('pairs'), list) and authored['pairs'],
            'Layout must retain every actual-vertex separating-plane certificate')
    return layout, translations, stool_sources


def verify_separating_planes(layout, vertices):
    """Recheck all independent halfspace proofs against actual imported vertices."""
    import numpy as np
    keys = sorted(vertices)
    required_pairs = {frozenset((first, second)) for i, first in enumerate(keys) for second in keys[i + 1:]
                      if first.split('/')[0] != second.split('/')[0]}
    required_keys = set(layout['component_input_sha256']) | {'same-stool/' + name for name in layout['stool_component_input_sha256']}
    require(set(keys) == required_keys, 'Actual furniture component inventory differs from certified layout')
    seen, minimum, max_deviation = set(), math.inf, 0.
    for row in layout['authored_layout']['pairs']:
        parts = row.get('parts', [])
        require(isinstance(parts, list) and len(parts) == 2, 'Invalid separation certificate component pair')
        pair = frozenset(parts)
        require(pair in required_pairs and pair not in seen and row.get('passed') is True,
                'Separation certificates are duplicate, missing, unknown or failed')
        seen.add(pair)
        plane = row['separating_plane']
        normal = np.asarray(plane['unit_normal_world'], dtype=float)
        first, second = np.asarray(vertices[parts[0]], dtype=float), np.asarray(vertices[parts[1]], dtype=float)
        require(normal.shape == (3,) and np.isfinite(normal).all() and abs(np.linalg.norm(normal) - 1) <= 1e-8
                and first.ndim == second.ndim == 2 and first.shape[1] == second.shape[1] == 3
                and len(first) and len(second) and np.isfinite(first).all() and np.isfinite(second).all(),
                'Invalid actual geometry or separating-plane unit normal')
        first_max, second_min = float((first @ normal).max()), float((second @ normal).min())
        declared_first, declared_second = plane['first_projection_max_m'], plane['second_projection_min_m']
        require(all(isinstance(x, (int, float)) and math.isfinite(x)
                    for x in (declared_first, declared_second, plane['gap_m']))
                and abs(plane['gap_m'] - (declared_second - declared_first)) <= 1e-10,
                'Invalid declared separating-plane bounds')
        require(first_max <= declared_first + 5e-6 and second_min >= declared_second - 5e-6,
                'Actual imported vertices violate the accepted layout halfspaces')
        gap = second_min - first_max
        require(gap >= layout['quality_gate']['required_cross_furniture_clearance_m'],
                'Actual imported furniture solids fail the required separation')
        minimum = min(minimum, gap)
        max_deviation = max(max_deviation, abs(first_max - declared_first), abs(second_min - declared_second))
    require(seen == required_pairs and len(seen) == layout['authored_layout']['cross_object_pair_count'],
            'Separation certificate set does not cover every cross-furniture component pair')
    return {'passed': True, 'actual_component_count': len(keys), 'cross_object_pair_count': len(seen),
            'required_clearance_m': layout['quality_gate']['required_cross_furniture_clearance_m'],
            'minimum_actual_vertex_plane_clearance_m': minimum,
            'maximum_projection_bound_deviation_m': max_deviation,
            'method': 'Every actual vertex lies in the independently certified separating halfspaces; all component triangles and enclosed solids are therefore separated.'}


def authored_delta(translation):
    from blender_scene import Y_UP_TO_Z_UP
    return Y_UP_TO_Z_UP.to_3x3() @ Vector(translation)


def actual_layout_vertices():
    from blender_scene import Y_UP_TO_Z_UP
    from blender_stool_scene import mesh_arrays
    # Inverse rigid coordinate conversion for validation only, without editing meshes.
    native_to_world = np.asarray(Y_UP_TO_Z_UP.to_3x3())
    vertices = {}
    roots = [('same-stool', bpy.context.scene.objects[STOOL_ROOT])]
    roots += [(identity, bpy.context.scene.objects[description[1]]) for identity, description in TARGETS.items()]
    for identity, root in roots:
        for obj in root.children:
            require(obj.type == 'MESH' and obj.get('source_part'), 'Furniture root contains an unexpected component')
            key = identity + '/' + obj['source_part']
            require(key not in vertices, 'Duplicate native furniture component')
            vertices[key] = mesh_arrays(obj)[0] @ native_to_world
    return vertices


def verify_stool(layout, stool_sources):
    from blender_stool import read_part
    from blender_stool_scene import mesh_arrays
    root = bpy.context.scene.objects[STOOL_ROOT]
    require(root.get('source_fit_json_sha256') == layout['stool_fit_json_sha256'], 'Accepted stool fit provenance differs')
    checked = {}
    for child in root.children:
        name = child['source_part']
        require(child.get('source_ply_sha256') == layout['stool_component_input_sha256'][name], 'Accepted stool component provenance differs')
        original_xyz, original_faces = read_part(stool_sources[name])
        xyz, faces = mesh_arrays(child)
        error = float(np.max(np.abs(xyz - original_xyz)))
        require(error <= 5e-7 and np.array_equal(faces, original_faces), 'Accepted stool differs from collision-evaluated PLY geometry')
        checked[name] = {'exact_input_face_indices': True, 'maximum_input_vertex_error_m': error}
    require(set(checked) == set(stool_sources), 'Accepted stool component inventory differs')
    return checked


def preserved_source(expected, replaced):
    from blender_stool_scene import object_snapshot
    checks = {}
    for name, before in expected.items():
        obj = bpy.context.scene.objects.get(name)
        require(obj is not None, f'Source object missing: {name}')
        actual = object_snapshot(obj)
        if name in replaced:
            for key in ('hide_render', 'hidden_in_view_layer', 'collections'):
                actual[key] = before[key]
        require(json.dumps(actual, sort_keys=True) == json.dumps(before, sort_keys=True), f'Source object changed: {name}')
        checks[name] = {'exact_geometry_transform_attributes_material_properties_preserved': True,
                        'hidden_observed_comparison': name in replaced}
    return checks


def input_hashes(source, fit_path, sources, evaluation, layout):
    return {'source_blend': sha(source), 'fit_json': sha(fit_path), 'evaluation_json': sha(evaluation),
            'layout_json': sha(layout),
            'components': {key: sha(path) for key, path in sources.items()}}


def viewport(camera):
    target = Vector(camera['target_blender_m'])
    eye = Vector(camera['eye_blender_m'])
    count = 0
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                space.overlay.show_overlays = False
                space.show_gizmo = True
                space.shading.type = 'SOLID'
                space.shading.light = 'STUDIO'
                space.shading.color_type = 'MATERIAL'
                space.region_3d.view_location = target
                space.region_3d.view_rotation = (target - eye).to_track_quat('-Z', 'Y')
                space.region_3d.view_distance = (target - eye).length
                space.region_3d.view_perspective = 'PERSP'
                space.lens = camera['lens_mm']
                count += 1
    return count


def render_views(output, originals, parts, camera_record):
    """Use the same temporary camera and lights for the before/complete pair."""
    scene = bpy.context.scene
    original_camera, original_world = scene.camera, scene.world
    temporary = []
    data = bpy.data.cameras.new('Dining render camera (temporary)')
    camera = bpy.data.objects.new(data.name, data)
    scene.collection.objects.link(camera)
    temporary.append(camera)
    scene.camera = camera
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'AgX'
    world = bpy.data.worlds.new('Dining neutral studio world (temporary)')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (.36, .38, .42, 1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = .35
    scene.world = world
    for name, location, energy, size in (
        ('Dining soft key', (1.6, -.8, 3.5), 650, 4.0),
        ('Dining soft fill', (-1.8, 1.0, 2.7), 400, 3.0)):
        light_data = bpy.data.lights.new(name, 'AREA')
        light_data.energy, light_data.shape, light_data.size = energy, 'DISK', size
        light = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light)
        light.location = location
        light.rotation_euler = (Vector((0, 2, .45)) - light.location).to_track_quat('-Z', 'Y').to_euler()
        temporary.append(light)

    def position(record):
        camera.location = record['eye_blender_m']
        camera.rotation_euler = (Vector(record['target_blender_m']) - camera.location).to_track_quat('-Z', 'Y').to_euler()
        data.lens = record['lens_mm']
        scene.render.resolution_x, scene.render.resolution_y = record['resolution']

    try:
        position(camera_record)
        for before in (True, False):
            bpy.data.collections[COMPARISON].hide_render = not before
            for obj in originals:
                obj.hide_render = not before
            for obj in parts:
                obj.hide_render = before
            scene.render.filepath = str(output / ('dining-before.png' if before else 'dining-complete.png'))
            bpy.ops.render.render(write_still=True)
        extra = {'dining-front.png': {'eye_blender_m': [0, -2.7, 1.2], 'target_blender_m': [0, 2.0, .55],
                                     'lens_mm': 56, 'resolution': [1500, 1050]},
                 'dining-underside.png': {'eye_blender_m': [2.6, -.3, .46], 'target_blender_m': [0, 2.0, .4],
                                         'lens_mm': 41, 'resolution': [1500, 1050]}}
        for filename, record in extra.items():
            position(record)
            scene.render.filepath = str(output / filename)
            bpy.ops.render.render(write_still=True)
        return extra
    finally:
        bpy.data.collections[COMPARISON].hide_render = True
        for obj in originals:
            obj.hide_render = True
        for obj in parts:
            obj.hide_render = False
        scene.camera, scene.world = original_camera, original_world
        for obj in temporary:
            block, kind = obj.data, obj.type
            bpy.data.objects.remove(obj, do_unlink=True)
            (bpy.data.cameras if kind == 'CAMERA' else bpy.data.lights).remove(block)
        bpy.data.worlds.remove(world)


def build(source, fit_dir, evaluation_path, layout_path, output):
    from blender_stool import read_part, check_closed, srgb_linear
    from blender_stool_scene import object_snapshot, mesh_arrays, source_state, unit_state
    fit, fit_path, sources, component_hashes, evaluation = input_bundle(fit_dir, evaluation_path)
    layout, translations, stool_sources = layout_bundle(layout_path, source, sha(fit_path), component_hashes)
    hashes = input_hashes(source, fit_path, sources, evaluation_path, layout_path)
    source_meshes = {key: read_part(path) for key, path in sources.items()}
    volumes = {key: check_closed(*arrays) for key, arrays in source_meshes.items()}
    bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=True, use_scripts=False)
    scene = bpy.context.scene
    require(unit_state()['system'] == 'METRIC' and unit_state()['scale_length'] == 1, 'Source scene must use metric metres')
    require(STOOL_ROOT in scene.objects and len(scene.objects[STOOL_ROOT].children) == 7,
            'Source scene must contain the accepted seven-part stool')
    original = {obj.name: object_snapshot(obj) for obj in scene.objects}
    original_faces, units = source_state(), unit_state()
    originals = {}
    for identity in TARGETS:
        matches = [obj for obj in scene.objects if obj.get('alvenx_id') == identity]
        require(len(matches) == 1, f'Source must contain exactly one observed {identity}')
        originals[identity] = matches[0]
    ambiguous = [obj for obj in scene.objects if obj.get('alvenx_id') == 'unresolved']
    require(len(ambiguous) == 1, 'Source must contain its ambiguous observed dining fragments')
    originals['unresolved'] = ambiguous[0]
    ambiguous_xyz, ambiguous_faces = mesh_arrays(ambiguous[0])
    ambiguous_record = {'object_name': ambiguous[0].name, 'identity': 'unresolved',
                        'classification': 'ambiguous observed dining fragments', 'restored_furniture': False,
                        'triangle_count': len(ambiguous_faces), 'vertex_count': len(ambiguous_xyz),
                        'bounds_min_blender_m': ambiguous_xyz.min(0).tolist(),
                        'bounds_max_blender_m': ambiguous_xyz.max(0).tolist(),
                        'before_visible': True, 'completed_visible': False,
                        'retention': 'unchanged native geometry/attributes/material/identity in the switchable comparison collection'}
    require(COMPARISON not in bpy.data.collections and COMPLETED not in bpy.data.collections, 'Source is already completed')
    comparison = bpy.data.collections.new(COMPARISON)
    scene.collection.children.link(comparison)
    completed = bpy.data.collections.new(COMPLETED)
    scene.collection.children.link(completed)
    roots, parts = {}, []
    for target in fit['objects']:
        identity = target['uuid']
        all_xyz = np.concatenate([source_meshes[identity + '/' + p['name']][0] for p in target['components']])
        origin = (all_xyz.min(0) + all_xyz.max(0)) / 2
        origin[2] = all_xyz[:, 2].min()
        collection = bpy.data.collections.new(TARGETS[identity][1] + ' - editable parts')
        completed.children.link(collection)
        root = bpy.data.objects.new(TARGETS[identity][1], None)
        collection.objects.link(root)
        root.location = origin
        root.empty_display_type, root.empty_display_size = 'PLAIN_AXES', .08
        root['dining_object_uuid'] = identity
        root['provenance_role'] = 'fitted/structural'
        root['source_fit_json_sha256'] = hashes['fit_json']
        root['source_evaluation_json_sha256'] = hashes['evaluation_json']
        root['native_axis_units'] = 'Z-up metres'
        root['coordinate_conversion'] = '(x,y,z) -> (x,-z,y); once at PLY import'
        root['component_count'] = len(target['components'])
        root['dimensions_world_blender_m'] = np.ptp(all_xyz, axis=0).tolist()
        root['fit_parameters_json'] = json.dumps({key: value for key, value in target.items() if key != 'components'})
        root['editing_help'] = 'Move/rotate/scale this root for whole-furniture edits; select child meshes and use native Transform/Dimensions or Edit Mode for component edits. Component parameters describe the accepted fitted shape and do not drive a procedural modifier.'
        roots[identity] = root
        for part in target['components']:
            key = identity + '/' + part['name']
            xyz, faces = source_meshes[key]
            center = xyz.mean(0)
            name = TARGETS[identity][1] + ' / ' + part['name']
            mesh = bpy.data.meshes.new(name + ' mesh')
            mesh.from_pydata((xyz - center).tolist(), [], faces.tolist())
            mesh.update()
            obj = bpy.data.objects.new(name, mesh)
            collection.objects.link(obj)
            obj.parent, obj.location = root, center - origin
            obj['dining_object_uuid'] = identity
            obj['source_part'] = part['name']
            obj['provenance_role'] = part['provenance_role']
            obj['source_ply_sha256'] = component_hashes[key]
            obj['source_fit_json_sha256'] = hashes['fit_json']
            obj['component_type'] = part['type']
            obj['component_parameters_json'] = json.dumps(part['parameters'])
            obj['input_component_metadata_json'] = json.dumps(part)
            rgb = part.get('color_rgb', part['parameters'].get('color_rgb', [151, 140, 119]))
            color = tuple(srgb_linear(c) for c in rgb) + (1,)
            mat = bpy.data.materials.new(name + ' measured neutral finish')
            mat.diffuse_color, mat.use_nodes = color, True
            bsdf = mat.node_tree.nodes.get('Principled BSDF')
            bsdf.inputs['Base Color'].default_value = color
            bsdf.inputs['Roughness'].default_value = .72
            mesh.materials.append(mat)
            parts.append(obj)
        observed_xyz, _ = mesh_arrays(originals[identity])
        offset = float(np.max(np.abs((observed_xyz.min(0) + observed_xyz.max(0)) / 2
                                     - (all_xyz.min(0) + all_xyz.max(0)) / 2)))
        root['observed_to_completed_bounds_center_difference_m'] = offset
    for target in originals.values():
        for collection in list(target.users_collection):
            collection.objects.unlink(target)
        comparison.objects.link(target)
        target.hide_set(True)
        target.hide_render = True
    comparison.hide_render = True
    bpy.context.view_layer.update()
    stool_checks = verify_stool(layout, stool_sources)
    root_transforms = {}
    for identity, root in roots.items():
        before = [list(row) for row in root.matrix_world]
        delta = authored_delta(translations[identity])
        root.location += delta
        root['authored_layout_translation_world_m'] = translations[identity]
        root['authored_layout_translation_blender_m'] = list(delta)
        root['source_layout_json_sha256'] = hashes['layout_json']
        root['authored_layout_transform_sha256'] = layout['layout_transform_sha256']
        bpy.context.view_layer.update()
        root_transforms[identity] = {'source_fit_matrix_blender': before,
                                     'authored_matrix_blender': [list(row) for row in root.matrix_world],
                                     'translation_world_m': translations[identity], 'translation_blender_m': list(delta)}
    collision_check = verify_separating_planes(layout, actual_layout_vertices())
    report_reference = os.path.relpath(layout_path, Path(__file__).resolve().parent).replace('\\', '/')
    camera = {'eye_blender_m': [2.8, -1.65, 2.12], 'target_blender_m': [0, 1.96, .48],
              'lens_mm': 54, 'resolution': [1500, 1100]}
    view_count = viewport(camera)
    require(view_count > 0, 'Native document must retain an editable 3D viewport')
    replaced = [obj.name for obj in originals.values()]
    check_paths(source, fit_dir, evaluation_path, layout_path, output, 'build')
    output.mkdir(parents=True, exist_ok=True)
    extra_views = render_views(output, list(originals.values()), parts, camera)
    checks = preserved_source(original, replaced)
    require(source_state() == original_faces and unit_state() == units, 'Source face lineage or native units changed')
    for obj in scene.objects:
        obj.select_set(False)
    for name in [STOOL_ROOT] + [obj.name for obj in roots.values()]:
        scene.objects[name].select_set(True)
    bpy.context.view_layer.objects.active = roots['same-table']
    inventory = []
    for identity, root in [('same-stool', scene.objects[STOOL_ROOT])] + list(roots.items()):
        children = sorted((obj for obj in root.children if obj.type == 'MESH'), key=lambda obj: obj.name)
        coordinates = np.concatenate([mesh_arrays(obj)[0] for obj in children])
        inventory.append({'uuid': identity, 'root_name': root.name, 'component_count': len(children),
                          'component_names': [obj.name for obj in children],
                          'triangle_count': sum(len(obj.data.polygons) for obj in children),
                          'bounds_min_blender_m': coordinates.min(0).tolist(),
                          'bounds_max_blender_m': coordinates.max(0).tolist(),
                          'dimensions_world_blender_m': np.ptp(coordinates, axis=0).tolist(),
                          'accepted_stool_preserved': identity == 'same-stool'})
    record = {'schema': 'alvenx.spatial.dining-scene-integration.v1', 'status': 'pending_fresh_reopen',
              'build_process_id': os.getpid(), 'input_sha256': hashes,
              'source_objects': original, 'source_face_snapshot_sha256': original_faces, 'source_units': units,
              'original_target_names': [originals[identity].name for identity in TARGETS],
              'comparison_source_names': replaced, 'comparison_collection': COMPARISON,
              'ambiguous_observed_dining_fragments': ambiguous_record,
              'completed_collection': COMPLETED, 'four_furniture_roots': [STOOL_ROOT] + [o.name for o in roots.values()],
              'furniture_inventory': inventory,
              'completed_roots': {identity: object_snapshot(obj) for identity, obj in roots.items()},
              'completed_parts': {obj.name: object_snapshot(obj) for obj in parts},
              'source_preservation': checks, 'component_positive_volume_m3': volumes,
              'evaluation_quality_gate': evaluation['quality_gate'], 'fixed_camera': camera,
              'authored_layout': {'independent_report_sha256': hashes['layout_json'],
                                  'independent_report_project_path': report_reference,
                                  'layout_transform_sha256': layout['layout_transform_sha256'],
                                  'translation_world_m': translations, 'root_transforms': root_transforms,
                                  'independent_quality_gate': layout['quality_gate'],
                                  'native_actual_separation': collision_check,
                                  'accepted_stool_exact_geometry': stool_checks,
                                  'policy': 'Frozen observation-frame fitted meshes remain unchanged; authored horizontal root translations determine the displayed/exported furniture layout.'},
              'additional_views': extra_views, 'saved_view_count': view_count,
              'editable_parameters': 'Native child meshes and root transforms are editable; recorded analytic component dimensions describe the fitted acceptance state, with no automatic parameter driver.'}
    bpy.ops.wm.save_as_mainfile(filepath=str(output / 'complete-dining.blend'), check_existing=False, compress=True)
    for obj in scene.objects:
        obj.select_set(obj.type in {'MESH', 'EMPTY'} and not obj.hide_render and not obj.hide_get())
    bpy.ops.export_scene.gltf(filepath=str(output / 'complete-dining.glb'), export_format='GLB', use_selection=True,
                              export_yup=True, export_extras=True, export_animations=False,
                              export_cameras=False, export_lights=False)
    require(input_hashes(source, fit_path, sources, evaluation_path, layout_path) == hashes, 'Input bytes changed during build')
    layout_bundle(layout_path, source, hashes['fit_json'], component_hashes)
    names = ('complete-dining.blend', 'complete-dining.glb', 'dining-before.png', 'dining-complete.png',
             'dining-front.png', 'dining-underside.png')
    record['artifacts'] = {name: {'sha256': sha(output / name), 'bytes': (output / name).stat().st_size} for name in names}
    save_json(output / 'scene-integration.json', record)
    print('BUILT', json.dumps(record['artifacts']))


def edit_verification(record):
    from blender_stool_scene import mesh_arrays, source_state
    results = {}
    bpy.context.preferences.edit.use_global_undo = True
    for identity in TARGETS:
        name = TARGETS[identity][1]
        root = bpy.context.scene.objects[name]
        require(root.type == 'EMPTY' and not root.hide_get(), 'Furniture root must be individually selectable')
        children = {obj.name: mesh_arrays(obj)[0] for obj in root.children}
        other = {obj.name: mesh_arrays(obj)[0] for obj in bpy.context.scene.objects
                 if obj.type == 'MESH' and obj.parent != root}
        for obj in bpy.context.scene.objects:
            obj.select_set(False)
        root.select_set(True)
        bpy.context.view_layer.objects.active = root
        require(root.select_get(), 'Native root selection failed')
        for child in root.children:
            child.select_set(True)
            require(child.select_get(), 'Native component selection failed')
            child.select_set(False)
        initial = np.array(root.matrix_world)
        bpy.ops.ed.undo_push(message=f'{name} before rigid edit')
        root.location += Vector((.15, .10, 0))
        root.rotation_euler.z += np.deg2rad(17)
        root.scale.x *= 1.05
        bpy.context.view_layer.update()
        edited = np.array(root.matrix_world)
        delta = edited @ np.linalg.inv(initial)
        maximum = 0.
        for part, xyz in children.items():
            expected = xyz @ delta[:3, :3].T + delta[:3, 3]
            maximum = max(maximum, float(np.max(np.abs(mesh_arrays(bpy.context.scene.objects[part])[0] - expected))))
        require(maximum <= 5e-7, 'Furniture root did not coherently move, rotate and resize every part')
        for part, xyz in other.items():
            require(np.array_equal(mesh_arrays(bpy.context.scene.objects[part])[0], xyz), 'Editing one root moved unrelated geometry')
        require(source_state() == record['source_face_snapshot_sha256'], 'Root edit changed source geometry/lineage')
        bpy.ops.ed.undo_push(message=f'{name} after rigid edit')
        undo = bpy.ops.ed.undo()
        require(np.array_equal(np.array(bpy.context.scene.objects[name].matrix_world), initial), 'Native undo failed')
        redo = bpy.ops.ed.redo()
        require(np.array_equal(np.array(bpy.context.scene.objects[name].matrix_world), edited), 'Native redo failed')
        bpy.ops.ed.undo()
        require(np.array_equal(np.array(bpy.context.scene.objects[name].matrix_world), initial), 'Second undo failed')
        require(source_state() == record['source_face_snapshot_sha256'], 'Undo/redo changed source geometry/lineage')
        preserved_source(record['source_objects'], record['comparison_source_names'])
        results[identity] = {'status': 'passed', 'translation_blender_m': [.15, .10, 0], 'rotation_z_degrees': 17,
                             'scale_x_factor': 1.05, 'selectable_components': len(children),
                             'maximum_affine_vertex_error_m': maximum,
                             'native_undo': sorted(undo), 'native_redo': sorted(redo), 'unrelated_meshes_unchanged': True}
        results[identity]['undo_restores_authored_layout'] = True
    return results


def reopen(source, fit_dir, evaluation_path, layout_path, output):
    from blender_stool import read_part, check_closed
    from blender_stool_scene import object_snapshot, mesh_arrays, source_state, unit_state
    from geometry_compare import compare_triangle_geometry
    import verify_blender_edit as verification
    fit, fit_path, sources, component_hashes, _ = input_bundle(fit_dir, evaluation_path)
    layout, translations, stool_sources = layout_bundle(layout_path, source, sha(fit_path), component_hashes)
    record_path = output / 'scene-integration.json'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    require(os.getpid() != record['build_process_id'], 'Must reopen in a fresh Blender process')
    require(input_hashes(source, fit_path, sources, evaluation_path, layout_path) == record['input_sha256'], 'Inputs changed after build')
    for name, artifact in record['artifacts'].items():
        require(sha(output / name) == artifact['sha256'], f'Artifact changed after build: {name}')
    bpy.ops.wm.open_mainfile(filepath=str(output / 'complete-dining.blend'), load_ui=True, use_scripts=False)
    require(unit_state() == record['source_units'] and source_state() == record['source_face_snapshot_sha256'],
            'Native source units, face lineage, ownership or geometry changed')
    native_source = preserved_source(record['source_objects'], record['comparison_source_names'])
    for name in record['comparison_source_names']:
        obj = bpy.context.scene.objects[name]
        require(obj.hide_get() and obj.hide_render and COMPARISON in {c.name for c in obj.users_collection},
                'Original target must remain hidden in its named comparison collection')
    for identity, expected in record['completed_roots'].items():
        require(object_snapshot(bpy.context.scene.objects[TARGETS[identity][1]]) == expected, 'Saved root metadata or transform changed')
    native_parts = {}
    for name, expected in record['completed_parts'].items():
        obj = bpy.context.scene.objects[name]
        require(json.dumps(object_snapshot(obj), sort_keys=True) == json.dumps(expected, sort_keys=True),
                f'Saved component changed: {name}')
        key = obj['dining_object_uuid'] + '/' + obj['source_part']
        original_xyz, original_faces = read_part(sources[key])
        xyz, faces = mesh_arrays(obj)
        authored_xyz = original_xyz + np.asarray(authored_delta(translations[obj['dining_object_uuid']]))
        error = float(np.max(np.abs(xyz - authored_xyz)))
        require(error <= 5e-7 and np.array_equal(faces, original_faces), 'Imported PLY geometry differs')
        native_parts[name] = {'exact_input_face_indices': True, 'maximum_input_vertex_error_m': error,
                              'comparison_frame': 'Frozen source PLY plus accepted authored root translation',
                              'closed_positive_volume_m3': check_closed(xyz, faces)}
    stool_checks = verify_stool(layout, stool_sources)
    separation = verify_separating_planes(layout, actual_layout_vertices())
    view_checks = []
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                require(not area.spaces.active.overlay.show_overlays, 'Saved default viewport shows helper overlays')
                view_checks.append({'screen': screen.name, 'overlays_hidden': True,
                                    'shading': area.spaces.active.shading.type})
    require(len(view_checks) == record['saved_view_count'] and len(view_checks) > 0,
            'Native document lost its saved editable 3D viewport')
    root_selection = {}
    for name in record['four_furniture_roots']:
        root = bpy.context.scene.objects.get(name)
        require(root is not None and root.type == 'EMPTY' and not root.hide_get() and not root.hide_select,
                'A furniture root is unavailable for independent selection')
        for obj in bpy.context.scene.objects:
            obj.select_set(False)
        root.select_set(True)
        bpy.context.view_layer.objects.active = root
        require(bpy.context.selected_objects == [root], 'Furniture root cannot be selected independently')
        root_selection[name] = True
    edits = edit_verification(record)
    restored_separation = verify_separating_planes(layout, actual_layout_vertices())
    visible = {obj.name: mesh_arrays(obj) for obj in bpy.context.scene.objects
               if obj.type == 'MESH' and not obj.hide_get() and not obj.hide_render}
    observed = verification.appearance_snapshot()
    for identity in (*TARGETS, 'same-stool', 'unresolved'):
        observed.pop(identity)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.import_scene.gltf(filepath=str(output / 'complete-dining.glb'))
    returned = {obj.name: obj for obj in bpy.context.scene.objects if obj.type == 'MESH'}
    require(set(returned) == set(visible), 'GLB visible mesh inventory differs or includes hidden source targets')
    geometry = {}
    for name, arrays in visible.items():
        check = compare_triangle_geometry(*arrays, *mesh_arrays(returned[name]), tolerance=5e-6)
        require(check['status'] == 'pass', f'GLB oriented triangle geometry changed: {name}')
        geometry[name] = check
    actual_observed = verification.appearance_snapshot()
    appearance = verification.check_appearance(observed, actual_observed)
    require(appearance['status'] == 'pass', 'GLB source appearance changed')
    for name, expected in record['completed_parts'].items():
        obj = returned[name]
        require(obj.parent and obj.parent.name == expected['parent']
                and obj.get('source_ply_sha256') == expected['properties']['source_ply_sha256'], 'GLB component parent or provenance changed')
    require(all(name in bpy.context.scene.objects for name in record['four_furniture_roots']), 'GLB furniture root inventory changed')
    glb_separation = verify_separating_planes(layout, actual_layout_vertices())
    require(input_hashes(source, fit_path, sources, evaluation_path, layout_path) == record['input_sha256'], 'Verification changed input bytes')
    layout_bundle(layout_path, source, record['input_sha256']['fit_json'], component_hashes)
    require(all(sha(output / name) == artifact['sha256'] for name, artifact in record['artifacts'].items()), 'Verification changed artifacts')
    record.update({'status': 'passed', 'fresh_process_native_reopen': True, 'native_source_preservation': native_source,
                   'native_parts': native_parts, 'native_edit_undo_redo': edits, 'saved_viewport': view_checks,
                   'four_roots_independently_selectable': root_selection,
                   'inter_furniture_actual_collision_gate': {'passed': True,
                       'independent_report_project_path': record['authored_layout']['independent_report_project_path'],
                       'independent_report_sha256': sha(layout_path), 'layout_transform_sha256': layout['layout_transform_sha256'],
                       'fresh_native': separation, 'after_edit_undo': restored_separation,
                       'visible_glb_roundtrip': glb_separation, 'accepted_stool_exact_geometry': stool_checks},
                   'glb_geometry': geometry, 'glb_visible_mesh_count': len(returned), 'glb_hidden_originals_excluded': True,
                   'glb_source_appearance': appearance,
                   'glb_scope': 'Visible oriented triangle geometry, object hierarchy and metadata; arbitrary Blender per-face lineage is native-only.'})
    save_json(record_path, record)
    print('VERIFIED', json.dumps({'status': record['status'], 'visible_meshes': len(returned),
                                 'maximum_glb_corner_error_m': max(x['maximum_matched_corner_distance_m'] for x in geometry.values())}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--fit', required=True, type=Path)
    parser.add_argument('--evaluation', required=True, type=Path)
    parser.add_argument('--layout', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('build', 'reopen'))
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None)
    require(bpy is not None, 'Run this entrypoint with Blender Python')
    paths = [getattr(args, key).resolve() for key in ('source', 'fit', 'evaluation', 'layout', 'output')]
    check_paths(*paths, args.mode)
    (build if args.mode == 'build' else reopen)(*paths)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise
