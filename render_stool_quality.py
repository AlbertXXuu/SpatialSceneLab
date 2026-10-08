"""Reproduce the fitted stool's fixed diagnostic views in background Blender. MIT.

blender --background --factory-startup --python-exit-code 1 --python render_stool_quality.py --
    --mesh examples/stool-fit/fitted/fitted_stool.ply
    --box-prior examples/stool-fit/input/scene.json --object same-stool
    --output .local/stool-render

The mesh is the ASCII PLY emitted by fit_stool.py. The required box prior supplies
the crop and camera frame. Output includes front/underside PNGs and a source-hash,
camera, crop and lighting record. No reconstruction or reference assets are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_stool import check_output, read_part, require, save_json, sha


VIEWS = {'front': [1.8, 1.25, 2.3], 'underside': [1.8, -0.16, 2.3]}
LIGHTS = (
    ('Key', (1.2, 1.5, 1.2), 18, 1.2),
    ('Fill', (-1.3, 0.7, 0.1), 7, 1.4),
    ('Low fill', (0.5, -0.5, 1.0), 4, 0.8),
)


def to_blender(point):
    return np.array([point[0], -point[2], point[1]])


def crop(vertices, faces, rotation, center, half):
    """Clip only at the six display planes, preserving interior geometry/winding."""
    local = (vertices - center) @ rotation
    out_vertices, out_faces = [], []
    for face in faces:
        polygon = list(local[face])
        for axis in range(3):
            for sign in (-1, 1):
                clipped = []
                for index, a in enumerate(polygon):
                    b = polygon[(index + 1) % len(polygon)]
                    da, db = sign * a[axis] - half[axis], sign * b[axis] - half[axis]
                    if da <= 0:
                        clipped.append(a)
                    if (da <= 0) != (db <= 0):
                        clipped.append(a + (b - a) * da / (da - db))
                polygon = clipped
                if not polygon:
                    break
            if not polygon:
                break
        if len(polygon) >= 3:
            start = len(out_vertices)
            out_vertices.extend(polygon)
            out_faces.extend((start, start + i, start + i + 1) for i in range(1, len(polygon) - 1))
    require(out_faces, 'No mesh triangles intersect the requested display crop')
    return np.array(out_vertices) @ rotation.T + center, out_faces


def render(args):
    mesh_path = args.mesh.resolve(strict=True)
    prior_path = args.box_prior.resolve(strict=True)
    output = args.output.resolve()
    check_output(mesh_path.parent, output, 'build')
    check_output(prior_path.parent, output, 'build')
    prior_scene = json.loads(prior_path.read_text(encoding='utf-8'))
    require(prior_scene.get('units') == 'm' and prior_scene.get('up_axis') == 'Y', 'Box prior must use Y-up metres')
    matches = [item for item in prior_scene['objects'] if item['uuid'] == args.object]
    require(len(matches) == 1, 'Object must identify exactly one observed box prior')
    observed = matches[0]
    require(observed.get('geometry_kind') == 'box', 'The selected prior must be a box')
    transform = np.array(observed['transform_world'], dtype=float)
    dimensions = np.array(observed['dimensions_m'], dtype=float)
    require(transform.shape == (4, 4) and np.isfinite(transform).all()
            and np.allclose(transform[3], [0, 0, 0, 1]), 'Invalid prior transform')
    require(dimensions.shape == (3,) and np.isfinite(dimensions).all() and (dimensions > 0).all(), 'Invalid box dimensions')
    rotation, center = transform[:3, :3], transform[:3, 3]
    require(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
            and np.isclose(np.linalg.det(rotation), 1, atol=1e-8), 'Prior transform must be a rigid rotation')
    half = dimensions / 2 + .06
    mesh_hash, prior_hash = sha(mesh_path), sha(prior_path)
    blender_vertices, faces = read_part(mesh_path)
    # read_part returns Z-up coordinates; crop in the explicit prior's Y-up frame.
    vertices = blender_vertices[:, [0, 2, 1]].copy()
    vertices[:, 2] *= -1
    clipped, clipped_faces = crop(vertices, faces, rotation, center, half)
    clipped = clipped[:, [0, 2, 1]]
    clipped[:, 1] *= -1

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 64
    scene.cycles.seed = 0
    scene.cycles.use_denoising = False
    scene.render.resolution_x = scene.render.resolution_y = 1000
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get('Background')
    background.inputs['Color'].default_value = (.68, .71, .74, 1)
    background.inputs['Strength'].default_value = .65
    material = bpy.data.materials.new('Neutral gray / flat normals')
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (.22, .23, .24, 1)
    bsdf.inputs['Roughness'].default_value = .85
    bsdf.inputs['Metallic'].default_value = 0
    data = bpy.data.meshes.new('Original fitted geometry / display crop')
    data.from_pydata(clipped.tolist(), [], clipped_faces)
    data.update()
    obj = bpy.data.objects.new('Fitted stool / structural completion', data)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    for polygon in data.polygons:
        polygon.use_smooth = False

    target = to_blender(center)
    for name, offset, energy, size in LIGHTS:
        light_data = bpy.data.lights.new(name, 'AREA')
        light_data.energy, light_data.shape, light_data.size = energy, 'DISK', size
        light = bpy.data.objects.new(name, light_data)
        bpy.context.collection.objects.link(light)
        light.location = to_blender(center + rotation @ np.array(offset))
        light.rotation_euler = (Vector(target) - light.location).to_track_quat('-Z', 'Y').to_euler()
    camera_data = bpy.data.cameras.new('Fixed diagnostic camera')
    camera = bpy.data.objects.new('Fixed diagnostic camera', camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type, camera_data.ortho_scale = 'ORTHO', 1.05
    camera_data.clip_start, camera_data.clip_end = .01, 100

    record = {
        'schema': 'spatial-scene-lab.stool-diagnostic-render.v1',
        'blender_version': bpy.app.version_string,
        'mesh': {'filename': mesh_path.name, 'sha256': mesh_hash, 'vertices': len(vertices), 'faces': len(faces)},
        'box_prior': {'filename': prior_path.name, 'sha256': prior_hash, 'object': observed},
        'crop': {'extra_margin_each_side_m': .06, 'half_extents_m': half.tolist(),
                 'method': 'All triangles clipped to six box planes; no caps, ownership or component filtering.',
                 'display_faces': len(clipped_faces)},
        'display': 'Neutral flat faces; original interior geometry and winding. No smoothing, filling, repair or denoising.',
        'projection': {'type': 'orthographic', 'scale_m': 1.05, 'resolution': [1000, 1000]},
        'lighting': {'world_color_linear': [.68, .71, .74], 'world_strength': .65, 'area_lights_name_prior_offset_energy_size': LIGHTS},
        'material': {'base_color_linear': [.22, .23, .24], 'roughness': .85, 'metallic': 0},
        'render': {'engine': 'Cycles CPU', 'samples': 64, 'seed': 0, 'view_transform': 'Standard', 'exposure': 0, 'gamma': 1},
        'views': {},
    }
    require(sha(mesh_path) == mesh_hash and sha(prior_path) == prior_hash, 'Render input changed during loading')
    output.mkdir(parents=True, exist_ok=True)
    views = VIEWS if args.view == 'both' else {args.view: VIEWS[args.view]}
    for name, offset in views.items():
        position = center + rotation @ np.array(offset)
        camera.location = to_blender(position)
        camera.rotation_euler = (Vector(target) - camera.location).to_track_quat('-Z', 'Y').to_euler()
        destination = output / f'{mesh_path.stem.replace("_", "-")}-{name}.png'
        scene.render.filepath = str(destination)
        bpy.ops.render.render(write_still=True)
        record['views'][name] = {'filename': destination.name, 'sha256': sha(destination),
                                 'camera_input_y_up_m': position.tolist(), 'target_input_y_up_m': center.tolist(),
                                 'offset_in_prior_axes_m': offset}
    save_json(output / 'render-record.json', record)
    print('RENDERED', json.dumps({key: value['filename'] for key, value in record['views'].items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', required=True, type=Path, help='Fitted ASCII PLY in Y-up metres')
    parser.add_argument('--box-prior', required=True, type=Path, help='Observed scene.json with units, up_axis and object boxes')
    parser.add_argument('--object', required=True, help='UUID of the observed stool box')
    parser.add_argument('--output', required=True, type=Path, help='New or empty render directory, separate from input directories')
    parser.add_argument('--view', choices=('front', 'underside', 'both'), default='both')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
    try:
        render(args)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
