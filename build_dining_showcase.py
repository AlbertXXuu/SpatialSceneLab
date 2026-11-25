"""Build a self-contained editable showcase over the frozen dining scene."""

import argparse
from hashlib import file_digest
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector


HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'examples/dining-complete/native/complete-dining.blend'
IDENTITIES = {
    'same-table': ('Completed Table', 'Showcase Table'),
    'same-chair-west': ('Completed Chair West', 'Showcase Chair West'),
    'same-chair-east': ('Completed Chair East', 'Showcase Chair East'),
    'same-stool': ('Fitted Stool (structural completion)', 'Showcase Stool'),
}
FURNITURE = 'Showcase furniture - editable assets'
RESEARCH = (
    'Completed dining - individually editable components',
    'Repaired stool - 7 editable structural parts',
    'Original dining observations + ambiguous fragments (hidden)',
    'Original stool - observed comparison (hidden)',
)


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read_assets(manifest):
    document = json.loads(manifest.read_text(encoding='utf-8-sig'))
    if set(document['assets']) != set(IDENTITIES):
        raise ValueError(f'Expected asset identities: {", ".join(IDENTITIES)}')
    for identity, asset in document['assets'].items():
        asset['blend'] = (manifest.parent / asset['blend']).resolve(strict=True)
        if not asset['blend'].is_file() or asset['blend'].suffix.lower() != '.blend':
            raise ValueError(f'{identity}: expected a native .blend file')
        if 'scale' in asset and 'target_height_m' in asset:
            raise ValueError(f'{identity}: use scale or target_height_m')
        asset['xy_offset_m'] = tuple(float(value) for value in asset.get('xy_offset_m', [0, 0]))
        asset['yaw_deg'] = float(asset.get('yaw_deg', 0))
        asset['floor_z_m'] = float(asset.get('floor_z_m', 0))
        asset['scale'] = float(asset.get('scale', 1))
        numbers = (*asset['xy_offset_m'], asset['yaw_deg'], asset['floor_z_m'], asset['scale'])
        if len(asset['xy_offset_m']) != 2 or not all(map(math.isfinite, numbers)) or asset['scale'] <= 0:
            raise ValueError(f'{identity}: invalid placement')
        if 'target_height_m' in asset:
            asset['target_height_m'] = float(asset['target_height_m'])
            if not math.isfinite(asset['target_height_m']) or asset['target_height_m'] <= 0:
                raise ValueError(f'{identity}: invalid target height')
    return document


def material_assets(objects):
    materials = {slot.material for obj in objects for slot in obj.material_slots if slot.material}
    trees = [material.node_tree for material in materials if material.use_nodes and material.node_tree]
    images, visited = set(), set()
    while trees:
        tree = trees.pop()
        if tree in visited:
            continue
        visited.add(tree)
        for node in tree.nodes:
            image = getattr(node, 'image', None)
            if image:
                images.add(image)
            nested = getattr(node, 'node_tree', None)
            if nested:
                trees.append(nested)
    return materials, images


def pack_images(images, source):
    for image in images:
        if image.source == 'GENERATED' or image.packed_files:
            continue
        if image.source != 'FILE':
            raise ValueError(f'{source.name}: unsupported external image source {image.name} ({image.source})')
        relative = Path(image.filepath.removeprefix('//'))
        candidates = (Path(bpy.path.abspath(image.filepath)), source.parent / relative,
                      source.parent / 'textures' / relative.name, source.parent / relative.name)
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f'{source.name}: missing texture {image.filepath}')
        image.filepath = str(path.resolve())
        image.pack()
        if not image.packed_files:
            raise RuntimeError(f'{source.name}: Blender could not pack {image.name}')
        image.filepath = f'//textures/{path.name}'


def load_meshes(source, names):
    scene = bpy.context.scene
    staging = bpy.data.collections.new('Temporary asset import')
    scene.collection.children.link(staging)
    with bpy.data.libraries.load(str(source), link=False) as (library, imported):
        source_names = list(library.objects)
        missing = set(names).difference(source_names)
        if missing:
            raise ValueError(f'{source.name}: objects not found: {sorted(missing)}')
        imported.objects = source_names.copy()
    for obj in imported.objects:
        staging.objects.link(obj)
    selected = [(name, obj) for name, obj in zip(source_names, imported.objects)
                if obj.type == 'MESH' and (name in names if names else not obj.hide_render)]
    if not selected or names and len(selected) != len(names):
        raise ValueError(f'{source.name}: expected visible mesh objects')

    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    components, corners = [], []
    for source_name, obj in selected:
        evaluated = obj.evaluated_get(depsgraph)
        world = evaluated.matrix_world.copy()
        mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True, depsgraph=depsgraph)
        if not mesh.polygons:
            raise ValueError(f'{source.name}: empty mesh {source_name}')
        for index, slot in enumerate(evaluated.material_slots):
            if slot.material and index < len(mesh.materials):
                mesh.materials[index] = slot.material
        components.append((source_name, mesh, world))
        corners.extend(world @ Vector(corner) for corner in evaluated.bound_box)
    low = Vector(min(point[axis] for point in corners) for axis in range(3))
    high = Vector(max(point[axis] for point in corners) for axis in range(3))
    if any(high[axis] - low[axis] <= 1e-6 for axis in range(3)):
        raise ValueError(f'{source.name}: expected a three-dimensional asset')
    for obj in imported.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(staging)
    return components, low, high


def place_asset(identity, asset, collection, anchor, template):
    root = bpy.data.objects.new(IDENTITIES[identity][1], None)
    collection.objects.link(root)
    root.empty_display_type = 'PLAIN_AXES'
    root.empty_display_size = 0.13
    components, low, high = template
    center = (low + high) * .5
    origin = Matrix.Translation(Vector((-center.x, -center.y, -low.z)))
    scale = asset['scale']
    if 'target_height_m' in asset:
        scale = asset['target_height_m'] / (high.z - low.z)
    offset_x, offset_y = asset['xy_offset_m']
    root.location = (anchor.x + offset_x, anchor.y + offset_y, asset['floor_z_m'])
    root.rotation_euler.z = math.radians(asset['yaw_deg'])
    root.scale = (scale,) * 3
    root['showcase_identity'] = identity
    root['showcase_role'] = 'external finished presentation asset'
    root['showcase_asset_title'] = asset.get('title', asset['blend'].stem)
    root['showcase_asset_license'] = asset.get('license', '')
    root['showcase_asset_source_url'] = asset.get('source_url', '')
    root['showcase_asset_source_file'] = asset['blend'].name
    root['showcase_asset_source_sha256'] = digest(asset['blend'])
    root['showcase_baseline_root'] = IDENTITIES[identity][0]
    root['showcase_floor_z_m'] = asset['floor_z_m']
    for name, template_mesh, world in components:
        part = bpy.data.objects.new(f'{root.name} / {name}', template_mesh.copy())
        collection.objects.link(part)
        part.parent = root
        part.matrix_basis = origin @ world
        part['showcase_source_object'] = name
    materials, images = material_assets(root.children)
    if not materials:
        raise ValueError(f'{asset["blend"].name}: furniture mesh has no material')
    pack_images(images, asset['blend'])
    return root


def inspect(scene, manifest=None):
    collection = bpy.data.collections.get(FURNITURE)
    if not collection or collection.hide_viewport or collection.hide_render:
        raise RuntimeError('Editable showroom furniture collection is missing or hidden')
    if manifest and scene.get('showcase_manifest_sha256') != digest(manifest):
        raise RuntimeError('Showcase manifest hash differs')
    if scene.get('showcase_source_sha256') != digest(SOURCE):
        raise RuntimeError('Frozen base scene hash differs')
    if not scene.camera or scene.camera.name != 'Dining Studio Camera':
        raise RuntimeError('Showcase studio camera is missing')
    if 'Review camera' not in scene.objects:
        raise RuntimeError('Original review camera is missing')
    for name in RESEARCH:
        comparison = bpy.data.collections.get(name)
        if not comparison or not comparison.hide_render or not comparison.hide_viewport:
            raise RuntimeError(f'Expected hidden research collection: {name}')

    roots = {obj.get('showcase_identity'): obj for obj in collection.objects if obj.parent is None}
    if set(roots) != set(IDENTITIES):
        raise RuntimeError('Showcase furniture identities differ')
    receipt = {}
    for identity, root in roots.items():
        parts = list(root.children)
        if root.type != 'EMPTY' or root.library or not parts or root.hide_render or root.hide_get():
            raise RuntimeError(f'{identity}: expected visible editable local furniture root')
        for part in parts:
            if part.type != 'MESH' or part.library or part.data.library or part.data.users != 1:
                raise RuntimeError(f'{identity}: expected independently editable local mesh data')
        baseline_name = IDENTITIES[identity][0]
        if root.get('showcase_baseline_root') != baseline_name or baseline_name not in scene.objects:
            raise RuntimeError(f'{identity}: baseline comparison root missing')
        source_hash = root.get('showcase_asset_source_sha256', '')
        if len(source_hash) != 64 or any(char not in '0123456789abcdef' for char in source_hash):
            raise RuntimeError(f'{identity}: invalid asset source hash')
        materials, images = material_assets(parts)
        if not materials or any(image.source != 'GENERATED' and not image.packed_files for image in images):
            raise RuntimeError(f'{identity}: material or packed image missing')
        receipt[identity] = {
            'root': root.name,
            'location_m': list(root.location),
            'mesh_parts': len(parts),
            'triangles': sum(len(face.vertices) - 2 for part in parts for face in part.data.polygons),
            'materials': sorted(material.name for material in materials),
            'packed_images': sorted(image.name for image in images if image.packed_files),
            'asset_sha256': source_hash,
        }
    return receipt


def build(manifest, output):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    document = read_assets(manifest)
    bpy.ops.wm.open_mainfile(filepath=str(SOURCE))
    scene = bpy.context.scene
    source_objects = list(scene.objects)
    anchors = {identity: scene.objects[baseline].matrix_world.translation.copy()
               for identity, (baseline, _) in IDENTITIES.items()}
    research_collections = [bpy.data.collections[name] for name in RESEARCH]
    research_objects = {obj for collection in research_collections for obj in collection.all_objects}
    if document.get('hide_environment', False):
        for obj in source_objects:
            if obj not in research_objects and obj.type in {'MESH', 'LIGHT', 'CAMERA'}:
                obj.hide_render = True
                obj.hide_set(True)
    for collection in research_collections:
        collection.hide_viewport = collection.hide_render = True

    collection = bpy.data.collections.new(FURNITURE)
    scene.collection.children.link(collection)
    roots, templates = {}, {}
    for identity, asset in document['assets'].items():
        names = tuple(asset.get('object_names', []))
        key = (asset['blend'], names)
        if key not in templates:
            templates[key] = load_meshes(*key)
        roots[identity] = place_asset(identity, asset, collection, anchors[identity], templates[key])

    sys.path.insert(0, str(HERE))
    sys.dont_write_bytecode = True
    from stage_dining_showcase import stage_showcase

    scene['showcase_manifest_sha256'] = digest(manifest)
    scene['showcase_source_sha256'] = digest(SOURCE)
    scene['showcase_purpose'] = 'Finished local furniture assets in the existing editable dining layout'
    bpy.context.view_layer.update()
    stage_showcase(scene, roots)
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    roots['same-table'].select_set(True)
    bpy.context.view_layer.objects.active = roots['same-table']
    bpy.context.view_layer.update()
    print(json.dumps(inspect(scene, manifest), indent=2))
    bpy.ops.wm.save_as_mainfile(filepath=str(output), compress=True, check_existing=False, relative_remap=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('build', 'verify'), default='build')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    output = args.output.resolve()
    manifest = args.manifest.resolve() if args.manifest else None
    if output.suffix.lower() != '.blend':
        parser.error('--output must be a .blend file')
    if args.mode == 'build':
        if not manifest:
            parser.error('--manifest is required to build')
        build(manifest, output)
    else:
        bpy.ops.wm.open_mainfile(filepath=str(output))
        print(json.dumps(inspect(bpy.context.scene, manifest), indent=2))


if __name__ == '__main__':
    main()
