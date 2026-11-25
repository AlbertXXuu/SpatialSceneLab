"""Set up the editable dining scene's studio presentation."""

import bpy
from mathutils import Vector


def stage_showcase(scene, roots):
    """Frame placed furniture in Blender world metres with a ground at Z = 0."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    corners = []
    for root in roots.values():
        for obj in (root, *root.children_recursive):
            if obj.type == 'MESH':
                evaluated = obj.evaluated_get(depsgraph)
                corners.extend(evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box)
    if not corners:
        raise ValueError('Studio framing requires placed furniture meshes')

    lower = Vector([min(p[axis] for p in corners) for axis in range(3)])
    upper = Vector([max(p[axis] for p in corners) for axis in range(3)])
    target = (lower + upper) / 2

    studio = bpy.data.collections.new('Dining Studio')
    scene.collection.children.link(studio)
    plane = bpy.data.meshes.new('Dining Studio Ground')
    plane.from_pydata([(-60, -60, 0), (60, -60, 0), (60, 60, 0), (-60, 60, 0)], [], [(0, 1, 2, 3)])
    ground = bpy.data.objects.new(plane.name, plane)
    studio.objects.link(ground)
    ground.location = (target.x, target.y, 0)
    finish = bpy.data.materials.new('Warm gray studio ground')
    finish.use_nodes = True
    finish.diffuse_color = (.46, .41, .36, 1)
    surface = finish.node_tree.nodes.get('Principled BSDF')
    surface.inputs['Base Color'].default_value = finish.diffuse_color
    surface.inputs['Roughness'].default_value = .76
    plane.materials.append(finish)

    world = bpy.data.worlds.new('Dining Studio World')
    world.use_nodes = True
    background = world.node_tree.nodes.get('Background')
    background.inputs['Color'].default_value = (.44, .42, .40, 1)
    background.inputs['Strength'].default_value = .25
    scene.world = world

    for label, offset, energy, size in (
        ('Key', (-2.7, -3.2, 4.4), 1050, 3.4),
        ('Fill', (3.2, -1.4, 2.8), 320, 3.2),
        ('Rim', (1.0, 3.3, 3.6), 680, 2.7),
    ):
        light = bpy.data.lights.new(f'Dining Studio {label}', 'AREA')
        light.energy, light.shape, light.size = energy, 'DISK', size
        lamp = bpy.data.objects.new(light.name, light)
        studio.objects.link(lamp)
        lamp.location = target + Vector(offset)
        lamp.rotation_euler = (target - lamp.location).to_track_quat('-Z', 'Y').to_euler()

    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x, scene.render.resolution_y = (1500, 1050)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'AgX'

    camera = bpy.data.cameras.new('Dining Studio Camera')
    camera.type, camera.sensor_fit, camera.clip_end = 'ORTHO', 'HORIZONTAL', 130
    camera_obj = bpy.data.objects.new(camera.name, camera)
    studio.objects.link(camera_obj)
    offset = Vector((3.0, -4.8, 2.15))
    rotation = (-offset).to_track_quat('-Z', 'Y')
    screen_corners = [rotation.inverted() @ (corner - target) for corner in corners]
    left, right = min(p.x for p in screen_corners), max(p.x for p in screen_corners)
    bottom, top = min(p.y for p in screen_corners), max(p.y for p in screen_corners)
    target += rotation @ Vector(((left + right) / 2, (bottom + top) / 2, 0))
    aspect = scene.render.resolution_x / scene.render.resolution_y
    camera.ortho_scale = max(right - left, (top - bottom) * aspect) * 1.20
    camera_obj.location, camera_obj.rotation_euler = target + offset, rotation.to_euler()
    scene.camera = camera_obj

    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                space.shading.type = 'SOLID'
                space.shading.color_type = 'MATERIAL'
                space.overlay.show_floor = False
                space.overlay.show_extras = False
                space.region_3d.view_location = target
                space.region_3d.view_rotation = rotation
                space.region_3d.view_distance = camera.ortho_scale
                space.region_3d.view_camera_zoom = 0
                space.region_3d.view_perspective = 'CAMERA'
