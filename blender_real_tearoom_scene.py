"""Local native tea-room with editable authored parts and retained observations. MIT.

Run in Blender with explicit --scene, --surface, --structural, --output, --mode.
Only the aggregate public-summary.json may be redistributed. Scan-derived native,
mesh, camera and rendered files stay local under the source's applicable rights.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time

import numpy as np

try:
    import bpy
    from mathutils import Matrix, Vector
except ImportError:
    bpy=None

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0,str(Path(__file__).resolve().parent))
from fit_stool import sha256
from build_real_tearoom import SCHEMA, sat_pair, topology, structural_contacts

ROTATION=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)
COMPARISON='Observed RGB-D surface — original world frame (hidden comparison)'
COMPLETED='Authored structural completion — editable identity roots'
STRUCTURES='Measured RoomPlan room envelope — cutaway review'
CAMERAS='Captured cameras — original transforms (hidden)'


def require(condition,message):
    if not condition:raise ValueError(message)


def source_path(base,filename):
    require(isinstance(filename,str) and filename and '\\' not in filename and ':' not in filename,
            'Mesh path must be relative')
    p=PurePosixPath(filename)
    require(not p.is_absolute() and '..' not in p.parts and p.suffix.lower()=='.ply','Unsafe mesh path')
    result=(base/filename).resolve()
    require(result.is_relative_to(base.resolve()) and result.is_file(),'Explicit mesh missing from bundle')
    return result


def save_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def verify_component_mesh(path,part):
    """Bind JSON component coordinates to the actual exported ASCII PLY."""
    lines=path.read_text(encoding='ascii').splitlines();end=lines.index('end_header')
    require(lines[:2]==['ply','format ascii 1.0'],'Expected authored ASCII PLY')
    nv=int(next(r for r in lines[:end] if r.startswith('element vertex')).split()[2])
    nf=int(next(r for r in lines[:end] if r.startswith('element face')).split()[2])
    vertices=np.array([[float(x) for x in r.split()[:3]] for r in lines[end+1:end+1+nv]])
    rows=[[int(x) for x in r.split()] for r in lines[end+1+nv:]]
    require(len(rows)==nf and all(len(r)==4 and r[0]==3 for r in rows),'Invalid triangular PLY')
    faces=np.array([r[1:] for r in rows])
    require(vertices.shape==np.asarray(part['vertices']).shape and np.allclose(vertices,part['vertices'],atol=5e-10,rtol=0)
        and np.array_equal(faces,part['faces']),'JSON geometry differs from bound component PLY')
    from build_real_tearoom import box
    predicted,predicted_faces=box(part['center'],part['dimensions'],part['rotation'])
    require(np.allclose(predicted,part['vertices'],atol=1e-8,rtol=0) and np.array_equal(predicted_faces,part['faces']),
            'Component geometry differs from its declared convex box parameters')


def inputs(scene_path,surface_path,structural_dir):
    scene=json.loads(scene_path.read_text(encoding='utf-8'))
    surface=json.loads(surface_path.read_text(encoding='utf-8'))
    record=json.loads((structural_dir/'structural.json').read_text(encoding='utf-8'))
    require(record.get('schema')==SCHEMA and record.get('units')=='m' and record.get('up_axis')=='Y',
            'Unsupported structural coordinates or schema')
    require(record['input_sha256']['scene']==sha256(scene_path),'Structural scene input changed')
    require(surface.get('schema')=='spatial-scene-lab.surface.v1' and surface.get('units')=='m'
            and surface.get('up_axis')=='Y','Unsupported observed surface coordinates or schema')
    observed_parts=surface.get('parts')
    require(isinstance(observed_parts,list) and observed_parts and all(isinstance(r,dict) for r in observed_parts),
            'Observed surface requires nonempty mesh records')
    identities=[r.get('id') for r in observed_parts]
    require(len(set(identities))==len(identities) and 'environment' in identities
            and set(identities).issubset({'environment',*[r['uuid'] for r in scene['objects']]}),
            'Observed mesh identities must be unique captured object regions plus environment')
    require(surface['input_scene_sha256']==sha256(scene_path),'Observed surface came from a different input scene')
    require(record['observed_reconstruction_accepted'] is False,'Completion must retain failed measured geometry status')
    require(record['layout_after']['passed'] and all(r['passed'] for r in record['floor_support']),
            'Structural collision/floor checks must pass independently of observed geometry gate')
    require(record['structural_contacts']['all_object_contact_graphs_connected'], 'Disconnected structural parts')
    require({r['uuid'] for r in record['objects']}=={r['uuid'] for r in scene['objects']},'Identity roots differ')
    hashes={'scene':sha256(scene_path),'surface':sha256(surface_path),'structural':sha256(structural_dir/'structural.json'),
            'native_builder':sha256(__file__)}
    for filename in ('build_real_tearoom.py','fit_stool.py','blender_scene.py','blender_stool_scene.py','geometry_compare.py'):
        hashes['helper/'+filename]=sha256(Path(__file__).resolve().parent/filename)
    for row in surface['parts']:
        p=source_path(surface_path.parent,row['path'])
        require(sha256(p)==row['sha256'],'Observed mesh changed')
        hashes['observed/'+row['id']]=sha256(p)
    for row in record['objects']:
        for part in row['components']:
            p=source_path(structural_dir,part['mesh'])
            require(sha256(p)==part['mesh_sha256'],'Authored component changed')
            verify_component_mesh(p,part)
            hashes['authored/'+part['key']]=sha256(p)
    return scene,surface,record,hashes


def arrays(obj):
    obj.data.calc_loop_triangles()
    local=np.asarray([v.co[:] for v in obj.data.vertices],float)
    matrix=np.asarray(obj.matrix_world,float)
    return local@matrix[:3,:3].T+matrix[:3,3],np.asarray([t.vertices[:] for t in obj.data.loop_triangles],int)


def snapshot(obj):
    import hashlib
    v,f=arrays(obj)
    from blender_stool_scene import object_snapshot
    details=object_snapshot(obj)
    return {'matrix':[list(r) for r in obj.matrix_world],
        'geometry_sha256':hashlib.sha256(v.astype('<f8').tobytes()+f.astype('<i4').tobytes()).hexdigest(),
        'vertices':len(v),'triangles':len(f),'parent':obj.parent.name if obj.parent else None,
        'properties':{k:str(obj[k]) for k in sorted(obj.keys())},
        'hidden':obj.hide_get(),'hide_render':obj.hide_render,
        'appearance':{'attributes':details['attributes'],'materials':details['materials']}}


def material(name,color):
    mat=bpy.data.materials.new(name);mat.use_nodes=True
    bsdf=mat.node_tree.nodes.get('Principled BSDF')
    c=np.asarray(color,float)/255
    c=np.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4)
    bsdf.inputs['Base Color'].default_value=(*c,1)
    bsdf.inputs['Roughness'].default_value=.72
    return mat


def newmesh(name,vertices,faces,collection,mat):
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(vertices,[],faces);mesh.update()
    obj=bpy.data.objects.new(name,mesh);collection.objects.link(obj);mesh.materials.append(mat)
    return obj


def setup_camera(scene,table,span=3.8):
    center=Vector(table)
    data=bpy.data.cameras.new('Review camera');data.type='ORTHO';data.ortho_scale=span*1.7
    camera=bpy.data.objects.new('Review camera',data);scene.collection.objects.link(camera)
    camera.location=center+Vector((-span,span*.9,span*1.1))
    camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler();scene.camera=camera
    for name,offset,power in [('Key',(1,-1,2),1700),('Fill',(-1,.5,1.5),1000)]:
        ld=bpy.data.lights.new(name,'AREA');ld.energy=power;ld.size=4
        light=bpy.data.objects.new(name,ld);scene.collection.objects.link(light)
        light.location=center+Vector(offset)*span
        light.rotation_euler=(center-light.location).to_track_quat('-Z','Y').to_euler()
    scene.world=bpy.data.worlds.new('Local neutral review');scene.world.color=(.6,.65,.72)
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=16
    scene.render.resolution_x=1400;scene.render.resolution_y=1000;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.film_transparent=False
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                area.spaces.active.overlay.show_overlays=False
                area.spaces.active.shading.type='MATERIAL'
                area.spaces.active.region_3d.view_location=center
                area.spaces.active.region_3d.view_distance=span*1.8
                area.spaces.active.region_3d.view_rotation=camera.rotation_euler.to_quaternion()
    return camera


def actual_parts(record):
    result=[]
    for row in record['objects']:
        for p in row['components']:
            obj=bpy.data.objects[row['name']+' / '+p['name']]
            v,f=arrays(obj)
            # Native output is rotated back to the original Y-up frame only
            # for comparison. Cameras and stored source arrays remain intact.
            result.append({**p,'vertices':v@ROTATION,'faces':f,'rotation':np.asarray(p['rotation'])})
    return result


def geometric_topology(vertices,faces):
    """Exact-coordinate seam welding for glTF normal/color vertex splits.

    No tolerance merge is performed. Oriented triangle-instance comparison
    independently checks multiplicity, winding and geometry before GLB use.
    """
    unique,inverse=np.unique(np.asarray(vertices,float),axis=0,return_inverse=True)
    return topology(unique,inverse[np.asarray(faces,int)])


def certify_actual(record):
    parts=actual_parts(record)
    lookup={p['key']:p for p in parts}
    checks=[]
    minimum=float('inf')
    for row in record['layout_after']['pairs']:
        a,b=[lookup[k] for k in row['parts']]
        checked=sat_pair(a,b,clearance=.00499)
        if row['intended_nested_assembly']:
            continue
        require(checked['passed'],'Actual native component collision differs from certified layout')
        minimum=min(minimum,checked['maximum_sat_projection_gap_m']);checks.append(checked)
    floor=[abs(float(p['vertices'][:,1].min()-record['floor_y_m'])) for p in parts if p['name'].startswith('leg_')]
    require(max(floor)<=.003,'Actual native feet moved away from floor')
    require(all(geometric_topology(p['vertices'],p['faces'])['passed'] for p in parts),'Actual native components lost closed volume')
    contacts=structural_contacts(parts)
    require(contacts['all_object_contact_graphs_connected'],'Actual native contact graph became disconnected')
    return {'passed':True,'verified_non_nested_component_pairs':len(checks),
        'minimum_actual_projection_gap_m':minimum,'maximum_actual_absolute_floor_gap_m':max(floor),
        'closed_components':len(parts),'connected_object_assemblies':len(contacts['objects']),
        'same_object_contact_pairs':contacts['contact_pair_count']}


def edit_checks(record):
    bpy.context.preferences.edit.use_global_undo=True
    results=[]
    for row in record['objects']:
        name=row['name']+' — editable completion'
        root=bpy.data.objects[name]
        initial=np.asarray(root.matrix_world).copy()
        children={obj.name:arrays(obj)[0] for obj in root.children}
        other={obj.name:snapshot(obj)['geometry_sha256'] for obj in bpy.context.scene.objects
               if obj.type=='MESH' and obj.parent!=root}
        for obj in bpy.context.scene.objects:obj.select_set(False)
        root.select_set(True);bpy.context.view_layer.objects.active=root
        require(root.select_get(),'Root selection failed')
        for child in root.children:
            child.select_set(True);require(child.select_get(),'Part selection failed');child.select_set(False)
        bpy.ops.ed.undo_push(message=name+' before edit')
        root.location+=Vector((.12,.08,0));root.rotation_euler.z+=np.deg2rad(13);root.scale.x*=1.04
        bpy.context.view_layer.update()
        edited=np.asarray(root.matrix_world).copy();delta=edited@np.linalg.inv(initial)
        error=max(float(np.max(np.abs(arrays(bpy.data.objects[k])[0]-(v@delta[:3,:3].T+delta[:3,3])))) for k,v in children.items())
        require(error<1e-6,'Root edit failed to transform all children coherently')
        require(all(snapshot(bpy.data.objects[k])['geometry_sha256']==h for k,h in other.items()),
                'Root edit changed unrelated geometry')
        bpy.ops.ed.undo_push(message=name+' after edit');bpy.ops.ed.undo()
        require(np.array_equal(np.asarray(bpy.data.objects[name].matrix_world),initial),'Undo did not restore root')
        bpy.ops.ed.redo();require(np.array_equal(np.asarray(bpy.data.objects[name].matrix_world),edited),'Redo failed')
        bpy.ops.ed.undo();require(np.array_equal(np.asarray(bpy.data.objects[name].matrix_world),initial),'Final undo failed')
        results.append({'name':row['name'],'passed':True,'parts':len(children),
            'maximum_affine_vertex_error_m':error,'undo_redo_passed':True,'unrelated_geometry_unchanged':True})
    return results


def build(scene_path,surface_path,structural_dir,output,*,render=True):
    started=time.time();scene,surface,record,hashes=inputs(scene_path,surface_path,structural_dir)
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),'Output must be new or empty')
    output.mkdir(parents=True,exist_ok=True)
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    collections={name:bpy.data.collections.new(name) for name in (COMPARISON,COMPLETED,STRUCTURES,CAMERAS)}
    for c in collections.values():bpy.context.scene.collection.children.link(c)
    observed=[]
    for row in surface['parts']:
        path=source_path(surface_path.parent,row['path']);bpy.ops.wm.ply_import(filepath=str(path))
        obj=bpy.context.active_object;obj.name='Observed / '+row['name']
        obj.matrix_world=Matrix(((1,0,0,0),(0,0,-1,0),(0,1,0,0),(0,0,0,1)))
        for c in tuple(obj.users_collection):c.objects.unlink(obj)
        collections[COMPARISON].objects.link(obj)
        obj['source_identity']=row['id'];obj['source_ply_sha256']=row['sha256'];obj['provenance_role']='observed_rgbd_surface'
        from blender_scene import add_material
        add_material(obj);observed.append(obj)
    roots=[];authored=[]
    for row in record['objects']:
        root=bpy.data.objects.new(row['name']+' — editable completion',None)
        collections[COMPLETED].objects.link(root)
        source=next(o for o in scene['objects'] if o['uuid']==row['uuid'])
        matrix=np.asarray(source['transform_world'],float);converted=np.eye(4)
        converted[:3,:3]=ROTATION@matrix[:3,:3];converted[:3,3]=ROTATION@(matrix[:3,3]+row['authored_translation_world_m'])
        root.matrix_world=Matrix(converted);root.empty_display_size=.15
        root['source_identity']=row['uuid'];root['category_prior']=row['category_prior']
        root['category_is_ground_truth']=False;root['observed_reconstruction_accepted']=False
        root['original_transform_world_json']=json.dumps(source['transform_world'])
        root['authored_translation_world_json']=json.dumps(row['authored_translation_world_m'])
        roots.append(root)
        for part in row['components']:
            v=np.asarray(part['vertices'])@ROTATION.T
            obj=newmesh(row['name']+' / '+part['name'],v,part['faces'],collections[COMPLETED],material(row['name']+' '+part['name'],part['color_rgb']))
            matrix_world=obj.matrix_world.copy();obj.parent=root;obj.matrix_world=matrix_world
            obj['source_identity']=row['uuid'];obj['part_identity']=part['key']
            obj['provenance_role']=part['provenance_role'];obj['source_ply_sha256']=part['mesh_sha256']
            obj['parameters_json']=json.dumps({'dimensions_m':part['dimensions'],'rotation_world':part['rotation']})
            obj['editability']='Object/component transforms; changing dimensions does not regenerate a CAD construction graph.'
            authored.append(obj)
    envelopes=[]
    for row in scene['structures']:
        t=np.asarray(row['transform_world'],float)
        if row.get('geometry_kind')=='mesh':v=np.asarray(row['vertices_local_m'])@t[:3,:3].T+t[:3,3];f=row['triangles']
        else:
            from build_real_tearoom import box
            v,f=box(t[:3,3],row['dimensions_m'],t[:3,:3])
        color=(150,155,155) if row['category']=='Floor' else (235,232,217)
        obj=newmesh('Envelope / '+row['name'],v@ROTATION.T,f,collections[STRUCTURES],material(row['name'],color))
        obj['source_identity']=row['uuid'];obj['provenance_role']='measured_roomplan_envelope'
        obj['cutaway_hidden']=row['category']!='Floor';envelopes.append(obj)
    captured=[]
    cv_to_blender=np.diag([1,-1,-1,1]);world_rotation=np.eye(4);world_rotation[:3,:3]=ROTATION
    for frame in scene['frames']:
        data=bpy.data.cameras.new('Captured '+str(frame['frame_index']));data.lens=25
        obj=bpy.data.objects.new(data.name,data);collections[CAMERAS].objects.link(obj)
        obj.matrix_world=Matrix(world_rotation@np.asarray(frame['camera_to_world_cv'])@cv_to_blender)
        obj['frame_index']=frame['frame_index'];obj['original_camera_to_world_cv_json']=json.dumps(frame['camera_to_world_cv'])
        obj['intrinsics_depth_json']=json.dumps(frame['intrinsics_depth'])
        obj['provenance_role']='captured_camera';obj.hide_set(True);obj.hide_render=True;captured.append(obj)
    units=bpy.context.scene.unit_settings;units.system='METRIC';units.scale_length=1.;units.length_unit='METERS'
    center=np.mean([np.asarray(r.location) for r in roots],axis=0)
    camera=setup_camera(bpy.context.scene,center)
    for obj in authored+envelopes:obj.hide_render=True
    if render:
        bpy.context.scene.render.filepath=str(output/'tea-room-before.png');bpy.ops.render.render(write_still=True)
    for obj in observed:obj.hide_set(True);obj.hide_render=True
    for obj in authored:obj.hide_set(False);obj.hide_render=False
    for obj in envelopes:obj.hide_set(obj['cutaway_hidden']);obj.hide_render=obj['cutaway_hidden']
    if render:
        bpy.context.scene.render.filepath=str(output/'tea-room-complete.png');bpy.ops.render.render(write_still=True)
        states={o.name:(o.hide_get(),o.hide_render) for o in observed+authored+envelopes}
        original_camera=(camera.location.copy(),camera.rotation_euler.copy(),camera.data.ortho_scale)
        focus_ids={r['uuid'] for r in record['objects'] if r['category_prior'] in ('Table','Chair')}
        focus_vertices=np.concatenate([arrays(o)[0] for o in authored if o['source_identity'] in focus_ids])
        focus_center=Vector((focus_vertices.min(0)+focus_vertices.max(0))/2)
        camera.location=focus_center+Vector((-3.8,3.42,4.18))
        camera.rotation_euler=(focus_center-camera.location).to_track_quat('-Z','Y').to_euler()
        camera.data.ortho_scale=4.3
        for o in observed+authored+envelopes:o.hide_set(True);o.hide_render=True
        for o in observed:
            if o['source_identity'] in focus_ids:o.hide_set(False);o.hide_render=False
        bpy.context.scene.render.filepath=str(output/'tea-room-table-before.png');bpy.ops.render.render(write_still=True)
        for o in observed:o.hide_set(True);o.hide_render=True
        for o in authored:
            if o['source_identity'] in focus_ids:o.hide_set(False);o.hide_render=False
        bpy.context.scene.render.filepath=str(output/'tea-room-table-complete.png');bpy.ops.render.render(write_still=True)
        for name,(hidden,hidden_render) in states.items():
            obj=bpy.data.objects[name];obj.hide_set(hidden);obj.hide_render=hidden_render
        camera.location,camera.rotation_euler,camera.data.ortho_scale=original_camera
    for obj in bpy.context.scene.objects:obj.select_set(False)
    target=next(r for r in roots if r['source_identity']==record['objects'][-1]['uuid'])
    target.select_set(True);bpy.context.view_layer.objects.active=target
    bpy.context.scene['world_frame']='Original captured Y-up world rotated exactly once to Blender Z-up; metres.'
    bpy.context.scene['completion_status']='Authored structure, original measured reconstruction gate failed; inspect retained comparison layer.'
    bpy.context.scene['source_rights']='All derived assets local only; no redistribution permission supplied by repository.'
    bpy.ops.wm.save_as_mainfile(filepath=str(output/'complete-tea-room.blend'),check_existing=False)
    snapshots={o.name:snapshot(o) for o in observed+authored+envelopes}
    certification=certify_actual(record)
    # Export full measured envelope even though wall meshes are hidden in the
    # saved cutaway view. The hidden original RGB-D comparison stays excluded.
    for obj in bpy.context.scene.objects:obj.select_set(False)
    for obj in authored+envelopes+roots:obj.hide_set(False);obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(output/'complete-tea-room.glb'),export_format='GLB',use_selection=True,
        export_yup=True,export_extras=True,export_animations=False,export_cameras=False,export_lights=False)
    require(inputs(scene_path,surface_path,structural_dir)[3]==hashes,'Inputs changed while building')
    verification={'schema':'alvenx.spatial.real-tearoom-native.v1','status':'pending_fresh_reopen',
        'build_process_id':os.getpid(),'input_sha256':hashes,'snapshots':snapshots,
        'captured_camera_snapshots':{o.name:{'matrix':[list(r) for r in o.matrix_world],
            'source_camera_json':o['original_camera_to_world_cv_json']} for o in captured},
        'object_roots':len(roots),'authored_components':len(authored),'observed_comparison_meshes':len(observed),
        'measured_envelope_meshes':len(envelopes),'captured_cameras':len(captured),
        'native_build_geometry':certification,'build_seconds':time.time()-started,
        'observed_reconstruction_accepted':False,'source_assets_local_only':True,
        'captured_world_frame_changed':False,'captured_camera_transforms_retained':len(captured),
        'render_policy':'Full-room before/after and same-camera Table0 plus four Chair-prior roots before/after. Focus views temporarily hide non-target objects; saved geometry and visibility restored.',
        'artifacts':{name:{'sha256':sha256(output/name),'bytes':(output/name).stat().st_size}
            for name in ('complete-tea-room.blend','complete-tea-room.glb','tea-room-before.png','tea-room-complete.png',
                         'tea-room-table-before.png','tea-room-table-complete.png') if (output/name).is_file()}}
    save_json(output/'scene-verification.json',verification)


def reopen(scene_path,surface_path,structural_dir,output):
    started=time.time();_,_,record,hashes=inputs(scene_path,surface_path,structural_dir)
    receipt=json.loads((output/'scene-verification.json').read_text(encoding='utf-8'))
    require(receipt['build_process_id']!=os.getpid(),'Verification requires a new Blender process')
    require(receipt['input_sha256']==hashes,'Input bytes changed after build')
    for name,row in receipt['artifacts'].items():require(sha256(output/name)==row['sha256'],'Artifact bytes changed')
    bpy.ops.wm.open_mainfile(filepath=str(output/'complete-tea-room.blend'),load_ui=True,use_scripts=False)
    require(bpy.context.scene.unit_settings.system=='METRIC' and bpy.context.scene.unit_settings.scale_length==1.,'World scale changed')
    for name,expected in receipt['snapshots'].items():
        require(json.dumps(snapshot(bpy.data.objects[name]),sort_keys=True)==json.dumps(expected,sort_keys=True),
                'Saved geometry/provenance/visibility changed: '+name)
    for name,row in receipt['captured_camera_snapshots'].items():
        obj=bpy.data.objects[name]
        require([list(r) for r in obj.matrix_world]==row['matrix'] and obj['original_camera_to_world_cv_json']==row['source_camera_json'],'Captured camera changed')
    native=certify_actual(record);edits=edit_checks(record);restored=certify_actual(record)
    # Only authored and measured envelope meshes go into the full GLB.
    expected={o.name:(arrays(o),{k:str(o[k]) for k in ('source_identity','provenance_role')})
        for o in bpy.context.scene.objects if o.type=='MESH' and 'Observed / ' not in o.name}
    roots={r['name']+' — editable completion' for r in record['objects']}
    for obj in list(bpy.data.objects):bpy.data.objects.remove(obj,do_unlink=True)
    bpy.ops.import_scene.gltf(filepath=str(output/'complete-tea-room.glb'))
    returned={o.name:o for o in bpy.context.scene.objects if o.type=='MESH'}
    require(set(returned)==set(expected),'GLB inventory omitted/added geometry or exported hidden observations')
    require(all(name in bpy.data.objects for name in roots),'GLB lost object roots')
    from geometry_compare import compare_triangle_geometry
    maximum=0.
    for name,(original,properties) in expected.items():
        obj=returned[name];comp=compare_triangle_geometry(*original,*arrays(obj),tolerance=5e-6)
        require(comp['status']=='pass','GLB oriented triangle geometry differs: '+name)
        # Oriented corner error is returned by the shared verifier.
        maximum=max(maximum,float(comp['maximum_matched_corner_distance_m']))
        require(all(str(obj.get(k))==v for k,v in properties.items()),'GLB identity/provenance changed')
    glb=certify_actual(record)
    require(inputs(scene_path,surface_path,structural_dir)[3]==hashes,'Reopen modified input bytes')
    for name,row in receipt['artifacts'].items():require(sha256(output/name)==row['sha256'],'Reopen modified artifacts')
    receipt.update(status='passed',fresh_process_reopen=True,native_reopen_geometry=native,
        edit_undo_redo=edits,after_undo_geometry=restored,glb_geometry=glb,
        glb_visible_meshes=len(returned),glb_maximum_oriented_corner_error_m=maximum,
        reopen_seconds=time.time()-started)
    save_json(output/'scene-verification.json',receipt)
    public={'schema':'alvenx.spatial.real-tearoom-native-summary.v1','status':'passed',
        'observed_reconstruction_accepted':False,'source_assets_local_only':True,
        'captured_world_frame_changed':False,'captured_camera_transforms_retained':receipt['captured_cameras'],
        'object_roots':receipt['object_roots'],'authored_closed_components':receipt['authored_components'],
        'observed_comparison_meshes':receipt['observed_comparison_meshes'],
        'measured_envelope_meshes':receipt['measured_envelope_meshes'],'captured_cameras':receipt['captured_cameras'],
        'glb_meshes':len(returned),'all_roots_edit_undo_redo_passed':all(r['passed'] for r in edits),
        'maximum_edit_affine_vertex_error_m':max(r['maximum_affine_vertex_error_m'] for r in edits),
        'glb_maximum_oriented_corner_error_m':maximum,'fresh_process_reopen':True,
        'native_geometry':native,'after_undo_geometry':restored,'glb_geometry':glb,
        'build_seconds':receipt['build_seconds'],'reopen_seconds':receipt['reopen_seconds'],
        'input_sha256':{k:v for k,v in hashes.items() if k in ('scene','surface','structural','native_builder')},
        'artifact_sha256':{k:v['sha256'] for k,v in receipt['artifacts'].items()},
        'limits':'Prior-driven authored scene; raw observation layer/cameras retained unchanged. No independent unseen-geometry accuracy claim or automatic CAD regeneration. All scan-derived assets stay local.'}
    save_json(output/'public-summary.json',public)
    print(json.dumps(public,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','surface','structural','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=('build','reopen'),required=True)
    p.add_argument('--no-render',action='store_true',help='Skip optional review PNG generation; native and GLB checks still run.')
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    paths=[getattr(args,k).resolve() for k in ('scene','surface','structural','output')]
    require(not any(paths[-1]==p or p.is_relative_to(paths[-1]) for p in paths[:-1]),'Output cannot contain input paths')
    if args.mode=='build':build(*paths,render=not args.no_render)
    else:reopen(*paths)


if __name__=='__main__':main()
