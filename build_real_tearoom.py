"""Author a local editable tea-room assembly from measurements and explicit priors.

This is structural completion, not a claim that unseen geometry was recovered.
Only Table0's upper plane is fitted to the frozen training frames. Original
observations, RoomPlan transforms and cameras are never rewritten. MIT.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys
import time

import numpy as np

from audit_real_tabletop import load_inputs, fit_plane, summarize, support_gate
from fit_stool import sha256, validate_output

TABLE_ID = '46C25048-47D3-45BE-BEDF-D3DF0DF758F6'
SCHEMA = 'alvenx.spatial.real-tearoom-structural.v1'
BOX_FACES = np.array([(0,1,3),(0,3,2),(4,6,7),(4,7,5),(0,4,5),(0,5,1),
                      (2,3,7),(2,7,6),(0,2,6),(0,6,4),(1,5,7),(1,7,3)], dtype=int)


def box(center, dimensions, rotation=np.eye(3)):
    center, dimensions, rotation = map(lambda x: np.asarray(x, float), (center, dimensions, rotation))
    if (center.shape != (3,) or dimensions.shape != (3,) or rotation.shape != (3,3)
            or not np.isfinite(np.r_[center, dimensions, rotation.ravel()]).all()
            or np.any(dimensions <= 0) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation),1.,atol=1e-6)):
        raise ValueError('Box needs finite positive dimensions and an orthogonal frame')
    vertices = np.array([[a,b,c] for a in (-1,1) for b in (-1,1) for c in (-1,1)]) * dimensions/2
    faces = BOX_FACES.copy()
    normal = np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
    inward = np.einsum('ij,ij->i',normal,vertices[faces].mean(1)) < 0
    faces[inward] = faces[inward][:,[0,2,1]]
    return vertices @ rotation.T + center, faces


def topology(vertices, faces):
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, int)
    edges = Counter(tuple(sorted((int(a),int(b)))) for f in faces for a,b in zip(f,np.roll(f,-1)))
    directed = Counter((int(a),int(b)) for f in faces for a,b in zip(f,np.roll(f,-1)))
    volume = float(np.einsum('ij,ij->i',vertices[faces[:,0]],
        np.cross(vertices[faces[:,1]],vertices[faces[:,2]])).sum()/6)
    closed = bool(edges and all(n==2 for n in edges.values()))
    wound = all(directed[(a,b)] == directed[(b,a)] == 1 for a,b in edges)
    return {'closed': closed, 'consistent_winding': wound, 'positive_volume_m3': volume,
            'passed': bool(closed and wound and volume > 1e-12)}


def sat_pair(first, second, clearance=.005):
    """Convex oriented-box SAT on actual triangle vertices, not bounding boxes.

    Every pass retains a separating plane for all actual vertices. Cross axes
    catch edge-edge intersections even when neither solid contains a vertex.
    """
    a,b = np.asarray(first['vertices']),np.asarray(second['vertices'])
    ra,rb = np.asarray(first['rotation']),np.asarray(second['rotation'])
    axes = [*ra.T,*rb.T,*[np.cross(x,y) for x in ra.T for y in rb.T]]
    best = (-np.inf,None,None)
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-10:
            continue
        axis = axis/norm
        pa,pb = a@axis,b@axis
        gap = float(pb.min()-pa.max())
        reverse = float(pa.min()-pb.max())
        if reverse > gap:
            gap,axis = reverse,-axis
            pa,pb = -pa,-pb
        if gap > best[0]:
            best = (gap,axis,(float(pa.max()),float(pb.min())))
    gap,axis,projections = best
    return {'parts': [first['key'],second['key']], 'passed': bool(gap >= clearance-1e-10),
            'positive_solid_intersection': bool(gap < -1e-8),
            'maximum_sat_projection_gap_m': gap,
            'separating_plane': {'unit_normal_world': axis.tolist(),
                'first_projection_max_m':projections[0], 'second_projection_min_m':projections[1]}
                if gap > 0 else None}


def point_box_distances(points, part):
    local = (np.asarray(points)-part['center'])@part['rotation']
    q = np.abs(local)-np.asarray(part['dimensions'])/2
    return np.abs(np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(1),0))


def parts_for_object(record, floor, plane=None, table_center=None):
    """Explicit operator-selected families; no geometric learning is asserted."""
    transform=np.asarray(record['transform_world'],float)
    rotation=transform[:3,:3]
    center=transform[:3,3]
    width,height,depth=record['dimensions_m']
    category=record['category']
    bottom=float(center[1]-height/2)
    parts=[]
    def add(name, xyz, dims, role='authored_prior_completion', color=(190,190,185), rot=None):
        r=rotation if rot is None else rot
        c=np.asarray(xyz)@rotation.T+center
        v,f=box(c,dims,r)
        parts.append({'name':name,'key':record['uuid']+'/'+name,'uuid':record['uuid'],
            'vertices':v,'faces':f,'center':c,'dimensions':np.asarray(dims), 'rotation':np.asarray(r),
            'provenance_role':role,'color_rgb':list(color),'editable':True})
    def legs(top_local, count=4, inset=.07):
        halfx=max(width/2-inset,.05);halfz=max(depth/2-inset,.05)
        low=floor-center[1]
        for i,(x,z) in enumerate([(-halfx,-halfz),(-halfx,halfz),(halfx,-halfz),(halfx,halfz)][:count]):
            add('leg_'+str(i+1),(x,(low+top_local)/2,z),(.035,top_local-low+.002,.035),color=(35,38,42))
    if category=='Table':
        a,b,c=plane
        # The visible upper face follows the training plane. Thickness and four
        # supports remain authored assumptions, including their hidden ends.
        normal=np.array([-a,1.,-b]);normal/=np.linalg.norm(normal)
        xaxis=np.array([1.,a,0.]);xaxis/=np.linalg.norm(xaxis)
        zaxis=np.cross(xaxis,normal)
        local_rotation=np.column_stack((xaxis,normal,zaxis))
        thickness=.028
        add('tabletop',(0,c-thickness/2,0),(width,thickness,depth),
            'training_plane_plus_authored_thickness',(210,196,158), rotation@local_rotation)
        # Each leg meets the tilted lower face with a 2 mm contact overlap.
        for i,(x,z) in enumerate([(-width/2+.07,-depth/2+.07),(-width/2+.07,depth/2-.07),
                                  (width/2-.07,-depth/2+.07),(width/2-.07,depth/2-.07)]):
            top=a*x+b*z+c-thickness+.004
            low=floor-center[1]
            add('leg_'+str(i+1),(x,(low+top)/2,z),(.05,top-low,.05),color=(88,84,76))
    elif category=='Chair':
        seat_y=floor-center[1]+min(.45,height*.56)
        color=(35,88,170) if record['name'] in ('Chair1','Chair3') else (37,40,44)
        # RoomPlan box yaw does not specify which end is the chair back. Select
        # its table-facing orientation explicitly from the captured centers.
        back_sign=1 if table_center is None or np.dot(center-np.asarray(table_center),rotation[:,2])>=0 else -1
        add('seat',(0,seat_y,0),(width*.94,.055,depth*.82),color=color)
        back_height=max(height-(seat_y+center[1]-floor)-.03,.14)
        add('back',(0,seat_y+back_height/2,back_sign*depth*.38),(width*.9,back_height,.045),color=color)
        legs(seat_y-.025,inset=.065)
        for i,x in enumerate((-width*.36,width*.36)):
            add('back_support_'+str(i+1),(x,seat_y+.15,back_sign*depth*.36),(.022,.35,.022),color=(30,32,35))
    elif category=='Sofa':
        # RoomPlan calls this Sofa; visual semantics remain a source prior.
        seat_y=floor-center[1]+.43
        add('seat',(0,seat_y,0),(width*.96,.12,depth*.85),color=(145,46,48))
        add('back',(0,seat_y+.22,depth*.38),(width*.96,.43,.09),color=(145,46,48))
        legs(seat_y-.055,inset=.1)
    elif category=='Sink':
        # Open basin built from five closed parts, not a filled bounding box.
        t=.016
        add('basin_bottom',(0,-height/2+t/2,0),(width,t,depth),color=(154,160,166))
        for i,x in enumerate((-width/2+t/2,width/2-t/2)):
            add('basin_side_x'+str(i),(x,0,0),(t,height,depth),color=(154,160,166))
        for i,z in enumerate((-depth/2+t/2,depth/2-t/2)):
            add('basin_side_z'+str(i),(0,0,z),(width-2*t,height,t),color=(154,160,166))
    else:
        t=.02
        # A 3 mm authored installation seam on each side prevents captured
        # adjacent cabinet boxes from becoming intersecting solid panels.
        width-=.006
        # Cabinet/appliance frame has accessible separately editable fronts.
        for name,xyz,dims in [('left',(-width/2+t/2,0,0),(t,height,depth)),
                              ('right',(width/2-t/2,0,0),(t,height,depth)),
                              ('bottom',(0,-height/2+t/2,0),(width-2*t,t,depth)),
                              ('top',(0,height/2-t/2,0),(width-2*t,t,depth)),
                              ('back',(0,0,depth/2-t/2),(width-2*t,height-2*t,t))]:
            add(name,xyz,dims,color=(205,202,190))
        if category=='Dishwasher':
            add('front',(0,0,-depth/2+t/2),(width-2*t,height-2*t,t),color=(220,223,225))
        else:
            for i,x in enumerate((-width/4,width/4)):
                add('door_'+str(i+1),(x,0,-depth/2+t/2),(width/2-t,height-2*t,t),color=(218,215,203))
    return parts


def layout(parts, names):
    rows=[]
    # Sink and enclosing kitchen storage form an explicitly allowed assembly.
    # Still retain all of its intersection diagnostics; never call it separated.
    intended={frozenset(('Sink0','Storage2'))}
    for a,b in combinations(parts,2):
        if a['uuid']==b['uuid']:
            continue
        row=sat_pair(a,b)
        row['intended_nested_assembly']=frozenset((names[a['uuid']],names[b['uuid']])) in intended
        rows.append(row)
    unexpected=[r for r in rows if not r['passed'] and not r['intended_nested_assembly']]
    return {'cross_object_pair_count':len(rows),'unexpected_clearance_failure_count':len(unexpected),
        'unexpected_positive_intersection_count':sum(r['positive_solid_intersection'] for r in unexpected),
        'passed':not unexpected,'required_clearance_m':.005,
        'scope':'Actual closed component vertices and triangle face axes; intended Sink0/Storage2 nesting reported separately.',
        'pairs':rows}


def serialize(part):
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in part.items()}


def structural_contacts(parts):
    """Build a contact graph using exact convex-component SAT intersection."""
    groups={p['uuid'] for p in parts};records=[]
    for identity in sorted(groups):
        own=[p for p in parts if p['uuid']==identity]
        edges=[];graph={p['key']:set() for p in own}
        for a,b in combinations(own,2):
            check=sat_pair(a,b,clearance=0.)
            if check['maximum_sat_projection_gap_m']<=1e-8:
                graph[a['key']].add(b['key']);graph[b['key']].add(a['key'])
                edges.append([a['key'],b['key']])
        reached=set();pending=[own[0]['key']]
        while pending:
            key=pending.pop()
            if key in reached:continue
            reached.add(key);pending.extend(graph[key]-reached)
        records.append({'uuid':identity,'components':len(own),'contact_pairs':edges,
                        'connected':len(reached)==len(own)})
    return {'all_object_contact_graphs_connected':all(r['connected'] for r in records),
            'contact_pair_count':sum(len(r['contact_pairs']) for r in records),'objects':records}


def write_ply(path, part):
    vertices,faces=part['vertices'],part['faces']
    with path.open('x',encoding='ascii',newline='\n') as f:
        f.write('ply\nformat ascii 1.0\ncomment authored completion; not observed geometry\n')
        f.write(f'element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write(f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
        for p in vertices:
            f.write('%.9f %.9f %.9f %d %d %d\n'%(*p,*part['color_rgb']))
        for face in faces:
            f.write('3 %d %d %d\n'%tuple(face))


def build(scene_path, observations_path, output):
    started=time.time()
    hashes={'scene':sha256(scene_path),'observations':sha256(observations_path),'builder':sha256(__file__)}
    helper_hashes={name:sha256(Path(__file__).resolve().parent/name) for name in ('audit_real_tabletop.py','fit_stool.py')}
    scene,index,points,frames,candidates,local,band,train,valid=load_inputs(scene_path,observations_path,TABLE_ID)
    coefficients,bounds=fit_plane(local,frames,train)
    floor_records=[r for r in scene['structures'] if r['category']=='Floor']
    if len(floor_records)!=1:
        raise ValueError('One measured Floor is required')
    floor_record=floor_records[0]
    ft=np.asarray(floor_record['transform_world'])
    fv=np.asarray(floor_record['vertices_local_m'])@ft[:3,:3].T+ft[:3,3]
    floor=float(fv[:,1].max())
    names={r['uuid']:r['name'] for r in scene['objects']}
    table_center=np.asarray(scene['objects'][index]['transform_world'])[:3,3]
    parts=[p for r in scene['objects'] for p in parts_for_object(r,floor,coefficients if r['uuid']==TABLE_ID else None,table_center)]
    before=layout(parts,names)
    original=[p['vertices'].copy() for p in parts]
    # Resolve chair-table interpenetration by a declared authored outward move.
    # No other roots, raw observations, cameras, or captured priors are moved.
    deltas={r['uuid']:np.zeros(3) for r in scene['objects']}
    for obj in scene['objects']:
        if obj['category']!='Chair':
            continue
        owned=[p for p in parts if p['uuid']==obj['uuid']]
        table_parts=[p for p in parts if p['uuid']==TABLE_ID]
        if all(sat_pair(a,b)['passed'] for a in owned for b in table_parts):
            continue
        direction=np.asarray(obj['transform_world'])[:3,3]-table_center
        direction[1]=0; direction/=np.linalg.norm(direction)
        initial=[p['vertices'].copy() for p in owned]
        lo,hi=0.,.4
        for _ in range(20):
            amount=(lo+hi)/2
            for p,v in zip(owned,initial):p['vertices']=v+direction*amount
            if all(sat_pair(a,b)['passed'] for a in owned for b in table_parts):hi=amount
            else:lo=amount
        delta=direction*(hi+.00005)
        deltas[obj['uuid']]=delta
        for p,v in zip(owned,initial):
            p['vertices']=v+delta;p['center']+=delta
    after=layout(parts,names)
    topologies={p['key']:topology(p['vertices'],p['faces']) for p in parts}
    if not all(r['passed'] for r in topologies.values()):
        raise ValueError('Authored component topology is invalid')
    contacts=structural_contacts(parts)
    if not contacts['all_object_contact_graphs_connected']:
        raise ValueError('Authored assembly has a disconnected component')
    supports=[]
    for p in parts:
        if p['name'].startswith('leg_'):
            error=float(p['vertices'][:,1].min()-floor)
            supports.append({'part':p['key'],'minimum_vertex_floor_gap_m':error,'passed':abs(error)<=.003})
    diagnostics=[]
    for i,obj in enumerate(scene['objects']):
        take=candidates[:,i] & (frames>30)
        selected=points[take]
        own=[p for p in parts if p['uuid']==obj['uuid']]
        if len(selected):
            # Shape residual is measured against pre-authoring root positions.
            residual=np.min(np.column_stack([point_box_distances(selected+deltas[obj['uuid']],p) for p in own]),axis=1)
            diagnostics.append({'name':obj['name'],'candidate_validation_points':len(selected),
                'mean_m':float(residual.mean()),'p95_m':float(np.quantile(residual,.95)),
                'within10mm':float(np.mean(residual<=.01)),
                'interpretation':'Candidate-box diagnostic, not instance ground truth or observed reconstruction acceptance.'})
    validation=summarize(local[valid],frames[valid],coefficients)
    output=validate_output(output,[scene_path,observations_path]);output.mkdir(parents=True,exist_ok=True)
    for p in parts:
        filename=names[p['uuid']]+'-'+p['name']+'.ply'
        p['mesh']=filename;write_ply(output/filename,p);p['mesh_sha256']=sha256(output/filename)
    record={'schema':SCHEMA,'units':'m','up_axis':'Y','input_sha256':hashes,
        'helper_sha256':helper_hashes,'rights':'Source and all derived geometry/renders remain local; source has no redistribution grant.',
        'method':'Explicit operator-selected structural families; Table0 upper plane alone uses frozen training frames <=30.',
        'observed_reconstruction_accepted':False,'floor_y_m':floor,
        'tabletop':{'coefficients_local':coefficients.tolist(),'training_bounds_xz_m':bounds,
            'validation':validation,'support_gate_passed':support_gate(validation),
            'unseen_geometry':'28 mm thickness and four 50 mm rectangular legs are authored assumptions, not measured truth.'},
        'original_object_priors':scene['objects'],'camera_frame_count':len(scene['frames']),
        'objects':[{'uuid':o['uuid'],'name':o['name'],'category_prior':o['category'],
            'category_is_ground_truth':False,'authored_translation_world_m':deltas[o['uuid']].tolist(),
            'components':[serialize(p) for p in parts if p['uuid']==o['uuid']]} for o in scene['objects']],
        'layout_before':before,'layout_after':after,'component_topology':topologies,'structural_contacts':contacts,'floor_support':supports,
        'observation_diagnostics':diagnostics,'elapsed_seconds':time.time()-started}
    if hashes!= {'scene':sha256(scene_path),'observations':sha256(observations_path),'builder':sha256(__file__)}:
        raise ValueError('Input bytes changed during completion')
    if helper_hashes!={name:sha256(Path(__file__).resolve().parent/name) for name in helper_hashes}:
        raise ValueError('Completion helper bytes changed during execution')
    (output/'structural.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    summary={'schema':'alvenx.spatial.real-tearoom-structural-summary.v1','status':'authored_completion',
        'source_sha256':hashes,'helper_sha256':helper_hashes,'object_roots':len(scene['objects']),'closed_components':len(parts),
        'component_triangles':sum(len(p['faces']) for p in parts),'camera_frames':len(scene['frames']),
        'observed_reconstruction_accepted':False,'tabletop_validation':{k:validation[k] for k in ('count','mean_m','p95_m','within10mm')},
        'tabletop_support_gate_passed':support_gate(validation),
        'layout_before_unexpected_intersections':before['unexpected_positive_intersection_count'],
        'layout_after_unexpected_intersections':after['unexpected_positive_intersection_count'],
        'layout_after_unexpected_clearance_failures':after['unexpected_clearance_failure_count'],
        'layout_after_passed':after['passed'],'cross_object_pairs':after['cross_object_pair_count'],
        'contact_graphs_connected':contacts['all_object_contact_graphs_connected'],'same_object_contact_pairs':contacts['contact_pair_count'],
        'maximum_authored_root_translation_m':max(float(np.linalg.norm(v)) for v in deltas.values()),
        'floor_support_parts':len(supports),'maximum_absolute_floor_gap_m':max(abs(r['minimum_vertex_floor_gap_m']) for r in supports),
        'all_floor_support_passed':all(r['passed'] for r in supports),'observation_diagnostics':diagnostics,
        'limits':'Prior-driven authored completion; only tabletop plane fitted. Category labels and unseen geometry are not ground truth. All scan-derived assets local only.'}
    (output/'public-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--observations',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();summary=build(a.scene,a.observations,a.output);print(json.dumps(summary,indent=2))
    return 0 if summary['layout_after_passed'] and summary['all_floor_support_passed'] and summary['contact_graphs_connected'] else 2


if __name__=='__main__':
    sys.exit(main())
