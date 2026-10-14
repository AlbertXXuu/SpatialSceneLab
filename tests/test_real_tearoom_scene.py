"""Authored-scene invariants and real geometric negative controls, no raw scan."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io

import numpy as np

import build_real_tearoom as author
import blender_real_tearoom_scene as native


def part(center,dimensions,rotation=np.eye(3),name='part'):
    vertices,faces=author.box(center,dimensions,rotation)
    return {'key':name,'vertices':vertices,'faces':faces,'rotation':rotation}


class GeometricCertificates(unittest.TestCase):
    def test_closed_box_positive_volume(self):
        v,f=author.box([3,-1,4],[.2,.3,.4])
        result=author.topology(v,f)
        self.assertTrue(result['passed']);self.assertAlmostEqual(result['positive_volume_m3'],.024)

    def test_missing_face_rejected(self):
        v,f=author.box([0,0,0],[1,1,1])
        self.assertFalse(author.topology(v,f[:-1])['passed'])

    def test_reversed_face_rejected(self):
        v,f=author.box([0,0,0],[1,1,1]);f[0]=f[0][::-1]
        self.assertFalse(author.topology(v,f)['passed'])

    def test_negative_dimension_rejected(self):
        with self.assertRaises(ValueError):author.box([0,0,0],[1,-1,1])

    def test_nonrigid_frame_rejected(self):
        with self.assertRaises(ValueError):author.box([0,0,0],[1,1,1],np.diag([1,2,1]))

    def test_reflection_frame_rejected(self):
        with self.assertRaises(ValueError):author.box([0,0,0],[1,1,1],np.diag([-1,1,1]))

    def test_projection_certificate_every_actual_vertex(self):
        a=part([0,0,0],[1,1,1],name='a');b=part([1.007,0,0],[1,1,1],name='b')
        result=author.sat_pair(a,b);self.assertTrue(result['passed'])
        proof=result['separating_plane'];normal=np.asarray(proof['unit_normal_world'])
        gap=(b['vertices']@normal).min()-(a['vertices']@normal).max()
        self.assertGreaterEqual(gap,.005)

    def test_gap_under_five_mm_rejected(self):
        self.assertFalse(author.sat_pair(part([0,0,0],[1,1,1]),part([1.004,0,0],[1,1,1]))['passed'])

    def test_touching_does_not_pass_clearance(self):
        result=author.sat_pair(part([0,0,0],[1,1,1]),part([1,0,0],[1,1,1]))
        self.assertFalse(result['passed']);self.assertFalse(result['positive_solid_intersection'])

    def test_edge_crossing_detected_without_contained_vertex(self):
        a=part([0,0,0],[3,.1,.1]);b=part([0,0,0],[.1,.1,3])
        # Every vertex lies outside the other solid, but the cross contains an
        # interior volume at the origin. Vertex-only collision misses this.
        self.assertTrue(np.all(np.max(np.abs(a['vertices'])-np.array([.1,.1,3])/2,axis=1)>0))
        self.assertTrue(np.all(np.max(np.abs(b['vertices'])-np.array([3,.1,.1])/2,axis=1)>0))
        self.assertTrue(author.sat_pair(a,b)['positive_solid_intersection'])

    def test_overlapping_aabbs_with_separate_rotated_solids(self):
        t=np.pi/4;r=np.array([[np.cos(t),0,np.sin(t)],[0,1,0],[-np.sin(t),0,np.cos(t)]])
        a=part([0,0,0],[2,.2,.1],r)
        offset=r[:,2]*.2;b=part(offset,[2,.2,.1],r)
        self.assertTrue(np.all(a['vertices'].max(0)>=b['vertices'].min(0)))
        self.assertTrue(np.all(b['vertices'].max(0)>=a['vertices'].min(0)))
        self.assertTrue(author.sat_pair(a,b)['passed'])

    def test_single_rotation_preserves_distance_and_orientation(self):
        r=native.ROTATION
        self.assertAlmostEqual(np.linalg.det(r),1.)
        np.testing.assert_array_equal(r@np.array([1,2,3]),[1,-3,2])

    def test_glb_exact_normal_seams_preserve_geometric_closure(self):
        vertices,faces=author.box([0,0,0],[1,1,1])
        split=vertices[faces].reshape(-1,3);indices=np.arange(len(split)).reshape(-1,3)
        self.assertFalse(author.topology(split,indices)['passed'])
        self.assertTrue(native.geometric_topology(split,indices)['passed'])
        self.assertFalse(native.geometric_topology(split,indices[:-1])['passed'])

    def test_geometric_closure_does_not_merge_close_distinct_corners(self):
        vertices,faces=author.box([0,0,0],[1,1,1])
        split=vertices[faces].reshape(-1,3);indices=np.arange(len(split)).reshape(-1,3)
        split[0,0]+=1e-7
        self.assertFalse(native.geometric_topology(split,indices)['passed'])


class AuthoringProvenance(unittest.TestCase):
    def record(self,category='Chair'):
        return {'uuid':'test','name':'Chair1','category':category,'dimensions_m':[.5,.85,.55],
                'transform_world':np.eye(4).tolist()}

    def test_prior_components_never_claim_observed_geometry(self):
        parts=author.parts_for_object(self.record(),-.425,table_center=[0,0,1])
        self.assertEqual(len(parts),8)
        self.assertTrue(all(p['provenance_role']=='authored_prior_completion' for p in parts))
        self.assertTrue(all(author.topology(p['vertices'],p['faces'])['passed'] for p in parts))

    def test_chair_back_selected_outward_from_table(self):
        parts=author.parts_for_object(self.record(),-.425,table_center=[0,0,1])
        self.assertLess(next(p for p in parts if p['name']=='back')['center'][2],0)

    def test_tabletop_plane_role_is_not_fully_observed(self):
        parts=author.parts_for_object(self.record('Table'),-.425,plane=[.001,-.002,.35])
        self.assertEqual(len(parts),5)
        self.assertEqual(parts[0]['provenance_role'],'training_plane_plus_authored_thickness')
        self.assertTrue(all(p['provenance_role']=='authored_prior_completion' for p in parts[1:]))

    def test_sink_open_basin_parts(self):
        parts=author.parts_for_object(self.record('Sink'),-.425)
        self.assertEqual(len(parts),5)
        self.assertTrue(all(author.topology(p['vertices'],p['faces'])['passed'] for p in parts))

    def test_ply_complete_triangles_with_colors(self):
        value=author.parts_for_object(self.record('Table'),-.425,plane=[0,0,.35])[0]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'part.ply';author.write_ply(path,value)
            text=path.read_text()
            self.assertIn('authored completion; not observed geometry',text)
            self.assertIn('element face 12',text)
            with self.assertRaises(FileExistsError):author.write_ply(path,value)

    def test_safe_source_path(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);(base/'part.ply').write_text('x')
            self.assertEqual(native.source_path(base,'part.ply'),(base/'part.ply').resolve())
            for value in ('../part.ply','/part.ply','C:/part.ply','a\\part.ply','file.json'):
                with self.assertRaises(ValueError):native.source_path(base,value)

    def test_candidate_distance_remains_translation_invariant(self):
        p=author.parts_for_object(self.record(),-.425)[0]
        samples=np.array([[0,.02,0],[0,.3,.3]])
        before=author.point_box_distances(samples,p)
        delta=np.array([.3,0,.2]);p['center']+=delta
        np.testing.assert_allclose(before,author.point_box_distances(samples+delta,p))

    def test_disconnected_component_detected(self):
        parts=author.parts_for_object(self.record(),-.425)
        parts[0]['vertices']+=np.array([0,3,0])
        self.assertFalse(author.structural_contacts(parts)['all_object_contact_graphs_connected'])

    def test_ply_json_coordinate_and_parameter_binding(self):
        value=author.parts_for_object(self.record('Table'),-.425,plane=[0,0,.35])[0]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'part.ply';author.write_ply(path,value)
            native.verify_component_mesh(path,value)
            bad=author.serialize(value);bad['vertices'][0][0]+=.002
            with self.assertRaises(ValueError):native.verify_component_mesh(path,bad)
            bad=author.serialize(value);bad['dimensions'][0]+=.002
            with self.assertRaises(ValueError):native.verify_component_mesh(path,bad)

    def test_structural_cli_failure_exit_code(self):
        summary={'layout_after_passed':False,'all_floor_support_passed':True,'contact_graphs_connected':True}
        with patch.object(author,'build',return_value=summary),patch('sys.argv',['build','--scene','s','--observations','o','--output','d']),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(author.main(),2)


class SourceContracts(unittest.TestCase):
    def test_surface_contract_wrong_schema_units_axis_empty_and_foreign_hash(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);scene=base/'scene.json';surface=base/'surface.json'
            scene.write_text(json.dumps({'objects':[{'uuid':'obj'}]}))
            structural={'schema':author.SCHEMA,'units':'m','up_axis':'Y',
                        'input_sha256':{'scene':author.sha256(scene)}}
            (base/'structural.json').write_text(json.dumps(structural))
            good={'schema':'spatial-scene-lab.surface.v1','units':'m','up_axis':'Y',
                  'input_scene_sha256':author.sha256(scene),'parts':[{'id':'environment'}]}
            mutations=[{'schema':'wrong'},{'units':'cm'},{'up_axis':'Z'},{'parts':[]},
                       {'input_scene_sha256':'0'*64},{'parts':[{'id':'environment'},{'id':'environment'}]},
                       {'parts':[{'id':'foreign-object'}]}]
            for change in mutations:
                surface.write_text(json.dumps({**good,**change}))
                with self.subTest(change=change),self.assertRaises(ValueError):native.inputs(scene,surface,base)


if __name__=='__main__':unittest.main()
