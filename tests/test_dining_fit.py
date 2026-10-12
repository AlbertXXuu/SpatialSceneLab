"""Dining numerical, input and export regressions; no reference asset reads."""
import copy
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from fit_dining import (TARGETS, box_mesh, box_sdf, fit_cushion,
                        half_ellipsoid_mesh, half_ellipsoid_residual, load_inputs,
                        slab_mesh, yaw_matrix)
from fit_stool import validate_output


def solid_assertions(test, vertices, faces):
    """Real export regression: each directed edge has exactly its reverse."""
    directed=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    undirected=np.sort(directed,axis=1)
    _, counts=np.unique(undirected,axis=0,return_counts=True)
    test.assertTrue(np.all(counts==2))
    actual=set(map(tuple,directed.tolist()))
    test.assertTrue(all((b,a) in actual for a,b in actual))
    tri=vertices[faces]
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    test.assertTrue(np.all(np.linalg.norm(cross,axis=1)>1e-12))
    # Translation to a nearby origin avoids cancellation in far-world assets.
    q=vertices-np.mean(vertices,axis=0)
    volume=np.einsum('ij,ij->i',q[faces[:,0]],np.cross(q[faces[:,1]],q[faces[:,2]])).sum()/6
    test.assertGreater(volume,0.)


class GeometryTests(unittest.TestCase):
    def test_exact_box_distance_and_rotation(self):
        p=np.array([[0,0,0],[1,0,0],[2,0,0],[2,2,0]])
        expected=[-1,0,1,np.sqrt(2)]
        np.testing.assert_allclose(box_sdf(p,[0,0,0],[2,2,2]),expected)
        r=yaw_matrix(.31);shift=np.array([3.1,.7,-2.])
        np.testing.assert_allclose(box_sdf(p@r.T+shift,shift,[2,2,2],r),expected,atol=1e-12)

    def test_box_export_has_eight_correct_corners_and_outward_winding(self):
        r=yaw_matrix(.43);center=np.array([.2,.7,-2.4]);dims=np.array([1.4,.08,.9])
        v,f=box_mesh({'center_m':center,'dimensions_m':dims,'rotation_world_3x3':r})
        self.assertEqual(v.shape,(8,3));self.assertEqual(f.shape,(12,3))
        np.testing.assert_allclose(np.abs((v-center)@r),np.tile(dims/2,(8,1)),atol=1e-12)
        solid_assertions(self,v,f)

    def test_each_cap_orientation_is_closed_positive_and_matches_parameters(self):
        for axis,sign in ((1,1),(2,-1)):
            with self.subTest(axis=axis,sign=sign):
                r=yaw_matrix(.23);center=np.array([1.2,.53,-2.1]);radii=np.array([.24,.16,.045])
                p={'center_m':center,'radii_m':radii,'rotation_world_3x3':r,'cap_axis':axis,'cap_sign':sign}
                v,f=half_ellipsoid_mesh(p,rings=12,slices=32)
                solid_assertions(self,v,f)
                q=(v-center)@r;norm=np.sum((q/radii)**2,axis=1)
                self.assertTrue(np.all(sign*q[:,axis]>=-1e-12))
                self.assertTrue(np.all((np.abs(norm-1)<1e-10)|(np.linalg.norm(q,axis=1)<1e-10)))
                np.testing.assert_allclose(half_ellipsoid_residual(v[:-1],center,radii,r,axis,sign),0,atol=1e-12)

    def test_rectangular_domed_control_is_manifold_after_axis_permutation(self):
        for axis,sign in ((1,1),(2,-1)):
            p={'center_m':[0,.5,-2.],'dimensions_m':[.5,.1,.4],
               'rotation_world_3x3':yaw_matrix(.1),'cap_axis':axis,'cap_sign':sign,
               'cap_height_m':.035,'profile_power':.5}
            v,f=slab_mesh(p,grid=8);solid_assertions(self,v,f)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='AlvenX-dining-tests-')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        transform=np.eye(4);transform[1,3]=.5
        floor=np.eye(4);floor[1,3]=-.025
        self.scene={'units':'m','up_axis':'Y','objects':[
            {'uuid':u,'category':'Table' if i==0 else 'Chair','dimensions_m':[1.,1.,1.],
             'transform_world':transform.tolist()} for i,u in enumerate(TARGETS)],
            'structures':[{'category':'Floor','dimensions_m':[3.,.05,3.],'transform_world':floor.tolist()}],
            'frames':[{'frame_index':0},{'frame_index':1}]}
        self.arrays={'points':np.tile([0.,.5,0.],(900,1)),
                     'colors':np.full((900,3),100,dtype=np.uint8),
                     'frame_index':np.arange(900,dtype=np.int32)%2,
                     'owner':np.repeat(np.arange(3,dtype=np.int32),300),
                     'candidates':np.repeat(np.eye(3,dtype=bool),300,axis=0)}

    def load(self,arrays=None,scene=None):
        sp=self.root/'scene.json';op=self.root/'observations.npz'
        sp.write_text(json.dumps(self.scene if scene is None else scene),encoding='utf-8')
        np.savez(op,**(self.arrays if arrays is None else arrays))
        return load_inputs(sp,op)

    def test_binary_integer_candidates_are_normalized_before_masking(self):
        a=copy.deepcopy(self.arrays);a['candidates']=a['candidates'].astype(np.int32)
        _,data,_,floor=self.load(a)
        self.assertEqual(data['candidates'].dtype,np.bool_)
        mask=(data['frame_index']%2==0)&data['candidates'][:,1]
        self.assertEqual(mask.dtype,np.bool_);self.assertEqual(len(data['points'][mask]),150)
        self.assertEqual(floor,0.)

    def test_overlap_candidates_do_not_remove_original_owner_validation(self):
        a=copy.deepcopy(self.arrays);a['candidates'][:300,1]=True
        _,data,_,_=self.load(a)
        self.assertEqual(np.sum((data['frame_index']%2==0)&data['candidates'][:,1]),300)
        self.assertEqual(np.sum((data['frame_index']%2==1)&(data['owner']==1)),150)

    def test_invalid_candidate_shape_values_and_rigid_prior_rejected(self):
        for bad in (np.zeros((900,2),bool),np.full((900,3),2),np.full((900,3),np.nan)):
            a=copy.deepcopy(self.arrays);a['candidates']=bad
            with self.assertRaises(ValueError):self.load(a)
        s=copy.deepcopy(self.scene);s['objects'][1]['transform_world'][0][0]=2
        with self.assertRaises(ValueError):self.load(scene=s)

    def test_output_refuses_existing_result_or_input_alias(self):
        p=self.root/'result';p.mkdir();(p/'fit.json').write_text('{}')
        with self.assertRaises(ValueError):validate_output(p,[])
        with self.assertRaises(ValueError):validate_output(self.root/'observations.npz',[self.root/'observations.npz'])

    def test_input_changed_during_read_refuses_final_result(self):
        import fit_dining
        scene=self.root/'scene.json';observations=self.root/'observations.npz'
        scene.write_bytes(b'original scene');observations.write_bytes(b'original observations')
        output=self.root/'changed-input-result'
        def changed_read(*_):
            scene.write_bytes(b'changed scene')
            return {},{},[],0.
        argv=['fit_dining.py','--scene',str(scene),'--observations',str(observations),'--output',str(output)]
        with patch('sys.argv',argv),patch.object(fit_dining,'load_inputs',side_effect=changed_read),contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as rejected:fit_dining.main()
        self.assertEqual(rejected.exception.code,2)
        self.assertIn('Input changed while reading',stderr.getvalue())
        self.assertFalse(output.exists())

    def test_input_changed_after_read_refuses_fit_record(self):
        import fit_dining
        scene=self.root/'scene.json';observations=self.root/'observations.npz'
        scene.write_bytes(b'original scene');observations.write_bytes(b'original observations')
        output=self.root/'changed-late-result'
        validate=validate_output
        def changed_after_read(path,inputs):
            validate(path,inputs)
            if path.exists():return
            if changed_after_read.called:observations.write_bytes(b'changed observations')
            changed_after_read.called=True
        changed_after_read.called=False
        argv=['fit_dining.py','--scene',str(scene),'--observations',str(observations),'--output',str(output)]
        with patch('sys.argv',argv),patch.object(fit_dining,'load_inputs',return_value=({}, {}, [], 0.)),patch.object(fit_dining,'validate_output',side_effect=changed_after_read),contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as rejected:fit_dining.main()
        self.assertEqual(rejected.exception.code,2)
        self.assertIn('Input changed during fitting',stderr.getvalue())
        self.assertFalse((output/'fit.json').exists())


@unittest.skipUnless(importlib.util.find_spec('scipy') is not None,'Optional SciPy fitter unavailable')
class NumericalControls(unittest.TestCase):
    def test_changed_cap_observations_change_fitted_pose_with_same_initial_prior(self):
        params={'center_m':[0,.5,0.],'radii_m':[.25,.055,.23],
                'rotation_world_3x3':np.eye(3),'cap_axis':1,'cap_sign':1}
        points,_=half_ellipsoid_mesh(params,rings=12,slices=36)
        points=points[(points[:,1]>.514)&(points[:,1]<.551)]
        center=np.array([0.,.5,0.]);radii=np.array([.26,.058,.24])
        c1,r1,_=fit_cushion(points,center,radii,np.eye(3),1,1)
        shift=np.array([.012,.002,-.009])
        c2,r2,_=fit_cushion(points+shift,center,radii,np.eye(3),1,1)
        np.testing.assert_allclose(c2-c1,shift,atol=.002)
        np.testing.assert_allclose(r1,[.25,.055,.23],atol=.003)
        np.testing.assert_allclose(r2,r1,atol=.004)


if __name__=='__main__':unittest.main()
