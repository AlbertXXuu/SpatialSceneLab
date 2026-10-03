import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

AVAILABLE = importlib.util.find_spec('open3d') is not None
if AVAILABLE:
    import reconstruct_mesh as mesh


@unittest.skipUnless(AVAILABLE, 'optional CPU Open3D dependency; local surface integration suite')
class SurfaceTests(unittest.TestCase):
    def make_scene(self, root):
        import importlib.util
        import scan_pipeline
        spec = importlib.util.spec_from_file_location('original_fixture', Path(__file__).resolve().parents[1] / 'fixtures/make_rgbd_fixture.py')
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        fixture.generate(root / 'input')
        scan_pipeline.reconstruct(root / 'input', root / 'points')
        return root / 'points/scene.json'

    def test_cross_boundary_face_remains_environment(self):
        def box(identity, x):
            transform = np.eye(4)
            transform[0, 3] = x
            return {'uuid': identity, 'geometry_kind': 'box', 'dimensions_m': [1, 1, 1],
                    'transform_world': transform.tolist()}
        vertices = np.array([[-.6, 0, 0], [-.55, .1, 0], [-.55, 0, .1], [.6, 0, 0], [0, 0, 0]])
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
        owners, _, _ = mesh.split_triangles(vertices, faces, [box('left', -.2), box('right', .2)], [],
                                            {'box_margin_m': 0, 'structure_tolerance_m': .025})
        np.testing.assert_array_equal(owners, [0, -1, -1])

    def test_empty_reference_cannot_report_zero_error_success(self):
        with tempfile.TemporaryDirectory(prefix='AlvenX-empty-pcd-') as tmp:
            root = Path(tmp)
            (root / 'pointcloud.pcd').write_text(
                'VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 0\nHEIGHT 1\nPOINTS 0\nDATA ascii\n')
            scene = root / 'scene.json'
            scene.write_text(json.dumps({'schema': 'spatial-scene-lab.observations.v1',
                                         'source': str(root), 'units': 'm', 'up_axis': 'Y'}))
            with self.assertRaisesRegex(ValueError, 'reference PCD must contain finite points'):
                mesh.reconstruct(scene, root.parent / (root.name + '-unused'))

    def test_changed_raw_frame_is_rejected_between_stages(self):
        with tempfile.TemporaryDirectory(prefix='AlvenX-changed-input-') as tmp:
            root = Path(tmp)
            scene = self.make_scene(root)
            with (root / 'input/frame_00000.json').open('a') as stream:
                stream.write(' ')
            with self.assertRaisesRegex(ValueError, 'input file changed'):
                mesh.reconstruct(scene, root / 'surface')
            self.assertFalse((root / 'surface/surface-scene.json').exists())

    def test_failed_mesh_write_cannot_publish_success_manifest(self):
        with tempfile.TemporaryDirectory(prefix='AlvenX-failed-write-') as tmp:
            root = Path(tmp)
            scene = self.make_scene(root)
            with patch.object(mesh.o3d.io, 'write_triangle_mesh', return_value=False):
                with self.assertRaisesRegex(OSError, 'could not write fused surface'):
                    mesh.reconstruct(scene, root / 'surface')
            self.assertFalse((root / 'surface/surface-scene.json').exists())
