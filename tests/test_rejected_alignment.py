"""Rejected alignment artifacts and unsupported derived poses fail closed."""
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
    import scan_pipeline as scan


@unittest.skipUnless(AVAILABLE, 'optional CPU Open3D dependency; mesh ingress checks')
class RejectedAlignmentTests(unittest.TestCase):
    def make_scene(self, root):
        spec = importlib.util.spec_from_file_location(
            'rejected_alignment_fixture', Path(__file__).resolve().parents[1] / 'fixtures/make_rgbd_fixture.py')
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        fixture.generate(root / 'input')
        scan.reconstruct(root / 'input', root / 'points')
        return root / 'points/scene.json'

    def test_rejected_or_malformed_control_fails_before_input_read_or_output(self):
        controls = [None, [], 'passed', {}, {'passed': 1}, {'passed': False},
                    {'passed': False, 'disposition': 'diagnostic_candidate'}]
        with tempfile.TemporaryDirectory(prefix='AlvenX-rejected-ingress-') as temporary:
            root = Path(temporary)
            scene_path = root / 'scene.json'
            scene = {'schema': 'spatial-scene-lab.observations.v1', 'units': 'm',
                     'up_axis': 'Y', 'source': str(root / 'unread')}
            for control in controls:
                with self.subTest(control=control):
                    scene['alignment_control'] = control
                    scene_path.write_text(json.dumps(scene), encoding='utf-8')
                    with patch.object(mesh.o3d.io, 'read_point_cloud', side_effect=AssertionError('must not read PCD')):
                        with self.assertRaisesRegex(ValueError, 'alignment'):
                            mesh.reconstruct(scene_path, root / 'surface')
                    self.assertFalse((root / 'surface').exists())

    def test_later_camera_mismatch_fails_before_any_integration_or_output(self):
        with tempfile.TemporaryDirectory(prefix='AlvenX-derived-camera-') as temporary:
            root = Path(temporary)
            scene_path = self.make_scene(root)
            scene = json.loads(scene_path.read_text(encoding='utf-8'))
            scene['frames'][-1]['camera_to_world_cv'][1][3] += .018
            for control in (None, {'passed': True}, {'passed': False, 'disposition': 'identity_fallback'}):
                with self.subTest(control=control):
                    if control is None:
                        scene.pop('alignment_control', None)
                    else:
                        scene['alignment_control'] = control
                    scene_path.write_text(json.dumps(scene), encoding='utf-8')
                    with patch.object(mesh.o3d.pipelines.integration, 'ScalableTSDFVolume') as volume:
                        with self.assertRaisesRegex(ValueError, 'camera.*raw|raw.*camera'):
                            mesh.reconstruct(scene_path, root / 'surface')
                        volume.assert_not_called()
                    self.assertFalse((root / 'surface').exists())

    def test_invalid_or_missing_camera_metadata_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix='AlvenX-invalid-camera-') as temporary:
            root = Path(temporary)
            scene_path = self.make_scene(root)
            original = json.loads(scene_path.read_text(encoding='utf-8'))
            for camera in (None, [], [[1]], 'invalid', np.full((4, 4), np.nan).tolist()):
                with self.subTest(camera=camera):
                    scene = json.loads(json.dumps(original))
                    scene['frames'][-1]['camera_to_world_cv'] = camera
                    scene_path.write_text(json.dumps(scene), encoding='utf-8')
                    with patch.object(mesh.o3d.pipelines.integration, 'ScalableTSDFVolume') as volume:
                        with self.assertRaisesRegex(ValueError, 'camera'):
                            mesh.reconstruct(scene_path, root / 'surface')
                        volume.assert_not_called()
                    self.assertFalse((root / 'surface').exists())

    def test_identity_fallback_retains_original_raw_cameras_and_surface(self):
        with tempfile.TemporaryDirectory(prefix='AlvenX-identity-fallback-') as temporary:
            root = Path(temporary)
            scene_path = self.make_scene(root)
            scene = json.loads(scene_path.read_text(encoding='utf-8'))
            scene['alignment_control'] = {'passed': False, 'disposition': 'identity_fallback'}
            scene_path.write_text(json.dumps(scene), encoding='utf-8')
            result = mesh.reconstruct(scene_path, root / 'surface')
            self.assertEqual(result['status'], 'pass')
            self.assertGreater(result['triangles'], 0)
            self.assertEqual(result['cameras'], scene['frames'])
            for frame in result['cameras']:
                raw = scan.load_frame(root / 'input' / frame['source'])
                np.testing.assert_array_equal(frame['camera_to_world_cv'], raw['camera_to_world_cv'])


if __name__ == '__main__':
    unittest.main()
