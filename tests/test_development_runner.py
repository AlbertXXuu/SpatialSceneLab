import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

AVAILABLE = importlib.util.find_spec('open3d') is not None
if AVAILABLE:
    import run_development


@unittest.skipUnless(AVAILABLE, 'CPU development runner requires Open3D')
class DevelopmentRunnerTests(unittest.TestCase):
    def test_reference_ids_map_by_uuid_not_array_order(self):
        with tempfile.TemporaryDirectory(prefix='AlvenX-reference-identity-') as tmp:
            path = Path(tmp) / 'labels.npz'
            np.savez(path, vertices=np.zeros((3, 3)), triangles=np.array([[0, 1, 2], [0, 1, 2]]),
                     triangle_owner=np.array([0, 1], dtype=np.int32), object_ids=np.array(['b', 'a']))
            _, _, owner = run_development.reference_arrays(path, ['a', 'b'])
            np.testing.assert_array_equal(owner, [1, 0])
            with self.assertRaisesRegex(ValueError, 'UUIDs must agree'):
                run_development.reference_arrays(path, ['a', 'other'])

    def test_failed_input_stays_in_task_denominators(self):
        with tempfile.TemporaryDirectory(prefix='AlvenX-development-failure-') as tmp:
            root = Path(tmp)
            suite = root / 'input'
            suite.mkdir()
            (suite / 'suite-manifest.json').write_text(json.dumps({'scenes': [{
                'scene_id': 'invalid-scan', 'target_id': 'target', 'factors': {},
                'scan': 'scan', 'reference_labels': 'reference/labels.npz'}]}), encoding='utf-8')
            result = run_development.run_suite(suite, root / 'output')
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['attempted_scenes'], 1)
            self.assertEqual(result['completed_scenes'], 0)
            self.assertEqual(result['attempted_method_tasks'], 3)
            self.assertEqual(result['completed_method_tasks'], 0)
            self.assertEqual(result['scenes'][0]['stage'], 'observations')
            self.assertTrue(all(row['mean_area_iou'] is None for row in result['aggregate']))
            self.assertTrue((root / 'output/invalid-scan/failure.txt').is_file())

    def test_reference_labels_reject_fractional_nan_and_wrong_shape(self):
        with tempfile.TemporaryDirectory(prefix='AlvenX-reference-labels-') as tmp:
            path = Path(tmp) / 'labels.npz'
            for owners in (np.array([.5]), np.array([np.nan]), np.array([[0]]), np.array([0, 0])):
                with self.subTest(owners=owners):
                    np.savez(path, vertices=np.zeros((3, 3)), triangles=np.array([[0, 1, 2]]),
                             triangle_owner=owners, object_ids=np.array(['a']))
                    with self.assertRaisesRegex(ValueError, 'one integer per triangle'):
                        run_development.reference_arrays(path, ['a'])

    def test_manifest_paths_and_output_cannot_overlap_or_escape(self):
        with tempfile.TemporaryDirectory(prefix='AlvenX-development-paths-') as tmp:
            root = Path(tmp).resolve()
            with self.assertRaisesRegex(ValueError, 'inside its suite'):
                run_development.inside(root, '../escape')
            (root / 'suite-manifest.json').write_text(json.dumps({'scenes': [{
                'scene_id': 'safe', 'target_id': 'target', 'factors': {}}]}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'must not overlap'):
                run_development.run_suite(root, root / 'output')

    def test_peak_memory_has_real_positive_units(self):
        self.assertGreater(run_development.peak_ram_bytes(), 0)

    def test_existing_output_is_preserved_instead_of_overwritten(self):
        with tempfile.TemporaryDirectory(prefix='AlvenX-development-existing-') as tmp:
            root = Path(tmp)
            suite, output = root / 'input', root / 'output'
            suite.mkdir()
            output.mkdir()
            sentinel = output / 'user-state.json'
            sentinel.write_text('preserve me', encoding='utf-8')
            (suite / 'suite-manifest.json').write_text(json.dumps({'scenes': [{
                'scene_id': 'safe', 'target_id': 'target', 'factors': {}}]}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'new or empty'):
                run_development.run_suite(suite, output)
            self.assertEqual(sentinel.read_text(encoding='utf-8'), 'preserve me')
