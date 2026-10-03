import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


AVAILABLE = importlib.util.find_spec('open3d') is not None
if AVAILABLE:
    import evaluate_partition
    import open3d as o3d
    import reevaluate_development
    import run_development


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False) + '\n', encoding='utf-8')


def _metrics():
    targets = []
    for index, identity in enumerate(('a', 'b')):
        targets.append({
            'index': index, 'object_id': identity, 'status': 'evaluated',
            'area_iou': .5, 'precision': .75, 'recall': .6,
            'false_positive_ratio': .2, 'false_negative_ratio': .4,
            'scorable_target_area_m2': .5, 'true_positive_area_m2': .3,
            'false_positive_area_m2': .1, 'false_negative_area_m2': .2,
            'unresolved_target_area_m2': 0.,
        })
    return {
        'schema': 'spatial-scene-lab.partition-evaluation.v1',
        'status': 'evaluated',
        'parameters': {'matching_tolerance_m': .08, 'samples_per_face': 7,
                       'ambiguity_margin_m': .002,
                       'sampling_rule': evaluate_partition.SAMPLING_RULE},
        'targets': targets,
        'coverage': {
            'mesh_area_m2': 1., 'scorable_area_m2': .8,
            'unscorable_area_m2': .2, 'ambiguous_area_m2': .05,
            'distance_rejected_area_m2': .15, 'scorable_fraction': .8,
            'faces': 2, 'samples': 14, 'scorable_samples': 12,
            'unscorable_samples': 2,
        },
    }


@unittest.skipUnless(AVAILABLE, 'CPU development sensitivity requires Open3D')
class DevelopmentSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='AlvenX-sensitivity-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.suite = self.root / 'suite'
        self.results = self.root / 'results'
        self.output = self.root / 'sensitivity'
        scan = self.suite / 'tiny' / 'scan'
        reference = self.suite / 'tiny' / 'reference'
        scene_root = self.results / 'tiny'
        points = scene_root / 'points'
        surface = scene_root / 'surface'
        for directory in (scan, reference, points, surface):
            directory.mkdir(parents=True)
        self.source_path = scan / 'source.txt'
        self.source_path.write_text('frozen scan observation\n', encoding='utf-8')
        self.vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                                  [2., 0., 0.], [3., 0., 0.], [2., 1., 0.]])
        self.triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        self.labels_path = reference / 'labels.npz'
        # Reference order differs from scene order; identity mapping is required.
        np.savez(self.labels_path, vertices=self.vertices, triangles=self.triangles,
                 triangle_owner=np.array([0, 1], dtype=np.int32),
                 object_ids=np.array(['b', 'a']))
        self.scene_path = points / 'scene.json'
        _json(self.scene_path, {
            'schema': 'spatial-scene-lab.observations.v1', 'source': str(scan),
            'units': 'm', 'up_axis': 'Y',
            'objects': [{'uuid': 'a'}, {'uuid': 'b'}],
            'input_files': {'source.txt': _sha256(self.source_path)},
        })
        self.mesh_path = surface / 'surface.ply'
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self.vertices),
            o3d.utility.Vector3iVector(self.triangles))
        self.assertTrue(o3d.io.write_triangle_mesh(str(self.mesh_path), mesh))
        self.assertTrue(o3d.io.write_triangle_mesh(str(reference / 'mesh.ply'), mesh))
        _json(surface / 'surface-scene.json', {
            'schema': 'spatial-scene-lab.surface.v1', 'status': 'pass',
            'units': 'm', 'up_axis': 'Y',
            'input_scene_sha256': _sha256(self.scene_path),
            'vertices': len(self.vertices), 'triangles': len(self.triangles),
        })
        self.owners = {'b0': [0, 1], 'b1': [1, 0], 'b2': [-2, 1]}
        methods = []
        for method, owner in self.owners.items():
            owner_path = scene_root / f'{method}-face-owner.npy'
            metrics_path = scene_root / f'{method}-metrics.json'
            np.save(owner_path, np.array(owner, dtype=np.int32))
            metrics = _metrics()
            _json(metrics_path, metrics)
            methods.append({
                'method': method, 'status': 'completed', 'parameters': {},
                'face_owner_sha256': _sha256(owner_path),
                'metrics_sha256': _sha256(metrics_path),
                'target': metrics['targets'][0], 'coverage': metrics['coverage'],
                'metrics_path': f'tiny/{method}-metrics.json',
            })
        project = Path(run_development.__file__).parent
        code_names = ('run_development.py', 'scan_pipeline.py', 'reconstruct_mesh.py',
                      'partition_baselines.py', 'evaluate_partition.py',
                      'fixtures/make_complex_fixture.py',
                      'fixtures/complex-development-scenes.json')
        generator_hash = _sha256(project / 'fixtures/make_complex_fixture.py')
        self.manifest_path = self.suite / 'suite-manifest.json'
        _json(self.manifest_path, {
            'schema': 'spatial-scene-lab.fixture-suite.v1',
            'generator_sha256': generator_hash, 'units': 'm', 'up_axis': 'Y',
            'split': 'development',
            'scenes': [{
                'scene_id': 'tiny', 'target_id': 'a', 'factors': {},
                'scan': 'tiny/scan', 'reference_labels': 'tiny/reference/labels.npz',
                'reference_mesh': 'tiny/reference/mesh.ply',
            }],
        })
        self.summary_path = self.results / 'development-results.json'
        self.summary = {
            'schema': 'spatial-scene-lab.development.v1', 'status': 'completed',
            'input_manifest_sha256': _sha256(self.manifest_path),
            'input_generator_sha256': generator_hash,
            'code_sha256': {name: _sha256(project / name) for name in code_names},
            'configuration': {
                'voxel_size_m': .035, 'sdf_truncation_m': .105,
                'matching_tolerance_m': .08, 'samples_per_face': 7,
                'methods': ['b0', 'b1', 'b2'],
            },
            'scenes': [{
                'scene_id': 'tiny', 'target_id': 'a', 'factors': {},
                'status': 'completed', 'vertices': 6, 'triangles': 2,
                'scene_sha256': _sha256(self.scene_path),
                'surface_manifest_sha256': _sha256(surface / 'surface-scene.json'),
                'surface_sha256': _sha256(self.mesh_path),
                'reference_sha256': _sha256(self.labels_path), 'methods': methods,
            }],
        }
        self._save_summary()

    def _save_summary(self):
        _json(self.summary_path, self.summary)
        _json(self.results / 'tiny/result.json', self.summary['scenes'][0])

    def _reevaluate(self):
        return reevaluate_development.reevaluate(self.results, self.suite, self.output)

    def test_reuses_frozen_arrays_and_maps_reference_uuids_for_all_six_settings(self):
        calls = []

        def evaluate(vertices, triangles, owner, ref_vertices, ref_triangles,
                     ref_owner, **configuration):
            np.testing.assert_array_equal(vertices, self.vertices)
            np.testing.assert_array_equal(triangles, self.triangles)
            np.testing.assert_array_equal(ref_vertices, self.vertices)
            np.testing.assert_array_equal(ref_triangles, self.triangles)
            np.testing.assert_array_equal(ref_owner, [1, 0])
            self.assertEqual(configuration['object_ids'], ['a', 'b'])
            calls.append((tuple(owner), configuration['matching_tolerance'],
                          configuration['samples_per_face']))
            return _metrics()

        with patch.object(evaluate_partition, 'evaluate_all_objects', side_effect=evaluate), \
                patch.object(run_development.scan_pipeline, 'reconstruct',
                             side_effect=AssertionError('observations must remain frozen')), \
                patch.object(run_development.reconstruct_mesh, 'reconstruct',
                             side_effect=AssertionError('surface must remain frozen')), \
                patch.object(run_development.partition_baselines, 'partition',
                             side_effect=AssertionError('partition must remain frozen')):
            summary = self._reevaluate()
        expected = {(tuple(owner), tolerance, samples)
                    for owner in self.owners.values()
                    for tolerance in (.04, .08, .12) for samples in (7, 49)}
        self.assertEqual(len(calls), 18)
        self.assertEqual(set(calls), expected)
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['findings'], [])
        self.assertEqual(len(summary['configurations']), 6)
        self.assertEqual(len(summary['comparisons']), 18)
        self.assertTrue(all(len(row['targets']) == 2 for row in summary['comparisons']))
        self.assertEqual(summary['baseline_consistency']['status'], 'pass')
        self.assertEqual(summary['baseline_consistency']['checked_method_tasks'], 3)
        self.assertEqual(summary['baseline_consistency']['mismatches'], [])
        saved = json.loads((self.output / 'sensitivity-results.json').read_text(encoding='utf-8'))
        self.assertEqual(saved, summary)
        with (self.output / 'sensitivity-metrics.csv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 36)

    def test_baseline_consistency_checks_secondary_object_and_coverage(self):
        changed = _metrics()
        changed['targets'][1]['area_iou'] = .7
        changed['coverage']['ambiguous_area_m2'] = .1
        with patch.object(evaluate_partition, 'evaluate_all_objects', return_value=changed):
            summary = self._reevaluate()
        self.assertEqual(summary['status'], 'consistency_failure')
        consistency = summary['baseline_consistency']
        self.assertEqual(consistency['status'], 'fail')
        self.assertEqual(consistency['checked_method_tasks'], 3)
        mismatches = json.dumps(consistency['mismatches'])
        self.assertIn('area_iou', mismatches)
        self.assertIn('ambiguous_area_m2', mismatches)
        self.assertTrue(consistency['mismatches'])

    def test_single_evaluation_failure_is_saved_and_other_settings_continue(self):
        calls = []

        def evaluate(*arrays, **configuration):
            calls.append(configuration)
            if len(calls) == 1:
                self.assertEqual(configuration['matching_tolerance'], .04)
                self.assertEqual(configuration['samples_per_face'], 7)
                raise RuntimeError('small synthetic evaluation failure')
            return _metrics()

        with patch.object(evaluate_partition, 'evaluate_all_objects', side_effect=evaluate):
            summary = self._reevaluate()
        self.assertEqual(len(calls), 18)
        self.assertEqual(summary['attempted_evaluations'], 18)
        self.assertEqual(summary['completed_evaluations'], 17)
        self.assertEqual(len(summary['comparisons']), 17)
        self.assertEqual(summary['failed_evaluations'], 1)
        self.assertEqual(summary['status'], 'evaluation_failure')
        self.assertEqual(summary['baseline_consistency']['status'], 'pass')
        self.assertEqual(summary['baseline_consistency']['checked_method_tasks'], 3)
        self.assertEqual(len(summary['findings']), 1)
        finding = summary['findings'][0]
        self.assertEqual(finding['check'], 'evaluation')
        self.assertEqual(finding['scene_id'], 'tiny')
        self.assertEqual(finding['method'], 'b0')
        self.assertEqual(finding['matching_tolerance_m'], .04)
        self.assertEqual(finding['samples_per_face'], 7)
        self.assertIn('small synthetic evaluation failure', finding['message'])
        saved = json.loads((self.output / 'sensitivity-results.json').read_text(encoding='utf-8'))
        self.assertEqual(saved, summary)

    def test_changed_frozen_file_stops_before_any_evaluation_and_saves_finding(self):
        owner_path = self.results / 'tiny/b1-face-owner.npy'
        np.save(owner_path, np.array([0, 0], dtype=np.int32))
        with patch.object(evaluate_partition, 'evaluate_all_objects') as evaluate:
            summary = self._reevaluate()
        evaluate.assert_not_called()
        self.assertEqual(summary['status'], 'integrity_failure')
        self.assertEqual(summary['baseline_consistency']['status'], 'not_run')
        self.assertTrue(summary['findings'])
        self.assertIn('b1', json.dumps(summary['findings']))
        saved = json.loads((self.output / 'sensitivity-results.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['findings'], summary['findings'])

    def test_missing_face_owner_anchor_fails_integrity_gate(self):
        del self.summary['scenes'][0]['methods'][0]['face_owner_sha256']
        self._save_summary()
        with patch.object(evaluate_partition, 'evaluate_all_objects') as evaluate:
            summary = self._reevaluate()
        evaluate.assert_not_called()
        self.assertEqual(summary['status'], 'integrity_failure')
        self.assertTrue(summary['findings'])
        self.assertIn('face_owner', json.dumps(summary['findings']))

    def test_changed_scan_file_stops_before_any_evaluation(self):
        self.source_path.write_text('changed observation\n', encoding='utf-8')
        with patch.object(evaluate_partition, 'evaluate_all_objects') as evaluate:
            summary = self._reevaluate()
        evaluate.assert_not_called()
        self.assertEqual(summary['status'], 'integrity_failure')
        self.assertTrue(summary['findings'])

    def test_requires_new_or_empty_output_without_overwriting_existing_files(self):
        self.output.mkdir()
        sentinel = self.output / 'user-state.txt'
        sentinel.write_text('retain prior work', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'new or empty'):
            self._reevaluate()
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'retain prior work')

    def test_output_cannot_overlap_suite_or_results(self):
        for output in (self.suite, self.suite / 'output', self.results,
                       self.results / 'output', self.root):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'overlap'):
                reevaluate_development.reevaluate(self.results, self.suite, output)


if __name__ == '__main__':
    unittest.main()
