"""Portable input audit: exact observations, bad contracts and output boundaries."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import audit_table0_inputs as audit
import scan_pipeline as scan


class TableInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='AlvenX-input-contract-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'tea_room'
        self.prepared = self.root / 'prepared'
        self.source.mkdir()
        self.prepared.mkdir()
        self.scene_path = self.prepared / 'scene.json'
        self.points_path = self.prepared / 'observations.npz'
        self.manifest = self.root / 'source-manifest.json'
        self.output = self.root / 'result'
        self.scene = {'schema': scan.SCHEMA, 'units': 'm', 'up_axis': 'Y', 'source': str(self.source),
                      'objects': [{'uuid': 'test-table', 'category': 'Table', 'dimensions_m': [1., .8, 1.],
                                   'transform_world': np.eye(4).tolist()}], 'frames': [],
                      'parameters': {'frame_step': 1, 'pixel_step': 1, 'min_confidence': 1,
                                     'max_depth_m': 6., 'depth_scale': 1000}}
        points, colors, pixels, frames = [], [], [], []
        for fid in [0, 1, 31, 32, 33, 34, 35]:
            pose = np.eye(4)
            pose[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
            pose[1, 3] = 1. if fid <= 30 else 1.018
            record = {'frame_index': fid, 'cameraPoseARFrame': pose.ravel().tolist(),
                      'intrinsics': [8., 0, 3., 0, 8., 3., 0, 0, 1.], 'time': fid * .75}
            frame_path = self.source / f'frame_{fid:05d}.json'
            frame_path.write_text(json.dumps(record), encoding='utf-8')
            Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(self.source / f'frame_{fid:05d}.jpg')
            Image.fromarray(np.full((4, 4), 680, dtype=np.uint16)).save(self.source / f'depth_{fid:05d}.png')
            Image.fromarray(np.full((4, 4), 2, dtype=np.uint8)).save(self.source / f'conf_{fid:05d}.png')
            loaded = scan.load_frame(frame_path)
            xyz, rgb, uv = scan.backproject_rgbd(loaded['depth_mm'], loaded['color'],
                                               loaded['intrinsics_depth'], loaded['camera_to_world_cv'],
                                               loaded['confidence'])
            points.append(xyz)
            colors.append(rgb)
            pixels.append(uv)
            frames.extend([fid] * len(xyz))
            self.scene['frames'].append({'frame_index': fid, 'source': frame_path.name,
                                         'color_size': list(loaded['color_size']), 'depth_size': [4, 4],
                                         'intrinsics_depth': loaded['intrinsics_depth'].tolist(),
                                         'camera_to_world_cv': loaded['camera_to_world_cv'].tolist(),
                                         'retained_observations': len(xyz)})
        self.arrays = {'points': np.concatenate(points), 'colors': np.concatenate(colors),
                       'pixel_uv': np.concatenate(pixels), 'frame_index': np.array(frames, dtype=np.int32),
                       'candidates': np.ones((len(frames), 1), dtype=bool)}
        entries = [{'path': 'tea_room/' + p.name, 'bytes': p.stat().st_size,
                    'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(self.source.iterdir())]
        self.manifest.write_text(json.dumps({'files': entries, 'upstream': {'commit': 'original-test-fixture'}}), encoding='utf-8')
        self.scene['input_files'] = {Path(e['path']).name: e['sha256'] for e in entries}
        self.save()

    def save(self):
        self.scene_path.write_text(json.dumps(self.scene), encoding='utf-8')
        np.savez_compressed(self.points_path, **self.arrays)

    def run_audit(self, output=None):
        return audit.audit(self.source, self.scene_path, self.points_path,
                           self.output if output is None else output, self.manifest, 'test-table')

    def test_exact_reprojection_and_numeric_only_summary(self):
        result = self.run_audit()
        self.assertTrue(result['passed'])
        self.assertTrue(result['input_unchanged'])
        self.assertEqual(result['observations'], 112)
        self.assertEqual(result['frames'], 7)
        self.assertEqual(result['source']['files'], 28)
        self.assertEqual(result['max_regenerated_point_component_error_m'], 0)
        self.assertLess(result['max_inverse_pixel_error'], 1e-12)
        self.assertEqual(result['counterfactuals']['original_axial']['training_count'], 32)
        self.assertEqual(result['counterfactuals']['original_axial']['validation_count'], 80)
        self.assertEqual(list(result['counterfactuals']), audit.CONVENTIONS)
        self.assertEqual([p.name for p in self.output.iterdir()], ['summary.json'])
        published = (self.output / 'summary.json').read_text(encoding='utf-8')
        for forbidden in ('camera_to_world_cv', 'training_plane_coefficients', 'plane_coefficients_local', str(self.source)):
            self.assertNotIn(forbidden, published)

    def test_changed_point_and_pixel_fail_before_output(self):
        for key in ('points', 'pixel_uv'):
            with self.subTest(key=key):
                original = self.arrays[key].copy()
                self.arrays[key][0, 0] += 1
                self.save()
                with self.assertRaisesRegex(ValueError, 'Stored observation'):
                    self.run_audit()
                self.assertFalse(self.output.exists())
                self.arrays[key] = original
                self.save()

    def test_wrong_intrinsics_metadata_cannot_pass(self):
        self.scene['frames'][1]['intrinsics_depth'][0][0] += .1
        self.save()
        with self.assertRaisesRegex(ValueError, 'intrinsics metadata'):
            self.run_audit()
        self.assertFalse(self.output.exists())

    def test_changed_raw_input_rejected_by_frozen_manifest(self):
        member = self.source / 'frame_00000.json'
        member.write_text(member.read_text() + ' ', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'checksum or size'):
            self.run_audit()
        self.assertFalse(self.output.exists())

    def test_existing_and_nested_outputs_preserve_inputs(self):
        for output in (self.source, self.source / 'results', self.prepared / 'results', self.root):
            with self.subTest(output=output):
                with self.assertRaisesRegex(ValueError, 'Output'):
                    self.run_audit(output)
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'must be new'):
            self.run_audit()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_source_mutation_during_audit_is_not_silent_success(self):
        original_fit = audit.baseline.fit_plane
        def mutate(*args, **kwargs):
            result = original_fit(*args, **kwargs)
            member = self.source / 'frame_00000.json'
            member.write_text(member.read_text() + ' ', encoding='utf-8')
            return result
        with patch.object(audit.baseline, 'fit_plane', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'checksum or size'):
                self.run_audit()
        self.assertFalse(self.output.exists())

    def test_cli_arguments_reach_local_audit(self):
        result = audit.main(['--input', str(self.source), '--scene', str(self.scene_path),
                             '--observations', str(self.points_path), '--manifest', str(self.manifest),
                             '--object', 'test-table', '--output', str(self.output)])
        self.assertEqual(result, 0)
        self.assertTrue((self.output / 'summary.json').is_file())


if __name__ == '__main__':
    unittest.main()
