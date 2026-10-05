import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import run_edit_suite as suite


class EditSuiteTests(unittest.TestCase):
    def make_bundle(self, root):
        name, primary, secondary = suite.TASKS[0]
        directory = root / name / 'surface'
        directory.mkdir(parents=True)
        source = directory / 'region.ply'
        source.write_bytes(b'input bytes, not a Blender mesh')
        parts = [{'id': identity, 'path': 'region.ply', 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
                 for identity in ('environment', primary, secondary)]
        path = directory / 'surface-scene.json'
        path.write_text(json.dumps({'parts': parts}), encoding='utf-8')
        return name, path, source

    def test_protocol_targets_are_explicit_and_unique(self):
        self.assertEqual(len(suite.TASKS), 6)
        self.assertEqual(len({task[0] for task in suite.TASKS}), 6)
        for _, primary, secondary in suite.TASKS:
            self.assertNotEqual(primary, secondary)
            self.assertNotIn('environment', (primary, secondary))

    def test_unknown_task_is_rejected_before_reading_inputs(self):
        with self.assertRaisesRegex(ValueError, 'not an S1'):
            suite.task_inputs('missing-root', ['unexpected-scene'])

    def test_part_content_drift_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='AlvenX-s1-drift-') as tmp:
            root = Path(tmp)
            name, _, source = self.make_bundle(root)
            self.assertEqual(len(suite.task_inputs(root, [name])), 1)
            source.write_bytes(b'changed input bytes')
            with self.assertRaisesRegex(ValueError, 'escaped or changed'):
                suite.task_inputs(root, [name])

    def test_duplicate_identity_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='AlvenX-s1-id-') as tmp:
            root = Path(tmp)
            name, path, _ = self.make_bundle(root)
            doc = json.loads(path.read_text())
            doc['parts'][-1]['id'] = doc['parts'][0]['id']
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'missing or duplicated'):
                suite.task_inputs(root, [name])

    def test_source_escape_is_rejected_even_with_valid_hash(self):
        with tempfile.TemporaryDirectory(prefix='AlvenX-s1-path-') as tmp:
            root = Path(tmp)
            name, path, source = self.make_bundle(root)
            outside = root / 'outside.ply'
            outside.write_bytes(source.read_bytes())
            doc = json.loads(path.read_text())
            doc['parts'][0]['path'] = '../../outside.ply'
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'escaped or changed'):
                suite.task_inputs(root, [name])

    def test_process_failure_keeps_log_and_exit_status(self):
        with tempfile.TemporaryDirectory(prefix='AlvenX-s1-exit-') as tmp:
            log = Path(tmp) / 'failed.log'
            result = suite.run_process([sys.executable, '-c', 'print("deliberate failure"); raise SystemExit(7)'], log, 5)
            self.assertEqual(result['status'], 'fail')
            self.assertEqual(result['exit_code'], 7)
            self.assertIn('deliberate failure', log.read_text())

    def test_timeout_is_counted_as_failure(self):
        with tempfile.TemporaryDirectory(prefix='AlvenX-s1-timeout-') as tmp:
            result = suite.run_process([sys.executable, '-c', 'import time; time.sleep(3)'], Path(tmp) / 'timeout.log', .1)
            self.assertEqual(result['status'], 'fail')
            self.assertTrue(result['timed_out'])

    def test_invalid_time_budget_cannot_start_a_process(self):
        for timeout in (0, -1, float('nan'), float('inf')):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, 'finite and positive'):
                suite.run_process(['missing executable'], 'unused.log', timeout)


if __name__ == '__main__':
    unittest.main()
