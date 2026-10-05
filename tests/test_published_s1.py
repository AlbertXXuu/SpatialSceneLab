import hashlib
import json
from pathlib import Path
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'reports/editing-s1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PublishedEditingTests(unittest.TestCase):
    def test_frozen_native_code_bytes_match_measured_versions(self):
        suite = json.loads((EVIDENCE / 'edit-suite-results.json').read_text())
        self.assertEqual((suite['status'], suite['completed_tasks'], suite['attempted_tasks']), ('pass', 6, 6))
        for name, digest in suite['code_sha256'].items():
            self.assertEqual(sha(ROOT / name), digest, name)
        controls = json.loads((EVIDENCE / 'controls.json').read_text())
        self.assertEqual((controls['status'], controls['passed'], controls['attempted']), ('pass', 15, 15))
        self.assertEqual(sha(ROOT / 'blender_edit.py'), controls['addon_sha256'])
        self.assertEqual(sha(ROOT / 'blender_edit_controls.py'), controls['script_sha256'])

    def test_figure_and_table_are_bound_to_raw_record(self):
        manifest = json.loads((EVIDENCE / 'figure-manifest.json').read_text())
        self.assertEqual(sha(EVIDENCE / 'edit-suite-results.json'), manifest['measured_results_sha256'])
        for name, record in manifest['outputs'].items():
            self.assertEqual(sha(EVIDENCE / name), record['sha256'], name)

    def test_download_archive_members_and_glb_identity_binding(self):
        root = ROOT / 'examples/edit-workflow'
        manifest = json.loads((root / 'manifest.json').read_text())
        self.assertEqual(sha(root / 'combined-study-editable.zip'), manifest['archive']['sha256'])
        with zipfile.ZipFile(root / 'combined-study-editable.zip') as archive:
            self.assertIsNone(archive.testzip())
            for name, record in manifest['members'].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), record['sha256'], name)
            sidecar = json.loads(archive.read('after.identity.json'))
            self.assertEqual(hashlib.sha256(archive.read('after.glb')).hexdigest(), sidecar['asset_sha256'])
            source_faces = [index for obj in sidecar['objects'] for index in obj['source_face_ids']]
            self.assertEqual(len(source_faces), 71967)
            self.assertEqual(len(set(source_faces)), 71967)
            self.assertFalse(sidecar['native_face_attributes_in_glb'])

    def test_original_failure_archive_is_unchanged_and_not_relabelled(self):
        root = ROOT / 'examples/edit-workflow'
        manifest = json.loads((root / 'original-failure-manifest.json').read_text())
        self.assertEqual(sha(root / 'original-verifier-failure.zip'), manifest['archive']['sha256'])
        repaired = json.loads((EVIDENCE / 'repaired-recheck.json').read_text())
        with zipfile.ZipFile(root / 'original-verifier-failure.zip') as archive:
            for name, record in manifest['members'].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), record['sha256'])
            original = json.loads(archive.read('blender-verification.json'))
            self.assertEqual(original['status'], 'fail')
            for name, digest in repaired['assets_sha256'].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)
        self.assertTrue(repaired['assets_unchanged'])
        self.assertTrue(all(check['status'] == 'pass' for check in repaired['checks'].values()))


if __name__ == '__main__':
    unittest.main()
