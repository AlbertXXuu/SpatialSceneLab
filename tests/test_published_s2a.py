"""Keep measured summaries, source bindings and the downloadable task intact."""
import hashlib
import json
from pathlib import Path
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/ownership-s2a'


def digest(data):
    return hashlib.sha256(data).hexdigest()


class PublishedS2a(unittest.TestCase):
    def test_delivered_evidence_bytes_match_manifest(self):
        manifest = json.loads((REPORT / 'manifest.json').read_text(encoding='utf-8'))
        for name, record in manifest['files'].items():
            data = (REPORT / name).read_bytes()
            self.assertEqual(len(data), record['bytes'], name)
            self.assertEqual(digest(data), record['sha256'], name)

    def test_measured_sources_and_all_scenes_are_retained(self):
        diagnostic = json.loads((REPORT / 'diagnostics-results.json').read_text())
        self.assertEqual(diagnostic['status'], 'completed')
        self.assertEqual(diagnostic['completed_scenes'], 6)
        self.assertEqual(diagnostic['findings'], [])
        self.assertEqual(diagnostic['source']['published_results_sha256'],
                         digest((ROOT / 'reports/development-s0-results.json').read_bytes()))
        for name, expected in diagnostic['code_sha256'].items():
            self.assertEqual(digest((ROOT / name).read_bytes()), expected, name)
        for filename, field, code in [('partition-export-results.json', 'exporter_sha256', 'export_partition.py'),
                                     ('native-lineage-results.json', 'verifier_sha256', 'verify_partition_import.py')]:
            result = json.loads((REPORT / filename).read_text())
            self.assertEqual(result['status'], 'pass')
            self.assertEqual(len(result['scenes']), 6)
            self.assertEqual(digest((ROOT / code).read_bytes()), result[field])

    def test_author_task_is_initial_scene_and_complete_archive(self):
        folder = ROOT / 'examples/ownership-b1'
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        archive = folder / manifest['archive']['path']
        self.assertEqual(digest(archive.read_bytes()), manifest['archive']['sha256'])
        with zipfile.ZipFile(archive) as z:
            self.assertEqual(set(z.namelist()), set(manifest['files']) | {'manifest.json'})
            for name, expected in manifest['files'].items():
                data = z.read(name)
                self.assertEqual(digest(data), expected['sha256'], name)
                self.assertEqual(len(data), expected['bytes'], name)
            self.assertEqual(z.read('blender_edit.py'), (ROOT / 'blender_edit.py').read_bytes())
            source = json.loads(z.read('surface/surface-scene.json'))
            self.assertEqual(source['triangles'], 96518)
            self.assertEqual(source['partition']['unresolved_triangles'], 3605)
            for part in source['parts']:
                self.assertEqual(digest(z.read('surface/' + part['path'])), part['sha256'])
                for index in part['source_indices'].values():
                    self.assertEqual(digest(z.read('surface/' + index['path'])), index['sha256'])
            editing = json.loads((REPORT / 'edit-suite-results.json').read_text())
            dining = next(t for t in editing['tasks'] if t['scene_id'] == manifest['scene_id'])
            self.assertEqual(digest(z.read('before.blend')), dining['assets']['before.blend']['sha256'])


if __name__ == '__main__':
    unittest.main()
