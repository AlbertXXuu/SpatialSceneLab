"""Check byte identities and truthful task denominators in published evidence."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'reports' / 'development-s0'


class PublishedEvidenceTests(unittest.TestCase):
    def test_figure_sources_and_assets_survive_git_checkout(self):
        manifest = json.loads((ASSETS / 'figure-manifest.json').read_text(encoding='utf-8'))
        for path, expected in (
            (ROOT / 'reports/development-s0-results.json', manifest['results_sha256']),
            (ASSETS / 'suite-manifest.json', manifest['input_manifest_sha256']),
        ):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        for name, record in manifest['files'].items():
            data = (ASSETS / name).read_bytes()
            self.assertEqual(len(data), record['bytes'])
            self.assertEqual(hashlib.sha256(data).hexdigest(), record['sha256'])

    def test_published_development_and_sensitivity_share_one_frozen_run(self):
        path = ROOT / 'reports/development-s0-results.json'
        development = json.loads(path.read_text(encoding='utf-8'))
        sensitivity = json.loads((ASSETS / 'sensitivity-results.json').read_text(encoding='utf-8'))
        self.assertEqual(sensitivity['source']['development_results_sha256'],
                         hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(development['attempted_scenes'], len(development['scenes']))
        self.assertEqual(development['attempted_method_tasks'], 18)
        self.assertEqual(development['completed_method_tasks'], 18)
        self.assertEqual(sensitivity['attempted_evaluations'], 108)
        self.assertEqual(sensitivity['completed_evaluations'], 108)
        self.assertEqual(sensitivity['baseline_consistency']['checked_method_tasks'], 18)
        self.assertEqual(sensitivity['baseline_consistency']['status'], 'pass')

    def test_native_failure_is_retained_alongside_limited_before_diagnosis(self):
        failure = json.loads((ASSETS / 'native-export-failure.json').read_text(encoding='utf-8'))
        diagnosis = json.loads((ASSETS / 'before-diagnostic.json').read_text(encoding='utf-8'))
        self.assertEqual(failure['status'], 'fail')
        self.assertEqual(diagnosis['assets_sha256']['before.glb'], failure['outputs']['before.glb']['sha256'])
        self.assertTrue(diagnosis['assets_unchanged'])
