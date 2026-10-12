"""Dining integration rejects destructive destinations and stale geometry gates."""
import json
from pathlib import Path
import tempfile
import unittest

from blender_dining_scene import (TARGETS, check_paths, component_path, input_bundle, layout_bundle,
                                 layout_transform_sha256, verify_separating_planes, sha)


class DiningInputBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source_dir = self.root / 'native-source'
        self.source_dir.mkdir()
        self.source = self.source_dir / 'accepted.blend'
        self.source.write_bytes(b'accepted source')
        self.fit_dir = self.root / 'fit'
        self.fit_dir.mkdir()
        objects, hashes = [], {}
        for identity, (category, _) in TARGETS.items():
            target_dir = self.fit_dir / identity
            target_dir.mkdir()
            parts = []
            for index in range(5):
                name = f'part_{index}'
                path = target_dir / f'{name}.ply'
                path.write_bytes(f'bounded mesh {identity} {index}'.encode())
                parts.append({'name': name, 'mesh': identity + '/' + path.name, 'type': 'box',
                              'parameters': {'dimensions_m': [1, 1, 1]},
                              'provenance_role': 'fitted/structural', 'editable': True})
                hashes[identity + '/' + name] = sha(path)
            objects.append({'uuid': identity, 'category': category, 'components': parts})
        self.fit = {'schema': 'alvenx.spatial.dining-fit.v1', 'units': 'm', 'up_axis': 'Y',
                    'quality_gate': {'passed': True}, 'objects': objects}
        self.fit_path = self.fit_dir / 'fit.json'
        self.fit_path.write_text(json.dumps(self.fit), encoding='utf-8')
        self.evaluation = {'schema': 'alvenx.spatial.dining-evaluation.v1', 'status': 'passed',
                           'quality_gate': {'passed': True}, 'fit_json_sha256': sha(self.fit_path),
                           'component_input_sha256': hashes,
                           'objects': [{'uuid': identity, 'observation_gate': {'passed': True},
                                        'structural_gate': {'passed': True}} for identity in TARGETS]}
        self.evaluation_path = self.root / 'evaluation.json'
        self.evaluation_path.write_text(json.dumps(self.evaluation), encoding='utf-8')
        stool_hashes, input_files = {}, []
        for name in ('seat', 'leg_1', 'leg_2', 'leg_3', 'brace_1_2', 'brace_2_3', 'brace_3_1'):
            path = self.root / (name + '.ply')
            path.write_bytes(('accepted stool ' + name).encode())
            stool_hashes[name] = sha(path)
            input_files.append({'path': path.name, 'sha256': sha(path)})
        translations = [{'uuid': identity, 'translation_world_m': delta} for identity, delta in (
            ('same-table', [0., 0., 0.]), ('same-chair-west', [-.12, -0., 0.]),
            ('same-chair-east', [.11, 0., -.015]), ('same-stool', [0., 0., 0.]))]
        self.layout = {'schema': 'alvenx.spatial.dining-layout.v1', 'status': 'passed', 'units': 'm', 'up_axis': 'Y',
                       'quality_gate': {'passed': True, 'all_actual_component_pairs_certified': True,
                                        'frozen_geometry_unchanged': True, 'required_cross_furniture_clearance_m': .005},
                       'fit_json_sha256': sha(self.fit_path), 'component_input_sha256': hashes,
                       'stool_source_blend_sha256': sha(self.source), 'stool_component_input_sha256': stool_hashes,
                       'input_files': input_files, 'objects': translations,
                       'layout_transform_sha256': layout_transform_sha256(translations),
                       'authored_layout': {'passed': True, 'failed_pair_count': 0, 'pairs': [{}]}}
        self.layout_path = self.root / 'layout.json'
        self.write_layout()

    def write_layout(self):
        self.layout_path.write_text(json.dumps(self.layout), encoding='utf-8')

    def read_layout(self):
        return layout_bundle(self.layout_path, self.source, sha(self.fit_path), self.evaluation['component_input_sha256'])

    def write_fit(self):
        self.fit_path.write_text(json.dumps(self.fit), encoding='utf-8')
        self.evaluation['fit_json_sha256'] = sha(self.fit_path)
        self.write_evaluation()

    def write_evaluation(self):
        self.evaluation_path.write_text(json.dumps(self.evaluation), encoding='utf-8')

    def test_valid_explicit_bundle_and_new_output(self):
        check_paths(self.source, self.fit_dir, self.evaluation_path, self.layout_path, self.root / 'output', 'build')
        fit, _, sources, hashes, evaluation = input_bundle(self.fit_dir, self.evaluation_path)
        self.assertEqual(len(sources), 15)
        self.assertEqual(hashes, evaluation['component_input_sha256'])
        self.assertEqual(fit['objects'][0]['uuid'], 'same-table')
        layout, translations, stool_sources = self.read_layout()
        self.assertEqual(translations['same-chair-west'], [-.12, -0., 0.])
        self.assertEqual(len(stool_sources), 7)
        self.assertEqual(layout['layout_transform_sha256'], layout_transform_sha256(layout['objects']))

    def test_source_and_fit_destinations_are_not_overwritten(self):
        for output in (self.source_dir, self.root, self.fit_dir, self.fit_dir / 'nested',
                       self.root / 'frozen' / 'result', self.root / 'inputs' / 'result'):
            with self.subTest(output=output), self.assertRaises(ValueError):
                check_paths(self.source, self.fit_dir, self.evaluation_path, self.layout_path, output, 'build')
        occupied = self.root / 'occupied'
        occupied.mkdir()
        keep = occupied / 'user-work.blend'
        keep.write_bytes(b'keep this')
        with self.assertRaises(ValueError):
            check_paths(self.source, self.fit_dir, self.evaluation_path, self.layout_path, occupied, 'build')
        self.assertEqual(keep.read_bytes(), b'keep this')
        self.assertEqual(self.source.read_bytes(), b'accepted source')

    def test_missing_input_and_reopen_output(self):
        with self.assertRaises(ValueError):
            check_paths(self.source, self.fit_dir, self.evaluation_path, self.layout_path, self.root / 'missing', 'reopen')
        with self.assertRaises(ValueError):
            check_paths(self.root / 'missing.blend', self.fit_dir, self.evaluation_path, self.layout_path, self.root / 'new', 'build')

    def test_layout_input_must_exist_and_cannot_be_overwritten(self):
        with self.assertRaisesRegex(ValueError, 'layout JSON'):
            check_paths(self.source, self.fit_dir, self.evaluation_path, self.root / 'absent.json', self.root / 'new', 'build')
        directory = self.root / 'layout-directory'
        directory.mkdir()
        layout_path = directory / 'layout.json'
        layout_path.write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'layout inputs'):
            check_paths(self.source, self.fit_dir, self.evaluation_path, layout_path, directory, 'build')

    def test_layout_rejects_failed_or_incomplete_actual_separation(self):
        for field, value in (('passed', False), ('all_actual_component_pairs_certified', False),
                             ('frozen_geometry_unchanged', False), ('required_cross_furniture_clearance_m', .0049)):
            with self.subTest(field=field):
                before = self.layout['quality_gate'][field]
                self.layout['quality_gate'][field] = value
                self.write_layout()
                with self.assertRaisesRegex(ValueError, 'actual-solid layout gate'):
                    self.read_layout()
                self.layout['quality_gate'][field] = before
        self.layout['authored_layout']['pairs'] = []
        self.write_layout()
        with self.assertRaisesRegex(ValueError, 'every actual-vertex'):
            self.read_layout()

    def test_layout_binds_frozen_fit_and_stool_bytes(self):
        self.layout['fit_json_sha256'] = 'stale'
        self.write_layout()
        with self.assertRaisesRegex(ValueError, 'exact frozen'):
            self.read_layout()
        self.layout['fit_json_sha256'] = sha(self.fit_path)
        self.write_layout()
        (self.root / 'seat.ply').write_bytes(b'changed accepted stool')
        with self.assertRaisesRegex(ValueError, 'stool component bytes changed'):
            self.read_layout()

    def test_authored_translation_hash_cannot_be_reused_for_another_layout(self):
        self.layout['objects'][1]['translation_world_m'][0] -= .01
        self.write_layout()
        with self.assertRaisesRegex(ValueError, 'transform hash'):
            self.read_layout()
        self.layout['layout_transform_sha256'] = layout_transform_sha256(self.layout['objects'])
        self.write_layout()
        self.assertEqual(self.read_layout()[1]['same-chair-west'][0], -.13)

    def test_layout_translations_are_horizontal_finite_and_leave_stool_unchanged(self):
        for delta in ([0, .01, 0], [float('nan'), 0, 0], [True, 0, 0], [0, 0]):
            with self.subTest(delta=delta):
                self.layout['objects'][1]['translation_world_m'] = delta
                self.write_layout()
                with self.assertRaisesRegex(ValueError, 'finite world-horizontal'):
                    self.read_layout()
        self.layout['objects'][1]['translation_world_m'] = [-.12, 0, 0]
        self.layout['objects'][3]['translation_world_m'] = [.01, 0, 0]
        self.write_layout()
        with self.assertRaisesRegex(ValueError, 'stool must remain unchanged'):
            self.read_layout()

    def test_native_separation_checks_every_vertex_and_every_component_pair(self):
        vertices = {'table/top': [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                    'chair/seat': [[1.006, 0, 0], [2, 0, 0], [1.006, 1, 0]]}
        certificate = {'parts': ['table/top', 'chair/seat'], 'passed': True,
                       'separating_plane': {'unit_normal_world': [1, 0, 0],
                                            'first_projection_max_m': 1., 'second_projection_min_m': 1.006,
                                            'gap_m': .006}}
        layout = {'component_input_sha256': dict.fromkeys(vertices, 'bound-input'),
                  'stool_component_input_sha256': {},
                  'quality_gate': {'required_cross_furniture_clearance_m': .005},
                  'authored_layout': {'pairs': [certificate], 'cross_object_pair_count': 1}}
        self.assertGreater(verify_separating_planes(layout, vertices)['minimum_actual_vertex_plane_clearance_m'], .005)
        vertices['table/top'].append([1.003, .5, 0])
        with self.assertRaisesRegex(ValueError, 'violate the accepted layout halfspaces'):
            verify_separating_planes(layout, vertices)
        vertices['table/top'].pop()
        layout['authored_layout']['pairs'].append(certificate)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            verify_separating_planes(layout, vertices)
        layout['authored_layout']['pairs'] = []
        with self.assertRaisesRegex(ValueError, 'cover every'):
            verify_separating_planes(layout, vertices)
        layout['authored_layout']['pairs'] = [certificate]
        vertices['chair/seat'][0][0] = vertices['chair/seat'][2][0] = 1.0049
        certificate['separating_plane']['second_projection_min_m'] = 1.0049
        certificate['separating_plane']['gap_m'] = .0049
        with self.assertRaisesRegex(ValueError, 'fail the required separation'):
            verify_separating_planes(layout, vertices)
        certificate['separating_plane']['unit_normal_world'] = [2, 0, 0]
        with self.assertRaisesRegex(ValueError, 'unit normal'):
            verify_separating_planes(layout, vertices)

    def test_escaping_component_paths(self):
        outside = self.root / 'outside.ply'
        outside.write_bytes(b'outside')
        for filename in ('../outside.ply', '/outside.ply', 'C:/outside.ply', 'same-table\\part_0.ply',
                         'same-table/part_0.ply. ', 'same-table/part_0.json'):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                component_path(self.fit_dir, filename)

    def test_failed_geometry_gate(self):
        self.evaluation['objects'][1]['structural_gate']['passed'] = False
        self.write_evaluation()
        with self.assertRaisesRegex(ValueError, 'Every target'):
            input_bundle(self.fit_dir, self.evaluation_path)

    def test_evaluated_domed_mesh_is_imported_without_regeneration(self):
        part = self.fit['objects'][1]['components'][0]
        part['type'] = 'domed_box'
        part['parameters'].update({'center_m': [0, 0, 0],
                                   'rotation_world_3x3': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                   'cap_axis': 1, 'cap_sign': 1, 'cap_height_m': .025,
                                   'profile_power': .5})
        source_path = self.fit_dir / part['mesh']
        original_bytes = source_path.read_bytes()
        self.write_fit()
        fit, _, sources, _, _ = input_bundle(self.fit_dir, self.evaluation_path)
        self.assertEqual(fit['objects'][1]['components'][0]['parameters']['cap_height_m'], .025)
        self.assertEqual(sources['same-chair-west/part_0'], source_path)
        self.assertEqual(source_path.read_bytes(), original_bytes)
        self.evaluation['quality_gate']['passed'] = False
        self.write_evaluation()
        with self.assertRaisesRegex(ValueError, 'evaluation must pass'):
            input_bundle(self.fit_dir, self.evaluation_path)

    def test_unimplemented_component_type_is_rejected(self):
        self.fit['objects'][1]['components'][0]['type'] = 'freeform_hull'
        self.write_fit()
        with self.assertRaisesRegex(ValueError, 'Unexpected component type'):
            input_bundle(self.fit_dir, self.evaluation_path)

    def test_gated_half_ellipsoid_component_keeps_supplied_mesh(self):
        part = self.fit['objects'][1]['components'][0]
        part['type'] = 'half_ellipsoid'
        part['parameters'] = {'center_m': [0, 0, 0], 'radii_m': [.2, .025, .2],
                              'rotation_world_3x3': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                              'cap_axis': 1, 'cap_sign': 1}
        self.write_fit()
        fit, _, sources, hashes, _ = input_bundle(self.fit_dir, self.evaluation_path)
        self.assertEqual(fit['objects'][1]['components'][0]['type'], 'half_ellipsoid')
        self.assertEqual(sha(sources['same-chair-west/part_0']), hashes['same-chair-west/part_0'])

    def test_stale_fit_hash(self):
        self.fit['assumptions'] = ['source changed after evaluation']
        self.fit_path.write_text(json.dumps(self.fit), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'exact fit'):
            input_bundle(self.fit_dir, self.evaluation_path)

    def test_stale_component_hash(self):
        path = self.fit_dir / 'same-table' / 'part_0.ply'
        path.write_bytes(b'changed geometry after evaluation')
        with self.assertRaisesRegex(ValueError, 'exact fit'):
            input_bundle(self.fit_dir, self.evaluation_path)

    def test_duplicate_components_and_target_identity(self):
        self.fit['objects'][0]['components'][1]['name'] = 'part_0'
        self.write_fit()
        with self.assertRaisesRegex(ValueError, 'unique'):
            input_bundle(self.fit_dir, self.evaluation_path)
        self.fit['objects'][0]['components'][1]['name'] = 'part_1'
        self.fit['objects'][1]['uuid'] = 'same-table'
        self.write_fit()
        with self.assertRaisesRegex(ValueError, 'exactly the three'):
            input_bundle(self.fit_dir, self.evaluation_path)

    def test_wrong_units_and_unaccepted_fit(self):
        for mutation in ({'up_axis': 'Z'}, {'quality_gate': {'passed': False}}):
            before = dict(self.fit)
            self.fit.update(mutation)
            self.write_fit()
            with self.assertRaises(ValueError):
                input_bundle(self.fit_dir, self.evaluation_path)
            self.fit = before


if __name__ == '__main__':
    unittest.main()
