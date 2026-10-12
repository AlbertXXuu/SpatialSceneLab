"""Layout collision cases that defeat vertex-containment and box-only checks."""
import hashlib
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


OPEN3D_AVAILABLE = importlib.util.find_spec("open3d") is not None
if OPEN3D_AVAILABLE:
    import evaluate_dining_layout as e


@unittest.skipUnless(OPEN3D_AVAILABLE, "optional Open3D runtime absent")
class DiningLayoutTests(unittest.TestCase):
    def part(self, mesh, uuid="first", name="part"):
        planes = e.convex_planes(mesh)
        self.assertIsNotNone(planes)
        vertices = np.asarray(mesh.vertices).copy()
        vertices.flags.writeable = False
        return {"uuid": uuid, "name": name, "key": uuid + "/" + name,
                "vertices": vertices, "triangles": np.asarray(mesh.triangles).copy(), "planes": planes}

    def box(self, size, center, uuid="first", name="part"):
        size, center = np.asarray(size), np.asarray(center)
        mesh = e.q.o3d.geometry.TriangleMesh.create_box(*size).translate(center - size / 2)
        return self.part(mesh, uuid, name)

    def test_crossing_boxes_without_any_contained_vertex_are_intersections(self):
        first = self.box([2, .2, .2], [0, 0, 0])
        second = self.box([.2, 2, .2], [0, 0, 0], "second")
        self.assertFalse(np.any(np.max(first["vertices"] @ second["planes"][:, :3].T + second["planes"][:, 3], axis=1) < 0))
        self.assertFalse(np.any(np.max(second["vertices"] @ first["planes"][:, :3].T + first["planes"][:, 3], axis=1) < 0))
        row = e.check_pair(first, second)
        self.assertFalse(row["passed"])
        self.assertEqual(row["classification"], "positive_solid_intersection")
        self.assertGreater(row["intersection"]["common_interior_ball_radius_lower_bound_m"], .09)

    def test_crossing_cylinders_without_contained_vertices_are_intersections(self):
        first_mesh = e.q.o3d.geometry.TriangleMesh.create_cylinder(radius=.1, height=2., resolution=32, split=1)
        second_mesh = e.q.o3d.geometry.TriangleMesh.create_cylinder(radius=.1, height=2., resolution=32, split=1)
        second_mesh.rotate(e.q.o3d.geometry.get_rotation_matrix_from_xyz([0, np.pi / 2, 0]), center=[0, 0, 0])
        first, second = self.part(first_mesh), self.part(second_mesh, "second")
        self.assertFalse(np.any(np.max(first["vertices"] @ second["planes"][:, :3].T + second["planes"][:, 3], axis=1) < 0))
        self.assertFalse(np.any(np.max(second["vertices"] @ first["planes"][:, :3].T + first["planes"][:, 3], axis=1) < 0))
        row = e.check_pair(first, second)
        self.assertFalse(row["passed"])
        self.assertTrue(row["intersection"]["positive_solid_intersection"])

    def test_touching_subthreshold_and_five_mm_separation_are_distinct(self):
        first = self.box([1, 1, 1], [0, 0, 0])
        for gap, passed in ((0., False), (.0049, False), (.005, True), (.007, True)):
            with self.subTest(gap=gap):
                second = self.box([1, 1, 1], [1 + gap, 0, 0], "second")
                row = e.check_pair(first, second, clearance=.005)
                self.assertEqual(row["passed"], passed)
                self.assertAlmostEqual(row["certified_clearance_lower_bound_m"], gap, places=10)

    def test_overlapping_aabbs_do_not_force_collision_for_rotated_solids(self):
        first_mesh = e.q.o3d.geometry.TriangleMesh.create_box(2, .1, .1).translate([-1, -.05, -.05])
        rotation = e.q.o3d.geometry.get_rotation_matrix_from_xyz([0, 0, np.pi / 4])
        first_mesh.rotate(rotation, center=[0, 0, 0])
        second_mesh = e.q.o3d.geometry.TriangleMesh(first_mesh).translate([-.2, .2, 0])
        first, second = self.part(first_mesh), self.part(second_mesh, "second")
        row = e.check_pair(first, second)
        self.assertEqual(row["broadphase_aabb_distance_lower_bound_m"], 0.)
        self.assertTrue(row["passed"])
        self.assertAlmostEqual(row["certified_clearance_lower_bound_m"], np.sqrt(.2 ** 2 + .2 ** 2) - .1, places=8)

    def test_separating_plane_certificate_is_recomputable_from_all_vertices(self):
        first = self.box([1, 1, 1], [0, 0, 0])
        second = self.box([1, 1, 1], [2, 2, 0], "second")
        row = e.check_pair(first, second)
        certificate = row["separating_plane"]
        normal = np.asarray(certificate["unit_normal_world"])
        self.assertAlmostEqual(np.linalg.norm(normal), 1., places=12)
        maximum = float((first["vertices"] @ normal).max())
        minimum = float((second["vertices"] @ normal).min())
        self.assertAlmostEqual(maximum, certificate["first_projection_max_m"], places=12)
        self.assertAlmostEqual(minimum, certificate["second_projection_min_m"], places=12)
        self.assertGreaterEqual(minimum - maximum, .005)

    def test_same_furniture_structural_contacts_are_allowed(self):
        first = self.box([1, 1, 1], [0, 0, 0], "same-table", "top")
        support = self.box([.2, 1, .2], [0, -.4, 0], "same-table", "leg")
        chair = self.box([.5, .5, .5], [3, 0, 0], "same-chair-west", "seat")
        report = e.check_layout([first, support, chair], {"same-table": np.zeros(3), "same-chair-west": np.zeros(3)})
        self.assertTrue(report["passed"])
        self.assertEqual(report["cross_object_pair_count"], 2)
        self.assertFalse(any(row["parts"] == ["same-table/top", "same-table/leg"] for row in report["pairs"]))

    def test_authored_root_translation_keeps_source_vertices_and_planes_unchanged(self):
        source = self.box([1, 1, 1], [0, 0, 0])
        vertices, planes = source["vertices"].copy(), source["planes"].copy()
        moved = e.transformed(source, [.1, 0, -.2])
        np.testing.assert_array_equal(source["vertices"], vertices)
        np.testing.assert_array_equal(source["planes"], planes)
        np.testing.assert_allclose(moved["vertices"], vertices + [.1, 0, -.2])
        self.assertFalse(np.shares_memory(source["vertices"], moved["vertices"]))
        self.assertLessEqual(np.max(moved["vertices"] @ moved["planes"][:, :3].T + moved["planes"][:, 3]), 1e-8)

    def test_vertical_or_mismatched_authored_layout_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="AlvenX-dining-layout-") as temporary:
            path = Path(temporary) / "layout.json"
            document = {"schema": "alvenx.spatial.dining-layout.v1", "objects": [
                {"uuid": uuid, "translation_world_m": [0, 0, 0]} for uuid in e.OBJECTS]}
            document["objects"][1]["translation_world_m"][1] = .001
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "horizontal"):
                e.load_translations(path)
            document["objects"][1]["translation_world_m"][1] = 0
            document["objects"][1]["uuid"] = "same-stool"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identities"):
                e.load_translations(path)

    def test_minimum_proof_rejects_different_obstacles_with_a_feasible_gap(self):
        moving = self.box([.2, .2, .2], [0, 0, 0], "same-chair-west", "seat")
        first = self.box([.2, .2, .2], [0, 0, 0], "same-table", "first")
        second = self.box([.2, .2, .2], [3, 0, 0], "same-table", "second")
        translations = {"same-chair-west": np.zeros(3), "same-table": np.zeros(3)}
        with self.assertRaisesRegex(ValueError, "no shared blocking pair"):
            e.minimum_shift_certificate([moving, first, second], translations, "same-chair-west",
                                        np.array([1., 0, 0]), 3., .0055)
        midway = {**translations, "same-chair-west": np.array([1.5, 0, 0])}
        self.assertTrue(e.check_layout([moving, first, second], midway)["passed"])

    def test_frozen_published_layout_failure_and_minimum_correction_are_real(self):
        repository = Path(__file__).resolve().parents[1]
        fit_dir, stool_dir = repository / "examples/dining-complete/fitted", repository / "examples/stool-fit/fitted"
        fit, _, parts, hashes = e.load_parts(fit_dir, stool_dir)
        original = {uuid: np.zeros(3) for uuid in e.OBJECTS}
        before = e.check_layout(parts, original)
        self.assertFalse(before["passed"])
        self.assertTrue(any(row["parts"] == ["same-table/leg_1", "same-chair-west/seat"] and
                            row["classification"] == "positive_solid_intersection" for row in before["pairs"]))
        translations, searches = e.propose_layout(fit, parts)
        after = e.check_layout(parts, translations)
        self.assertTrue(after["passed"])
        self.assertEqual(after["cross_object_pair_count"], 507)
        self.assertGreaterEqual(after["minimum_certified_clearance_m"], .0055)
        for search in searches:
            self.assertLessEqual(search["minimum_shift_bracket_m"][1] - search["minimum_shift_bracket_m"][0], 1e-6)
            certificate = search["minimum_shift_certificate"]
            self.assertEqual(certificate["blocked_shift_interval_m"], [0., search["minimum_shift_bracket_m"][0]])
            for witness in certificate["endpoint_witnesses"]:
                a, b = np.asarray(witness["first_point_m"]), np.asarray(witness["second_point_m"])
                self.assertLess(np.linalg.norm(a - b) + certificate["numerical_distance_headroom_m"], .0055)
                for key, point in zip(certificate["parts"], (a, b)):
                    part = next(part for part in parts if part["key"] == key)
                    delta = (np.asarray(search["outward_unit_direction_world"]) * witness["shift_m"] if part["uuid"] == search["uuid"]
                             else certificate["fixed_object_translations_world_m"][part["uuid"]])
                    moved = e.transformed(part, delta)
                    self.assertLessEqual(float((moved["planes"][:, :3] @ point + moved["planes"][:, 3]).max()), 1e-8)
        for path, digest in hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


@unittest.skipUnless(OPEN3D_AVAILABLE, "optional Open3D runtime absent")
class DiningLayoutCliTests(unittest.TestCase):
    def check_exit_and_saved_report(self, status, passed, expected):
        with tempfile.TemporaryDirectory(prefix="AlvenX-layout-cli-") as temporary:
            output = Path(temporary) / "diagnostics.json"
            failed_pair = {"parts": ["same-table/leg_1", "same-chair-west/seat"], "passed": False}
            report = {"schema": "alvenx.spatial.dining-layout.v1", "status": status,
                      "quality_gate": {"passed": passed}, "objects": [],
                      "original_observed_layout": {"pairs": [failed_pair]},
                      "authored_layout": {"minimum_certified_clearance_m": .003}}
            arguments = ["evaluate_dining_layout.py", "--fit-dir", "fitted", "--stool-fit-dir", "stool",
                         "--propose", "--output", str(output)]
            console = io.StringIO()
            with mock.patch.object(sys, "argv", arguments), mock.patch.object(e, "evaluate", return_value=report), contextlib.redirect_stdout(console):
                with self.assertRaises(SystemExit) as stopped:
                    e.main()
            self.assertEqual(stopped.exception.code, expected)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
            self.assertEqual(json.loads(console.getvalue())["original_failed_pairs"], [failed_pair])

    def test_failed_cli_keeps_colliding_pairs_and_exits_nonzero(self):
        for status, passed in (("failed", False), ("passed", False), ("failed", True)):
            with self.subTest(status=status, quality_gate=passed):
                self.check_exit_and_saved_report(status, passed, 2)

    def test_passed_cli_keeps_report_and_exits_zero(self):
        self.check_exit_and_saved_report("passed", True, 0)


if __name__ == "__main__":
    unittest.main()
