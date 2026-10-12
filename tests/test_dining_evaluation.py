"""Adversarial acceptance checks with small independently known box geometry."""
import copy
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
    import evaluate_dining as e


@unittest.skipUnless(OPEN3D_AVAILABLE, "optional Open3D geometry runtime absent")
class DiningEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="AlvenX-dining-evaluation-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fit_dir = self.root / "fit"
        self.fit_dir.mkdir()
        self.scene_path = self.root / "scene.json"
        self.observation_path = self.root / "observations.npz"
        self.scene = {"units": "m", "up_axis": "Y", "objects": [
            {"uuid": uuid, "category": "Table" if index == 0 else "Chair"} for index, uuid in enumerate(e.TARGETS)],
            "structures": [{"category": "Floor", "transform_world": np.eye(4).tolist(), "dimensions_m": [10, .02, 10]}]}
        self.scene["structures"][0]["transform_world"][1][3] = -.01
        self.scene_path.write_text(json.dumps(self.scene), encoding="utf-8")
        self.fit = {"schema": "alvenx.spatial.dining-fit.v1", "units": "m", "up_axis": "Y", "objects": [],
                    "quality_gate": {"passed": True}, "input_files": {}}
        self.points, self.frames, self.owners, self.point_parts = [], [], [], []
        for owner, uuid in enumerate(e.TARGETS):
            offset = owner * 2
            components = [self.component(uuid, "top" if owner == 0 else "seat", [offset, .51, 0], [.9, .08, .7], "top" if owner == 0 else "seat")]
            for index, (x, z) in enumerate(((-.36, -.26), (-.36, .26), (.36, -.26), (.36, .26))):
                components.append(self.component(uuid, f"leg_{index}", [offset + x, .24, z], [.08, .48, .08], "floor_support"))
            if owner:
                components.append(self.component(uuid, "back", [offset, .84, -.3], [.9, .74, .08], "back"))
            primary = components[0]["name"]
            self.fit["objects"].append({"uuid": uuid, "category": self.scene["objects"][owner]["category"],
                "source_owner_index": owner, "mesh": f"{uuid}.ply", "components": components,
                "contact_floor": {"floor_y_m": 0., "feet": [{"name": f"leg_{index}"} for index in range(4)],
                                  "joints": [{"parts": [primary, item["name"]]} for item in components[1:]]}})
            for item in components:
                p = item["parameters"]
                center, size = np.asarray(p["center_m"]), np.asarray(p["dimensions_m"])
                # Exterior side-face centres give every primitive distinct evidence.
                point = center + np.array([size[0] / 2, 0, 0])
                if item["role"] in ("top", "seat"):
                    point = center + np.array([0, size[1] / 2, 0])
                for frame in (0, 2, 1):
                    self.points.append(point)
                    self.frames.append(frame)
                    self.owners.append(owner)
                    self.point_parts.append(uuid + "/" + item["name"])
        self.points = np.asarray(self.points)
        self.frames, self.owners = np.asarray(self.frames), np.asarray(self.owners)
        self.save_observations()
        self.save_meshes()
        self.save_fit()

    def component(self, uuid, name, center, dimensions, role):
        return {"name": name, "role": role, "primitive_type": "box", "mesh": f"{uuid}__{name}.ply",
                "parameters": {"center_m": center, "dimensions_m": dimensions, "rotation_world_3x3": np.eye(3).tolist()}}

    def save_observations(self):
        candidates = self.owners[:, None] == np.arange(len(e.TARGETS))[None, :]
        np.savez(self.observation_path, points=self.points, frame_index=self.frames, owner=self.owners, candidates=candidates)

    def save_meshes(self):
        for obj in self.fit["objects"]:
            meshes = []
            for component in obj["components"]:
                p = component["parameters"]
                size = np.asarray(p["dimensions_m"])
                mesh = e.q.o3d.geometry.TriangleMesh.create_box(*size)
                mesh.translate(np.asarray(p["center_m"]) - size / 2)
                e.q.o3d.io.write_triangle_mesh(str(self.fit_dir / component["mesh"]), mesh)
                meshes.append(mesh)
            assembly = e.q.o3d.geometry.TriangleMesh()
            for mesh in meshes:
                assembly += mesh
            e.q.o3d.io.write_triangle_mesh(str(self.fit_dir / obj["mesh"]), assembly)

    def save_fit(self):
        self.fit["input_files"] = {name: {"path": "Z:/unavailable/old-machine-path", "sha256": e.q.sha256(path)}
                                   for name, path in (("scene", self.scene_path), ("observations", self.observation_path))}
        (self.fit_dir / "fit.json").write_text(json.dumps(self.fit), encoding="utf-8")

    def evaluate(self):
        # Reference sampling is exercised separately; acceptance uses all raw input.
        return e.evaluate(self.scene_path, self.observation_path, self.fit_dir, report_base=self.root)

    def test_known_solids_pass_without_using_embedded_machine_paths(self):
        report = self.evaluate()
        self.assertTrue(report["quality_gate"]["passed"])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["component_input_sha256"]), 17)
        self.assertEqual(report["inputs"]["scene"]["path"], "scene.json")
        json.dumps(report, allow_nan=False)

    def test_mismatched_target_and_owner_identity_rejected(self):
        for failure in ("uuid", "owner", "category"):
            with self.subTest(failure=failure):
                original = copy.deepcopy(self.fit)
                if failure == "uuid":
                    self.fit["objects"][0]["uuid"] = "same-stool"
                elif failure == "owner":
                    self.fit["objects"][0]["source_owner_index"] = 2
                else:
                    self.fit["objects"][0]["category"] = "Chair"
                self.save_fit()
                with self.assertRaisesRegex(ValueError, "identities|owner|category"):
                    self.evaluate()
                self.fit = original

    def test_changed_original_input_invalidates_declared_fit_hash(self):
        with self.scene_path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        report = self.evaluate()
        self.assertFalse(report["quality_gate"]["input_content_matches_fit_record"])
        self.assertEqual(report["status"], "failed")

    def test_changed_ply_fails_actual_parameter_and_assembly_checks(self):
        component = self.fit["objects"][0]["components"][1]
        path = self.fit_dir / component["mesh"]
        mesh, _ = e.q.read_mesh(path)
        mesh.translate([.01, 0, 0])
        e.q.o3d.io.write_triangle_mesh(str(path), mesh)
        report = self.evaluate()
        structural = report["objects"][0]["structural_gate"]
        self.assertFalse(structural["checks"]["parameters_match_actual_triangles"])
        self.assertFalse(structural["checks"]["assembly_equals_components"])
        self.assertFalse(report["quality_gate"]["passed"])

    def test_mixed_triangle_winding_cannot_pass_with_positive_total_volume(self):
        path = self.fit_dir / self.fit["objects"][0]["components"][1]["mesh"]
        mesh, _ = e.q.read_mesh(path)
        faces = np.asarray(mesh.triangles).copy()
        faces[0] = faces[0, ::-1]
        e.q.o3d.io.write_triangle_mesh(str(path), e.q.arrays_mesh(np.asarray(mesh.vertices), faces))
        row = self.evaluate()["objects"][0]
        self.assertFalse(row["structural_gate"]["checks"]["all_parts_watertight_outward_positive_volume"])

    def test_validation_outlier_cannot_be_excluded_by_fitter(self):
        validation = np.flatnonzero((self.frames == 1) & (self.owners == 0))
        self.points[validation[0]] += [0, .03, 0]
        self.fit["objects"][0]["exclusions"] = [int(validation[0])]
        self.save_observations()
        self.save_fit()
        report = self.evaluate()
        row = report["objects"][0]
        self.assertFalse(row["observation_gate"]["passed"])
        self.assertEqual(row["exclusions"]["count_from_validation_gate"], 0)
        self.assertEqual(row["observed_point_mesh_distance"]["validation_odd"]["sample_count"], len(validation))
        self.assertGreater(row["observed_point_mesh_distance"]["validation_odd"]["p95_distance_m"], .01)

    def test_component_without_training_support_fails_even_with_pass_flag(self):
        remove = np.asarray(self.point_parts) == "same-table/leg_0"
        keep = ~remove | (self.frames % 2 == 1)
        self.points, self.frames, self.owners = self.points[keep], self.frames[keep], self.owners[keep]
        self.save_observations()
        self.save_fit()
        row = self.evaluate()["objects"][0]
        self.assertTrue(row["observation_gate"]["passed"])
        self.assertFalse(row["structural_gate"]["checks"]["every_part_has_measured_training_support"])
        leg = next(item for item in row["components"] if item["name"] == "leg_0")
        self.assertEqual(leg["training_support"]["within_10mm_count"], 0)

    def test_candidate_even_support_is_measured_independently_of_hard_owner(self):
        selected = (np.asarray(self.point_parts) == "same-table/leg_0") & (self.frames % 2 == 0)
        self.owners[selected] = -1
        candidates = self.owners[:, None] == np.arange(len(e.TARGETS))[None, :]
        candidates[selected, 0] = True
        np.savez(self.observation_path, points=self.points, frame_index=self.frames, owner=self.owners, candidates=candidates)
        self.save_fit()
        row = self.evaluate()["objects"][0]
        self.assertTrue(row["observation_gate"]["passed"])
        self.assertTrue(row["structural_gate"]["passed"])
        self.assertGreater(row["training_support_pool"]["candidate_even_point_count"], row["training_support_pool"]["original_owner_even_point_count"])
        self.assertEqual(next(item for item in row["components"] if item["name"] == "leg_0")["training_support"]["within_10mm_count"], 2)

    def test_malformed_candidate_membership_is_rejected(self):
        np.savez(self.observation_path, points=self.points, frame_index=self.frames, owner=self.owners,
                 candidates=np.zeros((len(self.points), 2), dtype=bool))
        self.save_fit()
        with self.assertRaisesRegex(ValueError, "malformed observation"):
            self.evaluate()

    def test_float_and_detached_component_fail_floor_or_contact(self):
        component = self.fit["objects"][0]["components"][1]
        component["parameters"]["center_m"][1] += .004
        self.save_meshes()
        self.save_fit()
        self.assertFalse(self.evaluate()["objects"][0]["structural_gate"]["checks"]["feet_within_3mm_of_measured_floor"])
        component["parameters"]["center_m"][1] -= .004
        component["parameters"]["center_m"][0] += 1
        self.save_meshes()
        self.save_fit()
        row = self.evaluate()["objects"][0]
        self.assertFalse(row["structural_gate"]["checks"]["all_declared_contacts_have_positive_overlap"])
        self.assertFalse(row["structural_gate"]["checks"]["intended_contact_graph_connects_every_component"])

    def test_missing_contact_cannot_be_hidden_by_deleting_joint(self):
        self.fit["objects"][0]["contact_floor"]["joints"].pop()
        self.save_fit()
        self.assertFalse(self.evaluate()["objects"][0]["structural_gate"]["checks"]["intended_contact_graph_connects_every_component"])

    def test_missing_family_part_cannot_be_hidden_with_role_changes(self):
        self.fit["objects"][1]["components"][-1]["role"] = "rail"
        self.save_fit()
        row = self.evaluate()["objects"][1]
        self.assertFalse(row["structural_gate"]["checks"]["selected_furniture_family_has_required_primary_parts"])

    def test_contact_witness_detects_crossing_solids_and_rejects_touch_or_inward_faces(self):
        first = e.q.o3d.geometry.TriangleMesh.create_box(2, .2, .2).translate([-1, -.1, -.1])
        second = e.q.o3d.geometry.TriangleMesh.create_box(.2, 2, .2).translate([-.1, -1, -.1])
        self.assertTrue(e.solid_contact(first, second)["passed"])
        touch = e.q.o3d.geometry.TriangleMesh.create_box(.2, .2, .2).translate([1, -.1, -.1])
        self.assertFalse(e.solid_contact(first, touch)["passed"])
        inward = e.q.arrays_mesh(np.asarray(first.vertices), np.asarray(first.triangles)[:, ::-1].copy())
        self.assertFalse(e.solid_contact(inward, second)["passed"])

    def test_nonconvex_solid_contact_uses_actual_triangles(self):
        torus = e.q.o3d.geometry.TriangleMesh.create_torus(torus_radius=1., tube_radius=.2)
        in_ring = e.q.o3d.geometry.TriangleMesh.create_box(.08, .08, .08).translate([.96, -.04, -.04])
        in_hole = e.q.o3d.geometry.TriangleMesh.create_box(.08, .08, .08).translate([-.04, -.04, -.04])
        self.assertIsNone(e.convex_planes(torus))
        self.assertTrue(e.solid_contact(torus, in_ring)["passed"])
        self.assertFalse(e.solid_contact(torus, in_hole)["passed"])

    def test_half_ellipsoid_parameter_check_rejects_changed_radius_or_clip_plane(self):
        sphere = e.q.o3d.geometry.TriangleMesh.create_sphere(radius=1., resolution=8)
        vertices, faces = np.asarray(sphere.vertices), np.asarray(sphere.triangles)
        upper = e.compact_mesh(vertices, faces[np.all(vertices[faces, 2] >= -1e-9, axis=1)])
        vertices, faces = np.asarray(upper.vertices), np.asarray(upper.triangles)
        ring = np.flatnonzero(abs(vertices[:, 2]) < 1e-9)
        ring = ring[np.argsort(np.arctan2(vertices[ring, 1], vertices[ring, 0]))]
        center = len(vertices)
        caps = np.asarray([[center, ring[(index + 1) % len(ring)], ring[index]] for index in range(len(ring))])
        radii = np.array([.3, .2, .1])
        mesh = e.q.arrays_mesh(np.vstack((vertices, [0., 0., 0.])) * radii, np.vstack((faces, caps)))
        self.assertTrue(mesh.is_watertight())
        component = {"primitive_type": "half_ellipsoid", "parameters": {"center_m": [0, 0, 0],
                     "radii_m": radii.tolist(), "rotation_world_3x3": np.eye(3).tolist(), "cap_axis": 2, "cap_sign": 1}}
        self.assertTrue(e.parameter_consistency(mesh, component)["passed"])
        component["parameters"]["radii_m"][2] += .01
        self.assertFalse(e.parameter_consistency(mesh, component)["passed"])
        component["parameters"]["radii_m"][2] -= .01
        component["parameters"]["center_m"][2] += .01
        self.assertFalse(e.parameter_consistency(mesh, component)["passed"])

    def test_mesh_paths_cannot_escape_explicit_fit_directory(self):
        self.fit["objects"][0]["components"][0]["mesh"] = "../scene.json"
        self.save_fit()
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.evaluate()

    def test_input_mutation_during_evaluation_is_rejected(self):
        original = e.q.distances
        mutated = False

        def mutate(scene, points):
            nonlocal mutated
            if not mutated:
                mutated = True
                with self.scene_path.open("a", encoding="utf-8") as stream:
                    stream.write("\n")
            return original(scene, points)

        with mock.patch.object(e.q, "distances", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed during evaluation"):
                self.evaluate()

    def test_mesh_mutation_between_read_and_hash_verification_is_rejected(self):
        original = e.q.read_mesh
        mutated = False

        def mutate(path):
            nonlocal mutated
            result = original(path)
            if not mutated:
                mutated = True
                with Path(path).open("ab") as stream:
                    stream.write(b"\n")
            return result

        with mock.patch.object(e.q, "read_mesh", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed during evaluation"):
                self.evaluate()

    def test_deterministic_npz_and_reference_hash_tampering(self):
        first, second = self.root / "first.npz", self.root / "second.npz"
        arrays = {"a": np.array([1, 2]), "b": np.eye(3)}
        e.deterministic_npz(first, **arrays)
        e.deterministic_npz(second, **arrays)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        reference = Path(__file__).resolve().parents[1] / "examples/dining-complete/evaluation-reference"
        if not reference.is_dir():
            self.skipTest("portable reference extraction not present yet")
        metadata = json.loads((reference / "reference.json").read_text(encoding="utf-8"))
        copied = self.root / "reference"
        copied.mkdir()
        for name in ("reference.json", "labels.npz", "rays.npz"):
            (copied / name).write_bytes((reference / name).read_bytes())
        metadata["files"][0]["sha256"] = "0" * 64
        (copied / "reference.json").write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            e.load_reference(copied)

    def test_deviated_and_malformed_reference_rays_rejected_after_rehash(self):
        reference = Path(__file__).resolve().parents[1] / "examples/dining-complete/evaluation-reference"
        if not reference.is_dir():
            self.skipTest("portable reference extraction not present")
        metadata = json.loads((reference / "reference.json").read_text(encoding="utf-8"))
        with np.load(reference / "rays.npz", allow_pickle=False) as data:
            original = {key: data[key] for key in data.files}
        for failure in ("depth", "frame"):
            with self.subTest(failure=failure):
                copied = self.root / ("reference-" + failure)
                copied.mkdir()
                (copied / "labels.npz").write_bytes((reference / "labels.npz").read_bytes())
                arrays = {key: value.copy() for key, value in original.items()}
                if failure == "depth":
                    arrays["exact_depth_m"] += .05
                else:
                    arrays["frame_index"][0] = 99999
                e.deterministic_npz(copied / "rays.npz", **arrays)
                current = copy.deepcopy(metadata)
                next(item for item in current["files"] if item["path"] == "rays.npz")["sha256"] = e.q.sha256(copied / "rays.npz")
                (copied / "reference.json").write_text(json.dumps(current), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "disagree|invalid packed"):
                    e.load_reference(copied)


@unittest.skipUnless(OPEN3D_AVAILABLE, "optional Open3D geometry runtime absent")
class DiningEvaluationCliTests(unittest.TestCase):
    def check_exit_and_saved_report(self, status, passed, expected):
        with tempfile.TemporaryDirectory(prefix="AlvenX-dining-cli-") as temporary:
            output = Path(temporary) / "diagnostics.json"
            report = {"schema": "alvenx.spatial.dining-evaluation.v1", "status": status,
                      "quality_gate": {"passed": passed}, "objects": [],
                      "diagnostics": {"actual_triangle_p95_m": .012, "validation_points_retained": 200}}
            arguments = ["evaluate_dining.py", "--scene", "scene.json", "--observations", "observations.npz",
                         "--fit-dir", "fitted", "--output", str(output)]
            console = io.StringIO()
            with mock.patch.object(sys, "argv", arguments), mock.patch.object(e, "evaluate", return_value=report), contextlib.redirect_stdout(console):
                with self.assertRaises(SystemExit) as stopped:
                    e.main()
            self.assertEqual(stopped.exception.code, expected)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
            self.assertEqual(json.loads(console.getvalue())["status"], status)

    def test_failed_cli_keeps_diagnostics_and_exits_nonzero(self):
        for status, passed in (("failed", False), ("passed", False), ("failed", True)):
            with self.subTest(status=status, quality_gate=passed):
                self.check_exit_and_saved_report(status, passed, 2)

    def test_passed_cli_keeps_report_and_exits_zero(self):
        self.check_exit_and_saved_report("passed", True, 0)


if __name__ == "__main__":
    unittest.main()
