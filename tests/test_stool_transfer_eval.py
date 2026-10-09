"""Independent evaluation controls: missing evidence, wrong geometry and false gates."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


OPEN3D_AVAILABLE = importlib.util.find_spec("open3d") is not None
if OPEN3D_AVAILABLE:
    import evaluate_stool_transfer as e


@unittest.skipUnless(OPEN3D_AVAILABLE, "optional Open3D geometry dependency is absent")
class TransferEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="AlvenX-stool-evaluator-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reference = self.root / "reference"
        (self.reference / "frame_labels").mkdir(parents=True)
        vertices = np.array([[-.5, -.5, 0], [.5, -.5, 0], [0., .5, 0],
                             [1.5, -.5, 0], [2.5, -.5, 0], [2., .5, 0]])
        np.savez(self.reference / "labels.npz", vertices=vertices,
                 triangles=np.array([[0, 1, 2], [3, 4, 5]]),
                 triangle_owner=np.array([0, 0]), object_ids=np.array(["same-stool"]))
        self.pose = np.eye(4)
        self.pose[2, 3] = 2
        self.metadata = {"units": "m", "up_axis": "Y", "intrinsics_depth": np.eye(3).tolist(),
                         "depth_resolution": [2, 1], "true_camera_poses_arkit": [self.pose.tolist()],
                         "object_statistics": [{"object_id": "same-stool", "valid_ray_count": 1}],
                         "target_primitives": [{"id": "first", "triangle_start": 0, "triangle_count": 1},
                                               {"id": "unobserved", "triangle_start": 1, "triangle_count": 1}]}
        self.frame = {"measurement_valid": np.array([[True, False]]),
                      "instance_owner": np.array([[0, 0]]), "exact_depth_m": np.array([[2., 2.]]),
                      "triangle_index": np.array([[0, 1]])}
        self.scene = self.root / "scene.json"
        self.scene.write_text(json.dumps({"units": "m", "up_axis": "Y",
                                          "objects": [{"uuid": "same-stool"}]}), encoding="utf-8")
        self.observations = self.root / "observations.npz"
        np.savez(self.observations, points=np.array([[0., 0., 0.], [0., 0., 0.]]),
                 owner=np.array([0, 0]), frame_index=np.array([0, 1]))
        self.save_reference()

    def save_reference(self):
        (self.reference / "physical-scene.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        np.savez(self.reference / "frame_labels" / "frame_00000.npz", **self.frame)

    def test_dynamic_component_count_and_explicit_unobserved_primitive(self):
        _, points, point_parts, parts, provenance = e.load_reference(self.reference)
        np.testing.assert_allclose(points, [[0., 0., 0.]])
        np.testing.assert_array_equal(point_parts, [0])
        self.assertEqual(len(parts), 2)
        self.assertEqual(provenance["valid_target_ray_hits"], 1)

    def test_generated_primitive_counts_are_checked_individually(self):
        del self.metadata["object_statistics"]
        self.metadata["target_primitives"][0]["valid_ray_count"] = 1
        self.metadata["target_primitives"][1]["valid_ray_count"] = 0
        self.save_reference()
        self.assertEqual(e.load_reference(self.reference)[-1]["valid_target_ray_hits"], 1)
        self.metadata["target_primitives"][0]["valid_ray_count"] = 0
        self.metadata["target_primitives"][1]["valid_ray_count"] = 1
        self.save_reference()
        with self.assertRaisesRegex(ValueError, "primitive ray count"):
            e.load_reference(self.reference)

    def test_incorrect_primitive_range_and_camera_pose_are_refused(self):
        for failure in ("range", "pose"):
            with self.subTest(failure=failure):
                old = copy.deepcopy(self.metadata)
                if failure == "range":
                    self.metadata["target_primitives"][0]["triangle_start"] = 1
                else:
                    self.metadata["true_camera_poses_arkit"][0][2][3] = 2.1
                self.save_reference()
                with self.assertRaises(ValueError):
                    e.load_reference(self.reference)
                self.metadata = old

    def test_missing_candidate_is_reported_instead_of_dropping_case(self):
        report = e.evaluate(self.reference, self.scene, self.observations, self.root / "missing")
        self.assertFalse(report["candidate_accepted"])
        self.assertEqual(report["fit_status"], "missing")
        self.assertEqual(report["valid_reference_ray_hit_recall"]["within"][0]["fraction"], 0.)
        parts = report["per_reference_primitive_ray_hit_recall"]
        self.assertEqual(parts[1]["sample_count"], 0)
        self.assertIsNone(parts[1]["within"][0]["fraction"])
        self.assertEqual(report["observed_primitive_balanced_recall"]["unobserved_primitive_ids"], ["unobserved"])
        json.dumps(report, allow_nan=False)

    def test_completely_unobserved_target_remains_a_failed_row(self):
        self.frame["measurement_valid"][:] = False
        self.metadata["object_statistics"][0]["valid_ray_count"] = 0
        self.save_reference()
        np.savez(self.observations, points=np.empty((0, 3)), owner=np.array([], dtype=int),
                 frame_index=np.array([], dtype=int))
        report = e.evaluate(self.reference, self.scene, self.observations, self.root / "missing")
        self.assertEqual(report["reference"]["valid_target_ray_hits"], 0)
        self.assertFalse(report["engineering_acceptance"]["passed"])
        self.assertIsNone(report["observed_point_mesh_distance"]["validation_odd"]["p95_distance_m"])
        self.assertEqual(report["observed_primitive_balanced_recall"]["observed_primitive_count"], 0)
        json.dumps(report, allow_nan=False)

    def test_forged_pass_flag_does_not_override_actual_triangle_error(self):
        fit_dir = self.root / "bad-fit"
        fit_dir.mkdir()
        mesh = e.q.arrays_mesh([[-.5, -.5, .02], [.5, -.5, .02], [0., .5, .02]], np.array([[0, 1, 2]]))
        e.q.o3d.io.write_triangle_mesh(str(fit_dir / "fitted_stool.ply"), mesh)
        fit = {"units": "m", "up_axis": "Y", "object_uuid": "same-stool", "source_owner_index": 0,
               "input_files": {"scene": {"sha256": e.q.sha256(self.scene)},
                               "observations": {"sha256": e.q.sha256(self.observations)}},
               "quality_gate": {"passed": True}, "heldout_observation_error": {"mean_m": 0.}}
        (fit_dir / "fit.json").write_text(json.dumps(fit), encoding="utf-8")
        report = e.evaluate(self.reference, self.scene, self.observations, fit_dir)
        self.assertTrue(report["fitter_quality_gate_passed"])
        self.assertFalse(report["candidate_accepted"])
        self.assertAlmostEqual(report["observed_point_mesh_distance"]["validation_odd"]["p95_distance_m"], .02, places=6)
        self.assertEqual(report["source_mesh_area_samples_to_physical_target"]["sample_count"], 100000)

    def test_actual_closed_geometry_and_joint_endpoint_identity(self):
        fit_dir = Path(__file__).resolve().parents[1] / "examples" / "stool-fit" / "fitted"
        fit = json.loads((fit_dir / "fit.json").read_text(encoding="utf-8"))
        mesh, _ = e.q.read_mesh(fit_dir / "fitted_stool.ply")
        original = e.check_components(fit_dir, fit, mesh)
        self.assertTrue(original["all_seven_closed"])
        self.assertTrue(original["all_nine_positive_solid_contacts"])
        self.assertTrue(original["assembly_equals_component_files"])
        tampered = copy.deepcopy(fit)
        tampered["editable_components"][1]["end_m"][0] += 1
        result = e.check_components(fit_dir, tampered, mesh)
        self.assertFalse(result["all_nine_positive_solid_contacts"])
        self.assertFalse(result["joints"][0]["positive_solid_contact"])

    def test_open_and_inward_meshes_cannot_prove_solid_contact(self):
        mesh = e.q.o3d.geometry.TriangleMesh.create_box()
        self.assertIsNotNone(e.convex_planes(mesh))
        faces = np.asarray(mesh.triangles)
        inward = e.q.arrays_mesh(np.asarray(mesh.vertices), faces[:, ::-1].copy())
        self.assertIsNone(e.convex_planes(inward))
        open_mesh = e.q.arrays_mesh(np.asarray(mesh.vertices), faces[:-1])
        self.assertIsNone(e.convex_planes(open_mesh))


if __name__ == "__main__":
    unittest.main()
