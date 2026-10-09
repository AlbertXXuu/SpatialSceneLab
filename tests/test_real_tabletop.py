"""Small independent controls for frozen real-table diagnostics and failure paths. MIT."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import audit_real_tabletop as audit


OPEN3D = importlib.util.find_spec("open3d") is not None


def fixture():
    """A sampled plane with five held-out frames, unrelated to the real scan."""
    x, z = np.meshgrid(np.linspace(-.3, .3, 7), np.linspace(-.3, .3, 7))
    points = np.column_stack([x.ravel(), np.full(x.size, .32), z.ravel()])
    frame_ids = [0, 1, 31, 32, 33, 34, 35]
    scene = {"units": "m", "up_axis": "Y", "objects": [
        {"uuid": "test-table", "category": "Table", "dimensions_m": [1, .8, 1],
         "transform_world": np.eye(4).tolist()}],
        "frames": [{"frame_index": frame} for frame in frame_ids]}
    arrays = {"points": np.tile(points, (len(frame_ids), 1)),
              "frame_index": np.repeat(frame_ids, len(points)).astype(np.int32),
              "candidates": np.ones((len(frame_ids) * len(points), 1), dtype=bool)}
    return scene, arrays


class InputsAndPlaneTests(unittest.TestCase):
    def setUp(self):
        # Explicit external isolation; the temporary directory is always cleaned.
        self.temp = tempfile.TemporaryDirectory(prefix="AlvenX-real-tabletop-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scene_path = self.root / "scene.json"
        self.points_path = self.root / "observations.npz"
        self.scene, self.arrays = fixture()

    def save(self):
        self.scene_path.write_text(json.dumps(self.scene), encoding="utf-8")
        np.savez(self.points_path, **self.arrays)

    def load(self):
        self.save()
        return audit.load_inputs(self.scene_path, self.points_path, "test-table")

    def test_training_does_not_read_changed_validation(self):
        result = self.load()
        _, _, _, frames, _, local, _, train, valid = result
        coefficients, _ = audit.fit_plane(local, frames, train)
        changed = local.copy()
        changed[valid, 1] += .07
        again, _ = audit.fit_plane(changed, frames, train)
        np.testing.assert_array_equal(again, coefficients)
        np.testing.assert_allclose(coefficients, [0, 0, .32], atol=1e-12)

    def test_malformed_points_candidates_and_frames(self):
        variants = [("points", np.zeros((343, 2))), ("points", np.full((343, 3), np.nan)),
                    ("points", np.ones((343, 3), dtype=complex)),
                    ("candidates", np.ones((343, 2), dtype=bool)),
                    ("candidates", np.ones((343, 1), dtype=int)),
                    ("frame_index", np.full(343, 9, dtype=int)),
                    ("frame_index", np.zeros(343, dtype=float))]
        for key, value in variants:
            with self.subTest(key=key, shape=value.shape, dtype=value.dtype):
                self.scene, self.arrays = fixture()
                self.arrays[key] = value
                with self.assertRaises(ValueError):
                    self.load()

    def test_units_rigid_boxes_and_frame_records(self):
        for variant in ("units", "scale", "reflection", "tilt", "camera", "dimension", "duplicate", "missing"):
            with self.subTest(variant=variant):
                self.scene, self.arrays = fixture()
                target = self.scene["objects"][0]
                if variant == "units":
                    self.scene["units"] = "cm"
                elif variant == "scale":
                    target["transform_world"][0][0] = 2
                elif variant == "reflection":
                    target["transform_world"][0][0] = -1
                elif variant == "tilt":
                    target["transform_world"] = [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
                elif variant == "camera":
                    self.scene["frames"][0]["camera_to_world_cv"] = np.zeros((4, 4)).tolist()
                elif variant == "dimension":
                    target["dimensions_m"][0] = 0
                elif variant == "duplicate":
                    self.scene["frames"].append(copy.deepcopy(self.scene["frames"][0]))
                else:
                    del target["transform_world"]
                with self.assertRaises(ValueError):
                    self.load()

    def test_empty_splits_single_frame_and_collinear_training_are_rejected(self):
        for variant in ("no_train", "no_validation", "single_frame", "collinear"):
            with self.subTest(variant=variant):
                self.scene, self.arrays = fixture()
                frames = self.arrays["frame_index"]
                if variant == "no_train":
                    self.arrays["candidates"][frames <= 30] = False
                elif variant == "no_validation":
                    self.arrays["candidates"][frames > 30] = False
                elif variant == "single_frame":
                    frames[frames == 1] = 0
                else:
                    self.arrays["points"][frames <= 30, 2] = self.arrays["points"][frames <= 30, 0]
                with self.assertRaises(ValueError):
                    self.load()

    def test_all_objects_and_candidate_points_are_excluded_from_background(self):
        table = self.scene["objects"][0]
        other = copy.deepcopy(table)
        other["transform_world"][0][3] = 3
        points = np.array([[0, 0, 0], [.57, 0, 0], [3, 0, 0], [3.57, 0, 0], [5, 0, 0], [6, 0, 0]])
        candidates = np.zeros((len(points), 2), dtype=bool)
        candidates[4, 0] = True  # Even a prior candidate outside its OBB cannot enter registration.
        np.testing.assert_array_equal(audit.background_mask(points, [table, other], candidates),
                                      [False, False, False, False, False, True])

    def test_one_correction_improves_one_frame_but_worsens_another(self):
        points = np.array([[0, .020, 0], [1, .020, 1], [0, -.005, 0], [1, -.005, 1]])
        frames = np.array([31, 31, 32, 32])
        transform = np.eye(4)
        transform[1, 3] = -.015
        changed = audit.apply_transforms(points, frames, {31: transform, 32: transform})
        before = audit.summarize(points, frames, np.zeros(3))
        after = audit.summarize(changed, frames, np.zeros(3))
        self.assertLess(after["frames"][0]["mean_m"], before["frames"][0]["mean_m"])
        self.assertGreater(after["frames"][1]["mean_m"], before["frames"][1]["mean_m"])
        self.assertEqual(before["count"], after["count"])
        np.testing.assert_array_equal(points[:, 1], [.020, .020, -.005, -.005])

    def test_output_must_be_new_and_not_an_input_alias(self):
        self.save()
        existing = self.root / "existing"
        existing.mkdir()
        for output in (existing, self.scene_path, self.points_path):
            with self.subTest(output=output), self.assertRaises(ValueError):
                audit.run(self.scene_path, self.points_path, "test-table", output)

    def test_improved_mean_can_hide_a_worse_upper_tail(self):
        frames = np.repeat([31, 32], [94, 6])
        points = np.zeros((100, 3))
        points[:, 1] = np.repeat([.020, .025], [94, 6])
        better, worse = np.eye(4), np.eye(4)
        better[1, 3], worse[1, 3] = -.015, .035
        changed = audit.apply_transforms(points, frames, {31: better, 32: worse})
        before = audit.summarize(points, frames, np.zeros(3))
        after = audit.summarize(changed, frames, np.zeros(3))
        self.assertLess(after["mean_m"], before["mean_m"])
        self.assertGreater(after["p95_m"], before["p95_m"])
        self.assertEqual(after["count"], before["count"])

    @unittest.skipUnless(OPEN3D, "requires existing optional Open3D environment")
    def test_empty_background_preserves_all_frames_and_original_denominator(self):
        self.arrays["points"][self.arrays["frame_index"] > 30, 1] += .02
        self.save()
        before = (self.scene_path.read_bytes(), self.points_path.read_bytes())
        output = self.root / "audit"
        report = audit.run(self.scene_path, self.points_path, "test-table", output)
        self.assertEqual(report["validation_count"], 245)
        self.assertEqual(report["plane_validation"], report["background_validation"])
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["registration_frames"]), 5)
        self.assertTrue(all(r["status"] == "insufficient_geometry" for r in report["registration_frames"]))
        self.assertTrue(all(not r["transform_applied"] for r in report["registration_frames"]))
        self.assertEqual(before, (self.scene_path.read_bytes(), self.points_path.read_bytes()))
        self.assertEqual({p.name for p in output.iterdir()}, {"plane-fit.json", "background-icp.json", "comparison.json"})
        published = json.loads((output / "comparison.json").read_text())
        self.assertEqual(published["source_scene_sha256"], audit.sha256(self.scene_path))
        for forbidden in ("coefficients", "estimated_transform", "points", "camera_to_world_cv"):
            self.assertNotIn('"' + forbidden + '"', (output / "comparison.json").read_text())


@unittest.skipUnless(OPEN3D, "requires existing optional Open3D environment")
class RegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import open3d
        cls.o3d = open3d

    def make_cloud(self, points, normals=None):
        cloud = self.o3d.geometry.PointCloud(self.o3d.utility.Vector3dVector(points))
        if normals is not None:
            cloud.normals = self.o3d.utility.Vector3dVector(normals)
        return cloud

    def test_three_background_planes_recover_a_known_small_rigid_shift(self):
        u, v = np.meshgrid(np.linspace(.1, .9, 15), np.linspace(.1, .9, 15))
        u, v = u.ravel(), v.ravel()
        points = np.vstack([np.column_stack([np.full(len(u), 2), u, v]),
                            np.column_stack([1 + u, np.full(len(u), -1), v]),
                            np.column_stack([1 + u, v, np.full(len(u), 2)])])
        normals = np.repeat(np.eye(3), len(u), axis=0)
        source = self.make_cloud(points + [0, .02, 0])
        target = self.make_cloud(points, normals)
        transform, record = audit.register_background(source, target, self.o3d)
        self.assertTrue(record["transform_applied"], record)
        np.testing.assert_allclose(transform[:3, 3], [0, -.02, 0], atol=1e-6)
        self.assertLess(record["rmse_after_m"], record["rmse_before_m"])

    def test_single_plane_is_unobservable_and_retains_identity(self):
        u, v = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        points = np.column_stack([u.ravel(), np.zeros(u.size), v.ravel()])
        target = self.make_cloud(points, np.tile([0., 1., 0.], (len(points), 1)))
        source = self.make_cloud(points + [0, .02, 0])
        transform, record = audit.register_background(source, target, self.o3d)
        self.assertFalse(record["transform_applied"])
        self.assertFalse(record["checks"]["pose_observability"])
        np.testing.assert_array_equal(transform, np.eye(4))

    def test_insufficient_points_and_registration_exception_retain_identity(self):
        tiny = self.make_cloud([[0., 0., 0.]])
        transform, record = audit.register_background(tiny, tiny, self.o3d)
        self.assertEqual(record["status"], "insufficient_geometry")
        np.testing.assert_array_equal(transform, np.eye(4))
        enough = self.make_cloud([[0., 0., 0.], [1., 0., 0.], [0., 0., 1.]])
        with mock.patch.object(self.o3d.pipelines.registration, "registration_icp", side_effect=RuntimeError("test failure")):
            transform, record = audit.register_background(enough, enough, self.o3d)
        self.assertEqual(record["status"], "registration_error")
        self.assertIn("test failure", record["failure"])
        np.testing.assert_array_equal(transform, np.eye(4))


if __name__ == "__main__":
    unittest.main()
