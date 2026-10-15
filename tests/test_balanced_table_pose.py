"""Independent geometry, score isolation and fail-closed pose controls. MIT."""
import copy
import importlib.util
import unittest
from unittest import mock

import numpy as np

import audit_real_tabletop as baseline
import balanced_table_pose as pose


def fixture(angle_degrees=.8, translation=(0., 0., 0.)):
    """Opposite-side horizontal surfaces and three independent room planes.

    A small camera-centred rotation puts the table above its training plane and
    the opposite-side chair below its own plane. A target-only Y shift therefore
    worsens the chair, while a consistent rigid motion can restore both.
    Later pairs independently reserve odd indices from every fit.
    """
    a, b = np.meshgrid(np.linspace(-.27, .27, 8), np.linspace(-.27, .27, 8))
    a, b = a.ravel(), b.ravel()
    surfaces = [
        ("table", np.column_stack([a + 1, np.full(len(a), .3), b]), [0., 1., 0.]),
        ("chair", np.column_stack([a - 1, np.full(len(a), .2), b]), [0., 1., 0.]),
        ("wall_x", np.column_stack([np.full(len(a), 2), a + .8, b]), [1., 0., 0.]),
        ("floor", np.column_stack([a, np.zeros(len(a)), b + 1]), [0., 1., 0.]),
        ("wall_z", np.column_stack([a, b + .8, np.full(len(a), -2)]), [0., 0., 1.]),
    ]
    target_transform = np.eye(4)
    target_transform[0, 3] = 1
    chair_transform = np.eye(4)
    chair_transform[0, 3] = -1
    camera = np.eye(4)
    camera[:3, 3] = [0, 1, 0]
    scene = {"units": "m", "up_axis": "Y", "objects": [
        {"uuid": "table", "category": "Table", "dimensions_m": [1., .8, 1.],
         "transform_world": target_transform.tolist()},
        {"uuid": "chair", "category": "Chair", "dimensions_m": [1., .8, 1.],
         "transform_world": chair_transform.tolist()}],
        "frames": [{"frame_index": frame, "camera_to_world_cv": camera.tolist()} for frame in [0, 1, 31, 32]]}
    true_one = np.vstack([np.repeat(pts, 2, axis=0) for _, pts, _ in surfaces])
    labels_one = np.repeat(np.arange(len(surfaces)), len(a) * 2)
    normal_one = np.vstack([np.tile(normal, (len(a) * 2, 1)) for _, _, normal in surfaces])
    truth = np.tile(true_one, (4, 1))
    labels = np.tile(labels_one, 4)
    normals = np.tile(normal_one, (4, 1))
    frames = np.repeat([0, 1, 31, 32], len(true_one)).astype(np.int32)
    points = truth.copy()
    rotation = pose._rotation(np.array([0., 0., np.deg2rad(angle_degrees)]))
    later = frames > 30
    points[later] = (truth[later] - camera[:3, 3]) @ rotation.T + camera[:3, 3] + translation
    candidates = np.zeros((len(points), 2), dtype=bool)
    candidates[labels == 0, 0] = True
    candidates[labels == 1, 1] = True
    local = baseline.local_points(points, target_transform)
    estimate = (frames > 30) & (labels == 0) & (np.arange(len(points)) % 2 == 0)
    records = [{"frame_index": 31, "transform_applied": True, "status": "accepted", "checks": {"support": True}},
               {"frame_index": 32, "transform_applied": False, "status": "rejected", "checks": {"support": False}}]
    args = [scene, 0, points, frames, candidates, local, np.array([0., 0., .3]), estimate, records]

    def independent(unused_scene, unused_index, observed, frame_ids, unused_candidates, active):
        groups = []
        for label, (name, _, _) in enumerate(surfaces[1:], 1):
            ids = np.flatnonzero((labels == label) & np.isin(frame_ids, list(active)) &
                                 (np.arange(len(observed)) % 2 == 0))
            groups.append({"name": name, "ids": ids, "positions": observed[ids],
                           "reference_positions": truth[ids], "normals": normals[ids],
                           "residuals": np.einsum("ij,ij->i", observed[ids] - truth[ids], normals[ids])})
        return groups
    return args, truth, labels, normals, independent


class BalancedPoseTests(unittest.TestCase):
    def run_fixture(self, args, groups):
        with mock.patch.object(pose, "_independent_groups", side_effect=groups):
            return pose.estimate_transforms(*args)

    def test_rigid_rotation_control_repairs_target_without_harming_chair(self):
        args, truth, labels, normals, groups = fixture()
        transforms, rows = self.run_fixture(args, groups)
        self.assertTrue(rows[0]["transform_applied"])
        self.assertEqual(rows[0]["rank"], 6)
        self.assertGreater(rows[0]["eigenvalue_ratio"], 1e-4)
        points, frames = args[2], args[3]
        source = frames == 31
        fixed = points[source] @ transforms[31][:3, :3].T + transforms[31][:3, 3]
        normal_error = np.abs(np.einsum("ij,ij->i", fixed - truth[source], normals[source]))
        self.assertLess(float(np.max(normal_error)), .0004)
        baseline.rigid_transform(transforms[31], "Test accepted motion")
        table = source & (labels == 0)
        chair = source & (labels == 1)
        target_shift = -float(np.median(points[table, 1] - .3))
        before = np.mean(np.abs(points[chair, 1] - truth[chair, 1]))
        target_only_after = np.mean(np.abs(points[chair, 1] + target_shift - truth[chair, 1]))
        self.assertGreater(target_only_after, before * 1.8)
        self.assertLess(float(np.mean(normal_error[labels[source] == 1])), before / 20)

    def test_later_odd_point_mutations_cannot_change_estimates(self):
        args, _, _, _, groups = fixture()
        first, rows = self.run_fixture(args, groups)
        changed = copy.deepcopy(args)
        odd = (changed[3] > 30) & (np.arange(len(changed[2])) % 2 == 1)
        changed[2][odd] += [4, -8, 5]
        changed[5][odd] += [4, -8, 5]
        second, changed_rows = self.run_fixture(changed, groups)
        self.assertEqual(rows, changed_rows)
        for frame in first:
            np.testing.assert_array_equal(first[frame], second[frame])

    def test_inputs_unchanged_and_rejected_frame_remains_identity(self):
        args, _, _, _, groups = fixture()
        before = copy.deepcopy(args)
        transforms, rows = self.run_fixture(args, groups)
        for i in [2, 3, 4, 5, 6, 7]:
            np.testing.assert_array_equal(args[i], before[i])
        self.assertEqual(args[0], before[0])
        self.assertEqual(args[8], before[8])
        self.assertEqual(set(transforms), {31, 32})
        np.testing.assert_array_equal(transforms[32], np.eye(4))
        self.assertEqual(rows[1]["status"], "target_quality_rejected")

    def test_translation_and_rotation_bounds_fail_closed(self):
        for angle, translation, failing in [(0., (.2, 0., 0.), "bounded_translation"),
                                             (5., (0., 0., 0.), "bounded_rotation")]:
            with self.subTest(failing=failing):
                args, _, _, _, groups = fixture(angle, translation)
                transforms, rows = self.run_fixture(args, groups)
                self.assertFalse(rows[0]["transform_applied"])
                self.assertFalse(rows[0]["checks"][failing])
                np.testing.assert_array_equal(transforms[31], np.eye(4))
                self.assertIsNotNone(rows[0]["estimated_transform"])

    def test_single_plane_degeneracy_is_not_a_six_dof_estimate(self):
        args, _, _, _, _ = fixture()
        transforms, rows = self.run_fixture(args, lambda *unused: [])
        self.assertFalse(rows[0]["transform_applied"])
        self.assertFalse(rows[0]["checks"]["full_rank"])
        self.assertFalse(rows[0]["checks"]["pose_observable"])
        np.testing.assert_array_equal(transforms[31], np.eye(4))

    def test_missing_target_common_support_does_not_solve_background_only(self):
        args, _, _, _, groups = fixture()
        args[5][args[3] <= 30, 1] += .3
        transforms, rows = self.run_fixture(args, groups)
        self.assertFalse(rows[0]["transform_applied"])
        self.assertEqual(rows[0]["status"], "missing_common_support")
        np.testing.assert_array_equal(transforms[31], np.eye(4))

    def test_frame_record_presence_duplicates_and_false_quality_rejected(self):
        for kind in ["missing", "duplicate", "unchecked"]:
            args, _, _, _, groups = fixture()
            if kind == "missing":
                args[8].pop()
            elif kind == "duplicate":
                args[8][1] = copy.deepcopy(args[8][0])
            else:
                args[8][0]["checks"] = {"support": False}
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.run_fixture(args, groups)

    def test_nonfinite_odd_estimation_and_camera_reflection_rejected(self):
        for kind in ["nan", "odd", "camera"]:
            args, _, _, _, groups = fixture()
            if kind == "nan":
                args[2][0, 0] = np.nan
            elif kind == "odd":
                later_table = np.flatnonzero((args[3] > 30) & args[4][:, 0])
                args[7][later_table[1]] = True
            else:
                args[0]["frames"][2]["camera_to_world_cv"][0][0] = -1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.run_fixture(args, groups)

    def test_original_quality_cannot_leak_raw_transforms_or_geometry(self):
        args, _, _, _, groups = fixture()
        args[8][0].update(estimated_transform=np.eye(4).tolist(), applied_transform=np.eye(4).tolist(),
                          support_extent_xz_m=[1, 2], unexpected_coordinates=[[8, 9, 10]])
        _, rows = self.run_fixture(args, groups)
        quality = rows[0]["original_normal_quality"]
        for forbidden in ["estimated_transform", "applied_transform", "support_extent_xz_m", "unexpected_coordinates"]:
            self.assertNotIn(forbidden, quality)
        self.assertEqual(quality["checks"], {"support": True})

    def test_no_accepted_frame_needs_no_optional_matcher(self):
        args, _, _, _, _ = fixture()
        args[8][0].update(transform_applied=False, checks={"support": False})
        with mock.patch.object(pose, "_independent_groups", side_effect=AssertionError("unexpected match")):
            transforms, rows = pose.estimate_transforms(*args)
        self.assertTrue(all(not row["transform_applied"] for row in rows))
        for transform in transforms.values():
            np.testing.assert_array_equal(transform, np.eye(4))

    @unittest.skipUnless(importlib.util.find_spec("open3d"), "Existing optional Open3D environment required")
    def test_actual_fixed_matcher_never_reads_changed_odd_coordinates(self):
        args, _, _, _, _ = fixture()
        # One training overlap discovers the neighbour; its other points keep
        # their original membership and provide independent fixed geometry.
        shared = np.flatnonzero((args[3] <= 30) & args[4][:, 1])[0]
        args[4][shared, 0] = True
        first = pose._independent_groups(args[0], 0, args[2], args[3], args[4], {31})
        self.assertTrue(any(group["name"] == "neighbour_1" and len(group["ids"]) for group in first))
        self.assertTrue(any(group["name"] == "background" and len(group["ids"]) for group in first))
        self.assertTrue(all(np.all(group["ids"] % 2 == 0) for group in first))
        changed = args[2].copy()
        odd = (args[3] > 30) & (np.arange(len(changed)) % 2 == 1)
        changed[odd] += [9, 8, 7]
        second = pose._independent_groups(args[0], 0, changed, args[3], args[4], {31})
        self.assertEqual(len(first), len(second))
        for before, after in zip(first, second):
            self.assertEqual(before["name"], after["name"])
            for key in ["ids", "positions", "reference_positions", "normals", "residuals"]:
                np.testing.assert_array_equal(before[key], after[key])


if __name__ == "__main__":
    unittest.main()
