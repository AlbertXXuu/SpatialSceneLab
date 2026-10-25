"""Independent numerical checks for RGB-only camera consistency diagnostics.

The synthetic observations come from non-coplanar world points projected through
two distinct camera-to-world transforms. No depth fitting or OpenCV is used.
"""
from __future__ import annotations

import copy
import unittest

import numpy as np

from rgb_pose_check import (
    fundamental_from_cameras,
    grid_coverage,
    rigid_camera,
    sampson_px,
    score_camera,
    validate_scenes,
)


def rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def camera(rotation, centre):
    value = np.eye(4)
    value[:3, :3] = rotation
    value[:3, 3] = centre
    return value


def project(world, camera_to_world, intrinsics):
    # Solve the coordinate change directly; do not share an epipolar formula
    # with the implementation under test.
    local = np.linalg.solve(
        camera_to_world[:3, :3],
        (world - camera_to_world[:3, 3]).T,
    ).T
    if not np.all(local[:, 2] > 0):
        raise AssertionError("Synthetic points must be in front of each camera")
    image = local @ intrinsics.T
    return image[:, :2] / image[:, 2, None]


class CameraConsistencyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261025)
        self.world = rng.uniform([-1.1, -.8, 3.], [1.3, .9, 7.], (80, 3))
        self.a = camera(rotation_z(.07) @ rotation_y(.13), [-.4, .1, -.2])
        self.b = camera(rotation_x(-.08) @ rotation_y(-.11), [.55, -.15, .25])
        self.ka = np.array([[815., 0., 329.], [0., 790., 242.], [0., 0., 1.]])
        self.kb = np.array([[1040., 0., 471.], [0., 970., 351.], [0., 0., 1.]])
        self.pa = project(self.world, self.a, self.ka)
        self.pb = project(self.world, self.b, self.kb)

    def test_nonplanar_projection_satisfies_camera_fundamental(self):
        self.assertEqual(np.linalg.matrix_rank(self.world - self.world.mean(axis=0)), 3)
        fundamental = fundamental_from_cameras(self.a, self.b, self.ka, self.kb)
        self.assertEqual(fundamental.shape, (3, 3))
        self.assertTrue(np.isfinite(fundamental).all())
        self.assertGreater(np.linalg.norm(fundamental), 0)
        singular = np.linalg.svd(fundamental, compute_uv=False)
        self.assertLess(singular[-1], singular[0] * 1e-12)
        residual = sampson_px(fundamental, self.pa, self.pb)
        self.assertEqual(residual.shape, (len(self.world),))
        np.testing.assert_allclose(residual, 0., atol=1e-9, rtol=0.)

    def test_wrong_pose_direction_is_detected(self):
        wrong = fundamental_from_cameras(
            np.linalg.inv(self.a), np.linalg.inv(self.b), self.ka, self.kb,
        )
        self.assertGreater(float(np.median(sampson_px(wrong, self.pa, self.pb))), 5.)

    def test_wrong_rotation_and_translation_direction_are_detected(self):
        rotated = self.b.copy()
        rotated[:3, :3] = rotated[:3, :3] @ rotation_x(.025)
        turned_baseline = self.b.copy()
        turned_baseline[:3, 3] += np.array([0., .15, -.05])
        for changed in (rotated, turned_baseline):
            with self.subTest(camera=changed.tolist()):
                wrong = fundamental_from_cameras(self.a, changed, self.ka, self.kb)
                self.assertGreater(float(np.median(sampson_px(wrong, self.pa, self.pb))), 1.)

    def test_swapping_cameras_intrinsics_and_pixels_preserves_error(self):
        perturbed = self.pb + np.column_stack([
            np.linspace(-2., 3., len(self.pb)), np.linspace(1., -.5, len(self.pb)),
        ])
        forward = fundamental_from_cameras(self.a, self.b, self.ka, self.kb)
        backward = fundamental_from_cameras(self.b, self.a, self.kb, self.ka)
        np.testing.assert_allclose(
            sampson_px(forward, self.pa, perturbed),
            sampson_px(backward, perturbed, self.pa),
            atol=1e-10, rtol=1e-10,
        )

    def test_translation_scale_is_unobservable_even_for_noisy_correspondences(self):
        original = fundamental_from_cameras(self.a, self.b, self.ka, self.kb)
        noisy = self.pb + np.array([.7, -1.1])
        reference = sampson_px(original, self.pa, noisy)
        for scale in (.3, 7.):
            aa, bb = self.a.copy(), self.b.copy()
            aa[:3, 3] *= scale
            bb[:3, 3] *= scale
            scaled = fundamental_from_cameras(aa, bb, self.ka, self.kb)
            with self.subTest(scale=scale):
                np.testing.assert_allclose(
                    sampson_px(scaled, self.pa, noisy), reference,
                    atol=1e-10, rtol=1e-10,
                )

    def test_identical_camera_and_pure_rotation_have_no_fundamental_matrix(self):
        pure_rotation = self.a.copy()
        pure_rotation[:3, :3] = self.a[:3, :3] @ rotation_y(.3)
        for other in (self.a.copy(), pure_rotation):
            with self.subTest(camera=other.tolist()):
                with self.assertRaises(ValueError):
                    fundamental_from_cameras(self.a, other, self.ka, self.kb)

    def test_tolerated_homogeneous_row_rounding_cannot_change_pose_scores(self):
        # Serialized ARFrame matrices can end in 0.99999994. This accepted
        # storage rounding is not a physical camera scale or baseline change.
        rounded_a, rounded_b = self.a.copy(), self.b.copy()
        rounded_a[3, 3] = .99999994
        rounded_b[3, 3] = 1.00000006
        rigid_camera(rounded_a)
        rigid_camera(rounded_b)
        expected = fundamental_from_cameras(self.a, self.b, self.ka, self.kb)
        actual = fundamental_from_cameras(rounded_a, rounded_b, self.ka, self.kb)
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)
        noisy = self.pb + np.array([.2, -.4])
        np.testing.assert_allclose(sampson_px(actual, self.pa, noisy),
                                   sampson_px(expected, self.pa, noisy),
                                   atol=1e-10, rtol=1e-10)

    def test_rigid_camera_accepts_rotation_translation_and_does_not_mutate(self):
        before = self.a.copy()
        result = rigid_camera(self.a.tolist())
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_allclose(result, self.a, atol=0., rtol=0.)
        np.testing.assert_array_equal(self.a, before)

    def test_rigid_camera_rejects_bad_shape_nonfinite_reflection_and_affine(self):
        reflection = self.a.copy()
        reflection[:3, 0] *= -1
        scale = self.a.copy()
        scale[:3, :3] *= 1.01
        shear = self.a.copy()
        shear[0, 1] += .02
        bottom = self.a.copy()
        bottom[3, 0] = .01
        nonfinite = self.a.copy()
        nonfinite[0, 3] = np.nan
        for bad in (np.eye(3), np.zeros((4, 4)), reflection, scale, shear, bottom, nonfinite):
            with self.subTest(value=np.asarray(bad).tolist()):
                with self.assertRaises(ValueError):
                    rigid_camera(bad)

    def test_fundamental_rejects_invalid_intrinsics(self):
        nonfinite = self.ka.copy()
        nonfinite[0, 0] = np.inf
        zero_focal = self.ka.copy()
        zero_focal[0, 0] = 0.
        for bad in (np.eye(4), np.zeros((3, 3)), nonfinite, zero_focal):
            with self.subTest(intrinsics=bad.tolist()):
                with self.assertRaises(ValueError):
                    fundamental_from_cameras(self.a, self.b, bad, self.kb)


class SampsonTests(unittest.TestCase):
    def test_known_pixel_distance_and_fundamental_scale_invariance(self):
        # Horizontal epipolar lines: residual = abs(y_a - y_b) / sqrt(2).
        fundamental = np.array([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        a = np.array([[2., 3.], [7., 11.], [-3., 8.]])
        b = np.array([[17., 5.], [-4., 8.], [1., 8.]])
        expected = np.array([2., 3., 0.]) / np.sqrt(2.)
        for scale in (1., -3., 7., 1e-6):
            with self.subTest(scale=scale):
                np.testing.assert_allclose(
                    sampson_px(fundamental * scale, a, b), expected,
                    atol=1e-12, rtol=1e-12,
                )

    def test_epipoles_return_infinity_instead_of_false_zero(self):
        fundamental = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]])
        values = sampson_px(fundamental, np.array([[0., 0.], [1., 0.]]),
                            np.array([[0., 0.], [2., 0.]]))
        self.assertTrue(np.isinf(values[0]))
        self.assertEqual(values[1], 0.)

    def test_invalid_correspondences_and_matrix_are_rejected(self):
        fundamental = np.array([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        points = np.array([[2., 3.], [4., 5.]])
        cases = [
            (np.eye(4), points, points),
            # The all-zero matrix defines no epipolar constraint; reject it
            # before computing the valid nonzero-matrix epipole degeneracy.
            (np.zeros((3, 3)), points, points),
            (np.full((3, 3), np.nan), points, points),
            (fundamental, points, points[:1]),
            (fundamental, points[:, 0], points[:, 0]),
            (fundamental, np.array([[np.inf, 1.], [2., 3.]]), points),
        ]
        for matrix, a, b in cases:
            with self.subTest(matrix_shape=matrix.shape, a_shape=a.shape, b_shape=b.shape):
                with self.assertRaises(ValueError):
                    sampson_px(matrix, a, b)


class GridCoverageTests(unittest.TestCase):
    def test_full_grid_repeated_points_and_empty_input(self):
        # A point in every 4-by-3 cell; this is image coverage, not a count of
        # independently observed scene surfaces.
        centres = np.array([[x, y] for y in (50., 150., 250.)
                            for x in (50., 150., 250., 350.)])
        full = grid_coverage(centres, 400, 300)
        self.assertEqual(full["occupied"], 12)
        self.assertEqual(full["total"], 12)
        repeat = grid_coverage(np.repeat(centres[:1], 50, axis=0), 400, 300)
        self.assertEqual(repeat["occupied"], 1)
        empty = grid_coverage(np.empty((0, 2)), 400, 300)
        self.assertEqual(empty["occupied"], 0)
        self.assertEqual(empty["total"], 12)


class FrozenInputTests(unittest.TestCase):
    def setUp(self):
        fingerprints = {"frame_00000.json": "a" * 64}
        self.manifest = {"files": [{"path": "tea_room/frame_00000.json", "sha256": "a" * 64}]}
        original = {"schema": "spatial-scene-lab.observations.v1", "source": "synthetic-capture",
                    "input_files": fingerprints, "frames": []}
        for i in range(62):
            transform = camera(np.eye(3), [i * .01, 0., 0.])
            original["frames"].append({
                "frame_index": i, "camera_to_world_cv": transform.tolist(),
                "intrinsics_depth": [[100., 0., 128.], [0., 100., 96.], [0., 0., 1.]],
                "color_size": [1920, 1440], "depth_size": [256, 192],
                "source": f"frame_{i:05}.json",
            })
        self.scenes = {"baseline": original}
        for name, method in (("balanced", "balanced_pose"), ("normal", "normal_translation")):
            candidate = copy.deepcopy(original)
            candidate["schema"] = "alvenx.real-table-alignment-candidate.v1"
            candidate["alignment_control"] = {
                "method": method, "disposition": "diagnostic_candidate",
                "passed": False, "is_camera_pose_estimate": False,
            }
            for i in (46, 47, 48, 49):
                candidate["frames"][i]["camera_to_world_cv"][1][3] += .005
            self.scenes[name] = candidate

    def test_frozen_identity_accepts_storage_rounding_without_extra_pose_change(self):
        self.scenes["balanced"]["frames"][3]["camera_to_world_cv"][3][3] = .99999994
        indexed = validate_scenes(self.scenes, self.manifest)
        self.assertEqual(set(indexed), {"baseline", "balanced", "normal"})
        self.assertEqual(set(indexed["balanced"]), set(range(62)))

    def test_mislabeled_fallback_and_swapped_candidates_are_rejected(self):
        fallback = copy.deepcopy(self.scenes)
        fallback["balanced"] = copy.deepcopy(fallback["baseline"])
        swapped = copy.deepcopy(self.scenes)
        swapped["balanced"], swapped["normal"] = swapped["normal"], swapped["balanced"]
        adopted = copy.deepcopy(self.scenes)
        adopted["balanced"]["alignment_control"]["passed"] = True
        for scenes in (fallback, swapped, adopted):
            with self.subTest(control=scenes["balanced"].get("alignment_control")):
                with self.assertRaises(ValueError):
                    validate_scenes(scenes, self.manifest)

    def test_duplicate_identity_changed_capture_and_extra_pose_change_are_rejected(self):
        duplicate = copy.deepcopy(self.scenes)
        duplicate["balanced"]["frames"][1]["frame_index"] = 0
        unrelated = copy.deepcopy(self.scenes)
        unrelated["normal"]["input_files"]["frame_00000.json"] = "b" * 64
        wrong_intrinsics = copy.deepcopy(self.scenes)
        wrong_intrinsics["balanced"]["frames"][46]["intrinsics_depth"][0][0] += 1.
        extra_pose = copy.deepcopy(self.scenes)
        extra_pose["balanced"]["frames"][45]["camera_to_world_cv"][0][3] += .01
        wrong_manifest = copy.deepcopy(self.manifest)
        wrong_manifest["files"][0]["sha256"] = "c" * 64
        cases = [(value, self.manifest) for value in (duplicate, unrelated, wrong_intrinsics, extra_pose)]
        cases.append((self.scenes, wrong_manifest))
        for i, (scenes, manifest) in enumerate(cases):
            with self.subTest(case=i):
                with self.assertRaises(ValueError):
                    validate_scenes(scenes, manifest)

    def test_failed_pose_and_epipoles_keep_the_original_scoring_denominator(self):
        a = camera(np.eye(3), [0., 0., 0.])
        p = np.array([[0., 0.], [1., 0.], [1., 1.]])
        q = np.array([[0., 0.], [2., 0.], [2., 1.]])
        values, report = score_camera(a, a, np.eye(3), np.eye(3), p, q)
        self.assertEqual(len(values), 3)
        self.assertTrue(np.isinf(values).all())
        self.assertEqual(report["count"], 3)
        self.assertEqual(report["finite_count"], 0)
        self.assertEqual(report["fraction_le_2px"], 0.)
        self.assertIsNone(report["median_px"])
        self.assertIsNone(report["p95_px"])
        self.assertTrue(report["error"])
        b = camera(np.eye(3), [0., 0., 1.])
        values, report = score_camera(a, b, np.eye(3), np.eye(3), p, q)
        self.assertTrue(np.isinf(values[0]))
        self.assertEqual(report["count"], 3)
        self.assertEqual(report["finite_count"], 2)
        self.assertAlmostEqual(report["fraction_le_2px"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
