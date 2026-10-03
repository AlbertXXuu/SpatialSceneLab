import importlib.util
import json
import unittest
from unittest.mock import patch

import numpy as np

import partition_baselines as baselines
import evaluate_partition as evaluation


AVAILABLE = importlib.util.find_spec("open3d") is not None
OWNERSHIP = {"box_margin_m": 0, "structure_tolerance_m": 0}


def box(x=0, width=1):
    matrix = np.eye(4)
    matrix[0, 3] = x
    return {"uuid": "object", "geometry_kind": "box", "dimensions_m": [width, 4, 4],
            "transform_world": matrix.tolist()}


def faces_mesh(faces):
    vertices = np.asarray(faces, dtype=np.float64).reshape(-1, 3)
    return vertices, np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)


def triangle(x=0, width=2, height=1, z=0):
    return [[x, 0, z], [x + width, 0, z], [x, height, z]]


class PartitionBaselineTests(unittest.TestCase):
    def test_released_b0_is_exactly_preserved_without_mutating_mesh(self):
        vertices, triangles = faces_mesh([
            [[-.6, 0, 0], [-.55, .1, 0], [-.55, 0, .1]],
            [[-.6, 0, 0], [-.55, .1, 0], [.6, 0, 0]],
            [[-.6, 0, 0], [-.55, .1, 0], [0, 0, 0]],
        ])
        original_vertices, original_triangles = vertices.copy(), triangles.copy()
        objects = [box(-.2), box(.2)]
        expected = [0, -1, -1]
        result = baselines.partition(vertices, triangles, objects, [], OWNERSHIP, "B0")
        np.testing.assert_array_equal(result["face_owner"], expected)
        np.testing.assert_array_equal(vertices, original_vertices)
        np.testing.assert_array_equal(triangles, original_triangles)
        self.assertEqual(result["face_owner"].dtype, np.int32)
        if AVAILABLE:
            import reconstruct_mesh
            released, _, _ = reconstruct_mesh.split_triangles(vertices, triangles, objects, [], OWNERSHIP)
            np.testing.assert_array_equal(result["face_owner"], released)

    def test_centroid_and_area_votes_expose_discretization_difference(self):
        # The centroid is inside this thin prior; only a minority of area is.
        vertices, triangles = faces_mesh([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]])
        objects = [box(1 / 3, .1)]
        owners = {method: baselines.partition(vertices, triangles, objects, [], OWNERSHIP, method)
                  for method in ("b0", "b1", "b2")}
        self.assertEqual(int(owners["b0"]["face_owner"][0]), -1)
        self.assertEqual(int(owners["b1"]["face_owner"][0]), 0)
        self.assertEqual(int(owners["b2"]["face_owner"][0]), -1)
        self.assertEqual(owners["b2"]["parameters"]["samples_per_face"], 49)
        self.assertEqual(owners["b2"]["parameters"]["min_majority_fraction"], .6)

    def test_b2_threshold_preserves_unresolved_and_never_consults_reference(self):
        vertices, triangles = faces_mesh([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]])
        ownership = dict(OWNERSHIP, b2_min_majority_fraction=.95, b2_samples_per_face=49)
        with patch.object(evaluation, "evaluate", side_effect=AssertionError("reference leak")):
            result = baselines.partition(vertices, triangles, [box(.25, .5)], [], ownership, "b2")
        self.assertEqual(int(result["face_owner"][0]), -2)
        self.assertFalse(result["parameters"]["reference_used"])
        # A large area of overlapping priors must not become an object vote.
        overlap = baselines.partition(vertices, triangles, [box(.5, 4), box(.5, 4)], [], OWNERSHIP, "b2")
        self.assertEqual(int(overlap["face_owner"][0]), -2)

    def test_fixed_area_sampling_is_deterministic_and_stays_inside_thin_faces(self):
        barycentric = baselines.barycentric_samples(49)
        np.testing.assert_array_equal(barycentric, baselines.barycentric_samples(49))
        np.testing.assert_allclose(barycentric.sum(axis=1), 1)
        self.assertTrue(np.all(barycentric > 0))
        vertices, triangles = faces_mesh([triangle(width=100, height=.02)])
        self.assertAlmostEqual(float(baselines.triangle_areas(vertices, triangles)[0]), 1)
        self.assertRaises(ValueError, baselines.barycentric_samples, 0)
        self.assertRaises(ValueError, baselines.barycentric_samples, 7.5)
        with self.assertRaisesRegex(ValueError, "majority"):
            baselines.partition(vertices, triangles, [], [], dict(OWNERSHIP, b2_min_majority_fraction=.5), "b2")


@unittest.skipUnless(AVAILABLE, "optional CPU Open3D dependency; independent physical mesh evaluation")
class AreaEvaluationTests(unittest.TestCase):
    def evaluate_same_mesh(self, faces, predicted, truth, target=0, **options):
        vertices, triangles = faces_mesh(faces)
        return evaluation.evaluate(vertices, triangles, np.asarray(predicted, np.int32),
                                   vertices, triangles, np.asarray(truth, np.int32), target, **options)

    def test_area_weights_are_not_face_or_sample_counts(self):
        faces = [triangle(width=4, height=4)] + [triangle(x=10 + i, width=.2, height=.2)
                                                for i in range(10)]
        result = self.evaluate_same_mesh(faces, [-2] + [0] * 10, [0] * 11)
        self.assertAlmostEqual(result["coverage"]["mesh_area_m2"], 8.2)
        self.assertAlmostEqual(result["target"]["recall"], .2 / 8.2)
        self.assertAlmostEqual(result["target"]["false_negative_ratio"], 8 / 8.2)
        self.assertAlmostEqual(result["target"]["area_iou"], .2 / 8.2)
        self.assertNotAlmostEqual(result["target"]["recall"], 10 / 11)

    def test_long_thin_and_large_triangle_both_use_physical_area(self):
        result = self.evaluate_same_mesh([triangle(width=100, height=.02),
                                         triangle(x=200, width=10, height=2)], [0, -1], [0, 0])
        self.assertAlmostEqual(result["target"]["recall"], 1 / 11)
        self.assertAlmostEqual(result["target"]["scorable_target_area_m2"], 11)

    def test_all_unknown_counts_as_missing_target(self):
        result = self.evaluate_same_mesh([triangle()], [-2], [0])
        self.assertEqual(result["target"]["area_iou"], 0)
        self.assertEqual(result["target"]["recall"], 0)
        self.assertEqual(result["target"]["false_negative_ratio"], 1)
        self.assertAlmostEqual(result["target"]["unresolved_target_area_m2"], 1)
        self.assertIsNone(result["target"]["precision"])

    def test_target_outside_candidates_still_counts_as_missed(self):
        result = self.evaluate_same_mesh([triangle(), triangle(x=5)], [0, -1], [0, 0],
                                         candidate_mask=np.array([True, False]))
        self.assertAlmostEqual(result["target"]["recall"], .5)
        self.assertAlmostEqual(result["target"]["false_negative_ratio"], .5)
        self.assertAlmostEqual(result["target"]["target_outside_candidate_area_m2"], 1)

    def test_false_positive_ratio_uses_same_target_area_denominator(self):
        result = self.evaluate_same_mesh([triangle(), triangle(x=5, width=6)], [0, 0], [0, -1])
        self.assertAlmostEqual(result["target"]["false_positive_ratio"], 3)
        self.assertAlmostEqual(result["target"]["precision"], .25)
        self.assertAlmostEqual(result["target"]["area_iou"], .25)

    def test_unscorable_area_is_reported_and_widens_bounds(self):
        vertices, triangles = faces_mesh([triangle(), triangle(x=5, width=6, z=2)])
        ref_vertices, ref_triangles = faces_mesh([triangle()])
        result = evaluation.evaluate(vertices, triangles, np.array([0, 0], np.int32),
                                     ref_vertices, ref_triangles, np.array([0], np.int32), 0,
                                     matching_tolerance=.01)
        self.assertAlmostEqual(result["coverage"]["scorable_area_m2"], 1)
        self.assertAlmostEqual(result["coverage"]["unscorable_area_m2"], 3)
        self.assertAlmostEqual(result["target"]["area_iou"], 1)
        bounds = result["target"]["conservative_bounds"]["area_iou"]
        self.assertAlmostEqual(bounds["lower"], .25)
        self.assertAlmostEqual(bounds["upper"], 1)
        self.assertFalse(result["matching"]["visibility_verified"])
        self.assertFalse(result["physical_scan_accuracy_verified"])

    def test_target_absent_or_unmatched_has_null_scores(self):
        absent = self.evaluate_same_mesh([triangle()], [0], [-1])
        self.assertEqual(absent["status"], "no_target")
        for score in ["area_iou", "precision", "recall", "false_negative_ratio", "false_positive_ratio"]:
            self.assertIsNone(absent["target"][score])
        vertices, triangles = faces_mesh([triangle(z=1)])
        ref_vertices, ref_triangles = faces_mesh([triangle()])
        unmatched = evaluation.evaluate(vertices, triangles, np.array([0], np.int32),
                                        ref_vertices, ref_triangles, np.array([0], np.int32), 0,
                                        matching_tolerance=.1)
        self.assertEqual(unmatched["status"], "no_scorable_target")
        self.assertIsNone(unmatched["target"]["area_iou"])
        self.assertAlmostEqual(unmatched["coverage"]["unscorable_area_m2"], 1)
        json.dumps(unmatched, allow_nan=False)

    def test_independent_labels_override_input_box_selection(self):
        vertices, triangles = faces_mesh([triangle()])
        predicted = baselines.partition(vertices, triangles, [box(.5, 4)], [], OWNERSHIP, "b2")["face_owner"]
        result = evaluation.evaluate(vertices, triangles, predicted, vertices, triangles,
                                     np.array([-1], np.int32), 0)
        self.assertEqual(result["status"], "no_target")
        self.assertAlmostEqual(result["target"]["false_positive_area_m2"], 1)

    def test_nearby_different_reference_labels_are_unscorable(self):
        vertices, triangles = faces_mesh([triangle()])
        reference_vertices, reference_triangles = faces_mesh([triangle(), triangle(z=.001)])
        result = evaluation.evaluate(vertices, triangles, np.array([0], np.int32),
                                     reference_vertices, reference_triangles, np.array([0, 1], np.int32), 0)
        self.assertAlmostEqual(result["coverage"]["ambiguous_area_m2"], 1)
        self.assertAlmostEqual(result["coverage"]["scorable_area_m2"], 0)
        self.assertIsNone(result["target"]["area_iou"])
        self.assertEqual(result["parameters"]["ambiguity_margin_m"], .002)
        resolved = evaluation.evaluate(vertices, triangles, np.array([0], np.int32),
                                       reference_vertices, reference_triangles, np.array([0, 1], np.int32), 0,
                                       ambiguity_margin=.0001)
        self.assertAlmostEqual(resolved["target"]["area_iou"], 1)
        self.assertAlmostEqual(resolved["coverage"]["ambiguous_area_m2"], 0)

    def test_recorded_method_failure_keeps_null_scores_and_failure_count(self):
        result = self.evaluate_same_mesh([triangle()], [-2], [0], method_failure="time budget exceeded")
        self.assertEqual(result["status"], "method_failure")
        self.assertEqual(result["failure_counts"]["method_failures"], 1)
        self.assertIsNone(result["target"]["area_iou"])
        self.assertAlmostEqual(result["target"]["false_negative_area_m2"], 1)

    def test_all_objects_matches_once_and_keeps_missing_target_attempts(self):
        vertices, triangles = faces_mesh([triangle(), triangle(x=5)])
        with patch.object(evaluation, "_measure", wraps=evaluation._measure) as measured:
            result = evaluation.evaluate_all_objects(vertices, triangles, np.array([0, -2], np.int32),
                                                      vertices, triangles, np.array([0, 1], np.int32),
                                                      object_ids=["first", "second", "absent"])
        self.assertEqual(measured.call_count, 1)
        self.assertEqual(result["object_attempts"], 3)
        self.assertEqual(result["scored_objects"], 2)
        self.assertEqual(result["unscored_objects"], 1)
        self.assertEqual(result["macro_target_area_iou"], .5)
        self.assertEqual(result["targets"][2]["status"], "no_target")
        self.assertIsNone(result["targets"][2]["area_iou"])
        json.dumps(result, allow_nan=False)

    def test_invalid_labels_and_topology_do_not_silently_score(self):
        vertices, triangles = faces_mesh([triangle()])
        with self.assertRaisesRegex(ValueError, "reference_owner"):
            evaluation.evaluate(vertices, triangles, np.array([0], np.int32), vertices, triangles,
                                np.array([-2], np.int32), 0)
        with self.assertRaisesRegex(ValueError, "outside object_ids"):
            evaluation.evaluate_all_objects(vertices, triangles, np.array([2], np.int32),
                                            vertices, triangles, np.array([0], np.int32), object_ids=["first"])
        with self.assertRaisesRegex(ValueError, "triangle index"):
            evaluation.evaluate(vertices, triangles + 10, np.array([0], np.int32), vertices, triangles,
                                np.array([0], np.int32), 0)


if __name__ == "__main__":
    unittest.main()
