"""Surface comparison regression and faults, independent of Blender. MIT."""
import copy
from collections import Counter
import unittest

import numpy as np

from geometry_compare import compare_triangle_geometry


class GeometryCompareTests(unittest.TestCase):
    def setUp(self):
        self.vertices = [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]
        self.faces = [[0, 1, 2], [0, 2, 3]]

    def compare(self, vertices=None, faces=None, tolerance=5e-5):
        return compare_triangle_geometry(self.vertices, self.faces,
                                         self.vertices if vertices is None else vertices,
                                         self.faces if faces is None else faces, tolerance)

    def test_face_order_vertex_order_and_cyclic_permutations_are_equivalent(self):
        returned = [self.vertices[index] for index in [2, 0, 3, 1]]
        result = self.compare(returned, [[2, 1, 0], [3, 0, 1]])
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["matched_face_instances"], 2)
        self.assertEqual(result["winding_ambiguous_matched_faces"], 0)

    def test_exporter_vertex_seam_splitting_preserves_triangle_geometry(self):
        returned = [self.vertices[index] for face in self.faces for index in face]
        self.assertEqual(self.compare(returned, [[0, 1, 2], [3, 4, 5]])["status"], "pass")

    def test_opt_in_correspondence_maps_permuted_faces_and_corners(self):
        returned_faces = [[2, 3, 0], [1, 2, 0]]
        result = compare_triangle_geometry(self.vertices, self.faces, self.vertices, returned_faces,
                                           return_correspondence=True)
        self.assertEqual(result["matched_actual_face_indices"], [1, 0])
        self.assertEqual(result["matched_actual_corner_permutations"], [[2, 0, 1], [2, 0, 1]])
        for expected, actual, corners in zip(self.faces, result["matched_actual_face_indices"],
                                            result["matched_actual_corner_permutations"]):
            self.assertEqual(expected, [returned_faces[actual][corner] for corner in corners])
        without = self.compare(faces=returned_faces)
        self.assertEqual({key: value for key, value in result.items()
                          if key not in {"matched_actual_face_indices", "matched_actual_corner_permutations"}}, without)
        missing = compare_triangle_geometry(self.vertices, self.faces, self.vertices, returned_faces[:1],
                                            return_correspondence=True)
        self.assertEqual(missing["matched_actual_face_indices"], [-1, 0])
        self.assertEqual(missing["matched_actual_corner_permutations"], [None, [2, 0, 1]])

    def test_observed_micrometre_vertex_ambiguity_does_not_collapse_faces(self):
        original = np.asarray([[0, 0, 0], [1, 0, 0], [1, 0, 1.6093254e-6], [0, 1, 0.]])
        faces = [[0, 1, 3], [1, 2, 3]]
        returned = original + [0, 0, 1.2839233e-6]
        nearest = np.linalg.norm(returned[:, None] - original[None, :], axis=2).argmin(axis=1)
        collapsed = Counter(tuple(sorted(nearest[index] for index in face)) for face in faces)
        expected = Counter(tuple(sorted(face)) for face in faces)
        self.assertNotEqual(collapsed, expected)  # Original bug is present in this reproducer.
        result = compare_triangle_geometry(original, faces, returned, faces)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["matched_face_instances"], 2)
        self.assertEqual(result["winding_ambiguous_matched_faces"], 1)
        self.assertLess(result["maximum_matched_corner_distance_m"], 2e-6)

    def test_position_error_above_tolerance_is_rejected(self):
        returned = copy.deepcopy(self.vertices)
        returned[0][2] = .000051
        self.assertEqual(self.compare(returned)["status"], "fail")

    def test_small_roundtrip_position_error_is_accepted(self):
        returned = np.asarray(self.vertices) + [1e-6, -1e-6, 1e-6]
        self.assertEqual(self.compare(returned)["status"], "pass")

    def test_missing_face_is_rejected(self):
        result = self.compare(faces=self.faces[:1])
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["missing_face_instances"], 1)

    def test_duplicate_face_is_rejected_with_multiplicity(self):
        result = self.compare(faces=self.faces + [self.faces[0]])
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["extra_face_instances"], 1)

    def test_missing_face_replaced_by_duplicate_keeps_count_but_fails(self):
        result = self.compare(faces=[self.faces[0], self.faces[0]])
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["expected_face_instances"], result["actual_face_instances"])
        self.assertEqual(result["missing_face_instances"], 1)
        self.assertEqual(result["extra_face_instances"], 1)

    def test_genuine_equal_multiplicity_duplicate_faces_are_accepted(self):
        faces = [self.faces[0], self.faces[0], self.faces[1]]
        result = compare_triangle_geometry(self.vertices, faces, self.vertices, faces[::-1])
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["matched_face_instances"], 3)

    def test_changed_diagonal_is_detected_despite_equal_vertex_cloud(self):
        self.assertEqual(self.compare(faces=[[0, 1, 3], [1, 2, 3]])["status"], "fail")

    def test_resolvable_winding_reversal_is_rejected(self):
        self.assertEqual(self.compare(faces=[[0, 2, 1], self.faces[1]])["status"], "fail")

    def test_matching_uses_augmenting_path_instead_of_greedy_nearest_face(self):
        base = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        tolerance = .001
        original = np.concatenate([base, base + [.8 * tolerance, 0, 0]])
        returned = np.concatenate([base + [.3 * tolerance, 0, 0], base + [-.8 * tolerance, 0, 0]])
        faces = [[0, 1, 2], [3, 4, 5]]
        result = compare_triangle_geometry(original, faces, returned, faces, tolerance)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["matched_face_instances"], 2)

    def test_hash_cell_boundary_does_not_change_matching(self):
        original = [[-1e-7, 0, 0], [1 - 1e-7, 0, 0], [-1e-7, 1, 0]]
        returned = np.asarray(original) + [2e-7, 0, 0]
        self.assertEqual(compare_triangle_geometry(original, [[0, 1, 2]], returned,
                                                  [[0, 1, 2]])["status"], "pass")

    def test_invalid_tolerance_and_vertices_fail_explicitly(self):
        for tolerance in [0, -1, float("nan"), float("inf")]:
            with self.subTest(tolerance=tolerance), self.assertRaisesRegex(ValueError, "tolerance"):
                self.compare(tolerance=tolerance)
        for coordinate in [float("nan"), float("inf")]:
            returned = copy.deepcopy(self.vertices)
            returned[0][0] = coordinate
            with self.subTest(coordinate=coordinate), self.assertRaisesRegex(ValueError, "non-finite"):
                self.compare(returned)

    def test_invalid_triangle_indices_and_shapes_are_rejected(self):
        for faces in [[[0, 1, 4]], [[-1, 1, 2]], [[0., 1., 2.]], [[0, 1]], []]:
            with self.subTest(faces=faces), self.assertRaises(ValueError):
                self.compare(faces=faces)


if __name__ == "__main__":
    unittest.main()
