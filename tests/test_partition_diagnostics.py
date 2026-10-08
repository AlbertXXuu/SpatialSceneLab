import importlib.util
import time
import unittest

import numpy as np

import partition_diagnostics as diagnostic


class AttributionControls(unittest.TestCase):
    def test_overlap_structure_outside_and_false_positive_are_distinct(self):
        owners = np.array([0, -1, -2, 1, -2, -1, 0, 0])
        raw = np.array([[1, 0], [1, 1], [1, 1], [0, 1],
                        [0, 1], [0, 0], [1, 0], [1, 0]], dtype=bool)
        evidence = {"raw_box_candidates": raw,
                    "structure_excluded": np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=bool)}
        reference = np.zeros((8, 3))
        reference[:6, 1] = 1
        reference[6, 2] = 1
        reference[7, 0] = .75
        unknown = np.array([0, 0, 0, 0, 0, 0, 0, .25])
        values = diagnostic.target_reason_areas(owners, 0, evidence, reference, unknown)
        np.testing.assert_array_equal(values[:7, :7], np.eye(7))
        self.assertEqual(values[7, 7], .75)
        self.assertEqual(values[7, 9], .25)
        np.testing.assert_allclose(values.sum(axis=1), np.ones(8))

    def test_inconsistent_prior_trace_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly once"):
            diagnostic.target_reason_areas(
                np.array([0]), 0,
                {"raw_box_candidates": np.array([[True]]), "structure_excluded": np.array([True])},
                np.array([[0., 1.]]), np.zeros(1))

    def test_changed_recorded_area_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "conservation mismatch"):
            diagnostic._check_close("FN", .4, .5)

    def test_elapsed_budget_is_rejected(self):
        with self.assertRaises(TimeoutError):
            diagnostic._deadline(time.perf_counter() - 1)

    @unittest.skipUnless(importlib.util.find_spec("open3d"), "optional CPU reference matcher")
    def test_face_trace_is_independent_of_batch_size_and_keeps_unknown(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                             [0, 0, 4], [1, 0, 4], [0, 1, 4]], dtype=float)
        triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        parameters = {"matching_tolerance_m": .08, "samples_per_face": 7, "ambiguity_margin_m": .002}
        results = [diagnostic.trace_reference_area(vertices, triangles, vertices[:3], triangles[:1],
                   np.array([0]), 1, parameters, batch_faces=batch) for batch in (1, 2)]
        for area, unknown, _ in results:
            np.testing.assert_allclose(area, [[0, .5], [0, 0]])
            np.testing.assert_allclose(unknown, [0, .5])
        np.testing.assert_array_equal(results[0][0], results[1][0])


if __name__ == "__main__":
    unittest.main()
