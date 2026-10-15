"""Real failure diagnostics preserve denominator and distinguish strata. MIT."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

import diagnose_table0 as diagnosis


class DiagnosisTests(unittest.TestCase):
    def test_raw_depth_discontinuity_includes_invalid_neighbour_without_deleting_center(self):
        depth = np.ones((7, 7))
        depth[3, 5] = 0
        uv = np.array([[3, 3], [1, 1], [5, 3]])
        np.testing.assert_array_equal(diagnosis.depth_edges(depth, uv), [True, False, True])
        self.assertEqual(len(diagnosis.depth_edges(depth, uv)), len(uv))

    def test_bad_pixel_indices_rejected_instead_of_wrapping(self):
        for uv in (np.array([[-1, 0]]), np.array([[3, 0]]), np.array([[.5, 1.]])):
            with self.assertRaises(ValueError):
                diagnosis.depth_edges(np.ones((3, 3)), uv)

    def test_summary_retains_outliers_in_same_population(self):
        residual = np.array([0., .015, -.080, .012])
        mask = np.ones(4, dtype=bool)
        stats = diagnosis.summary(mask, residual, np.ones(4), np.full(4, .5),
                                  np.full(4, 2), np.array([False, False, True, False]), np.ones(4))
        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["within10mm"], .25)
        self.assertGreater(stats["p95_m"], .05)
        self.assertEqual(stats["depth_edge_fraction"], .25)
        self.assertEqual(diagnosis.summary(~mask, residual, residual, residual, residual, mask, residual), {"count": 0})

    def test_shared_spatial_cells_expose_view_offset_without_changing_points(self):
        local = np.array([[.01, 0., .01], [.02, 0., .01], [.03, 0., .02]] * 2)
        frames = np.array([4] * 3 + [46] * 3)
        residual = np.array([0., 0., 0., .017, .017, .017])
        original = local.copy()
        rows = diagnosis.shared_cells(local, frames, np.ones(6, bool), residual)
        row = next(r for r in rows if r["reference"] == 4 and r["other"] == 46)
        self.assertEqual(row["common_cells"], 1)
        self.assertEqual(row["signed_median_delta_m"], .017)
        self.assertEqual(row["positive_fraction"], 1.)
        np.testing.assert_array_equal(local, original)

    def test_existing_output_fails_before_reading_any_input(self):
        with tempfile.TemporaryDirectory(prefix="AlvenX-table0-diagnosis-") as temp:
            output = Path(temp) / "retained"
            output.mkdir()
            with self.assertRaises(ValueError):
                diagnosis.diagnose(Path(temp)/"absent-scene", Path(temp)/"absent-obs", output)


if __name__ == "__main__":
    unittest.main()
