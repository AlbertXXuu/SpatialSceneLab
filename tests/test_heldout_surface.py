import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('open3d'), 'optional Open3D runtime absent')
class HeldoutSurfaceTests(unittest.TestCase):
    def test_off_axis_ray_uses_axial_depth_at_the_first_surface(self):
        import open3d as o3d
        from evaluate_heldout_surface import measured_rays

        frame = {
            'depth_mm': np.array([[0, 2000]], dtype=np.uint16),
            'confidence': np.array([[0, 2]], dtype=np.uint8),
            'camera_to_world_cv': np.eye(4),
            'intrinsics_depth': np.eye(3),
        }
        rays, depth, observed = measured_rays(frame, {'max_depth_m': 6, 'min_confidence': 1})
        surface = o3d.t.geometry.RaycastingScene()
        for z in (2, 3):
            vertices = o3d.core.Tensor([[-4, -4, z], [8, -4, z], [-4, 8, z]], dtype=o3d.core.Dtype.Float32)
            triangles = o3d.core.Tensor([[0, 1, 2]], dtype=o3d.core.Dtype.UInt32)
            surface.add_triangles(vertices, triangles)
        hits = surface.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()

        np.testing.assert_array_equal(rays, [[0, 0, 0, 1, 0, 1]])
        np.testing.assert_array_equal(observed, [[2, 0, 2]])
        np.testing.assert_allclose(hits, depth)

    def test_misses_stay_in_support_denominator_and_tail_uses_common_hits(self):
        from evaluate_heldout_surface import RayTotals

        totals = RayTotals()
        totals.add(np.array([2.0, 2.0, 2.0]), np.array([[2.0, np.inf, 2.008], [2.0, 2.014, np.inf]]))
        report = totals.result()

        self.assertEqual(report['valid_depth_rays'], 3)
        self.assertEqual(report['common_hit_rays'], 1)
        self.assertEqual(report['baseline']['within_10mm'], 2)
        self.assertEqual(report['candidate']['within_10mm'], 1)
        self.assertEqual(report['candidate']['within_20mm_percent_of_valid'], 100 * 2 / 3)
        self.assertEqual(report['candidate']['common_hit_absolute_error_p95_mm'], 0.0)
