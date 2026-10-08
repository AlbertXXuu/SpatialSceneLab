"""Bounded stool-fit validation and numerical controls; no fixture/reference reads. MIT."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from fit_stool import (components, contact_checks, cylinder_mesh, cylinder_sdf,
                       initialize, load_inputs, self_test, validate_output, write_ply)


SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


def measured_scene():
    target = np.eye(4)
    target[:3, 3] = [.05, .3, -.1]
    floor = np.eye(4)
    floor[1, 3] = -.03
    return {"units": "m", "up_axis": "Y", "objects": [
        {"uuid": "same-stool", "category": "Chair", "dimensions_m": [.7, .6, .7],
         "transform_world": target.tolist()}], "structures": [
        {"category": "Floor", "dimensions_m": [2., .06, 2.], "transform_world": floor.tolist()}],
        "frames": [{"frame_index": 0}, {"frame_index": 1}]}


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="AlvenX-stool-fit-tests-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scene_path = self.root / "scene.json"
        self.points_path = self.root / "observations.npz"
        self.scene = measured_scene()
        self.arrays = {"points": np.tile([.05, .3, -.1], (240, 1)),
                       "colors": np.full((240, 3), 128, dtype=np.uint8),
                       "frame_index": np.arange(240, dtype=np.int32) % 2,
                       "owner": np.zeros(240, dtype=np.int32)}

    def load(self, arrays=None, scene=None):
        self.scene_path.write_text(json.dumps(self.scene if scene is None else scene), encoding="utf-8")
        np.savez(self.points_path, **(self.arrays if arrays is None else arrays))
        return load_inputs(self.scene_path, self.points_path, "same-stool")

    def test_valid_owner_selection_and_split(self):
        arrays = copy.deepcopy(self.arrays)
        arrays["owner"][:10] = -1
        result = self.load(arrays)
        self.assertEqual(len(result[5]), 230)
        self.assertEqual(result[1], 0)
        self.assertEqual(result[4], 0)
        self.assertEqual(int(np.sum(result[-1] % 2 == 0)), 115)

    def test_malformed_observation_shapes_values_and_indices_are_rejected(self):
        cases = [
            ("points", np.zeros((240, 2))),
            ("points", np.full((240, 3), np.nan)),
            ("points", np.zeros((240, 3), dtype=complex)),
            ("colors", np.zeros((239, 3), dtype=np.uint8)),
            ("colors", np.full((240, 3), 256)),
            ("colors", np.full((240, 3), .5)),
            ("colors", np.full((240, 3), np.inf)),
            ("frame_index", np.zeros((240, 1), dtype=int)),
            ("frame_index", np.full(240, 5, dtype=int)),
            ("frame_index", np.full(240, .5)),
            ("owner", np.zeros(239, dtype=int)),
            ("owner", np.ones(240, dtype=int)),
            ("owner", np.full(240, -3, dtype=int)),
        ]
        for field, bad in cases:
            with self.subTest(field=field, shape=bad.shape, dtype=bad.dtype):
                arrays = copy.deepcopy(self.arrays)
                arrays[field] = bad
                with self.assertRaises(ValueError):
                    self.load(arrays)

    def test_missing_array_and_insufficient_target_or_validation_support(self):
        for variant in ("missing", "no_target", "no_validation"):
            with self.subTest(variant=variant):
                arrays = copy.deepcopy(self.arrays)
                if variant == "missing":
                    del arrays["colors"]
                elif variant == "no_target":
                    arrays["owner"][:] = -1
                else:
                    arrays["frame_index"][:] = 0
                with self.assertRaises(ValueError):
                    self.load(arrays)

    def test_invalid_scene_units_target_box_floor_or_frame_records(self):
        for variant in ("units", "duplicate_target", "dimension", "scale", "no_floor", "tilted_floor", "nan_floor", "duplicate_frame"):
            with self.subTest(variant=variant):
                scene = copy.deepcopy(self.scene)
                if variant == "units":
                    scene["units"] = "cm"
                elif variant == "duplicate_target":
                    scene["objects"].append(copy.deepcopy(scene["objects"][0]))
                elif variant == "dimension":
                    scene["objects"][0]["dimensions_m"][0] = 0
                elif variant == "scale":
                    scene["objects"][0]["transform_world"][0][0] = 2
                elif variant == "no_floor":
                    scene["structures"] = []
                elif variant == "tilted_floor":
                    scene["structures"][0]["transform_world"][1][1] = 0
                elif variant == "nan_floor":
                    scene["structures"][0]["dimensions_m"][1] = float("nan")
                else:
                    scene["frames"][1]["frame_index"] = 0
                with self.assertRaises(ValueError):
                    self.load(scene=scene)

    def test_output_never_overwrites_inputs_or_existing_artifacts(self):
        self.load()
        output = self.root / "new-output"
        self.assertEqual(validate_output(output, [self.scene_path, self.points_path]), output)
        self.assertFalse(output.exists())
        output.mkdir()
        validate_output(output, [self.scene_path, self.points_path])
        marker = output / "keep.txt"
        marker.write_text("existing artifact", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "never overwritten"):
            validate_output(output, [self.scene_path, self.points_path])
        self.assertEqual(marker.read_text(), "existing artifact")
        with self.assertRaises(ValueError):
            validate_output(self.scene_path, [self.scene_path, self.points_path])

    def test_cli_requires_all_three_paths(self):
        script = Path(__file__).resolve().parents[1] / "fit_stool.py"
        result = subprocess.run([sys.executable, "-B", str(script)], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--scene", result.stderr)
        self.assertIn("--observations", result.stderr)
        self.assertIn("--output", result.stderr)

    def test_mesh_write_is_exclusive_even_after_directory_precheck(self):
        path = self.root / "seat.ply"
        path.write_text("existing source", encoding="ascii")
        with self.assertRaises(FileExistsError):
            write_ply(path, [], [], [], [], [])
        self.assertEqual(path.read_text(), "existing source")


class CylinderAndContactTests(unittest.TestCase):
    def test_exact_capped_cylinder_distances_and_rigid_transform(self):
        points = np.array([[1., 0, 0], [0, 1, 0], [0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]])
        expected = [0, 0, -1, 1, 1, np.sqrt(2)]
        np.testing.assert_allclose(cylinder_sdf(points, [0, -1, 0], [0, 1, 0], 1), expected, atol=1e-12)
        rotation = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])
        shift = np.array([.2, -.7, 1.3])
        transformed = cylinder_sdf(points @ rotation.T + shift,
                                   np.array([0, -1, 0]) @ rotation.T + shift,
                                   np.array([0, 1, 0]) @ rotation.T + shift, 1)
        np.testing.assert_allclose(transformed, expected, atol=1e-12)

    def test_degenerate_cylinder_rejected(self):
        for end, radius in (([0, 0, 0], 1), ([0, 1, 0], 0)):
            with self.assertRaises(ValueError):
                cylinder_sdf(np.zeros((1, 3)), [0, 0, 0], end, radius)

    def test_generated_parts_are_closed_outward_and_contact_at_nine_joints(self):
        x = [0., 0., .6, .06, .32]
        for angle in [0., 2.1, 4.2]:
            x.extend([.27 * np.cos(angle), .27 * np.sin(angle), .19 * np.cos(angle), .19 * np.sin(angle), .025])
        x.extend([.23, .014, .24, .015, .22, .013])
        parts = components(np.array(x), 0.)
        contacts = contact_checks(parts, 0.)
        self.assertEqual(len(contacts["joints"]), 9)
        self.assertTrue(contacts["all_nine_intended_joints_in_solid_contact"])
        self.assertEqual(len(parts), 7)
        for part in parts:
            vertices, faces = cylinder_mesh(part)
            edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
            _, incidence = np.unique(edges, axis=0, return_counts=True)
            self.assertTrue(np.all(incidence == 2))
            area = np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                            vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1)
            self.assertTrue(np.all(area > 0))
            volume = np.einsum("ij,ij->i", vertices[faces[:, 0]], np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]])).sum() / 6
            self.assertGreater(volume, 0.)


@unittest.skipUnless(SCIPY_AVAILABLE, "optional SciPy fitting dependency is absent")
class FittingControlTests(unittest.TestCase):
    def test_changed_observations_change_fit_under_same_prior(self):
        result = self_test()
        np.testing.assert_allclose(result["observed_parameter_shift"], [.03, -.02, .012], atol=.0015)

    def test_missing_third_leg_support_is_rejected(self):
        theta = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        seat = [[r * np.cos(t), y, r * np.sin(t)] for t in theta
                for r, y in ((.28, .45), (.28, .47), (.28, .49), (.28, .515), (.1, .515), (.2, .515))]
        # Only two observed legs. Splitting either narrow cluster does not create support for a third.
        legs = [[cx + .012 * np.cos(t), y, cz + .012 * np.sin(t)]
                for cx, cz in [(-.2, -.12), (.2, -.12)]
                for y in np.linspace(.04, .39, 30) for t in [0., 1.5, 3., 4.5]]
        source = np.asarray(seat + legs)
        with self.assertRaisesRegex(ValueError, "Three independently supported low-leg clusters"):
            initialize(source, np.array([0., .27, 0.]), np.array([.6, .54, .6]), 0.)


if __name__ == "__main__":
    unittest.main()
