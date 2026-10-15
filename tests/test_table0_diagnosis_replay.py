"""Reject default adoption, broken rays and changing populations in replay. MIT."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

import reproduce_table0_diagnosis as replay


class ReplayExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="AlvenX-table0-replay-")
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.points, self.output = root/"points", root/"candidate"
        self.points.mkdir(); self.output.mkdir()
        self.arrays = {"points": np.array([[0., .3, 1.], [.2, .4, 1.2]]),
                       "frame_index": np.array([4, 47]), "pixel_uv": np.array([[0, 1], [2, 3]]),
                       "candidates": np.ones((2, 1), bool)}
        self.original = {"schema": "spatial-scene-lab.observations.v1",
                         "observations": "observations.npz", "frames": [
                             {"frame_index": frame, "camera_to_world_cv": np.eye(4).tolist()} for frame in (4, 47)]}
        np.savez(self.points/"observations.npz", **self.arrays)
        self.write(self.points/"scene.json", self.original)
        shutil.copyfile(self.points/"observations.npz", self.output/"observations.npz")
        canonical = {**self.original, "alignment_control": {"passed": False, "disposition": "identity_fallback"}}
        self.write(self.output/"scene.json", canonical)
        self.write(self.output/"alignment.json", {"passed": False})
        self.candidate = json.loads(json.dumps(self.original))
        self.candidate.update(schema="alvenx.real-table-alignment-candidate.v1", observations="candidate-observations.npz")
        self.candidate["frames"][1]["camera_to_world_cv"][1][3] = -.01
        self.write(self.output/"candidate-scene.json", self.candidate)
        self.changed = {k: v.copy() for k, v in self.arrays.items()}
        self.changed["points"][1, 1] -= .01
        np.savez(self.output/"candidate-observations.npz", **self.changed)

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_completed_rejected_candidate_keeps_original_default_and_rays(self):
        result = replay.verify_exports(self.points, self.output)
        self.assertTrue(result["canonical_original_bundle_unchanged"])
        self.assertFalse(result["candidate_adopted"])
        self.assertEqual(result["candidate_changed_camera_count"], 1)
        self.assertLess(result["candidate_camera_space_max_component_error_m"], 1e-12)

    def test_rejected_default_cannot_silently_contain_candidate_points(self):
        shutil.copyfile(self.output/"candidate-observations.npz", self.output/"observations.npz")
        with self.assertRaisesRegex(ValueError, "default observation"):
            replay.verify_exports(self.points, self.output)

    def test_camera_point_mismatch_is_rejected(self):
        self.candidate["frames"][1]["camera_to_world_cv"][1][3] = 0
        self.write(self.output/"candidate-scene.json", self.candidate)
        with self.assertRaisesRegex(ValueError, "camera rays"):
            replay.verify_exports(self.points, self.output)

    def test_candidate_observation_pointer_and_original_pixel_ids_are_preserved(self):
        self.candidate["observations"] = "observations.npz"
        self.write(self.output/"candidate-scene.json", self.candidate)
        with self.assertRaises(ValueError):
            replay.verify_exports(self.points, self.output)
        self.candidate["observations"] = "candidate-observations.npz"
        self.write(self.output/"candidate-scene.json", self.candidate)
        self.changed["pixel_uv"][1] += 1
        np.savez(self.output/"candidate-observations.npz", **self.changed)
        with self.assertRaisesRegex(ValueError, "pixels or denominator"):
            replay.verify_exports(self.points, self.output)


if __name__ == "__main__":
    unittest.main()
