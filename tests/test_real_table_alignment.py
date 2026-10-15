"""Disjoint-sample, rejection, provenance and replay tests. MIT."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import audit_real_table_alignment as audit


def fixture():
    x, z = np.meshgrid(np.linspace(-.3, .3, 8), np.linspace(-.3, .3, 8))
    flat = np.column_stack([x.ravel(), np.full(x.size, .32), z.ravel()])
    ids = [0, 1, 31, 32, 33, 34, 35]
    points = np.tile(flat, (len(ids), 1))
    frames = np.repeat(ids, len(flat)).astype(np.int32)
    points[frames > 30, 1] += .018
    scene = {"units": "m", "up_axis": "Y", "objects": [
        {"uuid": "test-table", "category": "Table", "dimensions_m": [1., .8, 1.],
         "transform_world": np.eye(4).tolist()}],
        "frames": [{"frame_index": frame, "camera_to_world_cv": np.eye(4).tolist()} for frame in ids]}
    return scene, {"points": points, "frame_index": frames,
                   "candidates": np.ones((len(points), 1), dtype=bool),
                   "colors": np.full((len(points), 3), .5)}


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="AlvenX-real-table-alignment-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scene, self.arrays = fixture()
        self.scene_path = self.root / "scene.json"
        self.points_path = self.root / "observations.npz"
        self.background_path = self.root / "background.json"
        self.protocol_path = self.root / "protocol.json"
        self.save()

    def save(self):
        self.scene_path.write_text(json.dumps(self.scene), encoding="utf-8")
        np.savez(self.points_path, **self.arrays)
        data = {"schema": "alvenx.real-table-background-control.v1",
                "source_scene_sha256": audit.sha256(self.scene_path),
                "source_observations_sha256": audit.sha256(self.points_path),
                "parameters": audit.baseline.PARAMETERS,
                "frames": [{"frame_index": frame, "transform_applied": False,
                            "estimated_transform": None} for frame in [31, 32, 33, 34, 35]]}
        self.background_path.write_text(json.dumps(data), encoding="utf-8")

    def freeze(self):
        return audit.freeze(self.scene_path, self.points_path, "test-table", self.background_path, self.protocol_path)

    def loaded(self):
        return audit.baseline.load_inputs(self.scene_path, self.points_path, "test-table")

    def test_estimation_and_score_are_disjoint_and_ambiguous_not_estimated(self):
        valid = np.ones(8, dtype=bool)
        candidates = np.ones((8, 2), dtype=bool)
        candidates[:4, 1] = False
        estimate, score = audit.partition(valid, candidates, 0)
        self.assertEqual(np.flatnonzero(estimate).tolist(), [0, 2])
        self.assertEqual(np.flatnonzero(score).tolist(), [1, 3, 5, 7])
        self.assertFalse(np.any(estimate & score))

    def test_scoring_mutation_cannot_change_estimates(self):
        scene, index, _, frames, candidates, local, _, train, valid = self.loaded()
        coefficients, _ = audit.baseline.fit_plane(local, frames, train)
        before = audit.estimate_transforms(scene, index, frames, candidates, local, train, valid, coefficients)
        changed = local.copy()
        changed[before[3]] += [5, 8, -3]
        after = audit.estimate_transforms(scene, index, frames, candidates, changed, train, valid, coefficients)
        self.assertEqual(before[1], after[1])
        for key in before[0]:
            np.testing.assert_array_equal(before[0][key], after[0][key])

    def test_only_normal_translation_is_estimated(self):
        scene, index, _, frames, candidates, local, _, train, valid = self.loaded()
        coefficients, _ = audit.baseline.fit_plane(local, frames, train)
        transforms, records, _, _ = audit.estimate_transforms(scene, index, frames, candidates, local, train, valid, coefficients)
        self.assertTrue(all(row["transform_applied"] for row in records))
        for transform in transforms.values():
            np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=0)
            np.testing.assert_allclose(transform[:3, 3], [0, -.018, 0], atol=1e-12)

    def test_tilted_plane_world_rotation_uses_normal_without_yaw(self):
        x, z = np.meshgrid(np.linspace(-.3, .3, 8), np.linspace(-.3, .3, 8))
        coefficients = np.array([.05, -.03, .3])
        normal = np.array([-.05, 1, .03]); normal /= np.linalg.norm(normal)
        points = np.column_stack([x.ravel(), .05*x.ravel()-.03*z.ravel()+.3, z.ravel()]) + .018*normal
        cells = {tuple(cell) for cell in np.floor(points[:, [0, 2]] / .05).astype(int)}
        rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        transform, row = audit.estimate_offset(points, cells, coefficients, rotation)
        self.assertTrue(row["transform_applied"])
        np.testing.assert_allclose(transform[:3, 3], -.018*(rotation@normal), atol=1e-12)
        np.testing.assert_array_equal(transform[:3, :3], np.eye(3))

    def test_rejections_preserve_estimate_but_apply_identity(self):
        x, z = np.meshgrid(np.linspace(-.3, .3, 8), np.linspace(-.3, .3, 8))
        points = np.column_stack([x.ravel(), np.full(x.size, .4), z.ravel()])
        cells = {tuple(cell) for cell in np.floor(points[:, [0, 2]] / .05).astype(int)}
        for case in ("excessive_shift", "noise", "few_samples", "no_shared_cells"):
            changed, support = points.copy(), cells
            if case == "noise":
                changed[:, 1] = np.linspace(.21, .39, len(points))
            elif case == "few_samples":
                changed = changed[:3]
            elif case == "no_shared_cells":
                support = {(999, 999)}
            transform, row = audit.estimate_offset(changed, support, np.array([0., 0., .3]), np.eye(3))
            self.assertFalse(row["transform_applied"], case)
            np.testing.assert_array_equal(transform, np.eye(4))
            if case != "no_shared_cells":
                self.assertIsNotNone(row["estimated_transform"])

    def test_protocol_rejects_input_or_parameter_drift_and_overwrite(self):
        self.freeze()
        with self.assertRaises(ValueError):
            self.freeze()
        self.arrays["points"][0, 0] += .001
        self.save()
        with self.assertRaises(ValueError):
            audit.evaluate(self.scene_path, self.points_path, "test-table", self.background_path, self.protocol_path, self.root/"out")
        self.assertFalse((self.root/"out").exists())

    def test_parameter_drift_invalid_values_and_input_alias_fail_closed(self):
        self.freeze()
        with mock.patch.dict(audit.PARAMETERS, {"maximum_offset_m": .09}):
            with self.assertRaises(ValueError):
                audit.evaluate(self.scene_path, self.points_path, "test-table", self.background_path, self.protocol_path, self.root/"out")
        with self.assertRaises(ValueError):
            audit.evaluate(self.scene_path, self.points_path, "test-table", self.background_path, self.protocol_path, self.points_path)
        self.arrays["points"][0, 0] = np.nan
        self.save()
        with self.assertRaises(ValueError):
            audit.freeze(self.scene_path, self.points_path, "test-table", self.background_path, self.root/"bad-protocol.json")
        self.assertFalse((self.root/"bad-protocol.json").exists())

    def test_control_mutation_during_evaluation_is_rejected(self):
        self.freeze()
        def changed_control(*unused):
            self.background_path.write_text("{}")
            return [{"name": "background", "passed": False}], "mock"
        with mock.patch.object(audit, "consistency_controls", side_effect=changed_control), self.assertRaises(ValueError):
            audit.evaluate(self.scene_path, self.points_path, "test-table", self.background_path, self.protocol_path, self.root/"out")
        self.assertFalse((self.root/"out").exists())

    def test_estimator_rejects_nonfinite_and_nonrigid_frame(self):
        for points, rotation in ((np.full((12, 3), np.nan), np.eye(3)),
                                 (np.ones((12, 3)), np.diag([-1, 1, 1]))):
            with self.assertRaises(ValueError):
                audit.estimate_offset(points, {(0, 0)}, np.zeros(3), rotation)

    def test_background_provenance_and_rigid_validation(self):
        data = json.loads(self.background_path.read_text())
        data["source_observations_sha256"] = "wrong"
        self.background_path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            audit.background_transforms(self.background_path, audit.fingerprints(self.scene_path, self.points_path))
        self.save()
        data = json.loads(self.background_path.read_text())
        data["frames"][0].update(transform_applied=True, estimated_transform=np.diag([2, 1, 1, 1]).tolist())
        self.background_path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            audit.background_transforms(self.background_path, audit.fingerprints(self.scene_path, self.points_path))

    def test_full_denominator_replay_and_failed_control_fail_closed(self):
        self.freeze()
        with mock.patch.object(audit, "consistency_controls", return_value=([{"name": "background", "passed": False}], "mock")):
            report = audit.evaluate(self.scene_path, self.points_path, "test-table", self.background_path, self.protocol_path, self.root/"out")
        self.assertTrue(report["target_support_gate_passed"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["comparisons"]["all_original_validation"]["normal_translation"]["count"], 320)
        self.assertEqual(report["counts"]["disjoint_score"], 160)
        self.assertEqual(report["estimation_scoring_overlap"], 0)
        with np.load(self.root/"out"/"observations.npz") as data:
            np.testing.assert_array_equal(data["frame_index"], self.arrays["frame_index"])
            np.testing.assert_array_equal(data["candidates"], self.arrays["candidates"])
            np.testing.assert_array_equal(data["colors"], self.arrays["colors"])
            np.testing.assert_array_equal(data["points"], self.arrays["points"])
        self.assertEqual((self.root/"out"/"observations.npz").read_bytes(), self.points_path.read_bytes())
        canonical = json.loads((self.root/"out"/"scene.json").read_text())
        self.assertEqual(canonical["frames"], self.scene["frames"])
        self.assertEqual(canonical["observations"], "observations.npz")
        self.assertEqual(canonical["alignment_control"]["disposition"], "identity_fallback")
        self.assertFalse(report["export"]["candidate_adopted"])
        self.assertTrue(all(not row["canonical_transform_applied"] for row in report["frames"]))
        with np.load(self.root/"out"/"candidate-observations.npz") as candidate:
            np.testing.assert_allclose(candidate["points"][:, 1], .32, atol=1e-12)
        candidate_scene = json.loads((self.root/"out"/"candidate-scene.json").read_text())
        self.assertEqual(candidate_scene["schema"], "alvenx.real-table-alignment-candidate.v1")
        self.assertEqual(candidate_scene["observations"], "candidate-observations.npz")
        self.assertEqual(candidate_scene["alignment_control"]["disposition"], "diagnostic_candidate")
        self.assertNotEqual(candidate_scene["frames"], self.scene["frames"])
        from scan_pipeline import load_observations
        with self.assertRaises(ValueError):
            load_observations(self.root/"out"/"candidate-scene.json")
        public = json.loads((self.root/"out"/"summary.json").read_text())
        self.assertNotIn("training_plane_coefficients", public)
        self.assertTrue(all("estimated_transform" not in row and "applied_transform" not in row for row in public["frames"]))
        with self.assertRaises(ValueError):
            audit.evaluate(self.scene_path, self.points_path, "test-table", self.background_path, self.protocol_path, self.root/"out")

    def test_accepted_candidate_requires_complete_gate_and_preserves_ray_contract(self):
        # Scanner rounded homogeneous rows must not scale a new translation.
        for frame in self.scene["frames"]:
            frame["camera_to_world_cv"][3][3] = .99999994
        self.scene.update(layers=[{"path": "not-exported.ply", "positions_sha256": "old"}],
                          observed_ply="observed.ply", export_voxel_size_m=.015)
        self.save()
        self.freeze()
        with mock.patch.object(audit, "consistency_controls", return_value=([{"name": "background", "passed": True}], "mock")):
            report = audit.evaluate(self.scene_path, self.points_path, "test-table", self.background_path,
                                    self.protocol_path, self.root/"accepted")
        self.assertTrue(report["passed"])
        self.assertEqual(report["export"]["canonical_disposition"], "accepted")
        canonical = json.loads((self.root/"accepted"/"scene.json").read_text())
        self.assertEqual(canonical["observations"], "observations.npz")
        for name in ("scene.json", "candidate-scene.json"):
            written = json.loads((self.root/"accepted"/name).read_text())
            self.assertNotIn("layers", written)
            self.assertNotIn("observed_ply", written)
        with np.load(self.root/"accepted"/"observations.npz") as data:
            corrected = data["points"]
        for original, changed in zip(self.scene["frames"], canonical["frames"]):
            ids = self.arrays["frame_index"] == original["frame_index"]
            before = audit.baseline.local_points(self.arrays["points"][ids], np.array(original["camera_to_world_cv"]))
            after = audit.baseline.local_points(corrected[ids], np.array(changed["camera_to_world_cv"]))
            np.testing.assert_allclose(after, before, atol=1e-12)

    def test_method_mismatch_fails_before_any_output(self):
        self.freeze()
        with self.assertRaises(ValueError):
            audit.evaluate(self.scene_path, self.points_path, "test-table", self.background_path,
                           self.protocol_path, self.root/"wrong", "unknown")
        self.assertFalse((self.root/"wrong").exists())

    def test_rejected_frame_keeps_rounded_original_camera_record_exactly(self):
        self.scene["frames"][-1]["camera_to_world_cv"][3][3] = .99999994
        self.arrays["points"][self.arrays["frame_index"] == 35, 1] += .06
        self.save()
        self.freeze()
        with mock.patch.object(audit, "consistency_controls", return_value=([{"name": "background", "passed": False}], "mock")):
            audit.evaluate(self.scene_path, self.points_path, "test-table", self.background_path,
                           self.protocol_path, self.root/"rounded")
        candidate = json.loads((self.root/"rounded/candidate-scene.json").read_text())
        self.assertEqual(candidate["frames"][-1], self.scene["frames"][-1])

    def test_missing_background_frame_and_malformed_data_fail_before_output(self):
        data = json.loads(self.background_path.read_text())
        data["frames"].pop()
        self.background_path.write_text(json.dumps(data))
        self.freeze()
        with self.assertRaises(ValueError):
            audit.evaluate(self.scene_path, self.points_path, "test-table", self.background_path, self.protocol_path, self.root/"out")
        self.assertFalse((self.root/"out").exists())

    def test_cli_failed_gate_returns_nonzero(self):
        args = ["audit", "--scene", "s", "--observations", "p", "--object", "t", "--background-record", "b", "--protocol", "f", "--output", "o"]
        failed = {"passed": False, "target_support_gate_passed": True, "neighbour_background_gate_passed": False, "counts": {}}
        with mock.patch("sys.argv", args), mock.patch.object(audit, "evaluate", return_value=failed):
            self.assertEqual(audit.main(), 2)


if __name__ == "__main__":
    unittest.main()
