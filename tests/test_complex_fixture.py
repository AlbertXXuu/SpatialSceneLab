"""Physical-reference independence and ScannerApp-format fixture acceptance."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

import numpy as np

import scan_pipeline as scan


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("complex_fixture", PROJECT / "fixtures/make_complex_fixture.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


class ComplexFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(fixture.CONFIG_PATH.read_text(encoding="utf-8"))
        workspace = PROJECT.parents[1]
        temporary_root = None
        if (workspace / "workspace.json").is_file():
            temporary_root = workspace / ".workspace/tmp/spatial-s0-fixtures"
            temporary_root.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="AlvenX-complex-fixture-", dir=temporary_root)
        cls.root = Path(cls.temporary.name)
        try:
            cls.manifest = fixture.generate_suite(cls.root, (32, 24), ["dev-contact-chair"])
        except BaseException:
            cls.temporary.cleanup()
            raise
        cls.scene_entry = cls.manifest["scenes"][0]
        cls.scan_dir = cls.root / cls.scene_entry["scan"]
        cls.reference = cls.root / "dev-contact-chair/reference"

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_six_distinct_multi_part_scenes_and_independent_reference(self):
        signatures = set()
        asset_scene_by_hash = {}
        for scene in self.config["scenes"]:
            with self.subTest(scene=scene["scene_id"]):
                vertices, triangles, owner, colours, objects, _ = fixture._build_scene(self.config, scene)
                self.assertTrue(np.isfinite(vertices).all())
                self.assertEqual(triangles.dtype, np.int32)
                self.assertEqual(owner.dtype, np.int32)
                self.assertEqual(set(np.unique(owner)), {-1, *range(len(objects))})
                self.assertGreater(len(triangles), 500)
                self.assertTrue(np.all(triangles >= 0))
                self.assertTrue(np.all(triangles < len(vertices)))
                self.assertTrue(np.all(np.linalg.norm(np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                                                              vertices[triangles[:, 2]] - vertices[triangles[:, 0]]), axis=1) > 1e-8))
                signatures.add(hashlib.sha256(vertices.tobytes() + triangles.tobytes()).hexdigest())
                wrong_boxes = copy.deepcopy(scene)
                for obj in wrong_boxes["objects"]:
                    obj["prior"] = {"offset_m": [9, -8, 7], "dimension_scale": [.1, 4, 6], "yaw_error_degrees": 113}
                changed = fixture._build_scene(self.config, wrong_boxes)
                np.testing.assert_array_equal(changed[0], vertices)
                np.testing.assert_array_equal(changed[1], triangles)
                np.testing.assert_array_equal(changed[2], owner)
                self.assertTrue(12 <= scene["trajectory"]["frames"] <= 20)
                for obj in scene["objects"]:
                    parts = fixture._furniture(obj)
                    geometry = b"".join((points * obj["scale"]).tobytes() + faces.tobytes() for points, faces, _ in parts)
                    key = hashlib.sha256(geometry).hexdigest()
                    self.assertIn(asset_scene_by_hash.get(key, scene["scene_id"]), [scene["scene_id"]])
                    asset_scene_by_hash[key] = scene["scene_id"]
        self.assertEqual(len(signatures), 6)

    def test_labels_mapping_input_separation_and_aligned_usdz(self):
        labels = read_npz(self.reference / "labels.npz")
        objects, structures = scan.parse_roomplan(self.scan_dir / "room.usdz")
        self.assertEqual(labels["object_ids"].dtype.kind, "U")
        self.assertEqual(labels["object_ids"].tolist(), [obj["uuid"] for obj in objects])
        self.assertEqual({obj["category"] for obj in structures}, {"Floor", "Wall"})
        self.assertFalse(any("label" in path.name or "truth" in path.name or "reference" in path.name for path in self.scan_dir.iterdir()))
        with zipfile.ZipFile(self.scan_dir / "room.usdz") as archive:
            for info in archive.infolist():
                payload_offset = info.header_offset + 30 + len(info.filename.encode("utf-8")) + len(info.extra)
                self.assertEqual(payload_offset % 64, 0)
                self.assertEqual(info.compress_type, zipfile.ZIP_STORED)
        for key in ["scan", "reference_mesh", "reference_labels"]:
            self.assertFalse(Path(self.scene_entry[key]).is_absolute())
            self.assertTrue((self.root / self.scene_entry[key]).exists())
        self.assertEqual(self.scene_entry["target_id"], "contact-chair")

    def test_optical_depth_matches_independent_physical_triangle_planes(self):
        labels = read_npz(self.reference / "labels.npz")
        all_observed_owners = set()
        for frame_index in [0, 4, 8, 12, 15]:
            name = f"{frame_index:05d}"
            frame = scan.load_frame(self.scan_dir / f"frame_{name}.json")
            points, _, uv = scan.backproject_rgbd(frame["depth_mm"], frame["color"], frame["intrinsics_depth"],
                                                 frame["camera_to_world_cv"], frame["confidence"])
            truth = read_npz(self.reference / f"frame_labels/frame_{name}.npz")
            ids = truth["triangle_index"][uv[:, 1], uv[:, 0]]
            self.assertTrue(np.all(ids >= 0))
            vertex_triplets = labels["vertices"][labels["triangles"][ids]].astype(float)
            normals = np.cross(vertex_triplets[:, 1] - vertex_triplets[:, 0], vertex_triplets[:, 2] - vertex_triplets[:, 0])
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            distance = np.abs(np.sum((points - vertex_triplets[:, 0]) * normals, axis=1))
            # Optical-axis 0.5mm quantisation may produce >0.5mm world-ray distance.
            self.assertLess(float(distance.max()), .0008)
            np.testing.assert_array_equal(truth["instance_owner"][uv[:, 1], uv[:, 0]], labels["triangle_owner"][ids])
            all_observed_owners.update(labels["triangle_owner"][ids].tolist())
        self.assertIn(0, all_observed_owners)
        self.assertIn(-1, all_observed_owners)

    def test_noise_pose_and_prior_perturbations_do_not_modify_physical_labels(self):
        scene = self.config["scenes"][-1]
        vertices, triangles, owners, _, objects, structures = fixture._build_scene(self.config, scene)
        priors = fixture._measured_priors(scene, objects, structures)
        target = priors[0]
        physical = objects[0]
        lower, upper = np.asarray(physical["local_bounds_m"])
        original = np.asarray(physical["transform_world"])
        original_center = original[:3, :3] @ ((lower + upper) / 2) + original[:3, 3]
        np.testing.assert_allclose(np.asarray(target["transform"])[:3, 3] - original_center, [.1, -.04, .07])
        self.assertFalse(np.allclose(np.asarray(target["transform"])[:3, :3], original[:3, :3]))
        self.assertFalse(np.allclose(target["dimensions"], upper - lower))
        self.assertGreater(scene["sensor"]["pose_translation_sigma_m"], 0)
        self.assertGreater(scene["sensor"]["depth_noise_sigma_m"], 0)
        self.assertEqual(vertices.shape[1], 3)
        self.assertEqual(triangles.shape[1], 3)
        self.assertEqual(len(triangles), len(owners))

    def test_noisy_measurements_keep_true_cameras_in_reference_only(self):
        with tempfile.TemporaryDirectory(prefix="noisy-", dir=self.root) as temporary:
            output = Path(temporary)
            fixture.generate_suite(output, (32, 24), ["dev-combined-study"])
            scan_dir = output / "dev-combined-study/scan"
            reference = output / "dev-combined-study/reference"
            physical = json.loads((reference / "physical-scene.json").read_text(encoding="utf-8"))
            frame = scan.load_frame(scan_dir / "frame_00000.json")
            truth_pose = np.asarray(physical["true_camera_poses_arkit"][0])
            measured = np.asarray(frame["record"]["cameraPoseARFrame"]).reshape(4, 4)
            self.assertGreater(np.linalg.norm(truth_pose[:3, 3] - measured[:3, 3]), .001)
            self.assertFalse(np.allclose(truth_pose[:3, :3], measured[:3, :3]))
            self.assertEqual(set(frame["record"]), {"frame_index", "cameraPoseARFrame", "intrinsics"})
            truth = read_npz(reference / "frame_labels/frame_00000.npz")
            valid = truth["measurement_valid"]
            error = frame["depth_mm"][valid].astype(float) / 1000 - truth["exact_depth_m"][valid]
            self.assertGreater(np.std(error), .003)
            geometric_valid = truth["triangle_index"] >= 0
            self.assertGreater(np.count_nonzero(geometric_valid & ~valid), 0)
            self.assertGreater(sum(f["target_valid_pixels"] for f in physical["frame_statistics"]), 0)
            self.assertEqual(int(frame["depth_mm"][0, 1]), 65535)

    def test_output_write_error_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory(prefix="write-failure-", dir=self.root) as temporary:
            output = Path(temporary) / "existing-file"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                fixture.generate_suite(output, (32, 24), ["dev-contact-chair"])
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")

    def test_generation_replay_overwrites_identically(self):
        before = {path.relative_to(self.root): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in self.root.rglob("*") if path.is_file()}
        repeated = fixture.generate_suite(self.root, (32, 24), ["dev-contact-chair"])
        after = {path.relative_to(self.root): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(repeated, self.manifest)
        self.assertEqual(before, after)

    def test_generator_source_identity_is_saved_with_reference(self):
        expected = hashlib.sha256(Path(fixture.__file__).read_bytes()).hexdigest()
        physical = json.loads((self.reference / "physical-scene.json").read_text(encoding="utf-8"))
        self.assertEqual(self.manifest["generator_sha256"], expected)
        self.assertEqual(physical["generator_sha256"], expected)

    def test_nonempty_unknown_output_is_preserved(self):
        for existing_manifest in [None, {"schema": "other"}, [], "invalid-json"]:
            with self.subTest(manifest=existing_manifest), tempfile.TemporaryDirectory(prefix="foreign-", dir=self.root) as temporary:
                output = Path(temporary)
                sentinel = output / "existing-data.txt"
                sentinel.write_text("preserve", encoding="utf-8")
                if existing_manifest is not None:
                    (output / "suite-manifest.json").write_text(
                        existing_manifest if existing_manifest == "invalid-json" else json.dumps(existing_manifest), encoding="utf-8")
                before = {path.name: path.read_bytes() for path in output.iterdir()}
                with self.assertRaises(ValueError):
                    fixture.generate_suite(output, (32, 24), ["dev-contact-chair"])
                self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})

    def test_preexisting_file_directory_and_broken_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="links-", dir=self.root) as temporary:
            base = Path(temporary)
            target = base / "target"
            target.mkdir()
            sentinel = target / "data.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            cases = [
                ("file", Path("dev-contact-chair/scan/frame_00000.json"), sentinel, False),
                ("directory", Path("dev-contact-chair/reference/frame_labels"), target, True),
                ("broken", Path("dev-contact-chair/reference/labels.npz"), target / "missing", False)
            ]
            for name, relative, link_target, directory in cases:
                with self.subTest(case=name):
                    output = base / name
                    output.mkdir()
                    (output / "suite-manifest.json").write_text(json.dumps({"schema": "spatial-scene-lab.fixture-suite.v1"}), encoding="utf-8")
                    link = output / relative
                    link.parent.mkdir(parents=True)
                    try:
                        link.symlink_to(link_target, target_is_directory=directory)
                    except OSError as error:
                        self.skipTest(f"symlink creation unavailable for this account: {error}")
                    try:
                        with self.assertRaisesRegex(ValueError, "link or junction"):
                            fixture.generate_suite(output, (32, 24), ["dev-contact-chair"])
                    finally:
                        link.unlink()
                    self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
            ancestor_link = base / "ancestor-link"
            ancestor_link.symlink_to(target, target_is_directory=True)
            try:
                with self.assertRaisesRegex(ValueError, "link or junction"):
                    fixture.generate_suite(ancestor_link / "new-output", (32, 24), ["dev-contact-chair"])
                self.assertFalse((target / "new-output").exists())
            finally:
                ancestor_link.unlink()

    @unittest.skipUnless(os.name == "nt", "Windows junction case")
    def test_windows_junction_in_reference_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="junction-", dir=self.root) as temporary:
            base = Path(temporary)
            target = base / "target"
            target.mkdir()
            sentinel = target / "data.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            output = base / "output"
            output.mkdir()
            (output / "suite-manifest.json").write_text(json.dumps({"schema": "spatial-scene-lab.fixture-suite.v1"}), encoding="utf-8")
            junction = output / "dev-contact-chair/reference/frame_labels"
            junction.parent.mkdir(parents=True)
            created = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(target)], capture_output=True)
            if created.returncode:
                self.skipTest("junction creation unavailable for this account")
            try:
                with self.assertRaisesRegex(ValueError, "link or junction"):
                    fixture.generate_suite(output, (32, 24), ["dev-contact-chair"])
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
            finally:
                junction.rmdir()  # Remove the junction itself; never traverse its target.

    def test_invalid_arguments_fail_before_output_creation(self):
        rejected = self.root / "must-not-exist"
        for resolution in [(0, 120), (160, 0), (True, 120), (160.0, 120), (641, 120), (160, 481), (160,)]:
            with self.subTest(resolution=resolution), self.assertRaises(ValueError):
                fixture.generate_suite(rejected, resolution)
        for scene_ids in [[], ["unknown"], ["dev-contact-chair", "dev-contact-chair"], "dev-contact-chair", [1]]:
            with self.subTest(scene_ids=scene_ids), self.assertRaises(ValueError):
                fixture.generate_suite(rejected, scene_ids=scene_ids)
        self.assertFalse(rejected.exists())


if __name__ == "__main__":
    unittest.main()
