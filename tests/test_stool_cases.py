"""Physical-reference independence and actual sensor-crop acceptance checks."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from fixtures import make_stool_cases as fixture
import scan_pipeline


PROJECT = Path(__file__).resolve().parents[1]
OPEN3D = importlib.util.find_spec("open3d") is not None


class StoolPhysicalCasesTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(fixture.CONFIG.read_text(encoding="utf-8"))

    def test_reference_independent_of_prior_boxes(self):
        wrong = copy.deepcopy(self.config)
        wrong["defaults"]["prior"] = {"offset_m": [3, 2, -1], "dimension_scale": [.3, 4, 6], "yaw_error_degrees": 81}
        for case in self.config["cases"]:
            with self.subTest(case=case["id"]):
                expected = fixture.physical_scene(self.config, case)
                changed = fixture.physical_scene(wrong, case)
                for index in range(4):
                    np.testing.assert_array_equal(expected[index], changed[index])

    def test_dynamic_primitive_ranges_cover_target_without_environment(self):
        for case in self.config["cases"]:
            with self.subTest(case=case["id"]):
                vertices, triangles, owners, _, _, _, primitives, _ = fixture.physical_scene(self.config, case)
                self.assertTrue(np.isfinite(vertices).all())
                self.assertTrue(np.all((triangles >= 0) & (triangles < len(vertices))))
                covered = []
                for primitive in primitives:
                    start, count = primitive["triangle_start"], primitive["triangle_count"]
                    covered.extend(range(start, start + count))
                self.assertEqual(covered, np.flatnonzero(owners == 0).tolist())
                self.assertEqual(len(primitives), 1 + 2 * case["design"]["legs"])
                # Every physical part is a closed triangle shell before combination.
                for _, _, faces in fixture.stool_parts(case["design"]):
                    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
                    _, counts = np.unique(edges, axis=0, return_counts=True)
                    self.assertTrue(np.all(counts == 2))


@unittest.skipUnless(OPEN3D, "requires the optional CPU Open3D environment")
class StoolSensorCasesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import open3d
        cls.config = json.loads(fixture.CONFIG.read_text(encoding="utf-8"))
        workspace = PROJECT.parents[1]
        parent = None
        if (workspace / "workspace.json").is_file():
            parent = workspace / ".workspace/tmp/spatial-stool-transfer-20261009"
            parent.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="AlvenX-stool-cases-", dir=parent)
        cls.root = Path(cls.temporary.name)
        try:
            case = next(c for c in cls.config["cases"] if c["id"] == "transfer-upper-crop")
            cls.entry = fixture.generate_case(cls.root, cls.config, case, open3d,
                {"config_sha256": fixture.sha256(fixture.CONFIG),
                 "generator_sha256": fixture.sha256(fixture.__file__),
                 "shared_geometry_sensor_helper_sha256": fixture.sha256(fixture.base.__file__)})
        except BaseException:
            cls.temporary.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_pixel_crop_removes_lower_structure_without_label_filtering(self):
        counts = self.entry["primitive_valid_rays"]
        self.assertGreater(counts["seat"], 100)
        self.assertEqual(sum(v for k, v in counts.items() if k != "seat"), 0)
        total = 0
        for index in range(self.entry["frames"]):
            depth = np.asarray(Image.open(self.root / self.entry["scan"] / f"depth_{index:05d}.png"))
            self.assertFalse(np.any(depth[48:]))
            with np.load(self.root / self.entry["reference_dir"] / "frame_labels" / f"frame_{index:05d}.npz") as labels:
                self.assertFalse(np.any(labels["measurement_valid"][48:]))
                total += int(np.count_nonzero((labels["instance_owner"] == 0) & labels["measurement_valid"]))
        self.assertEqual(total, self.entry["target_valid_rays"])

    def test_scan_format_separates_reference_and_occupied_destination_is_safe(self):
        scan = self.root / self.entry["scan"]
        objects, structures = scan_pipeline.parse_roomplan(scan / "room.usdz")
        self.assertEqual([o["uuid"] for o in objects], ["same-stool"])
        self.assertEqual({s["category"] for s in structures}, {"Floor", "Wall"})
        self.assertFalse(any("reference" in p.name or "label" in p.name or "truth" in p.name for p in scan.iterdir()))
        frame = json.loads((scan / "frame_00000.json").read_text())
        self.assertEqual(set(frame), {"frame_index", "cameraPoseARFrame", "intrinsics"})
        before = fixture.sha256(scan / "depth_00000.png")
        with self.assertRaises(FileExistsError):
            fixture.generate(self.root)
        self.assertEqual(before, fixture.sha256(scan / "depth_00000.png"))


if __name__ == "__main__":
    unittest.main()
