import copy
import json
from pathlib import Path
import tempfile
import struct
import unittest

import numpy as np
import scene_lab as lab


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/three-objects.json"


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.scene = lab.load_scene(FIXTURE)

    def test_known_three_objects_and_rotated_edit_roundtrip(self):
        updated = lab.edit(self.scene, "block-b", [25, 0, 0], [75, 90, 60], "cm")
        self.assertEqual(updated["objects"][0], self.scene["objects"][0])
        self.assertEqual(updated["objects"][2], self.scene["objects"][2])
        np.testing.assert_allclose(updated["objects"][1]["translation"], [.25, 0, .3])
        np.testing.assert_allclose(updated["objects"][1]["dimensions"], [.75, .9, .6])
        self.assertEqual(updated["objects"][1]["rotation_xyzw"], self.scene["objects"][1]["rotation_xyzw"])
        # tempfile's context finally removes only this test's unique owned directory.
        with tempfile.TemporaryDirectory(prefix=".spatial-test-", dir=FIXTURE.parents[1]) as tmp:
            path = Path(tmp) / "scene.glb"
            lab.export_fixture(updated, path)
            result = lab.verify_fixture(updated, path)
            self.assertEqual(result["objects"], 3)
            self.assertLess(result["max_vertex_error_m"], 1e-6)
            doc, binary = lab.read_glb(path)
            self.assertEqual(doc["asset"]["extras"]["source_scene"], updated)
            self.assertEqual(len(binary), 168)

    def test_meter_and_centimeter_scenes_export_identical_physical_geometry(self):
        centimeter = copy.deepcopy(self.scene)
        centimeter["units"] = "cm"
        for obj in centimeter["objects"]:
            obj["translation"] = (np.asarray(obj["translation"]) * 100).tolist()
            obj["dimensions"] = (np.asarray(obj["dimensions"]) * 100).tolist()
        with tempfile.TemporaryDirectory(prefix=".spatial-test-", dir=FIXTURE.parents[1]) as tmp:
            path = Path(tmp) / "cm.glb"
            lab.export_fixture(centimeter, path)
            # Deliberately verify against the meter source, rather than exported metadata.
            self.assertEqual(lab.verify_fixture(self.scene, path)["status"], "pass")

    def test_wrong_unit_relabel_is_detected(self):
        wrong = copy.deepcopy(self.scene)
        wrong["units"] = "cm"  # No numeric conversion: real 100x error.
        with tempfile.TemporaryDirectory(prefix=".spatial-test-", dir=FIXTURE.parents[1]) as tmp:
            path = Path(tmp) / "wrong.glb"
            lab.export_fixture(wrong, path)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                lab.verify_fixture(self.scene, path)

    def test_wrong_axis_conversion_is_detected(self):
        with tempfile.TemporaryDirectory(prefix=".spatial-test-", dir=FIXTURE.parents[1]) as tmp:
            path = Path(tmp) / "wrong.glb"
            lab.export_fixture(self.scene, path)
            doc, binary = lab.read_glb(path)
            # Remove +Z-up -> +Y-up rotation from all node matrices.
            conversion = np.eye(4); conversion[:3, :3] = lab.LAB_TO_GLTF.T
            for node in doc["nodes"]:
                matrix = np.asarray(node["matrix"]).reshape(4, 4).T
                node["matrix"] = np.einsum("ij,jk->ik", conversion, matrix).T.reshape(-1).tolist()
            lab.write_glb(path, doc, binary)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                lab.verify_fixture(self.scene, path)

    def test_source_hash_corruption_is_detected(self):
        with tempfile.TemporaryDirectory(prefix=".spatial-test-", dir=FIXTURE.parents[1]) as tmp:
            path = Path(tmp) / "wrong.glb"
            lab.export_fixture(self.scene, path, lab.sha256(FIXTURE))
            doc, binary = lab.read_glb(path)
            doc["asset"]["extras"]["source_package_sha256"] = "0" * 64
            lab.write_glb(path, doc, binary)
            with self.assertRaisesRegex(ValueError, "hash changed"):
                lab.verify_fixture(self.scene, path, lab.sha256(FIXTURE))

    def test_invalid_edits_and_conventions_fail(self):
        for dimensions in [[0, 1, 1], [-1, 1, 1], [float("nan"), 1, 1]]:
            with self.assertRaises(ValueError):
                lab.edit(self.scene, "block-b", [0, 0, 0], dimensions, "m")
        with self.assertRaisesRegex(ValueError, "unknown object"):
            lab.edit(self.scene, "missing", [0, 0, 0], [1, 1, 1], "m")
        bad = copy.deepcopy(self.scene); bad["frame"] = "unknown"
        with self.assertRaises(ValueError):
            lab.validate(bad)

    def test_nonfinite_binary_vertices_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix=".spatial-test-", dir=FIXTURE.parents[1]) as tmp:
            path = Path(tmp) / "nan.glb"
            lab.export_fixture(self.scene, path)
            doc, binary = lab.read_glb(path)
            broken = bytearray(binary)
            struct.pack_into("<f", broken, 0, float("nan"))
            lab.write_glb(path, doc, bytes(broken))
            with self.assertRaisesRegex(ValueError, "non-finite"):
                lab.verify_fixture(self.scene, path)
        bad = copy.deepcopy(self.scene); bad["objects"][0]["rotation_xyzw"] = [0, 0, 0, 2]
        with self.assertRaises(ValueError):
            lab.validate(bad)


if __name__ == "__main__":
    unittest.main()
