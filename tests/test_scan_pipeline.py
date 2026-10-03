import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

import scan_pipeline as scan


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]
spec = importlib.util.spec_from_file_location("rgbd_fixture", PROJECT / "fixtures/make_rgbd_fixture.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def box(name, translation=(0, 0, 0), dimensions=(1, 1, 1)):
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    return {"name": name, "uuid": name, "category": "Table", "geometry_kind": "box",
            "transform_world": matrix.tolist(), "dimensions_m": list(dimensions)}


class RGBDTests(unittest.TestCase):
    def test_physical_front_plane_millimetres_axes_pose_and_intrinsic_scale(self):
        # An AR camera at (1,2,3) looks along -Z. A plane 2m ahead is z=1.
        # Pixel (0,0) is 2/3m left and 1/2m above the optical centre.
        pose = np.eye(4)
        pose[:3, 3] = [1, 2, 3]
        record = {"cameraPoseARFrame": pose.ravel().tolist(),
                  "intrinsics": [6, 0, 2, 0, 8, 2, 0, 0, 1]}
        k, cv_pose = scan.camera_from_record(record, (6, 4), (3, 2))
        depth = np.full((2, 3), 2000, dtype=np.uint16)
        color = np.full((2, 3, 3), [220, 10, 40], dtype=np.uint8)
        points, colors, pixels = scan.backproject_rgbd(depth, color, k, cv_pose)
        np.testing.assert_allclose(points[0], [1 - 2 / 3, 2.5, 1], atol=1e-12)
        np.testing.assert_allclose(points[4], [1, 2, 1], atol=1e-12)
        np.testing.assert_array_equal(pixels[4], [1, 1])
        np.testing.assert_array_equal(colors[0], [220, 10, 40])
        self.assertFalse(np.allclose(points[0], [1 - 2000 / 3, 502, -1997]))

    def test_rotated_camera_faces_world_negative_x(self):
        pose = np.eye(4)
        pose[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        pose[:3, 3] = [4, 2, -1]
        record = {"cameraPoseARFrame": pose.ravel().tolist(), "intrinsics": np.eye(3).ravel().tolist()}
        k, cv_pose = scan.camera_from_record(record, (1, 1), (1, 1))
        points, _, _ = scan.backproject_rgbd(np.array([[1000]], np.uint16), np.zeros((1, 1, 3)), k, cv_pose)
        np.testing.assert_allclose(points[0], [3, 2, -1], atol=1e-12)
        # Transposing this row-major capture would put translation in the last
        # row, and must fail rather than produce a self-consistent wrong scene.
        wrong = dict(record, cameraPoseARFrame=pose.T.ravel().tolist())
        with self.assertRaisesRegex(ValueError, "homogeneous"):
            scan.camera_from_record(wrong, (1, 1), (1, 1))

    def test_invalid_sensor_values_and_confidence_do_not_create_points(self):
        depth = np.array([[0, 65535, 1000, 4000, np.nan, -1, 2000]], dtype=float)
        confidence = np.array([[2, 2, 0, 2, 2, 2, 1]], dtype=np.uint8)
        points, _, pixels = scan.backproject_rgbd(depth, np.zeros((1, 7, 3)), np.eye(3), np.eye(4), confidence, 1, 3)
        np.testing.assert_array_equal(pixels, [[6, 0]])
        np.testing.assert_allclose(points, [[12, 0, 2]])
        with self.assertRaisesRegex(ValueError, "confidence shape"):
            scan.backproject_rgbd(depth, np.zeros((1, 7, 3)), np.eye(3), np.eye(4), np.zeros((2, 2)))

    def test_overlap_stays_ambiguous_and_structure_is_not_furniture(self):
        objects = [box("left", (-.2, 0, 0)), box("right", (.2, 0, 0))]
        floor = box("floor", (0, -.5, 0), (10, .0001, 10))
        floor["category"] = "Floor"
        points = np.array([[-.6, 0, 0], [0, 0, 0], [.6, 0, 0], [0, -.5, 0], [2, 0, 0]])
        owner, candidates = scan.assign_owners(points, objects, [floor], 0, .025)
        np.testing.assert_array_equal(owner, [0, -2, 1, -1, -1])
        np.testing.assert_array_equal(candidates[1], [True, True])
        self.assertFalse(candidates[3].any())

    def test_rotated_narrow_obb_classifies_in_local_coordinates(self):
        # A 2m local-X box rotated 90 degrees around Y extends along world Z.
        obj = box("rotated", (3, 1, -2), (2, .2, .2))
        matrix = np.asarray(obj["transform_world"])
        matrix[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        obj["transform_world"] = matrix.tolist()
        owner, _ = scan.assign_owners(np.array([[3, 1, -2.8], [3.8, 1, -2]]), [obj], box_margin_m=0)
        np.testing.assert_array_equal(owner, [0, -1])

    def test_usd_row_vector_translation_dimensions_and_uuid(self):
        with tempfile.TemporaryDirectory(prefix="AlvenX-rgbd-", dir=self.temp_root) as tmp:
            fixture.generate(tmp)
            objects, structures = scan.parse_roomplan(Path(tmp) / "room.usdz")
            target = next(x for x in objects if x["uuid"] == "synthetic-target")
            np.testing.assert_allclose(np.asarray(target["transform_world"])[:3, 3], [-.55, -.12, -2.0])
            np.testing.assert_allclose(target["dimensions_m"], [.65, .7, .55])
            self.assertEqual({x["category"] for x in structures}, {"Floor", "Wall"})
            bad = Path(tmp) / "bad.usdz"
            with zipfile.ZipFile(Path(tmp) / "room.usdz") as old, zipfile.ZipFile(bad, "w") as new:
                for member in old.namelist():
                    content = old.read(member)
                    if member == "room.usda":
                        content = content.replace(b'upAxis = "Y"', b'upAxis = "Z"')
                    new.writestr(member, content)
            with self.assertRaisesRegex(ValueError, "Y-up"):
                scan.parse_roomplan(bad)

    def test_usd_negative_triangle_index_and_count_are_rejected(self):
        # Original minimal triangular floor, independently authored here.
        root = '#usda 1.0\n(metersPerUnit = 1\nupAxis = "Y")\ndef Xform "room" { def Xform "Parametric_grp" { def Xform "Floor_grp" (prepend references = @./assets/floor.usda@) {} } }'
        component = '''#usda 1.0
def Xform "Floor0" (customData = { string Category = "Floor" string UUID = "negative-control-floor" }) {
def Mesh "Floor0" {
point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
int[] faceVertexIndices = INDICES
int[] faceVertexCounts = COUNTS
matrix4d xformOp:transform = ((1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1))
uniform token[] xformOpOrder = ["xformOp:transform"]
} }
'''
        for indices, counts, rejection in [("[0, 1, 2]", "[3]", None),
                                           ("[-1, 1, 2]", "[3]", "invalid RoomPlan mesh"),
                                           ("[0, 1, 2]", "[-3]", "only triangular RoomPlan Floor")]:
            with self.subTest(indices=indices, counts=counts):
                memory_archive = io.BytesIO()
                with zipfile.ZipFile(memory_archive, "w") as archive:
                    archive.writestr("room.usda", root)
                    archive.writestr("assets/floor.usda", component.replace("INDICES", indices).replace("COUNTS", counts))
                memory_archive.seek(0)
                if rejection:
                    with self.assertRaisesRegex(ValueError, rejection):
                        scan.parse_roomplan(memory_archive)
                else:
                    _, structures = scan.parse_roomplan(memory_archive)
                    self.assertEqual(structures[0]["triangles"], [[0, 1, 2]])

    @classmethod
    def setUpClass(cls):
        workspace_file = WORKSPACE / "workspace.json"
        is_workspace = (workspace_file.is_file() and json.loads(workspace_file.read_text(
            encoding="utf-8"))["workspace"]["name"] == "AlvenX")
        directory = WORKSPACE / ".workspace/tmp" if is_workspace else None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
        cls.owned_temp = tempfile.TemporaryDirectory(prefix="AlvenX-spatial-local-", dir=directory)
        cls.temp_root = Path(cls.owned_temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.owned_temp.cleanup()

    def test_synthetic_ray_groundtruth_and_saved_real_observation_edit(self):
        with tempfile.TemporaryDirectory(prefix="rgbd-", dir=self.temp_root) as tmp:
            path = Path(tmp)
            fixture.generate(path / "input")
            result = scan.reconstruct(path / "input", path / "before", pixel_step=1,
                                      voxel_size_m=0, box_margin_m=.003, structure_tolerance_m=.003)
            self.assertEqual(result["frames"], 2)
            scene, before = scan.load_observations(path / "before/scene.json")
            target = next(i for i, obj in enumerate(scene["objects"]) if obj["uuid"] == "synthetic-target")
            other = next(i for i, obj in enumerate(scene["objects"]) if obj["uuid"] == "synthetic-other")
            # Independently ray-cast semantic labels, not labels made by the
            # production inverse projection/box assignment under test.
            expected = np.empty(len(before["points"]), dtype=np.int16)
            for frame_index in range(2):
                truth = np.load(path / f"input/groundtruth_labels_{frame_index:05d}.npy")
                selected = before["frame_index"] == frame_index
                uv = before["pixel_uv"][selected]
                expected[selected] = truth[uv[:, 1], uv[:, 0]]
            truth_to_actual = {0: target, 1: other}
            for truth_id, actual_id in truth_to_actual.items():
                self.assertGreater(np.count_nonzero(expected == truth_id), 100)
                np.testing.assert_array_equal(before["owner"][expected == truth_id], actual_id)
                self.assertTrue(np.all(expected[before["owner"] == actual_id] == truth_id))
            np.testing.assert_array_equal(before["owner"][expected == -1], -1)
            # Actual front face, independently fixed at box centre + half Z.
            target_points = before["points"][before["owner"] == target]
            self.assertLess(abs(target_points[:, 2].max() - (-2 + .55 / 2)), .0006)
            edit = scan.edit_observations(path / "before/scene.json", "synthetic-target", [.3, 0, .1], path / "after")
            self.assertEqual(edit["status"], "pass")
            self.assertTrue(edit["complete_geometric_selection"])
            after_scene, after = scan.load_observations(path / "after/scene.json")
            np.testing.assert_array_equal(before["points"][before["owner"] == other], after["points"][after["owner"] == other])
            self.assertEqual(edit["checks"]["selected_sample_ids_remaining_in_environment"], 0)
            # Negative control: moving only the proxy box leaves measured
            # points at the old location and must fail the geometric check.
            rejected = scan.verify_edit(scene, before, after_scene, before, target, [.3, 0, .1])
            self.assertEqual(rejected["status"], "fail")
            tampered = {k: v.copy() for k, v in after.items()}
            tampered["points"][np.flatnonzero(before["owner"] == other)[0], 0] += .1
            self.assertEqual(scan.verify_edit(scene, before, after_scene, tampered, target, [.3, 0, .1])["status"], "fail")

    def test_ambiguous_selected_candidate_prevents_complete_claim(self):
        scene = {"objects": [box("a"), box("b")], "edits": []}
        before = {"points": np.array([[0., 0, 0], [.2, .2, .2]]),
                  "colors": np.zeros((2, 3), np.uint8), "owner": np.array([0, -2]),
                  "candidates": np.array([[True, False], [True, True]])}
        after = copy.deepcopy(before)
        after["points"][0] += [1, 0, 0]
        after_scene = copy.deepcopy(scene)
        after_scene["objects"][0]["transform_world"][0][3] += 1
        result = scan.verify_edit(scene, before, after_scene, after, 0, [1, 0, 0])
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["complete_geometric_selection"])
        self.assertEqual(result["checks"]["unresolved_target_candidate_observations"], 1)

    def test_scan_with_no_valid_confidence_cannot_report_reconstruction_success(self):
        with tempfile.TemporaryDirectory(prefix="invalid-rgbd-", dir=self.temp_root) as tmp:
            path = Path(tmp)
            fixture.generate(path / "input")
            for confidence_path in (path / "input").glob("conf_*.png"):
                Image.fromarray(np.zeros((32, 48), np.uint8)).save(confidence_path)
            with self.assertRaisesRegex(ValueError, "no valid RGB-D observations"):
                scan.reconstruct(path / "input", path / "output")
            self.assertFalse((path / "output/scene.json").exists())

    def test_edit_rejects_target_identity_dimensions_or_scene_metadata_corruption(self):
        scene = {"schema": scan.SCHEMA, "units": "m", "up_axis": "Y", "source": "original-input",
                 "objects": [box("target"), box("other")], "structures": [box("wall")], "edits": []}
        before = {"points": np.array([[0., 0, 0], [2., 0, 0]]),
                  "colors": np.zeros((2, 3), np.uint8), "owner": np.array([0, 1]),
                  "candidates": np.array([[True, False], [False, True]])}
        after = copy.deepcopy(before)
        after["points"][0] += [.3, 0, 0]
        edited_scene = copy.deepcopy(scene)
        edited_scene["objects"][0]["transform_world"][0][3] += .3
        self.assertEqual(scan.verify_edit(scene, before, edited_scene, after, 0, [.3, 0, 0])["status"], "pass")
        for field, value in [("uuid", "wrong-uuid"), ("dimensions_m", [100, 100, 100]), ("category", "Wrong")]:
            with self.subTest(target_field=field):
                corrupted = copy.deepcopy(edited_scene)
                corrupted["objects"][0][field] = value
                self.assertEqual(scan.verify_edit(scene, before, corrupted, after, 0, [.3, 0, 0])["status"], "fail")
        for field, value in [("structures", []), ("source", "other-input"), ("units", "cm"),
                             ("up_axis", "Z"), ("schema", "other-schema")]:
            with self.subTest(scene_field=field):
                corrupted = copy.deepcopy(edited_scene)
                corrupted[field] = value
                self.assertEqual(scan.verify_edit(scene, before, corrupted, after, 0, [.3, 0, 0])["status"], "fail")

    def test_voxel_selection_keeps_real_color_and_no_averaged_position(self):
        points = np.array([[.001, 0, 0], [.002, 0, 0], [1, 0, 0]])
        np.testing.assert_array_equal(scan.voxel_select(points, .01), [0, 2])


if __name__ == "__main__":
    unittest.main()
