import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import verify_partition_import as native

AVAILABLE = importlib.util.find_spec("open3d") is not None
if AVAILABLE:
    import open3d as o3d
    import export_partition as export


@unittest.skipUnless(AVAILABLE, "Open3D CPU surface dependencies are optional")
class PartitionExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="AlvenX-s2a-export-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "results" / "dev-example"
        (self.source / "surface").mkdir(parents=True)
        (self.source / "points").mkdir()
        self.output = self.root / "export" / "dev-example" / "surface"
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        triangles = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 2, 4]], dtype=np.int32)
        self.mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                            o3d.utility.Vector3iVector(triangles))
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(np.arange(15).reshape(5, 3) / 255)
        self.mesh.compute_vertex_normals()
        self.mesh_path = self.source / "surface" / "surface.ply"
        o3d.io.write_triangle_mesh(str(self.mesh_path), self.mesh)
        self.observation = {"schema": "spatial-scene-lab.observations.v1", "units": "m", "up_axis": "Y",
                            "objects": [{"uuid": "chair", "name": "Chair", "category": "Chair"},
                                        {"uuid": "table", "name": "Table", "category": "Table"}]}
        self.owner_path = self.source / "b1-face-owner.npy"
        np.save(self.owner_path, np.array([-1, -2, 0, 1], dtype=np.int32))
        self.surface = {"schema": "spatial-scene-lab.surface.v1", "status": "pass", "units": "m", "up_axis": "Y",
                        "vertices": 5, "triangles": 4, "parts": [{"id": "environment", "path": "original.ply",
                                                                 "sha256": export.sha256(self.mesh_path), "triangles": 4}]}
        (self.source / "surface" / "original.ply").write_bytes(self.mesh_path.read_bytes())
        self.record = {"scene_id": "dev-example", "status": "completed", "vertices": 5, "triangles": 4,
                       "surface_sha256": export.sha256(self.mesh_path),
                       "methods": [{"method": "b1", "status": "completed", "face_owner_sha256": export.sha256(self.owner_path),
                                    "parameters": {"method": "b1", "reference_used": False, "mesh_changed": False}}]}
        self.freeze()

    def freeze(self):
        export.save_json(self.source / "points" / "scene.json", self.observation)
        self.record["scene_sha256"] = export.sha256(self.source / "points" / "scene.json")
        self.surface["input_scene_sha256"] = self.record["scene_sha256"]
        export.save_json(self.source / "surface" / "surface-scene.json", self.surface)
        self.record["surface_manifest_sha256"] = export.sha256(self.source / "surface" / "surface-scene.json")

    def run_export(self):
        return export.export_scene(self.source, self.record, self.output, "a" * 64)

    def test_export_retains_each_source_face_geometry_color_and_owner_once(self):
        # No reference directory or labels exist in this fixture.
        result = self.run_export()
        self.assertTrue(result["partition"]["all_triangles_accounted_for"])
        seen = []
        source_vertices = np.asarray(self.mesh.vertices)
        source_triangles = np.asarray(self.mesh.triangles)
        source_colors = np.asarray(self.mesh.vertex_colors)
        owners = np.load(self.owner_path)
        for part in result["parts"]:
            current = o3d.io.read_triangle_mesh(str(self.output / part["path"]))
            face_map = np.load(self.output / part["source_indices"]["face"]["path"])
            vertex_map = np.load(self.output / part["source_indices"]["vertex"]["path"])
            np.testing.assert_array_equal(vertex_map[np.asarray(current.triangles)], source_triangles[face_map])
            np.testing.assert_array_equal(np.asarray(current.vertices), source_vertices[vertex_map])
            np.testing.assert_array_equal(np.asarray(current.vertex_colors), source_colors[vertex_map])
            np.testing.assert_array_equal(owners[face_map], part["owner_label"])
            self.assertEqual(part["native_source_face_offset"], len(seen))
            seen.extend(face_map.tolist())
        self.assertEqual(sorted(seen), list(range(4)))
        self.assertEqual(export.sha256(self.owner_path), export.sha256(self.output / "b1-face-owner.npy"))
        unresolved = next(part for part in result["parts"] if part["id"] == "unresolved")
        self.assertEqual(unresolved["owner_label"], -2)
        self.assertEqual(unresolved["ownership_state"], "unresolved")
        self.assertEqual(unresolved["name"], "Unresolved (-2)")
        self.assertEqual(unresolved["role"], "observed_object_region")

    def test_each_frozen_source_hash_is_checked_before_output(self):
        for relative in ("surface/surface.ply", "surface/surface-scene.json", "points/scene.json", "b1-face-owner.npy"):
            with self.subTest(relative=relative):
                path = self.source / relative
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"drift")
                    with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                        self.run_export()
                    self.assertFalse(self.output.exists())
                finally:
                    path.write_bytes(original)

    def test_labels_must_be_integer_one_per_face_and_within_objects(self):
        for labels in (np.array([0, 1]), np.array([[-1, -2, 0, 1]]), np.array([-1., -2., 0., 1.]),
                       np.array([-3, -2, 0, 1]), np.array([-1, -2, 0, 2]), np.array([False] * 4)):
            with self.subTest(labels=labels):
                np.save(self.owner_path, labels)
                self.record["methods"][0]["face_owner_sha256"] = export.sha256(self.owner_path)
                with self.assertRaisesRegex(ValueError, "owners must|owner label"):
                    self.run_export()
                self.assertFalse(self.output.exists())

    def test_count_mismatch_is_rejected(self):
        for document, key in ((self.surface, "triangles"), (self.record, "vertices")):
            original = document[key]
            document[key] = original + 1
            self.freeze()
            with self.assertRaisesRegex(ValueError, "count differs"):
                self.run_export()
            document[key] = original
        self.assertFalse(self.output.exists())

    def test_duplicate_reserved_and_unsafe_object_ids_are_rejected(self):
        for identity in ("chair", "CHAIR", "environment", "Unresolved", "../escape", "C:drive", "NUL", "a/b"):
            with self.subTest(identity=identity):
                self.observation["objects"][1]["uuid"] = identity
                self.freeze()
                with self.assertRaisesRegex(ValueError, "identities|filename"):
                    self.run_export()
                self.assertFalse(self.output.exists())

    def test_part_path_escape_and_duplicates_are_rejected_with_valid_hash(self):
        original = copy.deepcopy(self.surface["parts"])
        outside = self.source / "outside.ply"
        outside.write_bytes(self.mesh_path.read_bytes())
        self.surface["parts"][0]["path"] = "../outside.ply"
        self.freeze()
        with self.assertRaisesRegex(ValueError, "escaped"):
            self.run_export()
        self.surface["parts"] = original + [dict(original[0], id="chair")]
        self.freeze()
        with self.assertRaisesRegex(ValueError, "path is duplicated"):
            self.run_export()

    def test_nonfinite_mesh_and_color_are_rejected(self):
        bad = o3d.geometry.TriangleMesh(self.mesh)
        np.asarray(bad.vertices)[0, 0] = np.nan
        with patch.object(export.o3d.io, "read_triangle_mesh", return_value=bad):
            with self.assertRaisesRegex(ValueError, "finite"):
                self.run_export()
        bad = o3d.geometry.TriangleMesh(self.mesh)
        np.asarray(bad.vertex_colors)[0, 0] = np.inf
        with patch.object(export.o3d.io, "read_triangle_mesh", return_value=bad):
            with self.assertRaisesRegex(ValueError, "finite"):
                self.run_export()

    def test_nonfinite_manifest_is_rejected(self):
        path = self.source / "points" / "scene.json"
        path.write_text(json.dumps(dict(self.observation, invalid=float("nan"))))
        self.record["scene_sha256"] = export.sha256(path)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.run_export()

    def test_output_cannot_overwrite_sources_or_prior_runs(self):
        for output in (self.source, self.source / "new-export", self.source.parent):
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                export.export_scene(self.source, self.record, output, "a" * 64)
        self.run_export()
        before = export.sha256(self.output / "surface-scene.json")
        with self.assertRaisesRegex(ValueError, "retain existing"):
            self.run_export()
        self.assertEqual(before, export.sha256(self.output / "surface-scene.json"))

    def test_empty_owner_group_does_not_create_empty_blender_part(self):
        np.save(self.owner_path, np.array([-1, -1, 0, 0], dtype=np.int32))
        self.record["methods"][0]["face_owner_sha256"] = export.sha256(self.owner_path)
        result = self.run_export()
        self.assertEqual([part["id"] for part in result["parts"]], ["environment", "chair"])
        self.assertEqual(result["partition"]["unresolved_triangles"], 0)

    def test_write_failure_does_not_publish_success_manifest(self):
        with patch.object(export.o3d.io, "write_triangle_mesh", return_value=False):
            with self.assertRaisesRegex(OSError, "could not write"):
                self.run_export()
        self.assertFalse((self.output / "surface-scene.json").exists())

    def test_native_lineage_preparation_rejects_mapping_tamper_even_with_new_hash(self):
        result = self.run_export()
        part = next(part for part in result["parts"] if part["id"] == "chair")
        mapping = part["source_indices"]["face"]
        path = self.output / mapping["path"]
        np.save(path, np.array([3]))  # Table's face cannot be assigned Chair's identity.
        mapping["sha256"] = export.sha256(path)
        export.save_json(self.output / "surface-scene.json", result)
        with self.assertRaisesRegex(ValueError, "owner label differs"):
            native.expected_initial(self.source, self.record, self.output / "surface-scene.json")

    def test_native_lineage_preparation_rejects_duplicate_mapping(self):
        result = self.run_export()
        part = result["parts"][0]
        mapping = part["source_indices"]["face"]
        path = self.output / mapping["path"]
        np.save(path, np.array([0, 0]))
        mapping.update(sha256=export.sha256(path), count=2)
        export.save_json(self.output / "surface-scene.json", result)
        with self.assertRaisesRegex(ValueError, "unique in-range"):
            native.expected_initial(self.source, self.record, self.output / "surface-scene.json")

    def test_batch_hash_gate_failure_retention_and_reproduction_manifest(self):
        doc = {"schema": "spatial-scene-lab.development.v1", "scenes": [self.record]}
        baseline = self.source.parent / "development-results.json"
        export.save_json(baseline, doc)
        publication = self.root / "publication.json"
        publication.write_bytes(baseline.read_bytes())
        publication.write_bytes(publication.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "differ from published"):
            export.export_bundles(self.source.parent, publication, self.root / "batch")
        # A newly generated development result can be its own explicit baseline.
        self.owner_path.write_bytes(b"corrupt frozen ownership")
        result = export.export_bundles(self.source.parent, baseline, self.root / "batch")
        self.assertEqual(result["status"], "fail")
        self.assertIn("SHA256 mismatch", result["scenes"][0]["error"])
        self.assertTrue((self.root / "batch" / "partition-export-results.json").is_file())


class NativeLineageComparisonTests(unittest.TestCase):
    def setUp(self):
        self.expected = {"ids": np.array([0, 1]), "owners": np.array(["unresolved", "chair"]),
                         "corners": np.arange(18, dtype=float).reshape(2, 3, 3) / 10,
                         "colors": np.arange(24, dtype=float).reshape(2, 3, 4) / 255}

    def test_same_faces_allow_only_declared_numeric_rounding(self):
        actual = copy.deepcopy(self.expected)
        actual["corners"][0, 0, 0] += 2e-7
        actual["colors"][0, 0, 0] += 2e-7
        self.assertEqual(native.compare_initial(self.expected, actual)["status"], "pass")

    def test_face_corner_identity_owner_color_and_nonfinite_faults_are_rejected(self):
        faults = []
        for key in ("ids", "owners", "corners", "colors"):
            actual = copy.deepcopy(self.expected)
            actual[key] = actual[key][::-1]
            faults.append((f"permuted {key}", actual))
        actual = copy.deepcopy(self.expected)
        actual["corners"] = actual["corners"][:, [1, 2, 0]]
        faults.append(("cyclic corner reassignment", actual))
        actual = copy.deepcopy(self.expected)
        actual["corners"][1] = actual["corners"][0]
        faults.append(("duplicated face", actual))
        actual = copy.deepcopy(self.expected)
        actual["colors"][0, 0, 0] = np.nan
        faults.append(("nonfinite color", actual))
        actual = copy.deepcopy(self.expected)
        actual["ids"] = np.array([0])
        faults.append(("missing face", actual))
        for name, actual in faults:
            with self.subTest(fault=name):
                self.assertEqual(native.compare_initial(self.expected, actual)["status"], "fail")


if __name__ == "__main__":
    unittest.main()
