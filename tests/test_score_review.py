import copy
import importlib
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

import score_review as review


class ReviewParsingTests(unittest.TestCase):
    def setUp(self):
        self.baseline = np.array([0, -2, 1, 0, -1, -2], dtype=np.int32)
        self.object_ids = ["chair", "table"]
        self.maps = {"environment": np.array([4]), "unresolved": np.array([1, 5]),
                     "chair": np.array([0, 3]), "table": np.array([2])}
        self.parts = []
        self.sources = []
        offset = 0
        for index, (name, mapping) in enumerate(self.maps.items()):
            label = {"environment": -1, "unresolved": -2, "chair": 0, "table": 1}[name]
            digest = str(index + 1) * 64
            self.parts.append({"id": name, "owner_label": label, "triangles": len(mapping),
                               "sha256": digest, "native_source_face_offset": offset,
                               "source_indices": {"face": {"count": len(mapping)}}})
            self.sources.append({"id": name, "sha256": digest, "offset": offset, "faces": len(mapping)})
            offset += len(mapping)
        parameters = {"method": "b1", "reference_used": False, "mesh_changed": False}
        self.outcome = {"scene_id": "dev-test", "triangles": 6, "vertices": 3,
                        "surface_sha256": "5" * 64, "surface_manifest_sha256": "6" * 64,
                        "scene_sha256": "7" * 64,
                        "methods": [{"method": "b1", "status": "completed", "parameters": parameters,
                                     "face_owner_sha256": "8" * 64}]}
        self.bundle = {"schema": "spatial-scene-lab.surface.v1", "status": "pass", "units": "m", "up_axis": "Y",
                       "scene_id": "dev-test", "triangles": 6, "vertices": 3, "parts": self.parts,
                       "input_scene_sha256": "7" * 64, "partition": {"parameters": parameters},
                       "provenance": {"reference_used": False, "mesh_changed": False,
                                      "published_results_sha256": "a" * 64, "source_surface_sha256": "5" * 64,
                                      "source_surface_manifest_sha256": "6" * 64,
                                      "source_observation_sha256": "7" * 64,
                                      "face_owner": {"sha256": "8" * 64}}}
        # Native IDs [0,1,2,3,4,5] mean frozen faces [4,1,5,0,3,2].
        # Move native face 1 to chair, leaving native face 2 unresolved.
        self.identity = {"schema": "spatial-scene-lab.edit-identity.v1", "units": "m", "up_axis_glb": "Y",
                         "input_manifest_sha256": "b" * 64, "asset_sha256": "c" * 64,
                         "sources": list(reversed(self.sources)),
                         "objects": [{"id": "table", "source_face_ids": [5]},
                                     {"id": "chair", "source_face_ids": [4, 1, 3]},
                                     {"id": "unresolved", "source_face_ids": [2]},
                                     {"id": "environment", "source_face_ids": [0]}]}

    def restore(self):
        return review.restore_owners(self.bundle, self.identity, self.maps, self.baseline, self.object_ids)

    def bindings(self):
        review.validate_bindings(self.bundle, self.identity, self.outcome, "a" * 64, "b" * 64, "c" * 64)

    def test_reordered_native_ids_map_to_frozen_faces_and_keep_unresolved(self):
        self.bindings()
        np.testing.assert_array_equal(self.restore(), [0, 0, 1, 0, -1, -2])
        self.assertEqual(self.restore().dtype, np.int32)

    def test_duplicate_within_and_between_objects_is_rejected(self):
        for value in (5, 4):
            with self.subTest(value=value):
                self.identity["objects"][0]["source_face_ids"].append(value)
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    self.restore()
                self.identity["objects"][0]["source_face_ids"].pop()

    def test_missing_source_id_is_rejected(self):
        self.identity["objects"][0]["source_face_ids"] = []
        with self.assertRaisesRegex(ValueError, "missing"):
            self.restore()

    def test_invalid_source_ids_are_rejected(self):
        for value in (-1, 6, True, 1.5, "5"):
            with self.subTest(value=value):
                self.identity["objects"][0]["source_face_ids"] = [value]
                with self.assertRaisesRegex(ValueError, "source face ID"):
                    self.restore()

    def test_unknown_missing_and_duplicate_target_are_rejected(self):
        for value in ("other", "chair"):
            with self.subTest(value=value):
                self.identity["objects"][0]["id"] = value
                with self.assertRaisesRegex(ValueError, "target IDs"):
                    self.restore()

    def test_frozen_object_order_cannot_change(self):
        self.object_ids.reverse()
        with self.assertRaisesRegex(ValueError, "owner order"):
            self.restore()

    def test_source_mapping_cannot_be_substituted_or_reordered(self):
        for mapping in (np.array([3, 0]), np.array([0, 4]), np.array([0, 0]),
                        np.array([0]), np.array([0., 3.]), np.array([0, 6])):
            with self.subTest(mapping=mapping):
                self.maps["chair"] = mapping
                with self.assertRaisesRegex(ValueError, "source face mapping"):
                    self.restore()

    def test_source_hash_offset_count_and_identity_are_bound(self):
        original = copy.deepcopy(self.identity)
        for key, value in (("sha256", "d" * 64), ("offset", 0), ("faces", 2), ("id", "other")):
            with self.subTest(key=key):
                self.identity = copy.deepcopy(original)
                self.identity["sources"][0][key] = value
                with self.assertRaises(ValueError):
                    self.restore()

    def test_manifest_glb_and_frozen_source_hash_mismatches_are_rejected(self):
        for container, key in ((self.identity, "input_manifest_sha256"), (self.identity, "asset_sha256"),
                               (self.bundle["provenance"], "published_results_sha256"),
                               (self.bundle["provenance"], "source_surface_sha256"),
                               (self.bundle["provenance"], "source_surface_manifest_sha256"),
                               (self.bundle["provenance"], "source_observation_sha256"),
                               (self.bundle["provenance"]["face_owner"], "sha256")):
            with self.subTest(key=key):
                original = container[key]
                container[key] = "d" * 64
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    self.bindings()
                container[key] = original

    def test_wrong_scene_and_owner_label_are_rejected(self):
        self.bundle["scene_id"] = "dev-other"
        with self.assertRaisesRegex(ValueError, "scene identity"):
            self.bindings()
        self.parts[2]["owner_label"] = 1
        with self.assertRaisesRegex(ValueError, "owner label"):
            self.restore()

    def test_module_import_does_not_load_open3d(self):
        script = ("import sys; import score_review; "
                  "assert 'open3d' not in sys.modules; "
                  "assert 'reevaluate_development' not in sys.modules")
        subprocess.run([sys.executable, "-c", script], cwd=Path(review.__file__).parent, check=True)

    def test_rigid_edits_do_not_change_ownership_or_fixed_mesh_scoring(self):
        expected = self.restore()
        self.identity["journal"] = [{"kind": "rigid_edit", "id": "chair",
                                     "translation_blender_z_up_m": [100, -50, 3],
                                     "rotation_world_z_degrees": 120}]
        np.testing.assert_array_equal(self.restore(), expected)
        vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        triangles = np.tile([0, 1, 2], (6, 1))
        scene = {"vertices": vertices, "triangles": triangles,
                 "reference_vertices": vertices.copy(), "reference_triangles": triangles.copy(),
                 "reference_owner": self.baseline.copy(), "model_ids": self.object_ids,
                 "methods": [{"method": "b1", "owners": self.baseline,
                              "original": {"parameters": {"matching_tolerance_m": .08,
                                                           "samples_per_face": 7, "ambiguity_margin_m": .002}}}]}
        calls = []

        def evaluate(*args, **kwargs):
            calls.append((args, kwargs))
            return {"macro_target_area_iou": .5,
                    "targets": [{"object_id": name, "area_iou": .5, "precision": .5, "recall": .5}
                                for name in self.object_ids]}

        evaluator = importlib.import_module("evaluate_partition")
        with patch.object(evaluator, "evaluate_all_objects", side_effect=evaluate):
            scores = review.measure_review(scene, self.restore())
        self.assertEqual(len(calls), 2)
        for args, kwargs in calls:
            self.assertIs(args[0], vertices)
            self.assertIs(args[1], triangles)
            self.assertEqual(kwargs["matching_tolerance"], .08)
            self.assertEqual(kwargs["samples_per_face"], 7)
            self.assertEqual(kwargs["ambiguity_margin"], .002)
        np.testing.assert_array_equal(calls[0][0][2], self.baseline)
        np.testing.assert_array_equal(calls[1][0][2], expected)
        self.assertEqual(scores["changes"]["faces"], 1)
        self.assertEqual(scores["changes"]["area_m2"], .5)


if __name__ == "__main__":
    unittest.main()
