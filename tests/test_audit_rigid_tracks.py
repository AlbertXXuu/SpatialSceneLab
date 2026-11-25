"""Synthetic protocol checks; never reads capture images or operator records."""
from __future__ import annotations

import contextlib
import copy
import io
import itertools
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import audit_rigid_tracks as audit


FRAMES = [4, 12, 45, 46, 47, 48, 49, 50]
ANCHORS = [4, 12, 45, 50]


def fixture():
    catalogue = {"tracks": [{"id": f"t{i}", "region": f"r{i % 3}"} for i in range(6)]}
    annotations = {}
    for op in ("A", "B"):
        tracks = []
        for i, t in enumerate(catalogue["tracks"]):
            views = []
            for f in FRAMES:
                views.append({"frame": f, "visibility": "clear",
                              "xy": [800. + i * 50 - f, 600. + i * 25],
                              "radius_px": 2., "identity_basis": f"fixed junction {i}",
                              "ambiguity": "", "identity_confirmed": True})
            tracks.append({**t, "views": views})
        annotations[op] = {"operator_id": op, "operator_type": "agent", "score_blind": True,
                           "input_frames": FRAMES, "tracks": tracks}
    return catalogue, annotations


def view(data, track=0, frame=46):
    return next(v for v in data["tracks"][track]["views"] if v["frame"] == frame)


def run_synthetic(catalogue, annotations, degenerate=False, changed=False):
    """Exercise actual evaluation and output with a closed synthetic source map."""
    pairs = [list(p) for p in itertools.combinations(FRAMES, 2)]
    protocol = {"frames": FRAMES, "anchor_frames": ANCHORS, "modified_frames": [46, 47, 48, 49],
                "min_track_views": 3, "min_track_anchor_views": 1,
                "min_tracks_per_modified_frame": 6, "min_distinct_rigid_regions": 3,
                "min_tracks_pair_for_descriptive_support": 6, "min_regions_pair_for_descriptive_support": 2,
                "physical_group_mapping": {f"r{i}": f"group{i}" for i in range(3)},
                "uncertainty": {"draws": 16, "seed": 20261025}, "pairs": pairs,
                "primary_pairs": [p for p in pairs if (p[0] in ANCHORS) != (p[1] in ANCHORS)],
                "source_images": [], "camera_inputs": {m: {"path": m} for m in ("original", "normal", "balanced")}}
    k = np.array([[1000., 0, 960.], [0, 1000., 720.], [0, 0, 1.]])
    cameras, sources, manifest = {}, {}, {"files": []}
    receipt = {"files": {name: {"path": name} for name in ("protocol", "catalogue", "A", "B", "source_manifest")}}
    for f in FRAMES:
        c = np.eye(4)
        c[0, 3] = f * .03
        c[1, 3] = (f % 3) * .02
        cameras[f] = {"camera_to_world_cv": c.tolist()}
        raw = c.copy()
        raw[:3, 1:3] *= -1
        filename = f"frame_{f:05}.json"
        sources[filename] = {"intrinsics": k.tolist(), "cameraPoseARFrame": raw.tolist()}
        receipt["files"][f"capture_{f}"] = {"path": filename, "sha256": str(f)}
        manifest["files"].append({"path": filename, "sha256": str(f)})
    indexed = {m: copy.deepcopy(cameras) for m in ("baseline", "normal", "balanced")}
    if changed:
        indexed["balanced"][46]["camera_to_world_cv"][1][3] += .03
    if degenerate:
        indexed["balanced"][46] = copy.deepcopy(cameras[4])
    sources.update({"sealed": receipt, "protocol": protocol, "catalogue": catalogue,
                    "A": annotations["A"], "B": annotations["B"], "source_manifest": manifest,
                    "original": {}, "normal": {}, "balanced": {}})
    outputs = {}
    def save(path, data, **kwargs):
        outputs[Path(path).name] = copy.deepcopy(data)
    # This allowlist raises on any accidental attempt to read real task data.
    with patch.object(audit, "read", side_effect=lambda p: copy.deepcopy(sources[str(p)])), \
            patch.object(audit, "verify_record"), \
            patch.object(audit, "validate_scenes", return_value=indexed), \
            patch.object(Path, "mkdir"), patch.object(audit, "write_json", side_effect=save), \
            patch.object(audit.np, "save", side_effect=save), contextlib.redirect_stdout(io.StringIO()):
        audit.evaluate(SimpleNamespace(seal="sealed", output="synthetic-output"))
    return outputs, cameras, indexed, k


class AnnotationContractTests(unittest.TestCase):
    def test_clear_needs_explicit_nonempty_physical_evidence_and_B_identity(self):
        cat, ann = fixture()
        view(ann["A"])["identity_basis"] = " \n "
        with self.assertRaisesRegex(ValueError, "physical identity"):
            audit.annotation_index(ann["A"], cat, FRAMES, "A")
        del view(ann["B"])["identity_confirmed"]
        with self.assertRaisesRegex(ValueError, "explicitly confirm"):
            audit.annotation_index(ann["B"], cat, FRAMES, "B")

    def test_all_proposals_and_unique_views_are_required(self):
        cat, ann = fixture()
        ann["A"]["tracks"].pop()
        with self.assertRaisesRegex(ValueError, "every proposed"):
            audit.annotation_index(ann["A"], cat, FRAMES, "A")
        cat, ann = fixture()
        ann["A"]["tracks"][0]["views"][-1] = copy.deepcopy(ann["A"]["tracks"][0]["views"][0])
        with self.assertRaisesRegex(ValueError, "unique frame"):
            audit.annotation_index(ann["A"], cat, FRAMES, "A")

    def test_identity_visibility_and_localization_are_separate_consensus_gates(self):
        _, ann = fixture()
        a, b = view(ann["A"]), view(ann["B"])
        for key, value, reason in (("identity_confirmed", False, "B_identity_unconfirmed"),
                                    ("visibility", "uncertain", "B_not_clear"),
                                    ("xy", [a["xy"][0] + 5, a["xy"][1]], "localization_disagreement")):
            changed = copy.deepcopy(b)
            changed[key] = value
            decision = audit.consensus_view(a, changed)
            self.assertFalse(decision["accepted"])
            self.assertIn(reason, decision["reasons"])
        b["xy"][0] += 4  # Exactly touching disks are permitted by the frozen <= rule.
        accepted = audit.consensus_view(a, b)
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["enclosing_radius_px"], 4)


class JointSamplingTests(unittest.TestCase):
    def test_seeded_uniform_area_draws_stay_in_disks_and_allow_empty_support(self):
        centres, radii = np.array([[10., 20.], [80., 50.]]), np.array([2., 7.])
        first = audit.joint_draws(centres, radii, 10000, 81)
        np.testing.assert_array_equal(first, audit.joint_draws(centres, radii, 10000, 81))
        distance = np.linalg.norm(first - centres, axis=-1)
        self.assertTrue(np.all(distance <= radii + 1e-12))
        # For an area-uniform disk E[(r/R)^2] = 1/2; a radius-uniform bug gives 1/3.
        self.assertAlmostEqual(float(np.mean((distance / radii) ** 2)), .5, delta=.015)
        self.assertEqual(audit.joint_draws([], [], 256, 81).shape, (256, 0, 2))


class SyntheticEvaluationTests(unittest.TestCase):
    def test_three_view_anchor_rule_and_consensus_support_are_shared(self):
        cat, ann = fixture()
        # t0: three modified views, no anchor; t1: an anchor but only two views.
        for op in ann.values():
            for t, keep in ((0, {46, 47, 48}), (1, {4, 46})):
                for v in op["tracks"][t]["views"]:
                    if v["frame"] not in keep:
                        v.update(visibility="not_identified", xy=None, radius_px=None)
        view(ann["B"], 2, 46)["identity_confirmed"] = False
        outputs, *_ = run_synthetic(cat, ann)
        result = outputs["results.json"]
        self.assertEqual(result["qualified_tracks"], ["t2", "t3", "t4", "t5"])
        self.assertEqual(result["verdict"], "INCONCLUSIVE")
        pair = next(p for p in result["pairs"] if p["pair"] == [4, 46])
        self.assertEqual(pair["track_ids"], ["t3", "t4", "t5"])
        self.assertEqual({s["count"] for group in pair["score_sets"].values() for s in group.values()}, {3})
        self.assertEqual(len(result["tracks"]), 6)
        self.assertEqual(len(result["pairs"]), 28)

    def test_zero_support_remains_null_not_zero_and_all_pairs_remain(self):
        cat, ann = fixture()
        for op in ann.values():
            for t in op["tracks"]:
                for v in t["views"]:
                    v.update(visibility="not_identified", xy=None, radius_px=None)
        outputs, *_ = run_synthetic(cat, ann)
        result = outputs["results.json"]
        self.assertEqual(result["verdict"], "INCONCLUSIVE")
        self.assertEqual(result["qualified_tracks"], [])
        self.assertEqual(len(result["pairs"]), 28)
        for pair in result["pairs"]:
            self.assertEqual(pair["count"], 0)
            for group in pair["score_sets"].values():
                for score in group.values():
                    self.assertIsNone(score["median_px"])
            self.assertIsNone(pair["deltas"]["balanced"]["draw_min_px"])

    def test_supported_degenerate_candidate_cannot_yield_success(self):
        outputs, *_ = run_synthetic(*fixture(), degenerate=True)
        result = outputs["results.json"]
        self.assertEqual(result["verdict"], "INCONCLUSIVE")
        self.assertIn("supported_primary_pair_not_numerically_comparable", result["stop_reasons"])
        pair = next(p for p in result["pairs"] if p["pair"] == [4, 46])
        self.assertTrue(pair["support_threshold_met"])
        self.assertFalse(pair["numerically_comparable"])
        self.assertIsNone(pair["score_sets"]["A"]["balanced"]["median_px"])

    def test_joint_observations_are_reused_across_pairs_and_anchor_controls(self):
        outputs, cameras, indexed, k = run_synthetic(*fixture(), changed=True)
        result, samples = outputs["results.json"], outputs["joint-draws.npy"]
        keys = {tuple(key): i for i, key in enumerate(outputs["joint-draw-index.json"])}
        controls = [p for p in result["pairs"] if p["kind"] == "anchor_control"]
        self.assertEqual(len(controls), 6)
        self.assertTrue(result["unchanged_anchor_control_pass"])
        for pair_id in ([4, 46], [12, 46]):
            pi = next(i for i, p in enumerate(result["pairs"]) if p["pair"] == pair_id)
            pair = result["pairs"][pi]
            for method_index, method in enumerate(("original", "normal", "balanced")):
                selected = cameras if method == "original" else indexed[method]
                a, b = pair_id
                f = audit.fundamental_from_cameras(selected[a]["camera_to_world_cv"], selected[b]["camera_to_world_cv"], k, k)
                ia, ib = [keys[t, a] for t in pair["track_ids"]], [keys[t, b] for t in pair["track_ids"]]
                expected = [np.median(audit.sampson_px(f, draw[ia], draw[ib])) for draw in samples]
                np.testing.assert_array_equal(outputs["draw-median-scores.npy"][pi, :, method_index], expected)
        for pair in controls:
            for op in ("A", "B", "midpoint"):
                self.assertEqual(pair["score_sets"][op]["original"], pair["score_sets"][op]["balanced"])


if __name__ == "__main__":
    unittest.main()
