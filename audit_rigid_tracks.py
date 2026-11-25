"""Bounded agent-labelled rigid-track audit; no fitting or candidate adoption. MIT.

Seal the catalogue, protocol and both independent annotations before evaluation.
Only NumPy and the existing rgb_pose_check numeric helpers are needed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re

import numpy as np

from rgb_pose_check import (digest, fundamental_from_cameras, grid_coverage,
                            intrinsic, sampson_px, stats, validate_scenes, write_json)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def frame_id(value):
    if type(value) is int:
        return value
    match = re.fullmatch(r"frame_(\d+)\.jpg", str(value))
    if match:
        return int(match[1])
    raise ValueError(f"invalid frame identity: {value!r}")


def annotation_index(data, catalogue, frames, operator):
    if (data.get("operator_id") != operator or data.get("operator_type") != "agent"
            or data.get("score_blind") is not True):
        raise ValueError("agent identity and score-blind declaration required")
    supplied = [frame_id(v) for v in data["input_frames"]]
    if sorted(supplied) != sorted(frames):
        raise ValueError("annotation input frame set differs")
    expected = {t["id"]: t for t in catalogue["tracks"]}
    tracks = data["tracks"]
    if len(tracks) != len(expected) or {t["id"] for t in tracks} != set(expected):
        raise ValueError("retain exactly every proposed track")
    indexed = {}
    labels = {"clear", "uncertain", "occluded", "out_of_frame", "not_identified"}
    for track in tracks:
        tid = track["id"]
        if track["region"] != expected[tid]["region"]:
            raise ValueError("annotation region differs from catalogue")
        views = track["views"]
        if len(views) != len(frames) or {frame_id(v["frame"]) for v in views} != set(frames):
            raise ValueError("every track must retain every unique frame")
        indexed[tid] = {}
        for v in views:
            fid = frame_id(v["frame"])
            if v["visibility"] not in labels:
                raise ValueError("unknown visibility")
            if not isinstance(v["identity_basis"], str) or not isinstance(v["ambiguity"], str):
                raise ValueError("identity and ambiguity descriptions required")
            p, radius = v["xy"], v["radius_px"]
            if (p is None) != (radius is None):
                raise ValueError("position and radius must both be present or absent")
            if p is not None:
                xy = np.asarray(p, dtype=float)
                if (xy.shape != (2,) or not np.isfinite(xy).all()
                        or np.any(xy < 0) or np.any(xy >= [1920, 1440])
                        or isinstance(radius, bool) or not np.isfinite(radius) or radius <= 0):
                    raise ValueError("finite in-image coordinates and positive radius required")
            if v["visibility"] == "clear" and p is None:
                raise ValueError("clear observation needs coordinates")
            if v["visibility"] == "clear" and not v["identity_basis"].strip():
                raise ValueError("clear observation needs physical identity evidence")
            if operator == "B" and ("identity_confirmed" not in v or
                    v["identity_confirmed"] is not None and type(v["identity_confirmed"]) is not bool):
                raise ValueError("B must explicitly confirm, deny or leave identity unknown")
            indexed[tid][fid] = v
    return indexed


def consensus_view(a, b):
    reasons = []
    if a["visibility"] != "clear":
        reasons.append("A_not_clear")
    if b["visibility"] != "clear":
        reasons.append("B_not_clear")
    if b["identity_confirmed"] is not True:
        reasons.append("B_identity_unconfirmed")
    distance = None
    if a["xy"] is not None and b["xy"] is not None:
        distance = float(np.linalg.norm(np.array(a["xy"]) - b["xy"]))
        if distance > a["radius_px"] + b["radius_px"]:
            reasons.append("localization_disagreement")
    else:
        reasons.append("missing_position")
    accepted = not reasons
    return {"accepted": accepted, "reasons": reasons, "distance_px": distance,
            "midpoint": ((np.array(a["xy"]) + b["xy"]) / 2).tolist() if accepted else None,
            "enclosing_radius_px": max(a["radius_px"], b["radius_px"]) + distance / 2 if accepted else None}


def joint_draws(centres, radii, draws, seed):
    rng = np.random.default_rng(seed)
    centres = np.asarray(centres, dtype=float).reshape(-1, 2)
    radii = np.asarray(radii, dtype=float)
    angle = rng.uniform(0, 2 * np.pi, (draws, len(centres)))
    radius = np.sqrt(rng.uniform(size=angle.shape)) * radii
    offsets = np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=-1)
    return centres[None, :, :] + offsets


def record(path):
    p = Path(path).resolve()
    return {"path": str(p), "sha256": digest(p)}


def verify_record(item):
    if digest(item["path"]) != item["sha256"]:
        raise ValueError(f"sealed input changed: {item['path']}")


def seal(args):
    p, cat = read(args.protocol), read(args.catalogue)
    if not 0 < len(cat["tracks"]) <= p["max_candidates"]:
        raise ValueError("candidate budget exceeded or empty catalogue")
    if len({t["id"] for t in cat["tracks"]}) != len(cat["tracks"]):
        raise ValueError("duplicate catalogue ID")
    for label, path in (("A", args.a), ("B", args.b)):
        annotation_index(read(path), cat, p["frames"], label)
    frozen = read(Path(args.protocol).parent / "catalogue-freeze.json")
    if digest(args.catalogue) != frozen["sha256"]:
        raise ValueError("catalogue changed after its pre-annotation freeze")
    for x in p["source_images"] + list(p["camera_inputs"].values()):
        verify_record(x)
    root = Path(__file__).parent
    files = {"protocol": record(args.protocol), "catalogue": record(args.catalogue),
             "A": record(args.a), "B": record(args.b), "script": record(__file__),
             "numeric_helpers": record(root / "rgb_pose_check.py"),
             "source_manifest": record(root / "docs/tea-room-source-manifest.json")}
    for name, filename, key in (("contract", "ANNOTATION_CONTRACT.md", "annotation_contract_sha256"),
                                ("addendum", "METHOD_ADDENDUM.md", "method_addendum_sha256")):
        files[name] = record(Path(args.protocol).parent / filename)
        if files[name]["sha256"] != p[key]:
            raise ValueError("pre-scoring method document changed")
    files["catalogue_freeze"] = record(Path(args.protocol).parent / "catalogue-freeze.json")
    for item in p["source_images"]:
        files[f"capture_{item['frame']}"] = record(Path(item["path"]).with_suffix(".json"))
    receipt = {"schema": "alvenx.rigid-track-seal.v1", "sealed_at_utc": datetime.now(timezone.utc).isoformat(), "files": files,
               "scope": "Input and evaluator freeze before a single fixed-candidate comparison"}
    write_json(args.output, receipt)
    print(json.dumps({"sealed": str(args.output), "tracks": len(cat["tracks"]), "scoring_run": False}))


def finite_delta(candidate, original):
    if candidate is None or original is None:
        return None
    return float(candidate - original)


def evaluate(args):
    receipt = read(args.seal)
    for item in receipt["files"].values():
        verify_record(item)
    files = receipt["files"]
    p, cat = read(files["protocol"]["path"]), read(files["catalogue"]["path"])
    for item in p["source_images"] + list(p["camera_inputs"].values()):
        verify_record(item)
    annotations = {op: annotation_index(read(files[op]["path"]), cat, p["frames"], op) for op in ("A", "B")}
    scenes = {("baseline" if name == "original" else name): read(v["path"]) for name, v in p["camera_inputs"].items()}
    indexed = validate_scenes(scenes, read(files["source_manifest"]["path"]))
    indexed["original"] = indexed.pop("baseline")
    intrinsics = {}
    manifest = {Path(v["path"]).name: v["sha256"] for v in read(files["source_manifest"]["path"])["files"]}
    for f in p["frames"]:
        raw = read(files[f"capture_{f}"]["path"])
        if manifest.get(Path(files[f"capture_{f}"]["path"]).name) != files[f"capture_{f}"]["sha256"]:
            raise ValueError("intrinsics source differs from pinned capture")
        intrinsics[f] = intrinsic(np.asarray(raw["intrinsics"]).reshape(3, 3))
        camera = np.array(raw["cameraPoseARFrame"]).reshape(4, 4)
        camera[:3, 1:3] *= -1
        if not np.array_equal(camera, indexed["original"][f]["camera_to_world_cv"]):
            raise ValueError("baseline camera differs from original capture")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "seal.json", receipt)
    tracks, observations, qualified = [], {}, []
    for t in cat["tracks"]:
        tid = t["id"]
        views = {f: consensus_view(annotations["A"][tid][f], annotations["B"][tid][f]) for f in p["frames"]}
        clear = [f for f, v in views.items() if v["accepted"]]
        anchors = sorted(set(clear) & set(p["anchor_frames"]))
        qualifies = len(clear) >= p["min_track_views"] and len(anchors) >= p["min_track_anchor_views"]
        if qualifies:
            qualified.append(tid)
        observations[tid] = views
        tracks.append({"id": tid, "region": t["region"], "physical_group": p["physical_group_mapping"][t["region"]],
                       "qualifies": qualifies, "clear_frames": clear, "anchor_frames": anchors, "views": views})
    by_id = {t["id"]: t for t in tracks}
    coverage = []
    for f in p["frames"]:
        ids = [tid for tid in qualified if observations[tid][f]["accepted"]]
        points = np.array([observations[t][f]["midpoint"] for t in ids]).reshape(-1, 2)
        coverage.append({"frame": f, "modified": f in p["modified_frames"], "qualifying_tracks": ids,
                         "count": len(ids), "regions": sorted({by_id[t]["region"] for t in ids}),
                         "physical_groups": sorted({by_id[t]["physical_group"] for t in ids}),
                         "anchor_connections": {str(a): [t for t in ids if observations[t][a]["accepted"]] for a in p["anchor_frames"]},
                         "grid": grid_coverage(points, 1920, 1440),
                         "individually_clear_counts": {op: sum(v[f]["visibility"] == "clear" for v in annotations[op].values()) for op in annotations}})
    keys = [(tid, f) for tid in qualified for f in p["frames"] if observations[tid][f]["accepted"]]
    key_index = {key: i for i, key in enumerate(keys)}
    centres = [observations[t][f]["midpoint"] for t, f in keys]
    radii = [observations[t][f]["enclosing_radius_px"] for t, f in keys]
    sampled = joint_draws(centres, radii, p["uncertainty"]["draws"], p["uncertainty"]["seed"])
    np.save(out / "joint-draws.npy", sampled, allow_pickle=False)
    write_json(out / "joint-draw-index.json", keys)
    pair_results, all_samples = [], []
    methods = ("original", "normal", "balanced")
    for a, b in p["pairs"]:
        ids = [t for t in qualified if observations[t][a]["accepted"] and observations[t][b]["accepted"]]
        regions = sorted({by_id[t]["region"] for t in ids})
        fs, errors = {}, {}
        for method in methods:
            try:
                fs[method] = fundamental_from_cameras(indexed[method][a]["camera_to_world_cv"], indexed[method][b]["camera_to_world_cv"], intrinsics[a], intrinsics[b])
            except ValueError as e:
                errors[method] = str(e)
        def score(x, y, method):
            values = sampson_px(fs[method], x, y) if method in fs else np.full(len(ids), np.inf)
            return {**stats(values), "values_px": [float(v) if np.isfinite(v) else None for v in values], "error": errors.get(method)}
        sets = {}
        for op in ("A", "B", "midpoint"):
            xy = lambda f: np.array([observations[t][f]["midpoint"] if op == "midpoint" else annotations[op][t][f]["xy"] for t in ids]).reshape(-1, 2)
            sets[op] = {m: score(xy(a), xy(b), m) for m in methods}
        draw_scores = np.full((p["uncertainty"]["draws"], 3), np.nan)
        if ids:
            ia, ib = [key_index[t, a] for t in ids], [key_index[t, b] for t in ids]
            for n, draw in enumerate(sampled):
                for j, method in enumerate(methods):
                    if method in fs:
                        draw_scores[n, j] = np.median(sampson_px(fs[method], draw[ia], draw[ib]))
        all_samples.append(draw_scores)
        deltas = {}
        for j, method in enumerate(methods[1:], 1):
            observed = {op: finite_delta(sets[op][method]["median_px"], sets[op]["original"]["median_px"]) for op in sets}
            draws = draw_scores[:, j] - draw_scores[:, 0]
            finite = draws[np.isfinite(draws)]
            values = [v for v in observed.values() if v is not None] + finite.tolist()
            deltas[method] = {"operator_delta_px": observed, "finite_draws": len(finite),
                              "draw_min_px": float(finite.min()) if len(finite) else None,
                              "draw_max_px": float(finite.max()) if len(finite) else None,
                              "sign_reversal_observed": min(values) < 0 < max(values) if values else None,
                              "all_observed_lower": all(v < 0 for v in values) if len(finite) == len(draws) and all(v is not None for v in observed.values()) else None}
        kind = "primary" if [a, b] in p["primary_pairs"] else "anchor_control" if a in p["anchor_frames"] and b in p["anchor_frames"] else "modified_context"
        comparable = bool(ids) and all(sets[op][m]["finite_count"] == len(ids) and sets[op][m]["median_px"] is not None for op in sets for m in methods)
        pair_results.append({"pair": [a, b], "kind": kind, "track_ids": ids, "count": len(ids),
                             "regions": regions, "physical_groups": sorted({by_id[t]["physical_group"] for t in ids}),
                             "support_threshold_met": len(ids) >= p["min_tracks_pair_for_descriptive_support"] and len(regions) >= p["min_regions_pair_for_descriptive_support"],
                             "numerically_comparable": comparable, "score_sets": sets, "deltas": deltas})
    np.save(out / "draw-median-scores.npy", np.asarray(all_samples), allow_pickle=False)
    failures = []
    if any(v["count"] < p["min_tracks_per_modified_frame"] for v in coverage if v["modified"]):
        failures.append("modified_frame_anchor_connected_track_coverage")
    if len({by_id[t]["region"] for t in qualified}) < p["min_distinct_rigid_regions"]:
        failures.append("overall_rigid_region_coverage")
    if any(len(v["physical_groups"]) < 2 for v in coverage if v["modified"]):
        failures.append("single_structure_or_missing_support_on_modified_frame")
    if any(d["sign_reversal_observed"] for pair in pair_results if pair["kind"] == "primary" for d in pair["deltas"].values()):
        failures.append("primary_pair_delta_sign_reversal")
    if any(pair["support_threshold_met"] and not pair["numerically_comparable"] for pair in pair_results if pair["kind"] == "primary"):
        failures.append("supported_primary_pair_not_numerically_comparable")
    controls = [pair for pair in pair_results if pair["kind"] == "anchor_control"]
    control_ok = all(pair["score_sets"][op][m] == pair["score_sets"][op]["original"] for pair in controls for op in ("A", "B", "midpoint") for m in methods[1:])
    if not control_ok:
        raise ValueError("unchanged camera control failed")
    result = {"schema": "alvenx.rigid-track-feasibility.result.v1", "verdict": "INCONCLUSIVE" if failures else "LOCAL_IMAGE_SUPPORT_ONLY",
              "stop_reasons": failures, "proposed_tracks": len(tracks), "qualified_tracks": qualified,
              "tracks": tracks, "coverage": coverage, "pairs": pair_results,
              "unchanged_anchor_control_pass": control_ok,
              "sensitivity_axes": ["protocol pair order", "draw", list(methods)],
              "versions": {"python": platform.python_version(), "numpy": np.__version__},
              "limits": ["Agent annotations, shared identity catalogue; not independent discovery or physical ground truth.",
                         "Named image regions do not establish distinct 3D planes.",
                         "Finite disk perturbations are sensitivity diagnostics, not confidence or worst-case bounds.",
                         "Sampson residual is local image geometry consistency, not physical scale or reconstruction accuracy.",
                         "No fitting, adoption, new capture, training or publication."]}
    for item in list(files.values()) + p["source_images"] + list(p["camera_inputs"].values()):
        verify_record(item)
    write_json(out / "results.json", result)
    print(json.dumps({"verdict": result["verdict"], "proposed": len(tracks), "qualified": len(qualified),
                      "modified_coverage": {v["frame"]: v["count"] for v in coverage if v["modified"]}, "stop_reasons": failures}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    first = sub.add_parser("seal")
    for key in ("protocol", "catalogue", "a", "b", "output"):
        first.add_argument("--" + key, type=Path, required=True)
    second = sub.add_parser("evaluate")
    for key in ("seal", "output"):
        second.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    (seal if args.action == "seal" else evaluate)(args)


if __name__ == "__main__":
    main()
