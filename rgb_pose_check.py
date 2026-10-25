"""Frozen-camera RGB epipolar cross-check; no depth, pose fitting or adoption. MIT.

Optional OpenCV is imported only by the image experiment. Numeric functions use
NumPy alone. See examples/real-tearoom-local/TABLE0-RGB.md for scope and replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

PAIRS = tuple((a, b) for a in (4, 12) for b in (46, 47, 48, 49)) + (
    (45, 46), (46, 47), (47, 48), (48, 49), (49, 50), (44, 45), (50, 51))
PARAMETERS = {"sift_features": 4000, "ratio": .75, "duplicate_cell_px": 8,
              "score_threshold_px": 2., "minimum_matches": 30,
              "minimum_occupied_cells_each_image": 6, "coverage_grid": [4, 3],
              "homography_threshold_px": 3., "plane_fraction": .8,
              "seed": 20261025, "local_y_perturbation_degrees": 2.}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def rigid_camera(value):
    c = np.asarray(value, dtype=float)
    if c.shape != (4, 4) or not np.isfinite(c).all():
        raise ValueError("finite 4x4 camera-to-world required")
    if not np.allclose(c[3], [0, 0, 0, 1], rtol=0, atol=1e-6):
        raise ValueError("homogeneous camera row required")
    r = c[:3, :3]
    if (not np.allclose(r.T @ r, np.eye(3), rtol=0, atol=2e-5)
            or abs(np.linalg.det(r)-1) > 2e-5):
        raise ValueError("proper rigid camera required")
    return c


def intrinsic(value):
    k = np.asarray(value, dtype=float)
    if (k.shape != (3, 3) or not np.isfinite(k).all()
            or not np.allclose(k[2], [0, 0, 1], rtol=0, atol=1e-12)
            or k[0, 0] <= 0 or k[1, 1] <= 0
            or abs(np.linalg.det(k)) < 1e-12):
        raise ValueError("finite non-singular pinhole intrinsics required")
    return k


def fundamental_from_cameras(camera_a, camera_b, K_a, K_b):
    a, b = rigid_camera(camera_a), rigid_camera(camera_b)
    ka, kb = intrinsic(K_a), intrinsic(K_b)
    # Captured homogeneous rows contain float32 roundoff (e.g. 0.99999994).
    # They denote rigid cameras, not projective transforms. Use their validated
    # R/centre blocks so that homogeneous-row roundoff cannot invent scale signal.
    rotation = np.linalg.solve(b[:3, :3], a[:3, :3])
    t = np.linalg.solve(b[:3, :3], a[:3, 3] - b[:3, 3])
    if np.linalg.norm(t) < 1e-10:
        raise ValueError("zero baseline: epipolar translation is unobservable")
    x, y, z = t
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    f = np.linalg.inv(kb).T @ skew @ rotation @ np.linalg.inv(ka)
    return f / np.linalg.norm(f)


def sampson_px(F, points_a, points_b):
    f = np.asarray(F, dtype=float)
    a, b = np.asarray(points_a, dtype=float), np.asarray(points_b, dtype=float)
    if (f.shape != (3, 3) or not np.isfinite(f).all() or np.linalg.norm(f) == 0
            or a.ndim != 2 or a.shape[1] != 2 or b.shape != a.shape
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("finite F and equally sized Nx2 pixel arrays required")
    f = f / np.linalg.norm(f)
    ah, bh = np.column_stack((a, np.ones(len(a)))), np.column_stack((b, np.ones(len(b))))
    fa, ftb = ah @ f.T, bh @ f
    numerator = np.abs(np.sum(bh * fa, axis=1))
    denom = np.sum(fa[:, :2]**2 + ftb[:, :2]**2, axis=1)
    return np.divide(numerator, np.sqrt(denom), out=np.full(len(a), np.inf), where=denom > 0)


def grid_coverage(points, width, height):
    p = np.asarray(points, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 2 or width <= 0 or height <= 0
            or not np.isfinite(p).all() or np.any(p < 0)
            or np.any(p[:, 0] >= width) or np.any(p[:, 1] >= height)):
        raise ValueError("finite in-image Nx2 pixels and positive image size required")
    cells = np.floor(p / [width, height] * [4, 3]).astype(int)
    return {"occupied": len(np.unique(cells, axis=0)), "total": 12}


def stats(values):
    v = np.asarray(values)
    if not len(v):
        return {"count": 0, "finite_count": 0, "median_px": None, "p95_px": None, "fraction_le_2px": None}
    def finite(x):
        return float(x) if np.isfinite(x) else None
    # Keep invalid/degenerate correspondences in the denominator.
    with np.errstate(invalid="ignore"):
        return {"count": len(v), "finite_count": int(np.isfinite(v).sum()),
                "median_px": finite(np.median(v)), "p95_px": finite(np.quantile(v, .95)),
                "fraction_le_2px": float(np.mean(v <= PARAMETERS["score_threshold_px"]))}


def validate_scenes(scenes, manifest):
    """Reject a mislabeled fallback, swapped method or unrelated capture."""
    baseline = scenes["baseline"]
    if baseline.get("schema") != "spatial-scene-lab.observations.v1":
        raise ValueError("original observation scene required")
    expected = {Path(f["path"]).name: f["sha256"] for f in manifest["files"]}
    files = baseline.get("input_files", {})
    if not files or any(expected.get(name) != sha for name, sha in files.items()):
        raise ValueError("scene source fingerprints differ from the pinned capture")
    indexed = {}
    for name, scene in scenes.items():
        if scene.get("source") != baseline.get("source") or scene.get("input_files") != files:
            raise ValueError("all scenes must identify the same captured input files")
        entries = scene.get("frames", [])
        indexed[name] = {f["frame_index"]: f for f in entries}
        if len(indexed[name]) != len(entries) or set(indexed[name]) != set(range(62)):
            raise ValueError("each frozen scene must contain 62 unique captured frame IDs")
        for frame in entries:
            rigid_camera(frame["camera_to_world_cv"])
        if name == "baseline":
            continue
        control = scene.get("alignment_control", {})
        if (scene.get("schema") != "alvenx.real-table-alignment-candidate.v1"
                or control.get("disposition") != "diagnostic_candidate"
                or control.get("method") != {"balanced": "balanced_pose", "normal": "normal_translation"}[name]
                or control.get("passed") is not False or control.get("is_camera_pose_estimate") is not False):
            raise ValueError("frozen rejected diagnostic candidate with correct method required")
        changed = []
        for i, frame in indexed[name].items():
            original = indexed["baseline"][i]
            if any(frame[k] != original[k] for k in ("intrinsics_depth", "color_size", "depth_size", "source")):
                raise ValueError("candidate changed capture intrinsics or frame identity")
            if not np.array_equal(np.array(frame["camera_to_world_cv"])[:3], np.array(original["camera_to_world_cv"])[:3]):
                changed.append(i)
        if sorted(changed) != [46, 47, 48, 49]:
            raise ValueError("frozen candidate must change only camera R/centre blocks 46-49")
    return indexed


def score_camera(a, b, ka, kb, p, q):
    try:
        values = sampson_px(fundamental_from_cameras(a, b, ka, kb), p, q)
        return values, {**stats(values), "error": None}
    except ValueError as exc:
        values = np.full(len(p), np.inf)
        return values, {**stats(values), "error": str(exc)}


def match_features(cv, first, second):
    pa, da = first
    pb, db = second
    if da is None or db is None or min(len(da), len(db)) < 2:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0), 0
    matcher = cv.BFMatcher(cv.NORM_L2)
    def ratio_map(left, right):
        return {m.queryIdx: (m.trainIdx, m.distance) for m, n in matcher.knnMatch(left, right, k=2)
                if m.distance < PARAMETERS["ratio"] * n.distance}
    ab, ba = ratio_map(da, db), ratio_map(db, da)
    accepted = [(i, j, dist) for i, (j, dist) in ab.items() if j in ba and ba[j][0] == i]
    accepted.sort(key=lambda v: (v[2], v[0], v[1]))
    seen, selected = set(), []
    for i, j, dist in accepted:
        key = tuple(np.floor(np.r_[pa[i], pb[j]] / PARAMETERS["duplicate_cell_px"]).astype(int))
        if key not in seen:
            seen.add(key)
            selected.append((i, j, dist))
    return (np.array([pa[i] for i, _, _ in selected]).reshape(-1, 2),
            np.array([pb[j] for _, j, _ in selected]).reshape(-1, 2),
            np.array([d for _, _, d in selected]), len(accepted))


def montage(images, a, b, path):
    from PIL import Image, ImageDraw
    size = (768, 576)
    canvas = Image.new("RGB", (1536, 616), "#eef3ff")
    for offset, image in zip((0, 768), images):
        canvas.paste(Image.fromarray(image).resize(size), (offset, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), "Descriptor-only matches; selected without camera scores. IDs index the frozen match arrays.", fill="#102040")
    order = np.lexsort((a[:, 0], a[:, 1])) if len(a) else []
    ids = [int(order[i]) for i in np.linspace(0, len(order)-1, min(8, len(order)), dtype=int)]
    for n, i in enumerate(ids):
        color = ("#ee15bf", "#0072ff", "#f09100", "#7f36ff")[n % 4]
        p = (a[i, 0]*.4, a[i, 1]*.4+40)
        q = (b[i, 0]*.4+768, b[i, 1]*.4+40)
        draw.line([p, q], fill=color, width=2)
        for x, y in (p, q):
            draw.ellipse((x-6, y-6, x+6, y+6), outline=color, width=2)
            draw.text((x+7, y+3), str(i), fill=color, stroke_width=1, stroke_fill="white")
    canvas.save(path)
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("input", "baseline", "balanced", "normal", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {k: getattr(args, k).resolve() for k in ("input", "baseline", "balanced", "normal", "output")}
    out = paths["output"]
    for key, p in paths.items():
        if key != "output" and (out == p or out in p.parents or (key == "input" and p in out.parents)):
            raise ValueError("output must be separate from input trees")
    if out.exists():
        raise ValueError("output must be new; frozen receipts are never overwritten")
    import cv2 as cv
    import PIL
    from PIL import Image
    from reproduce_real_tearoom import verify_source
    cv.setNumThreads(1)
    cv.ocl.setUseOpenCL(False)
    cv.setRNGSeed(PARAMETERS["seed"])
    root = Path(__file__).resolve().parent
    manifest = root / "docs/tea-room-source-manifest.json"
    source_receipt = verify_source(paths["input"], manifest)
    scenes = {name: json.loads(paths[name].read_text(encoding="utf-8")) for name in ("baseline", "balanced", "normal")}
    frames = validate_scenes(scenes, json.loads(manifest.read_text(encoding="utf-8")))
    ids = sorted(set(i for pair in PAIRS for i in pair))
    records, cams, rgb_paths = {}, {}, {}
    for i in ids:
        records[i] = json.loads((paths["input"] / f"frame_{i:05}.json").read_text())
        intrinsic(np.asarray(records[i]["intrinsics"]).reshape(3, 3))
        rgb_paths[i] = paths["input"] / f"frame_{i:05}.jpg"
        cams[i] = {name: rigid_camera(frames[name][i]["camera_to_world_cv"]) for name in frames}
        raw = np.array(records[i]["cameraPoseARFrame"]).reshape(4, 4)
        raw[:3, 1:3] *= -1
        if not np.array_equal(raw, cams[i]["baseline"]):
            raise ValueError("baseline camera differs from original capture")
    hashes = {name: digest(paths[name]) for name in scenes}
    hashes.update({f"rgb_{i}": digest(rgb_paths[i]) for i in ids})
    code_hash = digest(__file__)
    protocol = {"schema": "alvenx.table0.rgb-cross-check.v1", "pairs": PAIRS, "parameters": PARAMETERS,
                "source": source_receipt, "input_hashes": hashes, "script_sha256": code_hash,
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv.__version__, "pillow": PIL.__version__},
                "selection": "RootSIFT mutual bidirectional ratio and duplicate cells only; no camera or depth filtering",
                "scope": "Image consistency of frozen candidates; original ARFrame pose may share RGB evidence; not physical truth"}
    out.mkdir(parents=True)
    write_json(out / "protocol.json", protocol)  # Before feature extraction/scoring.
    started = time.perf_counter()
    images, features, ks, feature_errors = {}, {}, {}, {}
    detector = cv.SIFT_create(nfeatures=PARAMETERS["sift_features"])
    for i in ids:
        images[i] = np.array(Image.open(rgb_paths[i]).convert("RGB"))
        if images[i].shape != (1440, 1920, 3):
            raise ValueError("fixed experiment expects unrotated original 1920x1440 RGB")
        try:
            kp, desc = detector.detectAndCompute(cv.cvtColor(images[i], cv.COLOR_RGB2GRAY), None)
        except cv.error as exc:
            kp, desc = [], None
            feature_errors[str(i)] = str(exc)
        if desc is not None:
            desc = np.sqrt(desc / np.maximum(desc.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)
        features[i] = (np.array([k.pt for k in kp]).reshape(-1, 2), desc)
        ks[i] = np.array(records[i]["intrinsics"]).reshape(3, 3)
    rows = []
    theta = np.deg2rad(PARAMETERS["local_y_perturbation_degrees"])
    rot = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    for a, b in PAIRS:
        pair_errors = []
        for i in (a, b):
            if str(i) in feature_errors:
                pair_errors.append(f"frame {i}: feature extraction failed")
        try:
            p, q, distances, before_dedup = match_features(cv, features[a], features[b])
        except cv.error as exc:
            p, q, distances, before_dedup = np.empty((0, 2)), np.empty((0, 2)), np.empty(0), 0
            pair_errors.append("matcher: " + str(exc))
        name = f"{a:05}-{b:05}"
        np.savez_compressed(out / f"matches-{name}.npz", a=p, b=q, descriptor_distance=distances)
        match_hash = hashlib.sha256(np.c_[p, q, distances].astype("<f8").tobytes()).hexdigest()
        scoring, arrays = {}, {}
        for method in scenes:
            arrays[method], scoring[method] = score_camera(cams[a][method], cams[b][method], ks[a], ks[b], p, q)
        wrong = cams[a]["baseline"].copy()
        wrong[:3, :3] = wrong[:3, :3] @ rot
        _, scoring["perturbed_2deg"] = score_camera(wrong, cams[b]["baseline"], ks[a], ks[b], p, q)
        scaled = cams[a]["baseline"].copy()
        scaled[:3, 3] = cams[b]["baseline"][:3, 3] + 3.7*(scaled[:3, 3]-cams[b]["baseline"][:3, 3])
        scaled_scores, scale_stats = score_camera(scaled, cams[b]["baseline"], ks[a], ks[b], p, q)
        scale_observable = not scale_stats["error"] and not scoring["baseline"]["error"]
        scale_observable &= bool(np.isfinite(scaled_scores).all() and np.isfinite(arrays["baseline"]).all())
        scale_delta = (float(np.max(np.abs(scaled_scores-arrays["baseline"]))) if len(p) else 0.) if scale_observable else None
        if scale_delta is not None and scale_delta > 1e-7:
            raise AssertionError("epipolar metric incorrectly observes translation length")
        unchanged = all(np.array_equal(cams[i]["baseline"][:3], cams[i]["balanced"][:3]) for i in (a, b))
        if unchanged and not np.array_equal(arrays["baseline"], arrays["balanced"]):
            raise AssertionError("unchanged camera scores changed")
        cv.setRNGSeed(PARAMETERS["seed"])
        try:
            h, inliers = cv.findHomography(p, q, cv.RANSAC, PARAMETERS["homography_threshold_px"], maxIters=2000, confidence=.995) if len(p) >= 4 else (None, None)
        except cv.error as exc:
            h, inliers = None, None
            pair_errors.append("homography: " + str(exc))
        fraction = float(inliers.mean()) if h is not None and inliers is not None else None
        ca, cb = grid_coverage(p, 1920, 1440), grid_coverage(q, 1920, 1440)
        sufficient = len(p) >= PARAMETERS["minimum_matches"] and min(ca["occupied"], cb["occupied"]) >= PARAMETERS["minimum_occupied_cells_each_image"]
        row = {"pair": [a, b], "matches_before_duplicate_cells": before_dedup, "matches": len(p),
               "matches_sha256": match_hash, "coverage_a": ca, "coverage_b": cb,
               "sufficient_numeric_coverage": sufficient, "homography_fraction": fraction,
               "single_homography_compatible": fraction is not None and fraction >= PARAMETERS["plane_fraction"],
               "errors": pair_errors,
               "evidence_status": "NUMERIC_COVERAGE_ONLY" if sufficient and not pair_errors and all(v["error"] is None and v["finite_count"] == len(p) for v in scoring.values()) else "INCONCLUSIVE",
               "scores": scoring, "unchanged_control": unchanged, "scale_invariance_max_px": scale_delta,
               "review_match_ids": montage((images[a], images[b]), p, q, out / f"review-{name}.png")}
        rows.append(row)
        print(json.dumps({"pair": [a, b], "n": len(p), "coverage": [ca["occupied"], cb["occupied"]],
                          "median_original_balanced": [scoring[k]["median_px"] for k in ("baseline", "balanced")]}), flush=True)
    if (verify_source(paths["input"], manifest) != source_receipt
            or any(digest(paths[name]) != hashes[name] for name in scenes)
            or digest(__file__) != code_hash):
        raise RuntimeError("inputs or code changed during experiment")
    write_json(out / "result.json", {"schema": protocol["schema"], "protocol_sha256": digest(out / "protocol.json"),
               "feature_counts": {str(i): len(features[i][0]) for i in ids}, "feature_errors": feature_errors, "pairs": rows,
               "candidate_adopted": False, "prior_geometry_gate": "FAIL", "all_inputs_preserved": True})
    write_json(out / "runtime.json", {"elapsed_seconds": time.perf_counter()-started, "versions": protocol["versions"]})


if __name__ == "__main__":
    main()
