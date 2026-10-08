"""Observation-only bounded fitting of the operator-selected seven-part stool family.

Inputs are scene box/floor priors and unmodified owner-assigned observations.
No reference asset or fixture configuration is read. All distances are in metres.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


PART_NAMES = ["seat", "leg_1", "leg_2", "leg_3", "brace_1_2", "brace_2_3", "brace_3_1"]
ASSUMPTIONS = [
    "Operator-selected family: one horizontal circular seat, three straight splayed cylindrical legs, and three horizontal cylindrical braces joining the three leg pairs.",
    "Y is up and units are metres; the measured scene Floor upper plane defines ground height.",
    "Legs continue through any unobserved regions to meet the seat and floor. Continuation and hidden surfaces are structural completion, not observed surface identity.",
    "Leg top centre ends at the seat mid-plane. Braces end on the fitted leg axes; deliberate solid overlap creates contact while preserving seven independently editable closed components.",
    "Each brace has its own fitted horizontal elevation and radius. No equal-angle, equal-radius, equal-height, or exact rotational-symmetry prior is imposed on the legs or braces.",
    "Object ownership is the supplied prior box assignment and may contain wrong or missing observations. It is not ground-truth instance segmentation.",
    "Even frame indices alone determine initialization, sampling, parameters, and color. Odd frame indices are a validation split, not an untouched final test set.",
    "Point-to-model distances do not prove unobserved surface accuracy or establish a unique reconstruction. Structural completion cannot be scored as observed evidence.",
]


def least_squares(*args, **kwargs):
    # Distance and input-validation tests run with the baseline NumPy dependency.
    try:
        from scipy.optimize import least_squares as scipy_least_squares
    except ImportError as error:
        raise RuntimeError("Fitting requires optional dependencies in requirements-fit.txt") from error
    return scipy_least_squares(*args, **kwargs)


def load_inputs(scene_path, observations_path, object_uuid):
    """Validate the supplied measurement bundle before selecting the target owner."""
    scene = json.loads(Path(scene_path).read_text(encoding="utf-8"))
    if not isinstance(scene, dict) or scene.get("up_axis") != "Y" or scene.get("units") != "m":
        raise ValueError("Fitter requires a scene with Y-up metre observations")
    objects = scene.get("objects")
    if not isinstance(objects, list) or not objects or any(not isinstance(o, dict) for o in objects):
        raise ValueError("Scene objects must be a nonempty list of object records")
    matches = [(i, o) for i, o in enumerate(objects) if o.get("uuid") == object_uuid]
    if len(matches) != 1 or matches[0][1].get("category") != "Chair":
        raise ValueError("Target UUID must identify exactly one Chair record")
    index, target = matches[0]

    def box(record, label):
        try:
            dimensions = np.asarray(record["dimensions_m"], dtype=float)
            transform = np.asarray(record["transform_world"], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{label} needs numeric dimensions_m and transform_world") from error
        if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
            raise ValueError(f"{label} dimensions must be three finite positive values")
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"{label} transform must be finite 4x4")
        rotation = transform[:3, :3]
        if (not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6) or
                not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or
                not np.isclose(np.linalg.det(rotation), 1., atol=1e-6) or
                not np.allclose(rotation[:, 1], [0, 1, 0], atol=1e-6)):
            raise ValueError(f"{label} transform must be a rigid upright Y-up transform")
        return transform[:3, 3], dimensions

    box_center, dims = box(target, "Target box")
    structures = scene.get("structures")
    if not isinstance(structures, list) or any(not isinstance(s, dict) for s in structures):
        raise ValueError("Scene structures must contain a measured Floor prior")
    floors = [s for s in structures if s.get("category") == "Floor"]
    if not floors:
        raise ValueError("A measured Floor prior is required")
    floor_planes = [c[1] + d[1] / 2 for c, d in [box(f, "Floor") for f in floors]]
    floor = max(floor_planes)
    if not floor < box_center[1] < floor + dims[1]:
        raise ValueError("Target box centre must be above and within one box height of the Floor")
    frame_records = scene.get("frames")
    if not isinstance(frame_records, list) or not frame_records:
        raise ValueError("Scene needs frame records with unique nonnegative integer frame_index")
    frame_ids = [f.get("frame_index") if isinstance(f, dict) else None for f in frame_records]
    if (any(type(f) is not int or f < 0 for f in frame_ids) or len(set(frame_ids)) != len(frame_ids)):
        raise ValueError("Scene frame_index values must be unique nonnegative integers")
    required = {"points", "colors", "frame_index", "owner"}
    with np.load(observations_path, allow_pickle=False) as raw:
        if not required.issubset(raw.files):
            raise ValueError("Observations require points, colors, frame_index, and owner arrays")
        points, colors, frames, owners = (raw[k] for k in ("points", "colors", "frame_index", "owner"))
        n = len(points) if points.ndim else 0
        for name, array, shape in (("points", points, (n, 3)), ("colors", colors, (n, 3)),
                                   ("frame_index", frames, (n,)), ("owner", owners, (n,))):
            if array.shape != shape or not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
                raise ValueError(f"{name} has an invalid numeric shape or type")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite")
        if np.any(colors < 0) or np.any(colors > 255) or np.any(colors != np.floor(colors)):
            raise ValueError("colors must be integer RGB values in [0,255]")
        if not np.issubdtype(frames.dtype, np.integer) or not np.isin(frames, frame_ids).all():
            raise ValueError("frame_index must contain integers declared by scene frames")
        if not np.issubdtype(owners.dtype, np.integer) or np.any(owners < -2) or np.any(owners >= len(objects)):
            raise ValueError("owner must contain valid object indices or -1/-2")
        target_mask = owners == index
        points, colors, frames = points[target_mask], colors[target_mask], frames[target_mask]
    if len(points) < 100:
        raise ValueError("Expected at least 100 finite target observations")
    train = frames % 2 == 0
    if np.sum(train) < 50 or np.sum(~train) < 20:
        raise ValueError("Need at least 50 even-frame training and 20 odd-frame validation observations")
    return scene, index, box_center, dims, float(floor), points, colors, frames


def validate_output(output, inputs):
    """Reject occupied destinations and aliases of input files before any write."""
    output = Path(output)
    resolved = output.resolve()
    if output.is_symlink() or any(resolved == Path(p).resolve() for p in inputs):
        raise ValueError("Output must be a new or empty directory separate from input files")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty; existing artifacts are never overwritten")
    return resolved


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cylinder_sdf(points, start, end, radius):
    """Exact signed Euclidean distance to one capped cylinder (negative inside)."""
    start, end = np.asarray(start), np.asarray(end)
    direction = end - start
    length = np.linalg.norm(direction)
    if length <= 1e-10 or radius <= 0:
        raise ValueError("Cylinder length and radius must be positive")
    axis = direction / length
    delta = np.asarray(points) - (start + end) / 2
    axial = delta @ axis
    radial = np.linalg.norm(delta - axial[:, None] * axis, axis=1)
    q = np.column_stack((radial - radius, np.abs(axial) - length / 2))
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0)


def components(x, floor):
    cx, cz, top, thick, radius = x[:5]
    seat = {"name": "seat", "start": np.array([cx, top - thick, cz]),
            "end": np.array([cx, top, cz]), "radius": radius}
    result = [seat]
    for i in range(3):
        bx, bz, tx, tz, r = x[5 + i * 5:10 + i * 5]
        top_y = top - thick / 2
        foot_y = floor
        # A tilted flat cap has a lower rim: lift its centre to make that rim touch floor.
        horizontal = np.hypot(tx - bx, tz - bz)
        for _ in range(8):
            foot_y = floor + r * horizontal / np.hypot(horizontal, top_y - foot_y)
        result.append({"name": PART_NAMES[i + 1], "start": np.array([bx, foot_y, bz]),
                       "end": np.array([tx, top_y, tz]), "radius": r})
    for i, (a, b) in enumerate(((1, 2), (2, 3), (3, 1))):
        y, r = x[20 + 2 * i:22 + 2 * i]
        ends = []
        for index in (a, b):
            leg = result[index]
            frac = (y - leg["start"][1]) / (leg["end"][1] - leg["start"][1])
            ends.append(leg["start"] + frac * (leg["end"] - leg["start"]))
        result.append({"name": PART_NAMES[4 + i], "start": ends[0], "end": ends[1], "radius": r})
    return result


def distances(points, x, floor):
    signed = np.column_stack([cylinder_sdf(points, p["start"], p["end"], p["radius"])
                              for p in components(x, floor)])
    # Outside all solids this is exact union distance; in overlap interiors the min-SDF
    # construction is an implicit residual rather than an exact union-boundary distance.
    return np.min(signed, axis=1), np.argmin(np.abs(signed), axis=1)


def deterministic_sample(points, frames, floor, top, cap=36):
    rng = np.random.default_rng(72491)
    height = np.clip(((points[:, 1] - floor) / (top - floor) * 12).astype(int), 0, 11)
    chosen = []
    for frame in np.unique(frames):
        for band in range(12):
            ids = np.flatnonzero((frames == frame) & (height == band))
            if len(ids):
                chosen.extend(rng.choice(ids, min(cap, len(ids)), replace=False))
    ids = np.sort(chosen)
    return ids, height


def optimizer_record(solution):
    return {"success": bool(solution.success), "status": int(solution.status),
            "message": str(solution.message), "nfev": int(solution.nfev),
            "njev": None if solution.njev is None else int(solution.njev),
            "cost": float(solution.cost), "optimality": float(solution.optimality),
            "active_mask": solution.active_mask.tolist()}


def seat_fit(points, box_center, dims, floor):
    height, span = dims[1], max(dims[0], dims[2])
    top_guess = np.median(points[points[:, 1] >= np.quantile(points[:, 1], .65), 1])
    upper = np.sort(points[points[:, 1] > floor + height * .5, 1])
    gaps = np.diff(upper)
    valid = np.flatnonzero((upper[:-1] < top_guess - height * .05) &
                          (upper[:-1] > top_guess - height * .3))
    if not len(valid):
        raise ValueError("Insufficient upper-height separation to initialize the seat band")
    boundary = (upper[valid[np.argmax(gaps[valid])]] + upper[valid[np.argmax(gaps[valid])] + 1]) / 2
    high = points[points[:, 1] > boundary]
    bottom_guess = np.quantile(high[:, 1], .015)
    r_guess = np.quantile(np.linalg.norm(high[:, [0, 2]] - box_center[[0, 2]], axis=1), .97)
    initial = np.array([box_center[0], box_center[2], top_guess, top_guess - bottom_guess, r_guess])
    lower = np.array([box_center[0] - span * .25, box_center[2] - span * .25,
                      floor + height * .65, height * .025, span * .2])
    upper_bound = np.array([box_center[0] + span * .25, box_center[2] + span * .25,
                            floor + height * 1.15, height * .25, span * .7])
    bands = np.clip(((high[:, 1] - boundary) / (top_guess - boundary) * 6).astype(int), 0, 5)
    weights = np.sqrt(len(high) / (6 * np.maximum(np.bincount(bands, minlength=6)[bands], 1)))

    def residual(x):
        a, b = [x[0], x[2] - x[3], x[1]], [x[0], x[2], x[1]]
        return cylinder_sdf(high, a, b, x[4]) * weights

    solution = least_squares(residual, np.clip(initial, lower + 1e-8, upper_bound - 1e-8),
                             bounds=(lower, upper_bound), loss="soft_l1", f_scale=.003,
                             x_scale="jac", max_nfev=160)
    return solution, lower, upper_bound, boundary


def kmeans_three(data):
    if len(np.unique(data, axis=0)) < 3:
        raise ValueError("Three distinct low-leg point clusters are required")
    rng = np.random.default_rng(1942)
    centers = [data[rng.integers(len(data))]]
    for _ in range(2):
        d = np.min(np.linalg.norm(data[:, None] - np.array(centers)[None], axis=2), axis=1)
        centers.append(data[np.argmax(d)])
    centers = np.array(centers)
    for _ in range(40):
        labels = np.argmin(np.linalg.norm(data[:, None] - centers[None], axis=2), axis=1)
        if np.any(np.bincount(labels, minlength=3) == 0):
            raise ValueError("Three independently supported low-leg clusters are required")
        centers = np.array([np.mean(data[labels == i], axis=0) for i in range(3)])
    return centers, labels


def initialize(points, box_center, dims, floor):
    seat, lo, hi, boundary = seat_fit(points, box_center, dims, floor)
    x = list(seat.x)
    lower, upper, names = list(lo), list(hi), ["seat_center_x", "seat_center_z", "seat_top_y", "seat_thickness", "seat_radius"]
    height = seat.x[2] - floor
    low = points[(points[:, 1] > floor + height * .04) & (points[:, 1] < floor + height * .28)]
    if len(low) < 18:
        raise ValueError("Too few low-leg observations to initialize three supported legs")
    centers, labels = kmeans_three(low[:, [0, 2]])
    separation = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    separation[np.diag_indices(3)] = np.inf
    if np.min(np.bincount(labels, minlength=3)) < 5 or np.min(separation) < height * .15:
        raise ValueError("Three independently supported low-leg clusters are required (>=5 points each and separation >=15% of fitted height)")
    order = np.argsort(np.arctan2(centers[:, 1] - seat.x[1], centers[:, 0] - seat.x[0]))
    for j, cluster in enumerate(order):
        p = low[labels == cluster]
        # The low band avoids the cross-brace elevation. Fit an axis estimate there.
        design = np.column_stack((np.ones(len(p)), p[:, 1] - floor))
        coefs = np.linalg.lstsq(design, p[:, [0, 2]], rcond=None)[0]
        foot = coefs[0]
        top = coefs[0] + coefs[1] * (seat.x[2] - seat.x[3] / 2 - floor)
        radial = top - seat.x[[0, 1]]
        if np.linalg.norm(radial) > seat.x[4] * .85:
            top = seat.x[[0, 1]] + radial / np.linalg.norm(radial) * seat.x[4] * .7
        radius = np.clip(np.median(np.linalg.norm(p[:, [0, 2]] - design @ coefs, axis=1)), height * .02, height * .075)
        x.extend([*foot, *top, radius])
        span = max(dims[0], dims[2])
        lower.extend([box_center[0] - span * .7, box_center[2] - span * .7,
                      box_center[0] - span * .6, box_center[2] - span * .6, height * .01])
        upper.extend([box_center[0] + span * .7, box_center[2] + span * .7,
                      box_center[0] + span * .6, box_center[2] + span * .6, height * .12])
        names.extend([f"leg_{j+1}_{suffix}" for suffix in ("foot_x", "foot_z", "top_x", "top_z", "radius")])
    below = points[points[:, 1] < boundary]
    counts, edges = np.histogram(below[:, 1], bins=40, range=(floor, seat.x[2] - seat.x[3]))
    eligible = (edges[:-1] > floor + height * .18) & (edges[1:] < floor + height * .7)
    peak = np.argmax(np.where(eligible, counts, -1))
    brace_y = (edges[peak] + edges[peak + 1]) / 2
    for i in range(3):
        x.extend([brace_y, height * .025])
        lower.extend([floor + height * .12, height * .0075])
        upper.extend([floor + height * .75, height * .08])
        names.extend([f"brace_{i+1}_height", f"brace_{i+1}_radius"])
    return np.array(x), np.array(lower), np.array(upper), names, optimizer_record(seat), boundary


def fit(points, frames, box_center, dims, floor):
    x, lower, upper, names, seat_record, boundary = initialize(points, box_center, dims, floor)
    initial = x.copy()
    ids, bands = deterministic_sample(points, frames, floor, x[2])
    selected = points[ids]
    # Balance both height and initial closest component; hold weights fixed during fitting.
    _, part = distances(selected, x, floor)
    hb = bands[ids]
    h_count = np.bincount(hb, minlength=12)
    p_count = np.bincount(part, minlength=7)
    weights = np.sqrt(.5 * len(ids) / (np.count_nonzero(h_count) * h_count[hb]) +
                      .5 * len(ids) / (np.count_nonzero(p_count) * p_count[part]))

    def residual(params):
        sdf, _ = distances(selected, params, floor)
        parts = components(params, floor)
        priors = []
        for leg in parts[1:4]:
            radius_to_seat = np.linalg.norm(leg["end"][[0, 2]] - params[:2])
            # Structural joint must lie inside the seat, and leg must splay outwards.
            priors.append(max(0., radius_to_seat + leg["radius"] - params[4]) * 6)
            foot_radius = np.linalg.norm(leg["start"][[0, 2]] - params[:2])
            priors.append(max(0., radius_to_seat - foot_radius) * 4)
        return np.concatenate((sdf * weights, priors))

    runs = []
    # Multiple observation-derived starting splay estimates reduce dependence on sparse low bands.
    for scale in (1., .75, 1.2):
        start = initial.copy()
        for i in range(3):
            top_ids = np.array([7 + 5 * i, 8 + 5 * i])
            start[top_ids] = initial[:2] + scale * (initial[top_ids] - initial[:2])
        start = np.clip(start, lower + 1e-9, upper - 1e-9)
        solution = least_squares(residual, start, bounds=(lower, upper), loss="soft_l1",
                                 f_scale=.004, x_scale="jac", max_nfev=240,
                                 ftol=1e-8, xtol=1e-8, gtol=1e-8)
        runs.append(solution)
    selected_run = int(np.argmin([result.cost for result in runs]))
    best = runs[selected_run]
    return best.x, {"method": "SciPy bounded trust-region least_squares; soft_l1 robust loss",
                   "robust_scale_m": .004, "training_sample_count": len(ids),
                   "sampling": "deterministic frame x 12-height strata, max 36 points per stratum",
                   "weighting": "equal mix of inverse height-band and initial closest-part frequency; fixed weights",
                   "seat_initialization": seat_record, "seat_band_boundary_y_m": boundary,
                   "multistart": [optimizer_record(run) for run in runs],
                   "selected_run": selected_run, "selected": optimizer_record(best),
                   "initial_parameters": dict(zip(names, initial.tolist())),
                   "parameters": dict(zip(names, best.x.tolist())),
                   "bounds": {name: {"lower": float(a), "upper": float(b)} for name, a, b in zip(names, lower, upper)},
                   "training_sample_index_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()}


def stats(values):
    if not len(values):
        return {"count": 0, "mean_m": None, "rmse_m": None, "p50_m": None, "p95_m": None, "within_10mm_fraction": None}
    return {"count": len(values), "mean_m": float(np.mean(values)), "rmse_m": float(np.sqrt(np.mean(values ** 2))),
            "p50_m": float(np.median(values)), "p95_m": float(np.quantile(values, .95)),
            "within_10mm_fraction": float(np.mean(values <= .010))}


def evaluate(points, frames, x, floor):
    sdf, part = distances(points, x, floor)
    error = np.abs(sdf)
    height = np.clip(((points[:, 1] - floor) / (x[2] - floor) * 12).astype(int), 0, 11)
    part_stats = {name: stats(error[part == i]) for i, name in enumerate(PART_NAMES)}
    bands = {str(i): stats(error[height == i]) for i in range(12)}
    mean = lambda records: float(np.mean([r["mean_m"] for r in records.values() if r["count"]]))
    return {"point_count_weighted": stats(error), "part_balanced_mean_m": mean(part_stats),
            "height_balanced_mean_m": mean(bands), "per_closest_part": part_stats, "height_bands": bands,
            "per_frame": {str(int(f)): {**stats(error[frames == f]),
                           "supported_points_within_15mm_by_part": {name: int(np.sum((frames == f) & (part == i) & (error <= .015)))
                                                                     for i, name in enumerate(PART_NAMES)}}
                          for f in np.unique(frames)}}


def cylinder_mesh(part, slices=80):
    a, b, r = part["start"], part["end"], part["radius"]
    axis = (b - a) / np.linalg.norm(b - a)
    seed = np.array([1., 0., 0.]) if abs(axis[0]) < .8 else np.array([0., 0., 1.])
    u = np.cross(axis, seed); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    theta = np.linspace(0, 2 * np.pi, slices, endpoint=False)
    circle = r * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)
    vertices = np.concatenate((a + circle, b + circle, a[None], b[None]))
    faces = []
    for i in range(slices):
        j = (i + 1) % slices
        faces.extend(((i, j, slices + j), (i, slices + j, slices + i),
                      (2 * slices, j, i), (2 * slices + 1, slices + i, slices + j)))
    return vertices, np.array(faces)


def write_ply(path, vertices, faces, vertex_parts, face_parts, colors):
    with open(path, "x", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\ncomment provenance fitted_structural_not_observed_identity\n")
        stream.write("comment seven editable closed parts with intentional joint overlap\n")
        stream.write(f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nproperty int part_id\n")
        stream.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nproperty int part_id\nend_header\n")
        for point, part in zip(vertices, vertex_parts):
            color = colors[part]
            stream.write("%.9f %.9f %.9f %d %d %d %d\n" % (*point, *color, part))
        for face, part in zip(faces, face_parts):
            stream.write("3 %d %d %d %d\n" % (*face, part))


def contact_checks(parts, floor):
    contacts = []
    seat = parts[0]
    for i in range(1, 4):
        signed = float(cylinder_sdf(parts[i]["end"][None], seat["start"], seat["end"], seat["radius"])[0])
        contacts.append({"parts": ["seat", PART_NAMES[i]], "joint_axis_inside_target_depth_m": -signed,
                         "contact": bool(signed < 0)})
    for j, pair in enumerate(((1, 2), (2, 3), (3, 1))):
        for endpoint, leg_idx in zip(("start", "end"), pair):
            leg = parts[leg_idx]
            signed = float(cylinder_sdf(parts[4 + j][endpoint][None], leg["start"], leg["end"], leg["radius"])[0])
            contacts.append({"parts": [PART_NAMES[4 + j], PART_NAMES[leg_idx]],
                             "joint_axis_inside_target_depth_m": -signed, "contact": bool(signed < 0)})
    return {"all_nine_intended_joints_in_solid_contact": all(c["contact"] for c in contacts),
            "joints": contacts, "mesh_topology": "7 closed separately indexed component shells; union has solid contact; no destructive welding",
            "floor_y_m": floor}


def self_test():
    p = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]])
    actual = cylinder_sdf(p, [0, -1, 0], [0, 1, 0], 1)
    np.testing.assert_allclose(actual, [0, 0, -1, 1, 1, np.sqrt(2)], atol=1e-12)
    theta = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    sides = np.array([[.28 * np.cos(t), y, .28 * np.sin(t)] for t in theta for y in [.45, .46, .48, .50, .515]])
    cap = np.array([[r * np.cos(t), .515, r * np.sin(t)] for t in theta for r in [.05, .14, .23]])
    # Add lower-band points solely to establish an observation-derived seat-band gap.
    source = np.concatenate((sides, cap, np.array([[.17, y, 0.] for y in np.linspace(.05, .39, 40)])))
    dims = np.array([.6, .54, .6]); center = np.array([0., .27, 0.])
    first = seat_fit(source, center, dims, 0)[0]
    shifted = source + np.array([.03, .012, -.02])
    second = seat_fit(shifted, center, dims, 0)[0]
    np.testing.assert_allclose(second.x[:3] - first.x[:3], [.03, -.02, .012], atol=.0015)
    return {"analytic_capped_cylinder_distance": "passed 6 exact cases",
            "changed_observations_change_fitted_seat": "passed; same scene prior and observations translated by [0.03,0.012,-0.02] m",
            "observed_parameter_shift": (second.x[:3] - first.x[:3]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-uuid", default="same-stool")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    started = time.time()
    try:
        validate_output(args.output, [args.scene, args.observations])
        scene, index, box_center, dims, floor, points, colors, frames = load_inputs(
            args.scene, args.observations, args.object_uuid)
        tests = self_test() if args.self_test else {"status": "not_requested"}
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    train = frames % 2 == 0
    try:
        x, optimization = fit(points[train], frames[train], box_center, dims, floor)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    parts = components(x, floor)
    training = evaluate(points[train], frames[train], x, floor)
    heldout = evaluate(points[~train], frames[~train], x, floor)
    _, assigned = distances(points[train], x, floor)
    default_color = np.median(colors[train], axis=0).astype(int)
    part_colors = [np.median(colors[train][assigned == i], axis=0).astype(int) if np.any(assigned == i) else default_color for i in range(7)]
    vertices, faces, vertex_parts, face_parts = [], [], [], []
    records = []
    offset = 0
    # Recheck after fitting so an intervening artifact cannot be overwritten.
    validate_output(args.output, [args.scene, args.observations])
    args.output.mkdir(parents=True, exist_ok=True)
    for i, part in enumerate(parts):
        v, f = cylinder_mesh(part)
        vertices.extend(v); faces.extend(f + offset)
        vertex_parts.extend([i] * len(v)); face_parts.extend([i] * len(f)); offset += len(v)
        record = {"part_id": i, "name": part["name"], "type": "capped_cylinder", "provenance_role": "fitted/structural",
                  "start_m": part["start"].tolist(), "end_m": part["end"].tolist(), "radius_m": float(part["radius"]),
                  "color_rgb": part_colors[i].tolist(), "editable": True,
                  "vertex_count": len(v), "face_count": len(f),
                  "training_closest_point_support": training["per_closest_part"][part["name"]]["count"],
                  "heldout_closest_point_support": heldout["per_closest_part"][part["name"]]["count"]}
        if i == 0:
            record.update({"center_m": ((part["start"] + part["end"]) / 2).tolist(), "thickness_m": float(x[3])})
        records.append(record)
        write_ply(args.output / (part["name"] + ".ply"), v, f, np.full(len(v), i), np.full(len(f), i), part_colors)
    write_ply(args.output / "fitted_stool.ply", np.asarray(vertices), np.asarray(faces), vertex_parts, face_parts, part_colors)
    contacts = contact_checks(parts, floor)
    reliable = (optimization["selected"]["success"] and contacts["all_nine_intended_joints_in_solid_contact"] and
                heldout["part_balanced_mean_m"] < .015 and heldout["height_balanced_mean_m"] < .015 and
                all(v["count"] >= 5 for v in training["per_closest_part"].values()))
    result = {"schema": "alvenx.spatial.observation-constrained-stool-fit.v1", "object_uuid": args.object_uuid,
              "provenance_role": "fitted/structural", "units": "m", "up_axis": "Y",
              "input_files": {"scene": {"path": str(args.scene.resolve()), "sha256": sha256(args.scene)},
                              "observations": {"path": str(args.observations.resolve()), "sha256": sha256(args.observations)},
                              "fitter": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)}},
              "read_scope": "Only the two specified input files; target box, Floor, ownership, XYZ, RGB, frame_index. No reference/fixture read.",
              "assumptions": ASSUMPTIONS, "source_owner_index": index,
              "source_observation_count": len(points), "training_frames": np.unique(frames[train]).tolist(),
              "heldout_frames": np.unique(frames[~train]).tolist(),
              "split_role": "Even frames train; odd frames validate on the same development example. This is not an untouched final test set.",
              "optimizer": optimization,
              "editable_components": records, "contact_checks": contacts,
              "distance_definition": "absolute min capped-cylinder signed-distance residual; exact outside the solids and on each exposed primitive surface; overlap-interior residual is not exact union-boundary Euclidean distance",
              "training_observation_error": training, "heldout_observation_error": heldout,
              "correctness_tests": tests, "quality_gate": {"passed": bool(reliable),
                 "criteria": "optimizer success; all nine contacts; odd-frame validation equal-part and equal-height mean <15mm; each part >=5 training closest-point supports",
                 "interpretation": "Validation thresholds declared for one development example; not a global shape-completeness, ground-truth-accuracy, or product-quality claim"},
              "elapsed_seconds": time.time() - started,
              "outputs": {"assembly_ply": "fitted_stool.ply", "editable_parts": [p["name"] + ".ply" for p in parts]}}
    with (args.output / "fit.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "quality_gate": result["quality_gate"],
                      "optimizer": optimization["selected"], "train": training["point_count_weighted"],
                      "heldout": heldout["point_count_weighted"], "heldout_part_balanced_mean_m": heldout["part_balanced_mean_m"],
                      "heldout_height_balanced_mean_m": heldout["height_balanced_mean_m"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2))
    return 0 if reliable else 2


if __name__ == "__main__":
    sys.exit(main())
