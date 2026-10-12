"""Check authored dining layout with actual immutable closed component meshes.

No physical reference is read. GJK works on the actual convex triangle vertices,
including edge crossings with no contained vertex. A pass always requires an
explicit separating plane with >=5mm projection gap for every cross-object pair.
Observed/fitted geometry stays frozen; only authored horizontal root deltas move.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np

import evaluate_dining as d
import evaluate_stool_quality as q
from evaluate_stool_transfer import convex_planes


OBJECTS = (*d.TARGETS, "same-stool")
CLEARANCE_M = .0055
SEARCH_TOLERANCE_M = 1e-6
AUTHORING_MARGIN_M = .00005


def transformed(part, delta):
    delta = np.asarray(delta, dtype=float)
    vertices = part["vertices"] + delta
    planes = part["planes"].copy()
    planes[:, 3] -= planes[:, :3] @ delta
    return {**part, "vertices": vertices, "planes": planes}


def support(first, second, direction):
    a = first[int(np.argmax(first @ direction))]
    b = second[int(np.argmin(second @ direction))]
    return np.concatenate((a - b, a, b))


def closest_simplex(simplex):
    """Enumerate simplex faces, solving each barycentric least-norm problem."""
    best = None
    for size in range(1, min(4, len(simplex)) + 1):
        for subset in combinations(range(len(simplex)), size):
            values = simplex[list(subset), :3]
            gram = values @ values.T
            system = np.block([[gram, np.ones((size, 1))], [np.ones((1, size)), np.zeros((1, 1))]])
            rhs = np.r_[np.zeros(size), 1.]
            weights = np.linalg.lstsq(system, rhs, rcond=None)[0][:-1]
            if np.any(weights < -1e-10):
                continue
            weights = np.maximum(weights, 0)
            weights /= weights.sum()
            closest = weights @ values
            squared = float(closest @ closest)
            if best is None or squared < best[0]:
                best = (squared, np.asarray(subset)[weights > 1e-10], weights[weights > 1e-10], closest)
    if best is None:
        raise ValueError("GJK failed to resolve a finite simplex")
    _, selected, weights, closest = best
    selected_points = simplex[selected]
    return selected_points, closest, weights @ selected_points[:, 3:6], weights @ selected_points[:, 6:9]


def gjk_distance(first, second):
    """Actual convex-solid closest points, with final bounds independently checked."""
    direction = second.mean(0) - first.mean(0)
    if np.linalg.norm(direction) < 1e-12:
        direction = np.array([1., 0, 0])
    simplex = support(first, second, direction)[None, :]
    previous = np.inf
    for iteration in range(256):
        simplex, closest, a, b = closest_simplex(simplex)
        squared = float(closest @ closest)
        if squared <= 1e-18:
            return {"closest_difference": closest, "first_point": a, "second_point": b,
                    "upper_bound_m": float(np.sqrt(squared)), "iterations": iteration + 1}
        candidate = support(first, second, -closest)
        improvement = squared - float(closest @ candidate[:3])
        if (improvement <= max(1e-15, squared * 1e-10) or
                np.any(np.linalg.norm(simplex[:, :3] - candidate[:3], axis=1) < 1e-12) or
                previous - squared <= 1e-20):
            return {"closest_difference": closest, "first_point": a, "second_point": b,
                    "upper_bound_m": float(np.sqrt(squared)), "iterations": iteration + 1}
        previous = squared
        simplex = np.vstack((simplex, candidate))
    raise ValueError("GJK iteration limit; separation is not established")


def plane_certificate(first, second, normal):
    normal = np.asarray(normal, dtype=float)
    length = float(np.linalg.norm(normal))
    if length <= 1e-14:
        return None
    normal = normal / length
    a, b = first @ normal, second @ normal
    gap = float(a.min() - b.max())
    if gap < 0:
        normal, a, b = -normal, -a, -b
        gap = float(a.min() - b.max())
    return {"unit_normal_world": (-normal).tolist(), "first_projection_max_m": float(-a.min()),
            "second_projection_min_m": float(-b.max()), "gap_m": gap,
            "plane_offset_m": float(-(a.min() + b.max()) / 2),
            "method": "minimum/maximum projections of every actual component vertex; all triangles and their enclosed convex solids remain in those halfspaces"}


def common_interior(first, second):
    """Strict shared point in the actual outward triangle halfspaces."""
    planes = np.vstack((first["planes"], second["planes"]))
    candidates = np.vstack((first["vertices"].mean(0), second["vertices"].mean(0),
                            (first["vertices"].mean(0) + second["vertices"].mean(0)) / 2))
    margins = -(candidates @ planes[:, :3].T + planes[:, 3]).max(1)
    point = candidates[int(np.argmax(margins))].copy()
    for _ in range(5000):
        signed = planes[:, :3] @ point + planes[:, 3]
        worst = int(np.argmax(signed))
        if signed[worst] < -1e-7:
            break
        point -= (signed[worst] + 4e-7) * planes[worst, :3]
    margin = float(-(planes[:, :3] @ point + planes[:, 3]).max())
    return {"positive_solid_intersection": margin > 1e-7,
            "common_interior_witness_m": point.tolist() if margin > 1e-7 else None,
            "common_interior_ball_radius_lower_bound_m": margin if margin > 1e-7 else None,
            "method": "a common point with strict signed margin to every actual outward triangle plane"}


def check_pair(first, second, clearance=CLEARANCE_M, *, details=True):
    va, vb = first["vertices"], second["vertices"]
    amin, amax, bmin, bmax = va.min(0), va.max(0), vb.min(0), vb.max(0)
    difference = np.maximum(amin - bmax, 0) - np.maximum(bmin - amax, 0)
    lower_aabb = float(np.linalg.norm(difference))
    certificate = plane_certificate(va, vb, difference) if lower_aabb > 0 else None
    gjk = None
    if certificate is None or certificate["gap_m"] < clearance:
        gjk = gjk_distance(va, vb)
        certificate = plane_certificate(va, vb, gjk["closest_difference"])
    lower = max(0., certificate["gap_m"]) if certificate else 0.
    passed = lower >= clearance - 1e-10
    row = {"parts": [first["key"], second["key"]], "passed": bool(passed),
           "required_clearance_m": clearance, "certified_clearance_lower_bound_m": lower,
           "broadphase_aabb_distance_lower_bound_m": lower_aabb,
           "separating_plane": certificate if lower > 0 else None}
    if passed:
        row["classification"] = "separated_at_required_clearance"
    elif lower > 1e-8:
        row["classification"] = "positive_gap_below_required_clearance"
    else:
        row["classification"] = "touching_or_intersecting"
        if details:
            row["intersection"] = common_interior(first, second)
            if row["intersection"]["positive_solid_intersection"]:
                row["classification"] = "positive_solid_intersection"
    if gjk is not None:
        row["gjk"] = {"convex_distance_upper_bound_m": gjk["upper_bound_m"], "iterations": gjk["iterations"],
                      "first_closest_point_m": gjk["first_point"].tolist(), "second_closest_point_m": gjk["second_point"].tolist()}
    return row


def check_layout(parts, translations, clearance=CLEARANCE_M, *, details=True, only_object=None):
    transformed_parts = [transformed(part, translations[part["uuid"]]) for part in parts]
    pairs = []
    for first, second in combinations(transformed_parts, 2):
        if first["uuid"] == second["uuid"]:
            continue
        if only_object and first["uuid"] != only_object and second["uuid"] != only_object:
            continue
        row = check_pair(first, second, clearance, details=details)
        pairs.append(row)
    return {"passed": bool(pairs and all(row["passed"] for row in pairs)), "cross_object_pair_count": len(pairs),
            "minimum_certified_clearance_m": min((row["certified_clearance_lower_bound_m"] for row in pairs), default=None),
            "failed_pair_count": sum(not row["passed"] for row in pairs), "pairs": pairs,
            "same_object_contacts": "excluded from inter-furniture checks; independently accepted structural contacts remain intact"}


def read_part(path, uuid, name):
    digest = q.sha256(path)
    mesh, status = q.read_mesh(path)
    planes = convex_planes(mesh)
    if status != "ok" or planes is None:
        raise ValueError(f"layout requires actual closed outward convex component triangles: {uuid}/{name}")
    vertices, triangles = np.asarray(mesh.vertices).copy(), np.asarray(mesh.triangles).copy()
    vertices.flags.writeable = False
    triangles.flags.writeable = False
    return {"uuid": uuid, "name": name, "key": uuid + "/" + name, "vertices": vertices,
            "triangles": triangles, "planes": planes, "path": Path(path), "sha256": digest}


def load_parts(fit_dir, stool_fit_dir):
    fit_dir, stool_fit_dir = Path(fit_dir), Path(stool_fit_dir)
    fit_path, stool_fit_path = fit_dir / "fit.json", stool_fit_dir / "fit.json"
    hashes = {fit_path: q.sha256(fit_path), stool_fit_path: q.sha256(stool_fit_path)}
    fit, stool = json.loads(fit_path.read_text(encoding="utf-8")), json.loads(stool_fit_path.read_text(encoding="utf-8"))
    if fit.get("schema") != "alvenx.spatial.dining-fit.v1" or {obj.get("uuid") for obj in fit["objects"]} != set(d.TARGETS):
        raise ValueError("frozen dining target identities mismatch")
    if stool.get("object_uuid") != "same-stool" or fit.get("units") != "m" or fit.get("up_axis") != "Y" or stool.get("units") != "m" or stool.get("up_axis") != "Y":
        raise ValueError("frozen stool identity or coordinates mismatch")
    parts = []
    for obj in fit["objects"]:
        for component in obj["components"]:
            parts.append(read_part(d.relative_mesh_path(fit_dir, component["mesh"]), obj["uuid"], component["name"]))
    for component in stool["editable_components"]:
        name = component["name"]
        parts.append(read_part(d.relative_mesh_path(stool_fit_dir, name + ".ply"), "same-stool", name))
    keys = [part["key"] for part in parts]
    if len(set(keys)) != len(keys):
        raise ValueError("component identity duplicated")
    hashes.update({part["path"]: part["sha256"] for part in parts})
    return fit, stool, parts, hashes


def preferred_outward(obj, table):
    direction = np.asarray(obj["transform_world"])[:3, 3] - np.asarray(table["transform_world"])[:3, 3]
    direction[1] = 0
    rotation = np.asarray(obj["transform_world"])[:3, :3]
    axes = rotation[:, [0, 2]].T.copy()
    axes[:, 1] = 0
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    scores = axes @ direction
    chosen = int(np.argmax(abs(scores)))
    result = axes[chosen] * (1 if scores[chosen] >= 0 else -1)
    return result, [0, 2][chosen]


def minimum_shift_certificate(parts, translations, uuid, direction, low, clearance):
    """Convexity proves a blocking pair stays too close on the whole [0,low]."""
    endpoints = []
    for distance in (0., low):
        trial = {key: value.copy() for key, value in translations.items()}
        trial[uuid] = direction * distance
        moved = {part["key"]: transformed(part, trial[part["uuid"]]) for part in parts}
        witnesses = {}
        for first, second in combinations(moved.values(), 2):
            if first["uuid"] == second["uuid"] or uuid not in (first["uuid"], second["uuid"]):
                continue
            row = check_pair(first, second, clearance, details=False)
            if row["passed"] or "gjk" not in row:
                continue
            witness = row["gjk"]
            a, b = np.asarray(witness["first_closest_point_m"]), np.asarray(witness["second_closest_point_m"])
            plane_error = max(float((first["planes"][:, :3] @ a + first["planes"][:, 3]).max()),
                              float((second["planes"][:, :3] @ b + second["planes"][:, 3]).max()), 0.)
            distance_upper = float(np.linalg.norm(a - b))
            if plane_error <= 1e-8 and distance_upper + 5e-8 < clearance:
                witnesses[tuple(row["parts"])] = {"shift_m": distance, "first_point_m": a.tolist(),
                    "second_point_m": b.tolist(), "distance_upper_bound_m": distance_upper,
                    "maximum_actual_triangle_halfspace_error_m": plane_error}
        endpoints.append(witnesses)
    common = endpoints[0].keys() & endpoints[1].keys()
    if not common:
        raise ValueError(f"restricted minimum is not certified for {uuid}; no shared blocking pair at both endpoints")
    selected = sorted(common)[0]
    return {"parts": list(selected), "blocked_shift_interval_m": [0., low],
            "endpoint_witnesses": [endpoint[selected] for endpoint in endpoints],
            "fixed_object_translations_world_m": {key: value.tolist() for key, value in translations.items() if key != uuid},
            "numerical_distance_headroom_m": 5e-8,
            "proof": "For two convex solids under linear translation, interpolate each pair of contained endpoint witness points. Every intermediate pair remains in the translated solids and has distance at most the larger endpoint distance, strictly below the required clearance."}


def propose_layout(fit, parts, clearance=CLEARANCE_M):
    translations = {uuid: np.zeros(3) for uuid in OBJECTS}
    table = next(obj for obj in fit["objects"] if obj["uuid"] == "same-table")
    searches = []
    for uuid in ("same-chair-west", "same-chair-east"):
        obj = next(obj for obj in fit["objects"] if obj["uuid"] == uuid)
        direction, local_axis = preferred_outward(obj, table)

        def acceptable(distance):
            trial = {key: value.copy() for key, value in translations.items()}
            trial[uuid] = direction * distance
            return check_layout(parts, trial, clearance, details=False, only_object=uuid)["passed"]

        low, high = 0., 0.
        if not acceptable(0):
            high = .025
            while high <= 2 and not acceptable(high):
                high *= 2
            if high > 2:
                raise ValueError(f"no separated outward layout within two metres for {uuid}")
            while high - low > SEARCH_TOLERANCE_M:
                middle = (low + high) / 2
                if acceptable(middle):
                    high = middle
                else:
                    low = middle
        distance = high + AUTHORING_MARGIN_M if high else 0.
        minimum_certificate = minimum_shift_certificate(parts, translations, uuid, direction, low, clearance) if high else None
        translations[uuid] = direction * distance
        searches.append({"uuid": uuid, "outward_unit_direction_world": direction.tolist(), "selected_chair_local_axis": local_axis,
                         "minimum_shift_bracket_m": [low, high], "applied_shift_m": distance,
                         "minimum_shift_certificate": minimum_certificate,
                         "authoring_precision_margin_m": AUTHORING_MARGIN_M if high else 0.,
                         "minimality_scope": "smallest feasible outward translation along the existing chair axis, with other furniture fixed; bisection bracket <=1micrometre"})
    return translations, searches


def load_translations(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data["objects"]
    if data.get("schema") != "alvenx.spatial.dining-layout.v1" or len(rows) != len(OBJECTS) or {row.get("uuid") for row in rows} != set(OBJECTS):
        raise ValueError("authored layout identities mismatch")
    result = {}
    for row in rows:
        delta = np.asarray(row["translation_world_m"], dtype=float)
        if delta.shape != (3,) or not np.isfinite(delta).all() or delta[1] != 0:
            raise ValueError("layout must contain finite horizontal world translations only")
        result[row["uuid"]] = delta
    return result


def evaluate(fit_dir, stool_fit_dir, *, layout_path=None, propose=False, stool_source=None, report_base=None):
    fit, stool, parts, hashes = load_parts(fit_dir, stool_fit_dir)
    original = {uuid: np.zeros(3) for uuid in OBJECTS}
    before = check_layout(parts, original)
    searches = []
    if propose:
        if layout_path:
            raise ValueError("choose explicit authored layout or proposal, not both")
        translations, searches = propose_layout(fit, parts)
    else:
        translations = load_translations(layout_path) if layout_path else original
    if layout_path:
        hashes[Path(layout_path)] = q.sha256(layout_path)
    if stool_source:
        hashes[Path(stool_source)] = q.sha256(stool_source)
    after = check_layout(parts, translations)
    for path, digest in hashes.items():
        if q.sha256(path) != digest:
            raise ValueError(f"immutable geometry or layout input changed during evaluation: {path}")
    objects = [{"uuid": uuid, "translation_world_m": translations[uuid].tolist()} for uuid in OBJECTS]
    transform_map = {row["uuid"]: [float(value) for value in row["translation_world_m"]] for row in objects}
    transform_sha = hashlib.sha256(json.dumps(transform_map, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    base = Path(report_base) if report_base else Path(fit_dir)
    passed = after["passed"]
    return {"schema": "alvenx.spatial.dining-layout.v1", "units": "m", "up_axis": "Y", "status": "passed" if passed else "failed",
            "quality_gate": {"passed": passed, "required_cross_furniture_clearance_m": .005,
                             "certification_target_with_roundtrip_margin_m": CLEARANCE_M,
                             "all_actual_component_pairs_certified": passed, "frozen_geometry_unchanged": True},
            "objects": objects, "layout_transform_sha256": transform_sha,
            "fit_json_sha256": hashes[Path(fit_dir) / "fit.json"],
            "component_input_sha256": {part["key"]: part["sha256"] for part in parts if part["uuid"] != "same-stool"},
            "stool_fit_json_sha256": hashes[Path(stool_fit_dir) / "fit.json"],
            "stool_component_input_sha256": {part["name"]: part["sha256"] for part in parts if part["uuid"] == "same-stool"},
            "stool_source_blend_sha256": hashes.get(Path(stool_source)) if stool_source else None,
            "input_files": [d.record(path, base) for path in hashes], "evaluator_sha256": q.sha256(__file__),
            "actual_closed_convex_component_count": len(parts), "original_observed_layout": before,
            "authored_layout": after, "translation_search": searches,
            "source_frame_policy": "all original observations and fitted component PLYs retain their original coordinates and bytes; authored root transforms are recorded separately",
            "limitations": ["The separation proof applies to the supplied closed convex component triangles and these recorded root translations.",
                "The minimum search is restricted to outward translation along each existing chair axis, with other furniture fixed; it is not an unrestricted layout optimisation.",
                "Positive common interior ball radius is a certified penetration lower bound, not the maximum penetration depth.",
                "This layout uses no evaluation reference geometry or furniture construction parameters."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--stool-fit-dir", type=Path, required=True)
    parser.add_argument("--stool-source", type=Path)
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--propose", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = q.new_report_path(args.output)
    report = evaluate(args.fit_dir, args.stool_fit_dir, layout_path=args.layout, propose=args.propose,
                      stool_source=args.stool_source, report_base=output.parent)
    q.write_new_report(output, report)
    print(json.dumps({"output": str(output), "status": report["status"], "objects": report["objects"],
                      "original_failed_pairs": [row for row in report["original_observed_layout"]["pairs"] if not row["passed"]],
                      "authored_minimum_clearance_m": report["authored_layout"]["minimum_certified_clearance_m"]}, indent=2))
    raise SystemExit(0 if report["status"] == "passed" and report["quality_gate"].get("passed") is True else 2)


if __name__ == "__main__":
    main()
