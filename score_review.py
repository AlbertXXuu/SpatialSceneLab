"""Score sidecar-declared ownership on the frozen B1 development mesh. MIT.

Ownership is recovered before reference labels are loaded. This measures the
declared final partition, not native geometry preservation or human effort.
The parsing helpers need NumPy only; Open3D is loaded by the scoring command.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np


def _same_hash(actual, expected, name):
    if (not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected)
            or actual != expected):
        raise ValueError(f"{name} SHA256 mismatch or missing anchor")


def validate_bindings(bundle, identity, outcome, published_hash, bundle_hash, glb_hash):
    """Bind metadata to already-hashed files and the frozen result record."""
    if (bundle.get("schema") != "spatial-scene-lab.surface.v1"
            or bundle.get("status") != "pass" or bundle.get("units") != "m"
            or bundle.get("up_axis") != "Y"
            or identity.get("schema") != "spatial-scene-lab.edit-identity.v1"
            or identity.get("units") != "m" or identity.get("up_axis_glb") != "Y"):
        raise ValueError("requires B1 metre/Y-up bundle and edit identity schemas")
    if (bundle.get("scene_id") != outcome.get("scene_id")
            or bundle.get("triangles") != outcome.get("triangles")
            or bundle.get("vertices") != outcome.get("vertices")):
        raise ValueError("bundle scene identity or frozen mesh counts differ")
    methods = [item for item in outcome["methods"] if item.get("method") == "b1"]
    if len(methods) != 1 or methods[0].get("status") != "completed":
        raise ValueError("requires one completed frozen B1 method")
    method = methods[0]
    provenance = bundle["provenance"]
    if (provenance.get("reference_used") is not False or provenance.get("mesh_changed") is not False
            or bundle["partition"].get("parameters") != method["parameters"]):
        raise ValueError("bundle must retain the frozen prior-only B1 parameters")
    for actual, expected, name in (
        (identity.get("input_manifest_sha256"), bundle_hash, "identity input manifest"),
        (identity.get("asset_sha256"), glb_hash, "identity GLB"),
        (provenance.get("published_results_sha256"), published_hash, "published results"),
        (provenance.get("source_surface_sha256"), outcome.get("surface_sha256"), "source mesh"),
        (provenance.get("source_surface_manifest_sha256"), outcome.get("surface_manifest_sha256"),
         "source surface manifest"),
        (provenance.get("source_observation_sha256"), outcome.get("scene_sha256"), "source observations"),
        (bundle.get("input_scene_sha256"), outcome.get("scene_sha256"), "bundle observations"),
        (provenance["face_owner"].get("sha256"), method.get("face_owner_sha256"), "frozen B1 ownership"),
    ):
        _same_hash(actual, expected, name)


def restore_owners(bundle, identity, face_maps, baseline, object_ids):
    """Map native IDs through initial B1 parts; never inspect reference labels."""
    baseline = np.asarray(baseline)
    if (baseline.ndim != 1 or baseline.dtype.kind not in "iu" or not len(baseline)
            or np.any(baseline < -2) or np.any(baseline >= len(object_ids))):
        raise ValueError("baseline must contain one valid integer owner per frozen face")
    if (len(set(object_ids)) != len(object_ids) or any(
            not isinstance(value, str) or not value or value in {"environment", "unresolved"}
            for value in object_ids)):
        raise ValueError("frozen object IDs must be unique and unreserved")
    count = len(baseline)
    if bundle.get("triangles") != count:
        raise ValueError("bundle triangle count differs from frozen ownership")
    labels = {"environment": -1, "unresolved": -2, **dict(zip(object_ids, range(len(object_ids))))}
    expected_parts = [name for name, label in labels.items() if np.any(baseline == label)]
    parts = bundle["parts"]
    if [part["id"] for part in parts] != expected_parts or set(face_maps) != set(expected_parts):
        raise ValueError("initial parts differ from frozen B1 owner order")
    sources = identity["sources"]
    source_ids = [source["id"] for source in sources]
    if len(set(source_ids)) != len(source_ids) or set(source_ids) != set(expected_parts):
        raise ValueError("identity sources differ from initial bundle parts")
    by_source = {source["id"]: source for source in sources}
    native_to_frozen = np.empty(count, dtype=np.int64)
    offset = 0
    for part in parts:
        name = part["id"]
        label = labels[name]
        expected = np.flatnonzero(baseline == label)
        mapping = np.asarray(face_maps[name])
        if (mapping.shape != expected.shape or mapping.dtype.kind not in "iu"
                or not np.array_equal(mapping, expected)):
            raise ValueError(f"source face mapping differs from frozen B1: {name}")
        if (type(part.get("owner_label")) is not int or part["owner_label"] != label
                or part.get("triangles") != len(expected)
                or part.get("native_source_face_offset") != offset
                or part["source_indices"]["face"].get("count") != len(expected)):
            raise ValueError(f"bundle owner label, count or native offset differs: {name}")
        source = by_source[name]
        if (source.get("offset") != offset or source.get("faces") != len(expected)):
            raise ValueError(f"identity source offset or count differs: {name}")
        _same_hash(source.get("sha256"), part.get("sha256"), f"identity source {name}")
        native_to_frozen[offset:offset + len(expected)] = mapping
        offset += len(expected)
    objects = identity["objects"]
    final_ids = [obj["id"] for obj in objects]
    if len(set(final_ids)) != len(final_ids) or set(final_ids) != set(expected_parts):
        raise ValueError("final objects contain duplicate, missing or unknown target IDs")
    reviewed = np.empty(count, dtype=np.int32)
    seen = np.zeros(count, dtype=bool)
    for obj in objects:
        values = obj.get("source_face_ids")
        if not isinstance(values, list) or any(type(value) is not int for value in values):
            raise ValueError("source face IDs must be integer lists")
        if any(value < 0 or value >= count for value in values):
            raise ValueError("source face ID is outside the frozen inventory")
        ids = np.asarray(values, dtype=np.int64)
        if len(np.unique(ids)) != len(ids) or np.any(seen[ids]):
            raise ValueError("duplicate final source face ID")
        seen[ids] = True
        reviewed[native_to_frozen[ids]] = labels[obj["id"]]
    if not seen.all():
        raise ValueError("missing final source face IDs")
    return reviewed


def measure_review(scene, owners):
    """Use fixed geometry for both partitions, regardless of exported transforms."""
    import evaluate_partition
    from partition_baselines import triangle_areas

    baseline = next(method for method in scene["methods"] if method["method"] == "b1")
    parameters = baseline["original"]["parameters"]
    scores = {}
    for name, labels in (("baseline", baseline["owners"]), ("reviewed", owners)):
        scores[name] = evaluate_partition.evaluate_all_objects(
            scene["vertices"], scene["triangles"], labels,
            scene["reference_vertices"], scene["reference_triangles"], scene["reference_owner"],
            matching_tolerance=parameters["matching_tolerance_m"],
            samples_per_face=parameters["samples_per_face"], object_ids=scene["model_ids"],
            ambiguity_margin=parameters["ambiguity_margin_m"])
    changed = owners != baseline["owners"]
    scores["changes"] = {"faces": int(changed.sum()),
                         "area_m2": float(triangle_areas(scene["vertices"], scene["triangles"])[changed].sum()),
                         "area_scope": "all changed frozen faces, including unscorable surface"}
    old = {target["object_id"]: target for target in scores["baseline"]["targets"]}
    fields = ("area_iou", "precision", "recall")
    scores["comparison"] = [{"object_id": target["object_id"], **{
        field: {"baseline": old[target["object_id"]][field], "reviewed": target[field],
                "delta": target[field] - old[target["object_id"]][field]
                if target[field] is not None and old[target["object_id"]][field] is not None else None}
        for field in fields}} for target in scores["reviewed"]["targets"]]
    before, after = [scores[name]["macro_target_area_iou"] for name in ("baseline", "reviewed")]
    scores["macro_target_area_iou"] = {"baseline": before, "reviewed": after,
                                       "delta": after - before if before is not None and after is not None else None}
    return scores


def score_review(results, suite, published_results, bundle, identity, glb, output):
    # Existing exporters/scorers import Open3D; parsing helpers above do not.
    import export_partition as export
    import reevaluate_development as frozen

    requested = [Path(value).absolute() for value in
                 (results, suite, published_results, bundle, identity, glb, output)]
    for path in requested:
        frozen._no_links(path)
    results, suite, published_results, bundle, identity, glb, output = [path.resolve() for path in requested]
    for source in (results, suite, bundle.parent, identity.parent, glb.parent):
        export.separate_output(output, source)
    if published_results == output or output in published_results.parents:
        raise ValueError("output must not contain the published result anchor")
    if glb.suffix.lower() != ".glb":
        raise ValueError("review asset must be a GLB file")
    published_hash = export.sha256(published_results)
    export.checked_file(results, "development-results.json", published_hash)
    published = export.read_json(published_results)
    manifest_path = export.checked_file(suite, "suite-manifest.json", published.get("input_manifest_sha256"))
    manifest = export.read_json(manifest_path)
    if (published.get("schema") != "spatial-scene-lab.development.v1" or published.get("status") != "completed"
            or manifest.get("schema") != "spatial-scene-lab.fixture-suite.v1"
            or published.get("input_generator_sha256") != manifest.get("generator_sha256")):
        raise ValueError("requires the completed frozen run and its original fixture suite")
    for name in ("evaluate_partition.py", "partition_baselines.py", "run_development.py"):
        export.checked_file(Path(__file__).parent, name, published["code_sha256"].get(name))
    doc, record = export.read_json(bundle), export.read_json(identity)
    records = {item["scene_id"]: item for item in manifest["scenes"]}
    outcomes = {item["scene_id"]: item for item in published["scenes"]}
    if (len(records) != len(manifest["scenes"]) or len(outcomes) != len(published["scenes"])
            or set(records) != set(outcomes) or doc.get("scene_id") not in records):
        raise ValueError("suite and result scene identities must correspond uniquely")
    outcome = outcomes[doc["scene_id"]]
    bundle_hash, identity_hash, glb_hash = [export.sha256(path) for path in (bundle, identity, glb)]
    validate_bindings(doc, record, outcome, published_hash, bundle_hash, glb_hash)
    _same_hash(doc["provenance"].get("exporter_sha256"), export.sha256(Path(export.__file__)), "bundle exporter")
    directory = export.member(results, doc["scene_id"])
    _, observation, _, baseline, _ = export.validate_source(directory, outcome)
    owner_file = doc["provenance"]["face_owner"]
    frozen_owner = np.load(export.checked_file(bundle.parent, owner_file["path"], owner_file["sha256"]),
                           allow_pickle=False)
    if not np.array_equal(frozen_owner, baseline):
        raise ValueError("bundle B1 owners differ from frozen result")
    face_maps = {}
    input_hashes = {"published_results": published_hash, "suite_manifest": export.sha256(manifest_path),
                    "bundle": bundle_hash, "identity": identity_hash, "glb": glb_hash,
                    "bundle_files": {owner_file["path"]: owner_file["sha256"]}}
    for part in doc["parts"]:
        export.checked_file(bundle.parent, part["path"], part["sha256"])
        input_hashes["bundle_files"][part["path"]] = part["sha256"]
        for kind in ("face", "vertex"):
            item = part["source_indices"][kind]
            path = export.checked_file(bundle.parent, item["path"], item["sha256"])
            input_hashes["bundle_files"][item["path"]] = item["sha256"]
            if kind == "face":
                face_maps[part["id"]] = np.load(path, allow_pickle=False)
    owners = restore_owners(doc, record, face_maps, baseline, [obj["uuid"] for obj in observation["objects"]])
    # Only now load the independent reference, after declared ownership is fixed.
    scene = frozen._check_scene(suite, results, records[doc["scene_id"]], outcome)
    scores = measure_review(scene, owners)
    original = next(method["original"] for method in scene["methods"] if method["method"] == "b1")
    mismatches = frozen._compare(original, scores["baseline"], doc["scene_id"], "b1")
    if mismatches:
        raise ValueError(f"recomputed baseline differs from frozen scores: {mismatches}")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "reviewed-face-owner.npy", owners, allow_pickle=False)
    summary = {"schema": "spatial-scene-lab.review-score.v1", "status": "evaluated",
               "scene_id": scene["scene_id"], "scope": "sidecar-declared ownership on the unchanged frozen mesh",
               "limitations": ["does not verify native geometry preservation",
                               "does not establish human operation, effort or usability"],
               "human_measurements": {"status": "not_measured", "operator": None, "active_seconds": None,
                                      "selection_operations": None, "undo_count": None},
               "baseline_consistency": "pass", "input_sha256": input_hashes,
               "frozen_artifacts": {**scene["hashes"], "reference_mesh_sha256": scene["reference_mesh_sha256"]},
               "code_sha256": {name: export.sha256(Path(__file__).parent / name) for name in
                               ("score_review.py", "export_partition.py", "reevaluate_development.py",
                                "evaluate_partition.py", "partition_baselines.py", "run_development.py")},
               "owner_array": {"path": "reviewed-face-owner.npy",
                               "sha256": export.sha256(output / "reviewed-face-owner.npy")}, **scores}
    export.save_json(output / "review-score.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "suite", "published-results", "bundle", "identity", "glb", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    try:
        result = score_review(**vars(args))
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        parser.exit(1, f"Review scoring rejected: {type(error).__name__}: {error}\n")
    print(f"{result['scene_id']}: {result['changes']['faces']} changed faces; "
          f"macro IoU {result['macro_target_area_iou']}")


if __name__ == "__main__":
    main()
