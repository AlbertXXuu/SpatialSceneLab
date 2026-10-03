"""Read-only diagnostic of the existing before.blend / before.glb comparison."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector
from mathutils.kdtree import KDTree
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_meshes():
    return {obj["alvenx_id"]: {
        "local_vertices": np.asarray([list(vertex.co) for vertex in obj.data.vertices], dtype=float),
        "matrix_world": obj.matrix_world.copy(),
    } for obj in bpy.context.scene.objects if obj.type == "MESH"}


def tree_of(vertices):
    tree = KDTree(len(vertices))
    for index, point in enumerate(vertices):
        tree.insert(Vector(point), index)
    tree.balance()
    return tree


def mapping(tree, vertices):
    return [tree.find(Vector(point))[1] for point in vertices]


def face_counter(faces, mapped):
    return Counter(tuple(sorted(mapped[index] for index in face)) for face in faces)


def coordinate_faces(vertices, faces):
    return Counter(tuple(sorted(tuple(vertices[index]) for index in face)) for face in faces)


def analyze(original, returned, source_raw, returned_raw):
    source = np.asarray(original["vertices"])
    actual = np.asarray(returned["vertices"])
    reference = tree_of(source)
    original_mapping = mapping(reference, source)
    actual_mapping = mapping(reference, actual)
    expected_faces = face_counter(original["faces"], original_mapping)
    actual_faces = face_counter(returned["faces"], actual_mapping)
    missing = expected_faces - actual_faces
    extra = actual_faces - expected_faces
    unique, counts = np.unique(source, axis=0, return_counts=True)
    close_pairs = []
    exact_groups = {tuple(point): [] for point, count in zip(unique, counts) if count > 1}
    for index, point in enumerate(source):
        key = tuple(point)
        if key in exact_groups:
            exact_groups[key].append(index)
        for _, other, distance in reference.find_range(Vector(point), 5e-5):
            if other > index and distance > 0:
                close_pairs.append((index, other, float(distance)))
    # Distinct source vertices within a few micrometres form positional
    # equivalence classes only for this diagnostic; this is not a new verifier.
    parent = list(range(len(source)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    for members in exact_groups.values():
        for other in members[1:]:
            parent[find(other)] = find(members[0])
    for left, right, distance in close_pairs:
        if distance <= 5e-6:
            parent[find(right)] = find(left)
    cluster = [find(index) for index in range(len(source))]
    clustered_original = face_counter(original["faces"], [cluster[index] for index in original_mapping])
    clustered_actual = face_counter(returned["faces"], [cluster[index] for index in actual_mapping])
    clustered_missing, clustered_extra = clustered_original - clustered_actual, clustered_actual - clustered_original
    aligned_vertices = [list(source_raw["matrix_world"] @ Vector(point))
                        for point in returned_raw["local_vertices"]]
    aligned_mapping = mapping(reference, aligned_vertices)
    aligned_faces = face_counter(returned["faces"], aligned_mapping)
    source_local = source_raw["local_vertices"]
    returned_local = returned_raw["local_vertices"]
    source_local_faces = coordinate_faces(source_local, original["faces"])
    returned_local_faces = coordinate_faces(returned_local, returned["faces"])
    local_missing, local_extra = source_local_faces - returned_local_faces, returned_local_faces - source_local_faces
    # Match missing/extra index triples that differ at only one nearby vertex.
    shared_edges = {}
    for face in extra:
        for edge in ((face[0], face[1]), (face[0], face[2]), (face[1], face[2])):
            shared_edges.setdefault(edge, []).append(face)
    examples = []
    nearby_missing = 0
    for face, count in missing.items():
        candidates = set()
        for edge in ((face[0], face[1]), (face[0], face[2]), (face[1], face[2])):
            candidates.update(shared_edges.get(edge, []))
        selected = None
        for candidate in candidates:
            left = list((Counter(face) - Counter(candidate)).elements())
            right = list((Counter(candidate) - Counter(face)).elements())
            if len(left) != 1 or len(right) != 1:
                continue
            separation = float(np.linalg.norm(source[left[0]] - source[right[0]]))
            if selected is None or separation < selected[0]:
                selected = (separation, candidate, left[0], right[0])
        if selected is not None and selected[0] <= 5e-6:
            nearby_missing += count
            if len(examples) < 12:
                separation, candidate, left, right = selected
                examples.append({"missing_face": face, "extra_face": candidate,
                                 "substituted_source_vertex_indices": [left, right],
                                 "source_vertex_separation_m": separation,
                                 "source_vertex_coordinates_m": [source[left].tolist(), source[right].tolist()]})
    changed_mapping = np.asarray(actual_mapping) != np.asarray(aligned_mapping)
    distances = np.asarray([distance for _, _, distance in close_pairs])
    return {
        "original_vertices": len(source), "returned_vertices": len(actual),
        "original_faces": len(original["faces"]), "returned_faces": len(returned["faces"]),
        "exact_duplicate_coordinate_groups": len(exact_groups),
        "exact_duplicate_excess_vertices": int(sum(len(group) - 1 for group in exact_groups.values())),
        "exact_duplicate_examples": [{"coordinate_m": coordinate, "indices": indices}
                                      for coordinate, indices in list(exact_groups.items())[:8]],
        "distinct_close_pairs_under_5e_5_m": len(close_pairs),
        "distinct_close_pairs_under_5e_6_m": int(np.count_nonzero(distances <= 5e-6)),
        "minimum_distinct_close_separation_m": float(distances.min()) if len(distances) else None,
        "original_mapping_nonidentity_vertices": int(np.count_nonzero(np.asarray(original_mapping) != np.arange(len(source)))),
        "original_mapping_unique_source_indices": len(set(original_mapping)),
        "returned_mapping_unique_source_indices": len(set(actual_mapping)),
        "original_mapped_degenerate_faces": sum(count for face, count in expected_faces.items() if len(set(face)) < 3),
        "returned_mapped_degenerate_faces": sum(count for face, count in actual_faces.items() if len(set(face)) < 3),
        "world_mapping_missing_face_instances": sum(missing.values()),
        "world_mapping_extra_face_instances": sum(extra.values()),
        "missing_faces_with_nearby_single_vertex_substitution": nearby_missing,
        "substitution_examples": examples,
        "cluster_5e_6_missing_face_instances": sum(clustered_missing.values()),
        "cluster_5e_6_extra_face_instances": sum(clustered_extra.values()),
        "local_coordinate_face_counter_missing_instances": sum(local_missing.values()),
        "local_coordinate_face_counter_extra_instances": sum(local_extra.values()),
        "source_matrix_normalized_missing_face_instances": sum((expected_faces - aligned_faces).values()),
        "source_matrix_normalized_extra_face_instances": sum((aligned_faces - expected_faces).values()),
        "returned_mapping_changes_after_source_matrix_normalization": int(np.count_nonzero(changed_mapping)),
        "matrix_world_max_absolute_difference": float(np.max(np.abs(np.asarray(source_raw["matrix_world"])
                                                                       - np.asarray(returned_raw["matrix_world"])))),
        "limits": "near-coordinate clustering is diagnostic and can conceal genuine tiny topology changes; local face equality is geometric, not preserved vertex/seam identity",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    sys.path.insert(0, str(Path(args.project).resolve()))
    import blender_scene
    assets = Path(args.assets).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("diagnostic JSON must be new")
    paths = [assets / name for name in ("before.blend", "before.glb")]
    hashes = {path.name: sha(path) for path in paths}
    bpy.ops.wm.open_mainfile(filepath=str(paths[0]), load_ui=False)
    before = blender_scene.world_geometry()
    before_raw = raw_meshes()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(paths[1]))
    returned = blender_scene.world_geometry()
    returned_raw = raw_meshes()
    result = {"blender": bpy.app.version_string, "assets_sha256": hashes,
              "original_verifier_reproduced": blender_scene.compare_geometry(before, returned),
              "objects": {identity: analyze(before[identity], returned[identity], before_raw[identity], returned_raw[identity])
                          for identity in before},
              "assets_unchanged": hashes == {path.name: sha(path) for path in paths}}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("ALVENX_CONNECTIVITY_DIAGNOSTIC " + json.dumps({identity: {
        key: value for key, value in report.items() if key in (
            "world_mapping_missing_face_instances", "world_mapping_extra_face_instances",
            "local_coordinate_face_counter_missing_instances", "source_matrix_normalized_missing_face_instances",
            "returned_mapping_changes_after_source_matrix_normalization")}
        for identity, report in result["objects"].items()}))


if __name__ == "__main__":
    main()
