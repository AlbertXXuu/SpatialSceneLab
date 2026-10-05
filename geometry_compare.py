"""One-to-one oriented triangle geometry comparison in metres. MIT.

Exporters may reorder vertices or split color/normal seams. Comparing vertex
indices after a nearest-point lookup is ambiguous for almost coincident source
vertices. This verifier instead matches complete triangle instances, including
multiplicity, under a maximum corner-distance tolerance. It compares geometric
surface incidence; it does not establish welded vertex, seam or attribute identity.
"""
from __future__ import annotations

from collections import deque
from itertools import product
import math

import numpy as np


_NEIGHBORS = tuple(product((-1, 0, 1), repeat=3))
_CYCLIC = np.asarray(((0, 1, 2), (1, 2, 0), (2, 0, 1)))
_REVERSED = np.asarray(((0, 2, 1), (2, 1, 0), (1, 0, 2)))


def _triangles(vertices, faces, tolerance):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("nonempty three-coordinate vertex array required")
    if not np.isfinite(vertices).all():
        raise ValueError("non-finite vertex coordinate")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("nonempty triangular face array required")
    if faces.dtype.kind not in "iu" or (faces < 0).any() or (faces >= len(vertices)).any():
        raise ValueError("face indices must be integers within the vertex array")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if not np.isfinite(vertices / tolerance).all():
            raise ValueError("coordinates and tolerance exceed spatial index range")
    return vertices[faces]


def _corner_errors(original, returned, permutations):
    # Rows are candidate triangles; each may match through a cyclic rotation.
    differences = original[None, None, :, :] - returned[:, permutations, :]
    distances = np.linalg.norm(differences, axis=-1)
    return distances.max(axis=-1).min(axis=-1)


def _maximum_matching(adjacency, returned_count):
    """Augmenting paths, rather than greedy matching, preserve multiplicity."""
    source_match = [-1] * len(adjacency)
    returned_match = [-1] * returned_count
    for start in range(len(adjacency)):
        queue = deque([start])
        visited_source = {start}
        parent_returned = {}
        endpoint = None
        while queue and endpoint is None:
            source = queue.popleft()
            for returned, _, _ in adjacency[source]:
                if returned in parent_returned:
                    continue
                parent_returned[returned] = source
                owner = returned_match[returned]
                if owner == -1:
                    endpoint = returned
                    break
                if owner not in visited_source:
                    visited_source.add(owner)
                    queue.append(owner)
        if endpoint is None:
            continue
        # Follow the discovered path backwards; no recursion-depth dependency.
        while endpoint != -1:
            source = parent_returned[endpoint]
            previous = source_match[source]
            source_match[source] = endpoint
            returned_match[endpoint] = source
            endpoint = previous
    return source_match, returned_match


def compare_triangle_geometry(expected_vertices, expected_faces, actual_vertices,
                              actual_faces, tolerance=5e-5, *, return_correspondence=False):
    """Compare oriented world-coordinate triangle multisets within tolerance.

    A match requires each corresponding corner to lie within ``tolerance``
    metres. Only cyclic corner permutations are eligible, preserving winding
    when geometry resolves it. Very thin or degenerate triangles can also match
    reversed permutations within that tolerance; those matches are explicitly
    counted as winding-ambiguous. No proximity clusters or source-index collapse
    are used. Candidate centroids are only a search accelerator, never a verdict.
    Opt-in correspondence lists use expected face order. Corner permutations
    index the matched actual face's corners; unmatched faces have -1 / None.
    """
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    original = _triangles(expected_vertices, expected_faces, tolerance)
    returned = _triangles(actual_vertices, actual_faces, tolerance)
    buckets = {}
    returned_centers = (returned / 3).sum(axis=1)
    original_centers = (original / 3).sum(axis=1)
    for index, center in enumerate(returned_centers):
        key = tuple(math.floor(float(value) / tolerance) for value in center)
        buckets.setdefault(key, []).append(index)
    adjacency = []
    for triangle, center in zip(original, original_centers):
        key = tuple(math.floor(float(value) / tolerance) for value in center)
        candidates = [index for offset in _NEIGHBORS
                      for index in buckets.get(tuple(key[axis] + offset[axis] for axis in range(3)), ())]
        edges = []
        if candidates:
            candidate_triangles = returned[candidates]
            distances = _corner_errors(triangle, candidate_triangles, _CYCLIC)
            eligible = np.flatnonzero(distances <= tolerance)
            if len(eligible):
                reversed_distances = _corner_errors(triangle, candidate_triangles[eligible], _REVERSED)
                edges = [(candidates[index], float(distances[index]), bool(reverse <= tolerance))
                         for index, reverse in zip(eligible, reversed_distances)]
                edges.sort(key=lambda edge: (edge[1], edge[0]))
        adjacency.append(edges)
    source_match, returned_match = _maximum_matching(adjacency, len(returned))
    matched = []
    for edges, index in zip(adjacency, source_match):
        if index != -1:
            matched.append(next(edge for edge in edges if edge[0] == index))
    missing = [index for index, match in enumerate(source_match) if match == -1]
    extra = [index for index, match in enumerate(returned_match) if match == -1]
    result = {
        "status": "pass" if not missing and not extra else "fail",
        "method": "one_to_one_oriented_triangle_geometry",
        "tolerance_m": tolerance,
        "expected_face_instances": len(original),
        "actual_face_instances": len(returned),
        "matched_face_instances": len(matched),
        "missing_face_instances": len(missing),
        "extra_face_instances": len(extra),
        "missing_face_indices_sample": missing[:10],
        "extra_face_indices_sample": extra[:10],
        "maximum_matched_corner_distance_m": max((edge[1] for edge in matched), default=None),
        "winding_ambiguous_matched_faces": sum(edge[2] for edge in matched),
        "scope": "oriented triangle geometry with multiplicity; welded vertices, seams and attributes unverified",
    }
    if return_correspondence:
        permutations = []
        for triangle, actual_index in zip(original, source_match):
            if actual_index == -1:
                permutations.append(None)
            else:
                errors = np.linalg.norm(triangle[None, :, :] - returned[actual_index][_CYCLIC],
                                        axis=-1).max(axis=-1)
                permutations.append(_CYCLIC[int(errors.argmin())].tolist())
        result["matched_actual_face_indices"] = source_match
        result["matched_actual_corner_permutations"] = permutations
    return result
