"""Direct partition controls on an unchanged reconstructed surface (S0). MIT.

Object boxes and finite structure exclusion are input priors, never reference
labels. All methods return one label per existing face and retain its geometry.
"""
from __future__ import annotations

import numpy as np

from scan_pipeline import assign_owners


SAMPLING_RULE = "hammersley-base2-square-root-triangle-v1"


def mesh_arrays(vertices, triangles):
    """Validate metre coordinates and integer topology without changing them."""
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be a finite Nx3 array")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in "iu":
        raise ValueError("triangles must be an integer Mx3 array")
    if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ValueError("triangle index is outside vertices")
    return vertices, triangles.astype(np.int64, copy=False)


def triangle_areas(vertices, triangles):
    corners = vertices[triangles]
    return np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                   corners[:, 2] - corners[:, 0]), axis=1) * 0.5


def barycentric_samples(count):
    """Fixed equal-weight, uniform-area quadrature; no labels or RNG are used.

    Hammersley points ((i+.5)/count, radical_inverse_base2(i+1)) in the
    unit square are mapped by sqrt(u) to barycentric triangle coordinates.
    Each point represents area/count. These are finite area estimates, not
    exact clipping; convergence must be checked when label boundaries matter.
    """
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("samples_per_face must be a positive integer")
    indices = np.arange(1, count + 1, dtype=np.uint64)
    inverse = np.zeros(count, dtype=np.float64)
    factor = 0.5
    while np.any(indices):
        inverse += (indices & 1) * factor
        indices >>= 1
        factor *= 0.5
    root = np.sqrt((np.arange(count) + 0.5) / count)
    return np.column_stack((1 - root, root * (1 - inverse), root * inverse))


def partition(vertices, triangles, objects, structures, ownership, method):
    """Return B0/B1/B2 labels: -1 environment, -2 unresolved, >=0 object.

    B0 exactly preserves the released three-vertices-same-object rule: all
    remaining faces go to environment. B1 uses the centroid's unique prior.
    B2 accepts the sampled majority (including environment) only at or above
    ``b2_min_majority_fraction`` (default .6); otherwise the face is unresolved.
    Overlap samples remain unresolved and cannot vote for an object. B2's
    ``b2_samples_per_face`` defaults to 49 and is configurable in ownership.
    Reference geometry/labels are deliberately absent from this interface.
    """
    vertices, triangles = mesh_arrays(vertices, triangles)
    method = str(method).lower()
    if method not in {"b0", "b1", "b2"}:
        raise ValueError("method must be b0, b1, or b2")
    margin = float(ownership["box_margin_m"])
    tolerance = float(ownership["structure_tolerance_m"])
    if not np.isfinite([margin, tolerance]).all() or min(margin, tolerance) < 0:
        raise ValueError("ownership tolerances must be finite and nonnegative")
    parameters = {"method": method, "box_margin_m": margin,
                  "structure_tolerance_m": tolerance,
                  "reference_used": False, "mesh_changed": False}
    if method == "b0":
        vertex_owner, _ = assign_owners(vertices, objects, structures, margin, tolerance)
        corners = vertex_owner[triangles]
        face_owner = np.full(len(triangles), -1, dtype=np.int32)
        same = (corners[:, 0] >= 0) & np.all(corners == corners[:, :1], axis=1)
        face_owner[same] = corners[same, 0]
        parameters["rule"] = "released-three-vertices-same-nonnegative-owner"
    elif method == "b1":
        face_owner, _ = assign_owners(vertices[triangles].mean(axis=1), objects,
                                      structures, margin, tolerance)
        parameters["rule"] = "centroid-unique-owner"
    else:
        count = ownership.get("b2_samples_per_face", 49)
        barycentric = barycentric_samples(count)
        threshold = float(ownership.get("b2_min_majority_fraction", 0.6))
        if not np.isfinite(threshold) or not 0.5 < threshold <= 1:
            raise ValueError("b2_min_majority_fraction must be in (.5, 1]")
        face_owner = np.full(len(triangles), -2, dtype=np.int32)
        # Bound temporary point/candidate arrays for large TSDF surfaces.
        chunk_faces = max(1, 65536 // count)
        for start in range(0, len(triangles), chunk_faces):
            stop = min(start + chunk_faces, len(triangles))
            points = np.einsum("sk,fkc->fsc", barycentric, vertices[triangles[start:stop]])
            sampled, _ = assign_owners(points.reshape(-1, 3), objects,
                                       structures, margin, tolerance)
            sampled = sampled.reshape(stop - start, count)
            best_count = np.zeros(stop - start, dtype=np.int64)
            best_label = np.full(stop - start, -2, dtype=np.int32)
            for label in [-2, -1, *range(len(objects))]:
                votes = np.count_nonzero(sampled == label, axis=1)
                better = votes > best_count
                best_count[better] = votes[better]
                best_label[better] = label
            accepted = best_count / count >= threshold
            face_owner[start:stop][accepted] = best_label[accepted]
        parameters.update(rule="uniform-area-majority-with-unresolved-threshold",
                          samples_per_face=int(count), sampling_rule=SAMPLING_RULE,
                          min_majority_fraction=threshold,
                          sample_weight="face_area_m2 / samples_per_face",
                          overlap_vote="unresolved (-2)")
    return {"face_owner": face_owner.astype(np.int32, copy=False), "parameters": parameters,
            "counts": {"faces": len(triangles),
                       "environment_faces": int(np.count_nonzero(face_owner == -1)),
                       "unresolved_faces": int(np.count_nonzero(face_owner == -2)),
                       "object_faces": int(np.count_nonzero(face_owner >= 0))}}
