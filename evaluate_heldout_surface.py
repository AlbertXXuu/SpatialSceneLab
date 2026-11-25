"""Compare fused surfaces against the same unused ScannerApp depth rays. MIT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import open3d as o3d

from reproduce_real_tearoom import sha256, verify_source
from scan_pipeline import SCHEMA, box_candidates, load_frame


MODELS = ('baseline', 'candidate')
THRESHOLDS_M = (.010, .020)
DEFAULT_MANIFEST = Path(__file__).resolve().parent / 'docs/tea-room-source-manifest.json'


class RayTotals:
    def __init__(self):
        self.valid = 0
        self.hits = np.zeros(len(MODELS), dtype=np.int64)
        self.support = np.zeros((len(MODELS), len(THRESHOLDS_M)), dtype=np.int64)
        self.common_errors = []

    def add(self, depth, hits):
        predicted = np.isfinite(hits) & (hits > 0)
        errors = np.abs(hits - depth)
        self.valid += len(depth)
        self.hits += predicted.sum(axis=1)
        self.support += np.column_stack([
            np.count_nonzero(predicted & (errors <= threshold), axis=1)
            for threshold in THRESHOLDS_M
        ])
        self.common_errors.append(errors[:, predicted.all(axis=0)])

    def result(self):
        common = np.concatenate(self.common_errors, axis=1)
        percentiles = np.percentile(common * 1000, [50, 95], axis=1).T if common.shape[1] else None
        result = {'valid_depth_rays': self.valid, 'common_hit_rays': common.shape[1]}
        for index, name in enumerate(MODELS):
            hits = int(self.hits[index])
            within_10, within_20 = map(int, self.support[index])
            result[name] = {
                'hits': hits,
                'misses': self.valid - hits,
                'hit_percent': 100 * hits / self.valid if self.valid else None,
                'within_10mm': within_10,
                'within_10mm_percent_of_valid': 100 * within_10 / self.valid if self.valid else None,
                'within_20mm': within_20,
                'within_20mm_percent_of_valid': 100 * within_20 / self.valid if self.valid else None,
                'common_hit_absolute_error_p50_mm': float(percentiles[index, 0]) if percentiles is not None else None,
                'common_hit_absolute_error_p95_mm': float(percentiles[index, 1]) if percentiles is not None else None,
            }
        return result


def prepare_holdout(source, observations_path):
    observations = json.loads(observations_path.read_text(encoding='utf-8'))
    if (observations.get('schema') != SCHEMA or observations.get('edits') != [] or
            Path(observations['source']).resolve() != source or 'alignment_control' in observations):
        raise ValueError('Expected fresh unedited observations from the supplied scan')

    by_index = {int(path.stem.removeprefix('frame_')): path for path in source.glob('frame_*.json')}
    train_frames = observations['frames']
    train = sorted(frame['frame_index'] for frame in train_frames)
    holdout = sorted(index for index in by_index if index % 2)
    if not train or not holdout or train != sorted(set(by_index) - set(holdout)):
        raise ValueError('Fusion input must include all and only even-index source frames')
    if any(frame['source'] != by_index[frame['frame_index']].name for frame in train_frames):
        raise ValueError('Fusion frame names differ from the source inventory')
    for name, expected in observations['input_files'].items():
        path = (source / name).resolve(strict=True)
        if source not in path.parents or sha256(path) != expected:
            raise ValueError(f'Fusion input changed: {name}')
    return observations, [(index, by_index[index]) for index in holdout]


def open_surface(directory, observations_hash, observations):
    manifest = json.loads((directory / 'surface-scene.json').read_text(encoding='utf-8'))
    if (manifest.get('status') != 'pass' or manifest.get('input_scene_sha256') != observations_hash or
            manifest.get('cameras') != observations['frames'] or manifest.get('alignment_control')):
        raise ValueError(f'{directory.name}: expected a surface fused from the supplied source observations')
    for name in ('depth_scale', 'min_confidence', 'max_depth_m'):
        if manifest['parameters'][name] != observations['parameters'][name]:
            raise ValueError(f'{directory.name}: unexpected fusion {name}')

    path = directory / 'surface.ply'
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) != manifest['vertices'] or len(mesh.triangles) != manifest['triangles']:
        raise ValueError(f'{directory.name}: unexpected vertex or triangle count')
    if not mesh.has_triangles() or not np.isfinite(np.asarray(mesh.vertices)).all():
        raise ValueError(f'{directory.name}: mesh must contain finite triangles')
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene, {
        'parameters': manifest['parameters'],
        'vertices': manifest['vertices'],
        'triangles': manifest['triangles'],
        'mesh_sha256': sha256(path),
        'surface_manifest_sha256': sha256(directory / 'surface-scene.json'),
        'fusion_elapsed_seconds': manifest.get('elapsed_seconds'),
    }


def measured_rays(frame, params):
    raw = frame['depth_mm']
    selected = ((raw > 0) & (raw != 65535) & (raw <= params['max_depth_m'] * 1000) &
                (frame['confidence'] >= params['min_confidence']))
    rows, cols = np.nonzero(selected)
    pixels = np.column_stack((cols, rows, np.ones(len(rows))))
    pose = frame['camera_to_world_cv']
    directions = pixels @ np.linalg.inv(frame['intrinsics_depth']).T @ pose[:3, :3].T
    origins = np.broadcast_to(pose[:3, 3], directions.shape)
    # The optical-axis component before rotation is one, so t_hit is axial depth.
    rays = np.column_stack((origins, directions)).astype(np.float32)
    depth = raw[rows, cols].astype(np.float64) / 1000
    return rays, depth, origins + directions * depth[:, None]


def evaluate(source, observations_path, baseline_path, candidate_path, manifest_path):
    started = time.perf_counter()
    source_check = verify_source(source, manifest_path)
    observations, frames = prepare_holdout(source, observations_path)
    observations_hash = sha256(observations_path)
    scenes, mesh_records = zip(*(open_surface(path, observations_hash, observations)
                                 for path in (baseline_path, candidate_path)))
    objects = observations['objects']
    names = [obj['name'] for obj in objects]
    if len(names) != len(set(names)):
        raise ValueError('RoomPlan object names must be unique for the per-object report')
    categories = {name: np.array([index for index, obj in enumerate(objects) if obj['category'] == name])
                  for name in sorted({obj['category'] for obj in objects})}
    total = RayTotals()
    by_object = {name: RayTotals() for name in names}
    by_category = {name: RayTotals() for name in categories}
    per_frame = []
    membership = {'none': 0, 'one': 0, 'competing': 0}
    for index, path in frames:
        frame = load_frame(path)
        if int(frame['record'].get('frame_index', index)) != index:
            raise ValueError(f'Held-out frame index differs from filename: {path.name}')
        rays, depth, points = measured_rays(frame, observations['parameters'])
        tensor = o3d.core.Tensor(rays)
        hits = np.stack([scene.cast_rays(tensor, nthreads=4)['t_hit'].numpy() for scene in scenes])
        total.add(depth, hits)
        frame_total = RayTotals()
        frame_total.add(depth, hits)
        per_frame.append({'frame_index': index, **frame_total.result()})

        inside = box_candidates(points, objects, margin_m=0)
        box_counts = inside.sum(axis=1)
        unique = box_counts == 1
        membership['none'] += int(np.count_nonzero(box_counts == 0))
        membership['one'] += int(np.count_nonzero(unique))
        membership['competing'] += int(np.count_nonzero(box_counts > 1))
        for column, name in enumerate(names):
            selected = unique & inside[:, column]
            by_object[name].add(depth[selected], hits[:, selected])
        for name, columns in categories.items():
            selected = unique & inside[:, columns].any(axis=1)
            by_category[name].add(depth[selected], hits[:, selected])

    return {
        'schema': 'spatial-scene-lab.heldout-surface.v1',
        'scope': 'Same-capture unused sensor depth, fixed captured poses; no independent physical surface or semantic labels.',
        'source_inventory': source_check,
        'observations_sha256': observations_hash,
        'evaluation_code_sha256': sha256(Path(__file__)),
        'open3d_version': o3d.__version__,
        'fusion_frame_indices': sorted(frame['frame_index'] for frame in observations['frames']),
        'heldout_frame_indices': [index for index, _ in frames],
        'max_depth_m': observations['parameters']['max_depth_m'],
        'minimum_sensor_confidence': observations['parameters']['min_confidence'],
        'method': 'First positive ray hit versus measured axial depth. Missing hits stay in the support denominator; error quantiles use rays hit by both meshes.',
        'strict_spatial_groups': 'Measured depth point inside exactly one unexpanded RoomPlan object box. Overlaps are excluded from object/category groups and retained in scene totals.',
        'roomplan_membership_counts': membership,
        'meshes': dict(zip(MODELS, mesh_records)),
        'scene': total.result(),
        'by_roomplan_category': {name: accumulator.result() for name, accumulator in by_category.items()},
        'by_roomplan_object': {name: accumulator.result() for name, accumulator in by_object.items()},
        'by_frame': per_frame,
        'evaluation_elapsed_seconds': time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve(strict=True)
    baseline, candidate = args.baseline.resolve(strict=True), args.candidate.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or source in output.parents or baseline == candidate:
        parser.error('Use distinct surface directories and a new report outside the raw scan directory')
    result = evaluate(source, args.observations.resolve(strict=True), baseline, candidate, args.source_manifest)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['scene'], indent=2))


if __name__ == '__main__':
    main()
