"""Audit prepared RGB-D contracts and fixed Table0 convention controls. MIT.

Reads only local prepared inputs. The output contains numeric evidence/hashes;
no points, camera matrices, fitted plane coordinates, images or assets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

import audit_real_tabletop as baseline
from reproduce_real_tearoom import TARGET, verify_source
import scan_pipeline as scan
from fit_stool import sha256


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / 'docs/tea-room-source-manifest.json'
CONVENTIONS = ['original_axial', 'radial_range', 'integer_plus_half', 'resize_pixel_centres']


def validate_paths(input_path, scene_path, observations_path, output_path, manifest_path):
    source = Path(input_path).resolve(strict=True)
    scene = Path(scene_path).resolve(strict=True)
    observations = Path(observations_path).resolve(strict=True)
    manifest = Path(manifest_path).resolve(strict=True)
    output_path = Path(output_path)
    output = output_path.resolve()
    if not source.is_dir() or not all(p.is_file() for p in (scene, observations, manifest)):
        raise ValueError('Prepared input directory, scene, observations and manifest files are required')
    if output_path.is_symlink() or output.exists():
        raise ValueError('Output directory must be new; previous evidence is never overwritten')
    for tree in (source, scene.parent, observations.parent):
        if output == tree or tree in output.parents or output in tree.parents:
            raise ValueError('Output and input trees must be separate')
    if output == manifest or output in manifest.parents:
        raise ValueError('Output must remain separate from the source manifest')
    if output == ROOT or ROOT in output.parents and not (ROOT / '.local') in output.parents:
        raise ValueError('Repository outputs must stay under its ignored .local directory')
    return source, scene, observations, output, manifest


def fingerprints(scene_path, observations_path, manifest_path):
    return {
        'scene_sha256': sha256(scene_path), 'observations_sha256': sha256(observations_path),
        'manifest_sha256': sha256(manifest_path),
        'code_sha256': {name: sha256(ROOT / name) for name in (
            Path(__file__).name, 'scan_pipeline.py', 'audit_real_tabletop.py',
            'reproduce_real_tearoom.py', 'fit_stool.py')},
    }


def checked_matrix(value, expected, label):
    try:
        actual = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{label} must match the raw source') from error
    if (actual.shape != expected.shape or not np.isfinite(actual).all() or
            not np.array_equal(actual, expected)):
        raise ValueError(f'{label} differs from the raw source')


def audit(input_path, scene_path, observations_path, output_path,
          manifest_path=DEFAULT_MANIFEST, object_uuid=TARGET):
    source, scene_path, observations_path, output, manifest_path = validate_paths(
        input_path, scene_path, observations_path, output_path, manifest_path)
    initial = fingerprints(scene_path, observations_path, manifest_path)
    source_check = verify_source(source, manifest_path)
    scene, index, points, frames, candidates, local, band, train, valid = baseline.load_inputs(
        scene_path, observations_path, object_uuid)
    if scene.get('schema') != scan.SCHEMA or Path(scene.get('source', '')).resolve() != source:
        raise ValueError('Observation scene must identify the supplied raw input source')
    if not isinstance(scene.get('input_files'), dict) or not scene['input_files']:
        raise ValueError('Observation scene needs its frozen raw input hashes')
    for name, expected in scene['input_files'].items():
        member = (source / name).resolve(strict=True)
        if source not in member.parents or sha256(member) != expected:
            raise ValueError('Observation scene raw input hash differs or escapes source')
    params = scene['parameters']
    if (type(params.get('pixel_step')) is not int or params['pixel_step'] < 1 or
            type(params.get('frame_step')) is not int or params['frame_step'] < 1 or
            type(params.get('min_confidence')) is not int or params['min_confidence'] not in (0, 1, 2) or
            params.get('depth_scale') != 1000):
        raise ValueError('Expected positive sampling steps, confidence 0/1/2 and millimetre depth scale')
    raw_frames = sorted(source.glob('frame_*.json'))[::params['frame_step']]
    if [p.name for p in raw_frames] != [f['source'] for f in scene['frames']]:
        raise ValueError('Declared frame inventory differs from raw source sampling')
    with np.load(observations_path, allow_pickle=False) as archive:
        uv, colors = archive['pixel_uv'].copy(), archive['colors'].copy()
    if (uv.shape != (len(points), 2) or not np.issubdtype(uv.dtype, np.integer) or
            colors.shape != points.shape or colors.dtype != np.uint8):
        raise ValueError('Observations require integer Nx2 pixels and uint8 Nx3 colours')
    variants = {name: points.copy() for name in CONVENTIONS[1:]}
    dimensions, levels, records, times = Counter(), Counter(), [], []
    max_inverse = max_orthogonal = 0.
    for metadata in scene['frames']:
        fid = metadata['frame_index']
        frame = scan.load_frame(source / metadata['source'])
        depth, k, pose = (frame[key] for key in ('depth_mm', 'intrinsics_depth', 'camera_to_world_cv'))
        checked_matrix(metadata['intrinsics_depth'], k, 'Depth intrinsics metadata')
        checked_matrix(metadata['camera_to_world_cv'], pose, 'Camera pose metadata')
        if (metadata['color_size'] != list(frame['color_size']) or
                metadata['depth_size'] != [depth.shape[1], depth.shape[0]] or
                frame['record']['frame_index'] != fid):
            raise ValueError('Frame index or image dimensions differ from raw source')
        confidence_levels, counts = np.unique(frame['confidence'], return_counts=True)
        if not np.isin(confidence_levels, [0, 1, 2]).all():
            raise ValueError('Raw confidence contains levels outside 0/1/2')
        for level, count in zip(confidence_levels, counts):
            levels[int(level)] += int(count)
        regenerated, rgb, pixels = scan.backproject_rgbd(
            depth, frame['color'], k, pose, frame['confidence'], params['min_confidence'],
            params['max_depth_m'], params['pixel_step'])
        mask = frames == fid
        if len(regenerated) != int(mask.sum()) or len(regenerated) != metadata['retained_observations']:
            raise ValueError('Retained observation count differs from raw source reprojection')
        if not np.array_equal(pixels, uv[mask]):
            raise ValueError('Stored observation pixels differ from raw source reprojection')
        if not np.array_equal(regenerated, points[mask]):
            raise ValueError('Stored observation points differ from raw source reprojection')
        if not np.array_equal(rgb, colors[mask]):
            raise ValueError('Stored observation colours differ from raw source reprojection')
        # Independent transcription of the pinned upstream row-major loader.
        upstream_pose = np.array(frame['record']['cameraPoseARFrame']).reshape(4, 4)
        upstream_pose[:3, 1:3] *= -1
        upstream_k = np.array(frame['record']['intrinsics']).reshape(3, 3)
        sx, sy = depth.shape[1] / frame['color_size'][0], depth.shape[0] / frame['color_size'][1]
        upstream_k[0, [0, 2]] *= sx
        upstream_k[1, [1, 2]] *= sy
        checked_matrix(k, upstream_k, 'Upstream depth intrinsics convention')
        checked_matrix(pose, upstream_pose, 'Upstream camera pose convention')
        camera = (points[mask] - pose[:3, 3]) @ np.linalg.inv(pose[:3, :3]).T
        projected = camera @ k.T
        if len(projected):
            max_inverse = max(max_inverse, float(np.max(np.abs(projected[:, :2] / projected[:, 2, None] - uv[mask]))))
        max_orthogonal = max(max_orthogonal, float(np.max(np.abs(pose[:3, :3].T @ pose[:3, :3] - np.eye(3)))))
        z = depth[pixels[:, 1], pixels[:, 0]].astype(float) / 1000
        rays = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(k).T
        centred_k = k.copy()
        centred_k[0, 2] += .5 * sx - .5
        centred_k[1, 2] += .5 * sy - .5
        alternative = {
            'radial_range': rays / np.linalg.norm(rays, axis=1)[:, None] * z[:, None],
            'integer_plus_half': np.column_stack([pixels + .5, np.ones(len(pixels))]) @ np.linalg.inv(k).T * z[:, None],
            'resize_pixel_centres': np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(centred_k).T * z[:, None],
        }
        for name, xyz in alternative.items():
            variants[name][mask] = xyz @ pose[:3, :3].T + pose[:3, 3]
        dimensions[f'RGB {frame["color_size"][0]}x{frame["color_size"][1]}; depth {depth.shape[1]}x{depth.shape[0]}'] += 1
        times.append(float(frame['record']['time']))
        records.append({'frame_index': fid, 'retained_observations': len(regenerated)})
    time_steps = np.diff(times)
    if not np.isfinite(times).all() or not len(time_steps) or not np.all(time_steps > 0):
        raise ValueError('Source frame timestamps must be finite and strictly increasing')
    transform = np.asarray(scene['objects'][index]['transform_world'])
    controls = {}
    for name, xyz in {'original_axial': points, **variants}.items():
        positions = baseline.local_points(xyz, transform)
        coefficients, _ = baseline.fit_plane(positions, frames, train)
        controls[name] = {
            'training_count': int(train.sum()), 'validation_count': int(valid.sum()),
            'validation': baseline.summarize(positions[valid], frames[valid], coefficients),
            'mean_point_displacement_m': float(np.linalg.norm(xyz[band] - points[band], axis=1).mean()),
            'adopted': name == 'original_axial',
        }
    if verify_source(source, manifest_path) != source_check or fingerprints(scene_path, observations_path, manifest_path) != initial:
        raise ValueError('Inputs, source inventory or audit implementation changed during evaluation')
    summary = {
        'schema': 'alvenx.table0-input-contract-summary.v1', 'passed': True,
        'target': object_uuid, 'source': source_check, 'fingerprints': initial,
        'input_unchanged': True, 'frames': len(records), 'observations': len(points),
        'dimensions': dict(dimensions), 'depth_format': 'uint16 millimetres',
        'confidence_level_counts': dict(levels), 'frame_records': records,
        'timestamps_strictly_increasing': True,
        'timestamp_interval_s': {'minimum': float(time_steps.min()), 'median': float(np.median(time_steps)), 'maximum': float(time_steps.max())},
        'max_regenerated_point_component_error_m': 0., 'max_regenerated_color_channel_error': 0,
        'max_inverse_pixel_error': max_inverse, 'max_pose_orthogonality_error': max_orthogonal,
        'max_upstream_k_component_difference': 0., 'max_upstream_pose_component_difference': 0.,
        'counterfactual_protocol': 'Original train/validation memberships and frozen estimator; one training-only plane per declared convention; no correction adopted.',
        'counterfactuals': controls, 'baseline_parameters': baseline.PARAMETERS.copy(),
        'limitations': [
            'Agreement with the pinned loader does not establish physical sensor calibration.',
            'A single saved frame time does not establish RGB/depth exposure synchronization.',
            'Convention controls are development diagnostics, not geometric acceptance or physical accuracy.',
            'Raw scan, camera transforms, points and plane coordinates are excluded from this summary.',
        ],
    }
    encoded = json.dumps(summary, indent=2, allow_nan=False) + '\n'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'summary.json').write_text(encoded, encoding='utf-8')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--object', default=TARGET)
    args = parser.parse_args(argv)
    result = audit(args.input, args.scene, args.observations, args.output, args.manifest, args.object)
    print(json.dumps({key: result[key] for key in ('passed', 'frames', 'observations', 'input_unchanged')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
