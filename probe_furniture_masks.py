"""Compare local 2D furniture masks with coarse RGB-D RoomPlan box proxies."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image
import torch
from torch.nn.functional import interpolate
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn

from scan_pipeline import backproject_rgbd, box_candidates, load_frame, parse_roomplan


SAMPLE_PERIOD = 4
SCORE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
CATEGORY_LABELS = {'Chair': 'chair', 'Table': 'dining table'}


def frame_result(frame_path, objects, targets, model, transform, labels, device):
    frame = load_frame(frame_path)
    points, _, uv = backproject_rgbd(
        frame['depth_mm'], frame['color'], frame['intrinsics_depth'],
        frame['camera_to_world_cv'], frame['confidence'], min_confidence=1,
    )
    # Zero margin makes membership and ambiguity explicit at RoomPlan box boundaries.
    boxes = box_candidates(points, objects, margin_m=0)
    box_count = boxes.sum(axis=1)
    exclusive = boxes & (box_count == 1)[:, None]
    with Image.open(frame_path.with_suffix('.jpg')) as rgb:
        input_rgb = transform(rgb.convert('RGB')).to(device)
    predictions = model([input_rgb])[0]

    class_rows, object_rows = {}, {}
    for category, label in labels.items():
        indices = targets[category]
        keep = (predictions['labels'] == label) & (predictions['scores'] >= SCORE_THRESHOLD)
        masks = predictions['masks'][keep]
        if len(masks):
            # Apply the capture's image-size scaling and preserve depth-pixel centres.
            resized = interpolate(masks, frame['depth_mm'].shape, mode='bilinear', align_corners=False)
            depth_masks = (resized[:, 0].cpu().numpy()[:, uv[:, 1], uv[:, 0]] >= MASK_THRESHOLD)
        else:
            depth_masks = np.zeros((0, len(uv)), dtype=bool)
        union = depth_masks.any(axis=0)
        matching_boxes = exclusive[:, indices].any(axis=1)
        class_rows[category] = {
            'instances': len(masks),
            'mask_depth_points': int(union.sum()),
            'exclusive_class_box_depth_points': int(matching_boxes.sum()),
            'mask_and_exclusive_class_box': int((union & matching_boxes).sum()),
            'mask_in_other_exclusive_box': int((union & (box_count == 1) & ~matching_boxes).sum()),
            'mask_in_multiple_boxes': int((union & (box_count > 1)).sum()),
            'mask_outside_all_boxes': int((union & (box_count == 0)).sum()),
        }
        for index in indices:
            proxy = exclusive[:, index]
            object_rows[objects[index]['name']] = {
                'category': category,
                'exclusive_box_depth_points': int(proxy.sum()),
                'mask_union_overlap': int((union & proxy).sum()),
                'best_instance_overlap': int((depth_masks & proxy).sum(axis=1).max(initial=0)),
            }
    return {
        'frame': int(frame_path.stem.removeprefix('frame_')),
        'valid_depth_points': len(uv),
        'classes': class_rows,
        'objects': object_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=Path, required=True, help='Local ScannerApp scan directory')
    parser.add_argument('--cache', type=Path, required=True, help='Torch Hub cache for official weights')
    parser.add_argument('--output', type=Path, required=True, help='New aggregate JSON file')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; select --device cpu to use the installed CPU runtime')

    objects, _ = parse_roomplan(args.scan / 'room.usdz')
    objects = [obj for obj in objects if obj['geometry_kind'] == 'box']
    frames = sorted(path for path in args.scan.glob('frame_*.json')
                    if path.stem.removeprefix('frame_').isdecimal())
    # Freeze the calendar-independent sample before reading any per-frame model scores.
    sample = [path for path in frames if int(path.stem.removeprefix('frame_')) % SAMPLE_PERIOD == 0]
    targets = {category: [i for i, obj in enumerate(objects) if obj['category'] == category]
               for category in CATEGORY_LABELS}
    if not sample or not any(targets.values()):
        parser.error('a non-empty fixed frame sample and at least one chair or table box are required')
    names = [objects[i]['name'] for indices in targets.values() for i in indices]
    if len(set(names)) != len(names):
        parser.error('target object names must be unique for the aggregate receipt')

    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    torch.hub.set_dir(str(args.cache.resolve()))
    filename = Path(urlsplit(weights.url).path).name
    checkpoint = args.cache / 'checkpoints' / filename
    if not checkpoint.exists():
        # TorchVision 0.20's constructor verifies the upstream filename hash on download.
        weights.get_state_dict(progress=False, check_hash=True)
    with checkpoint.open('rb') as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    if not checkpoint_sha256.startswith(Path(filename).stem.rsplit('-', 1)[1]):
        raise ValueError('cached model fails the official TorchVision checkpoint hash prefix')
    model = maskrcnn_resnet50_fpn(weights=weights, box_score_thresh=SCORE_THRESHOLD).to(args.device).eval()
    labels = {category: weights.meta['categories'].index(label) for category, label in CATEGORY_LABELS.items()}
    transform = weights.transforms()
    rows = []
    with torch.inference_mode():
        for frame_path in sample:
            rows.append(frame_result(frame_path, objects, targets, model, transform, labels, args.device))
            print(f'Processed fixed frame {rows[-1]["frame"]:05d}', flush=True)
    receipt = {
        'schema': 'alvenx.furniture-mask-feasibility.v1',
        'method': {
            'model': 'TorchVision Mask R-CNN ResNet-50 FPN COCO_V1',
            'torch': torch.__version__,
            'torchvision': torchvision.__version__,
            'weights_url': weights.url,
            'weights_sha256': checkpoint_sha256,
            'device': args.device,
            'original_frame_count': len(frames),
            'sample_rule': 'original numeric frame index modulo 4 equals 0',
            'detection_score_threshold': SCORE_THRESHOLD,
            'mask_threshold_after_depth_resize': MASK_THRESHOLD,
            'min_depth_confidence': 1,
            'max_depth_m': 6,
            'box_margin_m': 0,
            'proxy': 'valid registered depth points inside exactly one RoomPlan object box',
            'limit': 'Coarse box agreement measures feasibility; it supplies no ground-truth masks or reconstruction score.',
        },
        'frames': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
