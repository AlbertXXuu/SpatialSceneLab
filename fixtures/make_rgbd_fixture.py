"""Create original MIT-licensed metric RGB-D rays for offline regression tests.

Ground truth is a physical scene with two solid boxes, a horizontal floor and
a back wall. Slab intersection is independent of scan_pipeline's inverse
camera projection. No third-party scan, photograph or geometry is included.
"""

import argparse
import json
from pathlib import Path
import struct
import zipfile

import numpy as np
from PIL import Image


BOXES = [
    {"name": "Target0", "uuid": "synthetic-target", "category": "Table",
     "center": [-.55, -.12, -2.0], "dimensions": [.65, .7, .55], "rgb": [225, 40, 30]},
    {"name": "Other0", "uuid": "synthetic-other", "category": "Storage",
     "center": [.6, -.15, -2.35], "dimensions": [.6, .65, .6], "rgb": [30, 210, 55]},
]


def _hit_box(origin, directions, box):
    half = np.asarray(box["dimensions"]) / 2
    lower, upper = np.asarray(box["center"]) - half, np.asarray(box["center"]) + half
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (lower - origin) / directions
        b = (upper - origin) / directions
    near = np.max(np.minimum(a, b), axis=-1)
    far = np.min(np.maximum(a, b), axis=-1)
    return np.where((far >= near) & (near > 0), near, np.inf)


def _component(box):
    x, y, z = box["center"]
    dimensions = ", ".join(str(float(v)) for v in box["dimensions"])
    matrix = f"((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), ({x}, {y}, {z}, 1))"
    return f'''#usda 1.0
(
    defaultPrim = "{box['name']}"
    metersPerUnit = 1
    upAxis = "Y"
)
def Xform "{box['name']}" (
    customData = {{
        string Category = "{box['category']}"
        string UUID = "{box['uuid']}"
    }}
)
{{
    def Cube "{box['name']}"
    {{
        double size = 1
        double3 xformOp:scale = ({dimensions})
        matrix4d xformOp:transform = {matrix}
        uniform token[] xformOpOrder = ["xformOp:transform", "xformOp:scale"]
    }}
}}
'''


def generate(output_path, resolution_scale=1):
    if not isinstance(resolution_scale, int) or not 1 <= resolution_scale <= 8:
        raise ValueError("resolution_scale must be an integer from 1 to 8")
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    width, height = 48 * resolution_scale, 32 * resolution_scale
    fx, fy, cx, cy = [v * resolution_scale for v in (40., 40., 23.5, 15.5)]
    yy, xx = np.mgrid[:height, :width]
    directions = np.stack([(xx - cx) / fx, -(yy - cy) / fy, -np.ones_like(xx)], axis=-1)
    cameras = [[-.25, 0, 0], [.25, 0, 0]]
    measured_surfaces = []
    groundtruth = {"licence": "MIT; original AlvenX synthetic test fixture",
                   "units": "m", "up_axis": "Y", "boxes": BOXES,
                   "floor_y": -.5, "back_wall_z": -3,
                   "depth_quantization_max_error_m": .0005,
                   "camera_positions": cameras, "resolution_scale": resolution_scale}
    for index, camera in enumerate(cameras):
        origin = np.asarray(camera)
        depth_m = np.full((height, width), 3.)
        labels = np.full((height, width), -1, dtype=np.int16)
        rgb = np.tile(np.array([45, 75, 170], dtype=np.uint8), (height, width, 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            floor_hit = (-.5 - origin[1]) / directions[:, :, 1]
        floor = (floor_hit > 0) & (floor_hit < depth_m)
        depth_m[floor] = floor_hit[floor]
        rgb[floor] = [150, 145, 135]
        for box_index, box in enumerate(BOXES):
            hits = _hit_box(origin, directions, box)
            closer = hits < depth_m
            depth_m[closer] = hits[closer]
            labels[closer] = box_index
            rgb[closer] = box["rgb"]
        depth_mm = np.rint(depth_m * 1000).astype(np.uint16)
        confidence = np.full((height, width), 2, dtype=np.uint8)
        depth_mm[0, 0] = 0
        depth_mm[0, 1] = 65535
        confidence[0, 2] = 0
        labels[0, :3] = -3  # known invalid, never part of observed geometry
        exact_points = origin + directions * depth_m[:, :, None]
        measured_surfaces.append(exact_points[labels != -3])
        name = f"{index:05d}"
        Image.fromarray(depth_mm).save(output / f"depth_{name}.png")
        Image.fromarray(confidence).save(output / f"conf_{name}.png")
        Image.fromarray(rgb).resize((width * 2, height * 2), Image.Resampling.NEAREST).save(
            output / f"frame_{name}.jpg", quality=100, subsampling=0)
        pose = np.eye(4)
        pose[:3, 3] = origin
        record = {"frame_index": index, "cameraPoseARFrame": pose.ravel().tolist(),
                  "intrinsics": [fx * 2, 0, cx * 2, 0, fy * 2, cy * 2, 0, 0, 1]}
        (output / f"frame_{name}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        np.save(output / f"groundtruth_labels_{name}.npy", labels)
    floor = {"name": "Floor0", "uuid": "synthetic-floor", "category": "Floor",
             "center": [0, -.50005, -1.5], "dimensions": [10, .0001, 10]}
    wall = {"name": "Wall0", "uuid": "synthetic-wall", "category": "Wall",
            "center": [0, 0, -3], "dimensions": [10, 10, .0001]}
    components = BOXES + [floor, wall]
    root = '#usda 1.0\n( metersPerUnit = 1\n upAxis = "Y" )\n'
    root += 'def Xform "room" {\n def Xform "Parametric_grp" {\n'
    for component in components:
        root += f' def Xform "{component["name"]}_grp" (prepend references = @./assets/{component["name"]}.usda@) {{ }}\n'
    root += '}\n}\n'
    with zipfile.ZipFile(output / "room.usdz", "w", compression=zipfile.ZIP_STORED) as archive:
        for member, content in [("room.usda", root)] + [(f'assets/{x["name"]}.usda', _component(x)) for x in components]:
            info = zipfile.ZipInfo(member, date_time=(2026, 10, 4, 0, 0, 0))
            # USDZ stores uncompressed members with 64-byte-aligned payloads.
            padding = (-(archive.fp.tell() + 30 + len(member.encode("utf-8")))) % 64
            if padding:
                if padding < 4:
                    padding += 64
                info.extra = struct.pack("<HH", 0x1986, padding - 4) + bytes(padding - 4)
            archive.writestr(info, content)
    (output / "groundtruth.json").write_text(json.dumps(groundtruth, indent=2) + "\n", encoding="utf-8")
    points = np.concatenate(measured_surfaces)
    header = ("# .PCD v0.7 - original analytic ray intersections\nVERSION 0.7\n"
              "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
              f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA ascii\n")
    with (output / "pointcloud.pcd").open("w", encoding="ascii", newline="\n") as stream:
        stream.write(header)
        np.savetxt(stream, points, fmt="%.9g")
    return groundtruth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution-scale", type=int, default=1)
    arguments = parser.parse_args()
    print(json.dumps(generate(arguments.output, arguments.resolution_scale), indent=2))
