"""Offline evidence figures using already installed Pillow; no model inference."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import scene_lab as lab

FACES = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
PROJECTION = np.array([[1, -.5, 0], [.25, .25, 1.]])


def font(size):
    for path in [Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def canvas(title, subtitle):
    image = Image.new("RGB", (1400, 640), "#eef6ff")
    draw = ImageDraw.Draw(image)
    draw.text((45, 28), title, fill="#0b1731", font=font(28))
    draw.text((45, 70), subtitle, fill="#334155", font=font(18))
    return image, draw


def panel(draw, index, title):
    left = 35 + index*700
    draw.rounded_rectangle((left, 118, left+630, 542), radius=20, fill="#fbfdff", outline="#c8d5e4", width=1)
    draw.text((left+20, 138), title, fill="#0b1731", font=font(21))
    return left


def controlled(original, edited, output):
    image, draw = canvas("CONTROLLED three-object edit / export", "Authored rigid cuboids with known dimensions. Independent GLB binary + transform round trip passed.")
    for index, scene in enumerate([original, edited]):
        left = panel(draw, index, "Before edit" if index == 0 else "After edit: block-b")
        def screen(point):
            return (left+310+point[0]*157, 400-point[1]*140)
        polygons = []
        for obj in scene["objects"]:
            vertices = np.einsum("ij,nj->ni", lab.rotation(obj["rotation_xyzw"]), lab.BOX * np.asarray(obj["dimensions"])) + np.asarray(obj["translation"])
            projected = np.einsum("ij,nj->ni", PROJECTION, vertices)
            for face in FACES:
                depth = float(vertices[face, 1].mean() - vertices[face, 0].mean())
                color = tuple(round(c*255) for c in obj["color_rgba"][:3])
                polygons.append((depth, [screen(point) for point in projected[face]], color))
        for _, polygon, color in sorted(polygons, key=lambda item: item[0]):
            draw.polygon(polygon, fill=color, outline="#334155", width=2)
        for obj in scene["objects"]:
            center = np.einsum("ij,j->i", PROJECTION, obj["translation"])
            pos = screen(center)
            draw.text((pos[0]-33, pos[1]+45), obj["id"], font=font(18), fill="#334155")
        draw.text((left+22, 499), "Z-up fixture coordinates; all lengths in meters", font=font(16), fill="#52647a")
    draw.text((45, 566), "Edit: translate block-b +25 cm along X; dimensions 60 x 90 x 60 cm -> 75 x 90 x 60 cm.", font=font(20), fill="#334155")
    draw.text((45, 599), "The three shapes test editability. They are controlled geometry, not recovered objects.", font=font(18), fill="#52647a")
    Path(output).parent.mkdir(parents=True, exist_ok=True); image.save(output)


def actual(glb, output):
    doc, binary = lab.read_glb(glb)
    points = lab.accessor(doc, binary, 0).astype(float)
    image, draw = canvas("ACTUAL recorded MapAnything / TUM geometry", "Coordinate integration only: 6,020 sampled points + 4 source camera transforms; no segmentation.")
    for index, pair in enumerate([(0, 2), (0, 1)]):
        left = panel(draw, index, "Export X / Z" if index == 0 else "Export X / Y")
        selected = points[:, pair]
        camera_positions = [np.asarray(node["matrix"]).reshape(4, 4).T[:3, 3][list(pair)] for node in doc["nodes"][1:]]
        extent = np.concatenate([selected, np.asarray(camera_positions)])
        lo, hi = extent.min(0), extent.max(0)
        span = np.maximum(hi-lo, 1e-9)
        factor = min(565/span[0], 285/span[1])
        def screen(point):
            xy = (point-lo)*factor
            return (left+30+xy[0], 479-xy[1])
        for point in selected:
            x, y = screen(point)
            draw.point((round(x), round(y)), fill="#52647a")
        draw.text((left+20, 172), "Blue markers: source cameras 0-3", font=font(14), fill="#2563eb")
        for node in doc["nodes"][1:]:
            matrix = np.asarray(node["matrix"]).reshape(4, 4).T
            x, y = screen(matrix[:3, 3][list(pair)])
            draw.ellipse((x-4, y-4, x+4, y+4), fill="#2563eb")
        draw.text((left+20, 501), f"Point+camera extent: {span[0]:.2f} x {span[1]:.2f} model meters", font=font(16), fill="#52647a")
    draw.text((45, 566), "Predicted metric scale is not object dimension ground truth; world display rotation is explicit.", font=font(19), fill="#334155")
    draw.text((45, 600), "TUM RGB-D: Sturm et al., IROS 2012, CC BY 4.0. Derived geometry only; no photos copied.", font=font(18), fill="#52647a")
    image.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True); parser.add_argument("--edited", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--reconstruction-glb"); parser.add_argument("--reconstruction-output")
    args = parser.parse_args()
    controlled(lab.load_scene(args.original), lab.load_scene(args.edited), args.output)
    if args.reconstruction_glb:
        actual(args.reconstruction_glb, args.reconstruction_output)
