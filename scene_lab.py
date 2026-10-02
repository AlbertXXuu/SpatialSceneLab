"""Small local scene experiment. MIT; requires only NumPy besides Python."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

UNIT = {"m": 1.0, "cm": 0.01}
# lab +Z-up -> glTF +Y-up; right handed, determinant +1.
LAB_TO_GLTF = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
# Original GeometryAudit display conversion, NOT gravity alignment.
MODEL_TO_GLTF = np.diag([1., -1., -1.])
BOX = np.array([[x, y, z] for x in [-.5, .5] for y in [-.5, .5] for z in [-.5, .5]], dtype="<f4")
TRIANGLES = np.array([0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5,
                      0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6,
                      0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3], dtype="<u2")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def rotation(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-6:
        raise ValueError("rotation_xyzw must be a finite unit quaternion")
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def vector(value, name, positive=False):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all() or (positive and np.any(result <= 0)):
        raise ValueError(f"{name} requires three finite {'positive ' if positive else ''}numbers")
    return result


def validate(scene):
    if scene.get("schema") != "spatial-scene-lab/1" or scene.get("units") not in UNIT:
        raise ValueError("unsupported schema or units (use explicit m or cm)")
    if scene.get("frame") != "lab-right-handed-z-up" or scene.get("kind") != "controlled_fixture":
        raise ValueError("editable fixture must declare lab-right-handed-z-up and controlled_fixture")
    if not scene.get("objects"):
        raise ValueError("scene has no objects")
    ids = set()
    for obj in scene["objects"]:
        if not isinstance(obj.get("id"), str) or not obj["id"] or obj["id"] in ids:
            raise ValueError("object IDs must be unique nonempty strings")
        ids.add(obj["id"])
        if obj.get("geometry") != "box" or obj.get("origin") != "authored_controlled_geometry":
            raise ValueError("this experiment accepts authored box geometry only")
        vector(obj["translation"], "translation")
        vector(obj["dimensions"], "dimensions", positive=True)
        rotation(obj["rotation_xyzw"])
        color = np.asarray(obj["color_rgba"], dtype=float)
        if color.shape != (4,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 1)):
            raise ValueError("color_rgba requires four finite values in [0, 1]")
    return scene


def load_scene(path):
    return validate(json.loads(Path(path).read_text(encoding="utf-8")))


def object_matrix(obj, units):
    transform = np.eye(4)
    transform[:3, :3] = np.einsum("ij,jk->ik", LAB_TO_GLTF, rotation(obj["rotation_xyzw"])) * (vector(obj["dimensions"], "dimensions") * UNIT[units])
    transform[:3, 3] = np.einsum("ij,j->i", LAB_TO_GLTF, vector(obj["translation"], "translation") * UNIT[units])
    return transform


def edit(scene, object_id, displacement, dimensions, input_units):
    validate(scene)
    edited = copy.deepcopy(scene)
    matching = [obj for obj in edited["objects"] if obj["id"] == object_id]
    if not matching:
        raise ValueError(f"unknown object: {object_id}")
    obj = matching[0]
    factor = UNIT[input_units] / UNIT[scene["units"]]
    obj["translation"] = (vector(obj["translation"], "translation") + vector(displacement, "displacement") * factor).tolist()
    obj["dimensions"] = (vector(dimensions, "dimensions", positive=True) * factor).tolist()
    edited.setdefault("edits", []).append({"object_id": object_id, "operation": "translate-and-set-dimensions", "input_units": input_units,
                                           "displacement": list(displacement), "dimensions": list(dimensions)})
    return validate(edited)


def write_glb(path, document, binary):
    binary += b"\0" * (-len(binary) % 4)
    document["buffers"] = [{"byteLength": len(binary)}]
    encoded = json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")
    encoded += b" " * (-len(encoded) % 4)
    data = struct.pack("<III", 0x46546C67, 2, 28 + len(encoded) + len(binary))
    data += struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
    data += struct.pack("<II", len(binary), 0x004E4942) + binary
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)


def read_glb(path):
    data = Path(path).read_bytes()
    if len(data) < 28 or struct.unpack_from("<III", data) != (0x46546C67, 2, len(data)):
        raise ValueError("invalid GLB header")
    offset, document, binary = 12, None, None
    while offset < len(data):
        size, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        if size % 4 or offset + size > len(data):
            raise ValueError("invalid GLB chunk")
        content = data[offset:offset+size]
        if kind == 0x4E4F534A:
            document = json.loads(content)
        elif kind == 0x004E4942:
            binary = content
        offset += size
    if document is None or binary is None:
        raise ValueError("GLB requires JSON and binary chunks")
    return document, binary


def accessor(document, binary, index):
    item = document["accessors"][index]
    view = document["bufferViews"][item["bufferView"]]
    dtype = {5126: "<f4", 5123: "<u2", 5125: "<u4", 5121: "u1"}[item["componentType"]]
    count = {"VEC3": 3, "SCALAR": 1}[item["type"]]
    if "byteStride" in view:
        raise ValueError("interleaved GLB arrays are outside this small reader")
    offset = view.get("byteOffset", 0) + item.get("byteOffset", 0)
    end = offset + np.dtype(dtype).itemsize * count * item["count"]
    if end > len(binary):
        raise ValueError("accessor exceeds buffer")
    return np.frombuffer(binary, dtype=dtype, count=count*item["count"], offset=offset).reshape(item["count"], count)


def export_fixture(scene, output, source_hash=None):
    validate(scene)
    positions, indices = BOX.tobytes(), TRIANGLES.tobytes()
    doc = {"asset": {"version": "2.0", "generator": "SpatialSceneLab local experiment", "extras": {
               "source_frame": scene["frame"], "output_units": "m", "kind": scene["kind"], "source_scene": scene,
               "source_to_gltf_rotation": LAB_TO_GLTF.tolist(),
               "source_package_sha256": source_hash or hashlib.sha256(json.dumps(scene, sort_keys=True).encode()).hexdigest(),
               "source_hash_basis": "source_file_bytes" if source_hash else "sorted_json_scene"}},
           "scene": 0, "scenes": [{"nodes": list(range(len(scene["objects"]))) }],
           "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(positions), "target": 34962},
                           {"buffer": 0, "byteOffset": len(positions), "byteLength": len(indices), "target": 34963}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3", "min": [-.5]*3, "max": [.5]*3},
                         {"bufferView": 1, "componentType": 5123, "count": 36, "type": "SCALAR"}],
           "nodes": [], "meshes": [], "materials": []}
    for index, obj in enumerate(scene["objects"]):
        doc["nodes"].append({"name": obj["id"], "mesh": index, "matrix": object_matrix(obj, scene["units"]).T.reshape(-1).tolist(),
                             "extras": {"origin": obj["origin"], "physical_scale_status": "authored_known_dimensions"}})
        doc["meshes"].append({"name": obj["id"], "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4, "material": index}]})
        doc["materials"].append({"doubleSided": True, "pbrMetallicRoughness": {"baseColorFactor": obj["color_rgba"], "metallicFactor": 0, "roughnessFactor": .8}})
    write_glb(output, doc, positions + indices)


def verify_fixture(scene, path, source_hash=None):
    """Decode binary vertex buffers and column-major transforms; compare physical vertices."""
    validate(scene)
    doc, binary = read_glb(path)
    if doc["asset"]["extras"]["output_units"] != "m" or len(doc["nodes"]) != len(scene["objects"]):
        raise ValueError("unit declaration or object count changed")
    if source_hash and doc["asset"]["extras"].get("source_package_sha256") != source_hash:
        raise ValueError("source package hash changed")
    max_error = 0.
    for obj, node in zip(scene["objects"], doc["nodes"]):
        if node["name"] != obj["id"]:
            raise ValueError("object identity changed")
        primitive = doc["meshes"][node["mesh"]]["primitives"][0]
        positions = accessor(doc, binary, primitive["attributes"]["POSITION"])
        indices = accessor(doc, binary, primitive["indices"]).ravel()
        if len(indices) != 36 or np.any(indices >= len(positions)):
            raise ValueError("invalid box topology")
        matrix = np.asarray(node["matrix"]).reshape(4, 4).T
        if not np.isfinite(positions).all() or not np.isfinite(matrix).all():
            raise ValueError("non-finite GLB geometry or transform")
        exported_world = np.einsum("ij,nj->ni", matrix[:3, :3], positions) + matrix[:3, 3]
        # Independent source-frame expectation, including actual units, rotation and dimensions.
        source = np.einsum("ij,nj->ni", rotation(obj["rotation_xyzw"]), BOX.astype(float) * np.asarray(obj["dimensions"])) + np.asarray(obj["translation"])
        expected = np.einsum("ij,nj->ni", LAB_TO_GLTF, source * UNIT[scene["units"]])
        error = float(np.max(np.abs(exported_world - expected)))
        if not np.isfinite(error):
            raise ValueError("non-finite geometry comparison")
        max_error = max(max_error, error)
    if max_error > 1e-6:
        raise ValueError(f"geometry/units/frame mismatch: {max_error} m")
    return {"status": "pass", "objects": len(scene["objects"]), "max_vertex_error_m": max_error, "sha256": sha256(path)}


def probe_environment():
    modules = {name: bool(importlib.util.find_spec(name)) for name in ["numpy", "torch", "depth_anything_3", "open3d", "gtsam", "rosbags", "vismatch", "trimesh", "matplotlib"]}
    gpu = None
    executable = shutil.which("nvidia-smi")
    if executable:
        result = subprocess.run([executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        gpu = {"returncode": result.returncode, "inventory": result.stdout.strip()}
    blender = shutil.which("blender")
    checked = []
    if platform.system() == "Windows":
        for root in [Path("C:/Program Files"), Path("C:/Program Files (x86)")]:
            for folder in root.glob("*Blender*"):
                checked.append(str(folder))
                found = list(folder.glob("*/blender.exe")) + list(folder.glob("blender.exe"))
                if found:
                    blender = str(found[0])
    missing = [name for name in ["torch", "depth_anything_3", "open3d", "gtsam", "rosbags", "vismatch"] if not modules[name]]
    return {"platform": platform.platform(), "python": sys.version, "executable": sys.executable, "numpy": np.__version__,
            "modules": modules, "gpu_inventory_only": gpu, "blender_executable": blender, "blender_program_files_folders": checked,
            "scarf": {"execution": "not_run", "reason": "offline stack unavailable" if missing else "not executed by environment probe",
                      "missing_imports": missing, "weights_downloaded": False, "six_frame_cuda_inference": "not_run"},
            "cuda_inference_this_experiment": False}


def reconstruction(artifact_root, output):
    """Read original hashed TUM output. Retain derived local audit, never redistribute photos."""
    start = time.perf_counter()
    root, output = Path(artifact_root).resolve(), Path(output).resolve()
    if root == output or root in output.parents:
        raise ValueError("output must be separate from read-only historical artifacts")
    manifest_path = root / "tum-p0/artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {item["path"]: item for item in manifest["entries"]}
    used = [f"run-518/raw-view-{i:02}.npz" for i in range(4)] + ["run-518/scene.glb", "selection.json"]
    verified = []
    for name in used:
        file = root / "tum-p0" / name
        actual = sha256(file)
        if actual != entries[name]["sha256"]:
            raise ValueError(f"source hash mismatch: {name}")
        verified.append({"path": name, "bytes": file.stat().st_size, "sha256": actual})
    selection = json.loads((root / "tum-p0/selection.json").read_text(encoding="utf-8"))
    samples, cameras, coordinate_errors = [], [], []
    for i in range(4):
        with np.load(root / "tum-p0/run-518" / f"raw-view-{i:02}.npz", allow_pickle=False) as data:
            world, camera = data["pts3d"][0], data["pts3d_cam"][0]
            pose = data["camera_poses"][0].astype(float)
            mask = data["non_ambiguous_mask"][0] & np.isfinite(world).all(-1) & np.isfinite(camera).all(-1)
            if not mask.any():
                raise ValueError("original reconstruction has no finite valid points")
            selected = np.flatnonzero(mask.reshape(-1))[::max(1, int(mask.sum()) // 1500)]
            world_sample = world.reshape(-1, 3)[selected].astype(float)
            camera_sample = camera.reshape(-1, 3)[selected].astype(float)
            calculated_world = np.einsum("ij,nj->ni", pose[:3, :3], camera_sample) + pose[:3, 3]
            coordinate_errors.append(float(np.max(np.abs(calculated_world - world_sample))))
            samples.append(world_sample)
            cameras.append({"view": i, "camera_to_world_opencv": pose.tolist(), "intrinsics": data["intrinsics"][0].tolist(),
                            "model_metric_scaling_factor": data["metric_scaling_factor"].tolist(), "selected_point_count": len(selected),
                            "rgb_timestamp": selection["frames"][i]["rgb_timestamp"],
                            "input_rgb_sha256": selection["frames"][i]["sha256"]})
    if max(coordinate_errors) > 1e-4:
        raise ValueError("camera/world frame mismatch in actual reconstruction")
    model_points = np.concatenate(samples)
    exported = np.einsum("ij,nj->ni", MODEL_TO_GLTF, model_points).astype("<f4")
    count = len(exported)
    # No texture/RGB arrays. A sampled unsegmented geometry/camera integration artifact.
    metadata = {"kind": "actual_reconstruction_coordinate_integration", "source_model": "MapAnything Apache recorded TUM P0",
                "source_manifest_sha256": sha256(manifest_path), "sources": verified,
                "units": "m", "scale_status": "model_predicted_metric_scale_unvalidated_for_object_dimensions",
                "source_world_frame": "MapAnything OpenCV world; optical cameras X-right Y-down Z-forward",
                "world_to_export_rotation": MODEL_TO_GLTF.tolist(), "gravity_alignment": "unknown",
                "segmentation": "none", "editable_recovered_objects": 0, "cameras": cameras}
    nodes = [{"name": "unsegmented_reconstruction_points", "mesh": 0, "extras": {"segmentation": "none"}}]
    for camera in cameras:
        # glTF camera local looks -Z, +Y-up. Both world and camera changes are explicit.
        world_change = np.eye(4); world_change[:3, :3] = MODEL_TO_GLTF
        camera_change = np.eye(4); camera_change[:3, :3] = MODEL_TO_GLTF
        converted = np.einsum("ij,jk,kl->il", world_change, np.asarray(camera["camera_to_world_opencv"]), camera_change)
        nodes.append({"name": f"source_camera_{camera['view']}", "matrix": converted.T.reshape(-1).tolist(),
                      "extras": {"camera_convention": "glTF optical basis; exact intrinsics in manifest"}})
    doc = {"asset": {"version": "2.0", "generator": "SpatialSceneLab", "extras": metadata}, "scene": 0,
           "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes,
           "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": exported.nbytes, "target": 34962}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": count, "type": "VEC3", "min": exported.min(0).tolist(), "max": exported.max(0).tolist()}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": 0}]}]}
    output.mkdir(parents=True, exist_ok=True)
    export_path = output / "reconstruction-integration.glb"
    write_glb(export_path, doc, exported.tobytes())
    parsed, binary = read_glb(export_path)
    recovered = accessor(parsed, binary, 0).astype(float)
    roundtrip = np.einsum("ij,nj->ni", MODEL_TO_GLTF.T, recovered)
    error = float(np.max(np.abs(roundtrip - model_points)))
    if error > 1e-5:
        raise ValueError("reconstruction export coordinate round trip failed")
    source_glb, _ = read_glb(root / "tum-p0/run-518/scene.glb")
    report = {"status": "pass", "sources": verified, "point_count": count, "camera_count": len(cameras),
              "original_glb_mesh_count": len(source_glb.get("meshes", [])), "original_glb_read": True,
              "camera_world_max_abs_error_model_m": max(coordinate_errors), "point_export_roundtrip_max_abs_error_model_m": error,
              "wall_seconds": time.perf_counter() - start, "new_gpu_inference": False, "metadata": metadata,
              "derived_glb_sha256": sha256(export_path), "limit": "coordinate integration only; no metric validation, semantic segmentation or recovered-object edit"}
    save_json(output / "reconstruction-checks.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export"); export.add_argument("scene"); export.add_argument("output")
    modify = commands.add_parser("edit"); modify.add_argument("scene"); modify.add_argument("output")
    modify.add_argument("--object", required=True); modify.add_argument("--translate", type=float, nargs=3, required=True)
    modify.add_argument("--dimensions", type=float, nargs=3, required=True); modify.add_argument("--unit", choices=list(UNIT), required=True)
    verify = commands.add_parser("verify"); verify.add_argument("scene"); verify.add_argument("glb")
    probe = commands.add_parser("environment"); probe.add_argument("output")
    original = commands.add_parser("reconstruction"); original.add_argument("--artifact-root", required=True); original.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "export":
        scene = load_scene(args.scene); export_fixture(scene, args.output, sha256(args.scene)); print(json.dumps(verify_fixture(scene, args.output, sha256(args.scene))))
    elif args.command == "edit":
        scene = load_scene(args.scene); updated = edit(scene, args.object, args.translate, args.dimensions, args.unit)
        updated["source_scene_sha256"] = sha256(args.scene); save_json(args.output, updated)
        print(json.dumps({"status": "edited", "object": args.object, "output": args.output}))
    elif args.command == "verify":
        print(json.dumps(verify_fixture(load_scene(args.scene), args.glb, sha256(args.scene))))
    elif args.command == "environment":
        result = probe_environment(); save_json(args.output, result); print(json.dumps(result))
    elif args.command == "reconstruction":
        result = reconstruction(args.artifact_root, args.output); print(json.dumps({k: v for k, v in result.items() if k not in ["metadata", "sources"]}))


if __name__ == "__main__":
    main()
