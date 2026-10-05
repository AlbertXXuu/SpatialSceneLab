"""Isolated native Blender input/export regressions. MIT.

python blender_edit_controls.py --blender /path/to/blender --output NEW-DIRECTORY
Each control runs in a fresh background Blender. This is scripted engineering
evidence, not interactive user validation or a whole-machine offline test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
import traceback


CASES = (
    "no_face_selection", "all_faces_selection", "self_destination", "missing_id", "multi_object_edit",
    "object_mode_selection_exactness", "nonfinite_transform", "source_sha_preflight",
    "duplicate_uuid_preflight", "duplicate_current_id", "hidden_region_export", "excluded_view_layer_export",
    "blocked_collection_export", "sidecar_write_failure", "pair_replace_failure",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fixture(directory):
    """Disconnected colored triangles make exact face transfer visible."""
    directory.mkdir(parents=True, exist_ok=True)
    parts = []
    for identity, count, offset, rgb in (
        ("environment", 4, 0, (192, 51, 31)),
        ("target", 2, 2, (33, 162, 81)),
        ("other", 2, 4, (41, 78, 213)),
    ):
        points = [(offset + index * .25 + x, y, z)
                  for index in range(count) for x, y, z in ((0, 0, 0), (.1, 0, 0), (0, .1, .03))]
        header = ("ply\nformat ascii 1.0\n" + f"element vertex {len(points)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                  + f"element face {count}\nproperty list uchar int vertex_indices\nend_header\n")
        body = "".join(" ".join(str(value) for value in (*point, *rgb)) + "\n" for point in points)
        body += "".join(f"3 {3 * index} {3 * index + 1} {3 * index + 2}\n" for index in range(count))
        asset = directory / f"{identity}.ply"
        asset.write_text(header + body, encoding="ascii")
        parts.append({"id": identity, "name": identity, "role": "environment" if identity == "environment"
                      else "observed_object_region", "path": asset.name, "sha256": digest(asset),
                      "triangles": count, "unmoved_candidate_boundary_triangles": 0})
    path = directory / "surface-scene.json"
    write_json(path, {"schema": "spatial-scene-lab.surface.v1", "units": "m", "up_axis": "Y", "parts": parts})
    return path


def glb_ids(path):
    with Path(path).open("rb") as stream:
        magic, version, length = struct.unpack("<4sII", stream.read(12))
        size, kind = struct.unpack("<II", stream.read(8))
        if magic != b"glTF" or version != 2 or length != Path(path).stat().st_size or kind != 0x4E4F534A:
            raise AssertionError("invalid native GLB header")
        document = json.loads(stream.read(size))
    return sorted(node.get("extras", {}).get("alvenx_id") for node in document["nodes"] if "mesh" in node)


def worker(case, output):
    import bpy
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import blender_edit as edit
    import verify_blender_edit as verify
    import blender_scene

    result = {"case": case, "status": "running", "blender": bpy.app.version_string,
              "script_sha256": digest(__file__), "addon_sha256": digest(edit.__file__),
              "execution": "one fresh background Blender process; scripted native operations"}
    started = time.perf_counter()
    source = fixture(output / "input")
    edit.register()
    bpy.ops.wm.read_factory_settings(use_empty=True)

    def expect_rejection(function, before, message):
        try:
            function()
        except (ValueError, OSError, KeyError) as error:
            text = str(error)
        else:
            raise AssertionError("invalid action was accepted")
        if message.lower() not in text.lower():
            raise AssertionError(f"unexpected diagnostic: {text}")
        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        verify.require(verify.check_state(before, verify.snapshot()))
        result["diagnostic"] = text
        result["state_unchanged"] = True

    def select_face(source_obj):
        for face in source_obj.data.polygons:
            face.select = face.index == 0

    def selection_state():
        return {"selected": sorted(obj.name for obj in bpy.context.selected_objects),
                "active": bpy.context.view_layer.objects.active.name if bpy.context.view_layer.objects.active else None,
                "visibility": {obj.name: [obj.hide_get(), obj.hide_select, obj.hide_viewport]
                               for obj in edit.managed_meshes()}}

    def link_blocked_collection(obj):
        collection = bpy.data.collections.new("Blocked observed region")
        bpy.context.scene.collection.children.link(collection)
        collection.objects.link(obj)
        for prior in list(obj.users_collection):
            if prior != collection:
                prior.objects.unlink(obj)
        bpy.context.view_layer.update()
        return collection

    try:
        if case in {"source_sha_preflight", "duplicate_uuid_preflight"}:
            doc = json.loads(source.read_text(encoding="utf-8"))
            if case == "source_sha_preflight":
                doc["parts"][-1]["sha256"] = "0" * 64
            else:
                doc["parts"][-1]["id"] = doc["parts"][0]["id"]
            write_json(source, doc)
            mesh = bpy.data.meshes.new("Existing unrelated mesh")
            mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
            obj = bpy.data.objects.new("Existing unrelated object", mesh)
            bpy.context.scene.collection.objects.link(obj)
            obj["sentinel"] = "unchanged"
            before = (set(bpy.data.objects), set(bpy.data.meshes), set(bpy.data.materials), list(obj.matrix_world))
            try:
                edit.import_surface(source)
            except ValueError as error:
                result["diagnostic"] = str(error)
            else:
                raise AssertionError("invalid manifest passed preflight")
            if (set(bpy.data.objects), set(bpy.data.meshes), set(bpy.data.materials), list(obj.matrix_world)) != before:
                raise AssertionError("preflight rejection mutated existing data")
            if obj.get("sentinel") != "unchanged" or edit.managed_meshes() or bpy.context.scene.get("ax_sources"):
                raise AssertionError("preflight rejection mutated provenance")
            result["state_unchanged"] = True
        else:
            edit.import_surface(source)
            env, target, other = (edit.object_by_id(identity) for identity in ("environment", "target", "other"))
            verify.activate(env)
            for face in env.data.polygons:
                face.select = False
            before = verify.snapshot()
            if case == "no_face_selection":
                expect_rejection(lambda: edit.reassign_selected(bpy.context, "target"), before, "subset")
            elif case == "all_faces_selection":
                for face in env.data.polygons:
                    face.select = True
                expect_rejection(lambda: edit.reassign_selected(bpy.context, "target"), before, "subset")
            elif case == "self_destination":
                select_face(env)
                expect_rejection(lambda: edit.reassign_selected(bpy.context, "environment"), before, "different")
            elif case == "missing_id":
                select_face(env)
                expect_rejection(lambda: edit.reassign_selected(bpy.context, "missing"), before, "identity")
            elif case == "multi_object_edit":
                target.select_set(True)
                bpy.ops.object.mode_set(mode="EDIT")
                result["objects_in_mode"] = len(bpy.context.objects_in_mode)
                if result["objects_in_mode"] != 2:
                    raise AssertionError("control did not enter multi-object Edit Mode")
                expect_rejection(lambda: edit.reassign_selected(bpy.context, "other"), before, "multi-object")
            elif case == "object_mode_selection_exactness":
                target.select_set(True)
                other.select_set(True)
                for vertex in env.data.vertices:
                    vertex.select = True
                for edge in env.data.edges:
                    edge.select = True
                select_face(env)
                chosen = env.data.attributes[edit.SOURCE_FACE].data[0].value
                transferred = edit.reassign_selected(bpy.context, "target")
                expected = {key: value.copy() for key, value in before.items()}
                changed = expected["ids"] == chosen
                expected["owners"][changed] = "target"
                expected["reviewed"][changed] = 1
                result["state"] = verify.require(verify.check_state(expected, verify.snapshot()))
                if transferred != [chosen] or len(edit.managed_meshes()) != 3:
                    raise AssertionError("stale selections expanded transfer or merged another object")
                result["selected_face_ids"] = transferred
                result["extra_selected_source_isolated"] = True
            elif case == "nonfinite_transform":
                rejected = []
                for label, translation, degrees, pivot in (
                    ("nan_translation", (float("nan"), 0, 0), 0, (0, 0, 0)),
                    ("infinite_rotation", (0, 0, 0), float("inf"), (0, 0, 0)),
                    ("infinite_pivot", (0, 0, 0), 15, (0, 0, float("inf"))),
                ):
                    expect_rejection(lambda: edit.rigid_edit(target, translation, degrees, pivot), before, "finite")
                    rejected.append(label)
                result["rejected_inputs"] = rejected
            elif case == "duplicate_current_id":
                other["alvenx_id"] = "target"
                asset = output / "rejected.glb"
                try:
                    edit.export_package(asset)
                except ValueError as error:
                    result["diagnostic"] = str(error)
                    if "duplicate observed object identity" not in str(error):
                        raise AssertionError("duplicate current identity lacks clear diagnostic")
                else:
                    raise AssertionError("duplicate current identity accepted for export")
                if asset.exists() or asset.with_suffix(".identity.json").exists():
                    raise AssertionError("duplicate current identity published a package")
                other["alvenx_id"] = "other"
                verify.require(verify.check_state(before, verify.snapshot()))
                result["state_unchanged_apart_from_injected_identity"] = True
            elif case in {"hidden_region_export", "excluded_view_layer_export", "blocked_collection_export"}:
                if case == "hidden_region_export":
                    other.hide_set(True)
                    other.hide_select = True
                else:
                    collection = link_blocked_collection(other)
                    if case == "excluded_view_layer_export":
                        bpy.context.view_layer.layer_collection.children[collection.name].exclude = True
                    else:
                        collection.hide_viewport = True
                    bpy.context.view_layer.update()
                ui_before = selection_state()
                asset = output / "package.glb"
                if case != "hidden_region_export":
                    expect_rejection(lambda: edit.export_package(asset), before, "active view layer")
                    if asset.exists() or asset.with_suffix(".identity.json").exists():
                        raise AssertionError("blocked export published an incomplete package")
                    if selection_state() != ui_before:
                        raise AssertionError("blocked export changed selection/visibility")
                else:
                    geometry = blender_scene.world_geometry()
                    appearance = verify.appearance_snapshot()
                    sidecar = edit.export_package(asset)
                    if selection_state() != ui_before:
                        raise AssertionError("export did not restore selection/visibility")
                    verify.require(verify.check_state(before, verify.snapshot()))
                    if glb_ids(asset) != ["environment", "other", "target"]:
                        raise AssertionError("hidden region was omitted from GLB")
                    record = json.loads(sidecar.read_text(encoding="utf-8"))
                    if record["asset_sha256"] != digest(asset) or record["native_face_attributes_in_glb"] is not False:
                        raise AssertionError("identity sidecar hash/scope mismatch")
                    sidecar_ids = sorted(face for obj in record["objects"] for face in obj["source_face_ids"])
                    if sidecar_ids != before["ids"].tolist():
                        raise AssertionError("sidecar omitted source-face lineage")
                    bpy.ops.wm.read_factory_settings(use_empty=True)
                    bpy.ops.import_scene.gltf(filepath=str(asset))
                    result["geometry"] = verify.require(blender_scene.compare_geometry(geometry, blender_scene.world_geometry()))
                    result["appearance"] = verify.require(verify.check_appearance(appearance, verify.appearance_snapshot()))
                    native_ids_present = any(obj.data.attributes.get(edit.SOURCE_FACE) is not None
                                             for obj in edit.managed_meshes())
                    if native_ids_present:
                        raise AssertionError("GLB native source-face claim differs from sidecar scope")
                    result["native_face_attributes_in_glb"] = False
                    result["sidecar_source_face_instances"] = len(sidecar_ids)
                    result["selection_visibility_restored"] = True
            elif case in {"sidecar_write_failure", "pair_replace_failure"}:
                saved = output / "prior.blend"
                asset = output / "package.glb"
                bpy.ops.wm.save_as_mainfile(filepath=str(saved))
                sidecar = edit.export_package(asset)
                preserved = {path: digest(path) for path in (saved, asset, sidecar)}
                edit.rigid_edit(target, (.35, .1, 0), 13, edit.geometry_center(target))
                before_failure = verify.snapshot()
                ui_before = selection_state()
                original_write = Path.write_text
                original_replace = edit.os.replace
                injected = []

                def failed_write(path, *args, **kwargs):
                    if str(path).endswith(".identity.json.partial"):
                        injected.append("sidecar_write")
                        raise OSError("injected sidecar write failure")
                    return original_write(path, *args, **kwargs)

                def failed_replace(source_path, target_path):
                    if Path(source_path).name == sidecar.name + ".partial":
                        injected.append("sidecar_replace")
                        raise OSError("injected second publication replace failure")
                    return original_replace(source_path, target_path)

                try:
                    if case == "sidecar_write_failure":
                        Path.write_text = failed_write
                    else:
                        edit.os.replace = failed_replace
                    expect_rejection(lambda: edit.export_package(asset), before_failure, "injected")
                finally:
                    Path.write_text = original_write
                    edit.os.replace = original_replace
                if not injected:
                    raise AssertionError("write failure injection did not trigger")
                result["injected_failure"] = injected
                result["prior_hashes_unchanged"] = {path.name: digest(path) == value for path, value in preserved.items()}
                if selection_state() != ui_before:
                    raise AssertionError("failed export changed selection/visibility")
                result["selection_visibility_restored"] = True
                if not all(result["prior_hashes_unchanged"].values()):
                    raise AssertionError("failed publication changed a previously valid saved asset")
                if list(output.glob("*.partial*")):
                    raise AssertionError("failed export left unfinished temporary publication")
            else:
                raise ValueError(f"unknown control: {case}")
        result["status"] = "pass"
    except Exception as error:
        result.update(status="fail", error=str(error), traceback=traceback.format_exc())
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "result.json", result)
        print("ALVENX_EDIT_CONTROL " + json.dumps({"case": case, "status": result["status"]}))
    if result["status"] != "pass":
        raise RuntimeError(f"native control failed: {case}: {result['error']}")


def main():
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new directory; prior failures are retained")
    parser.add_argument("--blender", help="required for the stdlib orchestrator")
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args(arguments)
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("output must be a new directory; retain previous evidence")
    output.mkdir(parents=True)
    if args.case:
        worker(args.case, output)
        return
    if not args.blender or not Path(args.blender).is_file():
        raise ValueError("provide an existing Blender executable")
    results = []
    started = time.perf_counter()
    for case in args.cases:
        case_output = output / case
        command = [str(Path(args.blender).resolve()), "--background", "--factory-startup", "--python-exit-code", "1",
                   "--python", str(Path(__file__).resolve()), "--", "--case", case, "--output", str(case_output)]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                       timeout=120, check=False)
            (output / f"{case}.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
            path = case_output / "result.json"
            record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
                "case": case, "status": "fail", "error": "worker did not produce result.json"}
            record["process_exit_code"] = completed.returncode
            if completed.returncode != 0:
                record["status"] = "fail"
        except subprocess.TimeoutExpired:
            record = {"case": case, "status": "fail", "error": "120-second native worker timeout"}
        results.append(record)
        print(f"{case}: {record['status']}", flush=True)
    summary = {"schema": "spatial-scene-lab.native-controls.v1", "status": "pass" if all(
        item["status"] == "pass" for item in results) else "fail", "script_sha256": digest(__file__),
        "addon_sha256": digest(Path(__file__).with_name("blender_edit.py")),
        "attempted": len(results), "passed": sum(item["status"] == "pass" for item in results),
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "scripted native small-mesh input/selection/export regressions; no interactive timing, render, undo or OS isolation claim",
        "results": results}
    write_json(output / "summary.json", summary)
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
