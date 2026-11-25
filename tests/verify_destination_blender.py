"""Native Blender Destination regression; no geometry/export acceptance. MIT.

Run each mode in a separate Blender process and a new output directory:
  blender --background --factory-startup --python-exit-code 1 --python
    tests/verify_destination_blender.py -- --mode exercise --output /new/exercise
  blender --background --factory-startup --python-exit-code 1 --python
    tests/verify_destination_blender.py -- --mode reopen
    --blend /new/exercise/destination.blend --output /new/reopen

Use legacy-prepare and legacy-reopen with a new directory for each to test a
real saved three-field dynamic Enum, rather than a similarly named custom
IDProperty. The second mode takes --blend /new/legacy-prepare/legacy.blend.

Every attempt writes a report, including partial checks and a traceback on failure.
The synthetic triangles exercise selection identity only, not face transfer,
appearance, export, or the visible popup's layout and pointer interaction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import blender_edit as edit

SCENE_NAME = "AX_Destination_Exercise"
OTHER_SCENE = "AX_Destination_Other"
IDS = ("source-a", "target-b", "target-c")
UNSET = "__ALVENX_SELECT_DESTINATION__"
KEY = "ax_target_id"
# Blender borrows callback strings. Keep the temporary old-style menu alive too.
LEGACY_ITEMS = []


def check(report, name, condition, **evidence):
    record = {"name": name, "status": "pass" if condition else "fail", **evidence}
    report["checks"].append(record)
    if not condition:
        raise AssertionError(json.dumps(record, ensure_ascii=False))


def rejects(report, name, action):
    try:
        action()
    except ValueError as error:
        check(report, name, True, error=str(error))
    else:
        check(report, name, False, error="Expected ValueError")


def make_mesh(scene, identity, name):
    mesh = bpy.data.meshes.new(name + " mesh")
    mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj["alvenx_id"] = identity
    scene.collection.objects.link(obj)
    return obj


def scene():
    # Undo/Redo and opening a blend invalidate old RNA references.
    return bpy.data.scenes[SCENE_NAME]


def state(value):
    return {"scene": value.name, "canonical_id": value.get(KEY, ""),
            "rna_enum": value.ax_target,
            "objects": [{"id": obj["alvenx_id"], "name": obj.name}
                        for obj in edit.managed_meshes(value)]}


def assert_selected(report, name, value, expected):
    current = state(value)
    resolved = edit.selected_target(value)
    check(report, name, current["canonical_id"] == expected
          and current["rna_enum"] == expected and resolved == expected,
          state=current, resolved=resolved)


def menu_ids(items):
    return [item[0] for item in items]


def exercise(report, output):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    edit.register()
    current = bpy.context.scene
    current.name = SCENE_NAME
    current["ax_sources"] = json.dumps([{"id": identity} for identity in IDS])
    for identity in IDS:
        make_mesh(current, identity, "A " + identity)

    check(report, "new scene has an explicit unselected state",
          current.ax_target == UNSET and current.get(KEY, "") == "",
          state=state(current))
    rejects(report, "unselected destination cannot resolve", lambda: edit.selected_target(current))
    # This is only a custom IDProperty, not Blender 5.x's system Enum storage.
    # The real legacy Enum persistence check has separate fresh-process modes.
    current["ax_target"] = 2
    check(report, "unrelated custom numeric property does not migrate", current.ax_target == UNSET,
          custom_numeric_property=current["ax_target"], canonical_id=current.get(KEY, ""))
    del current["ax_target"]
    current.ax_target = "target-b"
    assert_selected(report, "RNA assignment stores canonical identity", current, "target-b")

    original = edit.target_items(current, bpy.context)
    original_labels = [item[1] for item in original]
    stable = True
    for _ in range(200):
        returned = edit.target_items(current, None)
        stable = stable and len(returned) == len(original) and all(
            before is after for before, after in zip(original, returned))
    gc.collect()
    check(report, "repeated callbacks retain borrowed enum tuples", stable
          and [item[1] for item in original] == original_labels,
          iterations=200, items=[list(item) for item in original])
    assert_selected(report, "repeated callbacks retain selection", current, "target-b")

    before = [obj["alvenx_id"] for obj in edit.managed_meshes(current)]
    first = edit.object_by_id("source-a", current)
    current.collection.objects.unlink(first)
    current.collection.objects.link(first)
    after = [obj["alvenx_id"] for obj in edit.managed_meshes(current)]
    check(report, "unlink/relink actually changes object iteration order", before != after,
          before=before, after=after)
    assert_selected(report, "reordering cannot retarget selection", current, "target-b")
    previous_label = next(item for item in original if item[0] == "target-b")
    edit.object_by_id("target-b", current).name = "Renamed destination B"
    renamed = edit.target_items(current, None)
    renamed_label = next(item for item in renamed if item[0] == "target-b")
    check(report, "renaming updates the label without invalidating old strings",
          "Renamed destination B" in renamed_label[1]
          and previous_label[1] == original_labels[2]
          and previous_label is not renamed_label,
          old=list(previous_label), new=list(renamed_label))
    assert_selected(report, "renaming preserves canonical target", current, "target-b")

    other = bpy.data.scenes.new(OTHER_SCENE)
    other["ax_sources"] = json.dumps([{"id": identity} for identity in reversed(IDS)])
    for identity in IDS:
        make_mesh(other, identity, "Other " + identity)
    other.ax_target = "target-c"
    bpy.context.window.scene = other
    wrong_context = edit.target_items(current, bpy.context)
    no_context = edit.target_items(current, None)
    check(report, "callback uses owner scene despite a different global context",
          bpy.context.scene.name == OTHER_SCENE
          and menu_ids(wrong_context) == [UNSET, *IDS]
          and all("Other " not in item[1] for item in wrong_context)
          and any("Renamed destination B" in item[1] for item in wrong_context),
          global_scene=bpy.context.scene.name, owner_scene=current.name,
          items=[list(item) for item in wrong_context])
    check(report, "None context yields the owner scene menu", no_context == wrong_context)
    assert_selected(report, "inactive owner scene resolves its own target", current, "target-b")
    assert_selected(report, "second scene keeps an independent selection", other, "target-c")
    bpy.context.window.scene = current

    target = edit.object_by_id("target-b", current)
    current.collection.objects.unlink(target)
    check(report, "missing target shows unavailable without losing saved identity",
          current.ax_target == UNSET and current[KEY] == "target-b"
          and "target-b" not in menu_ids(edit.target_items(current, None)),
          state=state(current), items=[list(item) for item in edit.target_items(current, None)])
    rejects(report, "missing target cannot resolve", lambda: edit.selected_target(current))
    current.collection.objects.link(target)
    assert_selected(report, "relinking the same identity restores the selection", current, "target-b")

    duplicate = make_mesh(current, "target-b", "Duplicate B")
    check(report, "ambiguous object identity is unavailable",
          current.ax_target == UNSET and current[KEY] == "target-b"
          and "target-b" not in menu_ids(edit.target_items(current, None)))
    rejects(report, "ambiguous object identity cannot resolve", lambda: edit.selected_target(current))
    bpy.data.objects.remove(duplicate, do_unlink=True)
    assert_selected(report, "removing duplicate restores canonical selection", current, "target-b")

    saved_registry = current["ax_sources"]
    current["ax_sources"] = json.dumps([{"id": "target-b"}, {"id": "target-b"}])
    check(report, "duplicate frozen registry is unavailable", current.ax_target == UNSET
          and menu_ids(edit.target_items(current, None)) == [UNSET])
    rejects(report, "duplicate registry cannot resolve", lambda: edit.selected_target(current))
    current["ax_sources"] = saved_registry
    assert_selected(report, "restoring registry restores canonical selection", current, "target-b")
    rejects(report, "invalid numeric menu value cannot replace the target",
            lambda: edit.set_target(current, 9999))
    assert_selected(report, "failed setter retains the previous target", current, "target-b")

    for cycle in range(2):
        edit.unregister()
        check(report, f"unregister removes RNA property, cycle {cycle + 1}",
              not hasattr(bpy.types.Scene, "ax_target") and current[KEY] == "target-b")
        edit.register()
        assert_selected(report, f"registration restores saved identity, cycle {cycle + 1}",
                        current, "target-b")

    # Do not dereference these objects after Undo; obtain all RNA data again.
    del target, first, other, current
    bpy.context.preferences.edit.use_global_undo = True
    scene().ax_target = "target-b"
    first_checkpoint = bpy.ops.ed.undo_push(message="Destination canonical B")
    scene().ax_target = "target-c"
    edit.object_by_id("target-c", scene()).name = "C after checkpoint"
    second_checkpoint = bpy.ops.ed.undo_push(message="Destination canonical C")
    check(report, "explicit native Undo checkpoints finish",
          first_checkpoint == second_checkpoint == {"FINISHED"},
          first=sorted(first_checkpoint), second=sorted(second_checkpoint))
    undo_result = bpy.ops.ed.undo()
    check(report, "native Undo finishes", undo_result == {"FINISHED"}, returned=sorted(undo_result))
    assert_selected(report, "Undo restores B with freshly acquired RNA", scene(), "target-b")
    check(report, "Undo restores the object's name",
          edit.object_by_id("target-c", scene()).name == "A target-c")
    check(report, "callback remains usable after Undo",
          "target-b" in menu_ids(edit.target_items(scene(), None)))
    redo_result = bpy.ops.ed.redo()
    check(report, "native Redo finishes", redo_result == {"FINISHED"}, returned=sorted(redo_result))
    assert_selected(report, "Redo restores C with freshly acquired RNA", scene(), "target-c")
    check(report, "Redo restores the object's name",
          edit.object_by_id("target-c", scene()).name == "C after checkpoint")
    check(report, "callback remains usable after Redo",
          any("C after checkpoint" in item[1] for item in edit.target_items(scene(), None)))

    scene().ax_target = UNSET
    check(report, "explicit unset clears the saved identity",
          scene().get(KEY, "") == "" and scene().ax_target == UNSET)
    scene().ax_target = "target-b"
    assert_selected(report, "final selected state before save", scene(), "target-b")
    blend = output / "destination.blend"
    saved = bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    check(report, "editable blend saved", saved == {"FINISHED"} and blend.is_file(),
          path=str(blend), bytes=blend.stat().st_size)
    report["saved_blend"] = str(blend)
    report["saved_state"] = state(scene())


def reopen(report, blend):
    source_report_path = blend.parent / "report.json"
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    check(report, "exercise completed successfully", source_report["status"] == "pass",
          exercise_report=str(source_report_path))
    check(report, "reopen uses a separate process", source_report["pid"] != os.getpid(),
          exercise_pid=source_report["pid"], reopen_pid=os.getpid())
    check(report, "reopen uses the exercised implementation",
          source_report["implementation_sha256"] == report["implementation_sha256"])
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    check(report, "canonical ID survives without registered RNA property",
          not hasattr(bpy.types.Scene, "ax_target") and scene().get(KEY) == "target-b",
          canonical_id=scene().get(KEY), scene=scene().name)
    edit.register()
    assert_selected(report, "fresh registration resolves saved canonical ID", scene(), "target-b")
    check(report, "saved renamed label survives",
          any("Renamed destination B" in item[1] for item in edit.target_items(scene(), None)))
    before = [obj["alvenx_id"] for obj in edit.managed_meshes(scene())]
    moved = edit.object_by_id(before[0], scene())
    scene().collection.objects.unlink(moved)
    scene().collection.objects.link(moved)
    after = [obj["alvenx_id"] for obj in edit.managed_meshes(scene())]
    check(report, "reopened object ordering actually changes", before != after, before=before, after=after)
    assert_selected(report, "reopened reorder cannot retarget saved selection", scene(), "target-b")
    bpy.context.window.scene = bpy.data.scenes[OTHER_SCENE]
    assert_selected(report, "reopened inactive scene keeps target B", scene(), "target-b")
    assert_selected(report, "reopened second scene keeps target C", bpy.context.scene, "target-c")
    check(report, "reopened callback supports None context",
          menu_ids(edit.target_items(scene(), None)) == [UNSET, *IDS])
    edit.unregister()
    check(report, "final unregister preserves persisted canonical identity",
          not hasattr(bpy.types.Scene, "ax_target") and scene().get(KEY) == "target-b")


def legacy_items(owner, _context):
    items = [(obj["alvenx_id"], obj.name, "Observed surface region")
             for obj in edit.managed_meshes(owner)]
    LEGACY_ITEMS.append(items)
    return items


def register_legacy():
    bpy.types.Scene.ax_target = bpy.props.EnumProperty(name="Destination", items=legacy_items)


def legacy_prepare(report, output):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    current = bpy.context.scene
    current.name = SCENE_NAME
    current["ax_sources"] = json.dumps([{"id": identity} for identity in IDS])
    for identity in IDS:
        make_mesh(current, identity, "Legacy " + identity)
    register_legacy()
    items = legacy_items(current, None)
    check(report, "temporary legacy property is a real three-field dynamic RNA Enum",
          current.bl_rna.properties["ax_target"].type == "ENUM"
          and all(len(item) == 3 for item in items),
          items=[list(item) for item in items])
    before = current.ax_target
    current.ax_target = "target-b"
    check(report, "legacy RNA assignment selects B without a canonical ID",
          before != "target-b" and current.ax_target == "target-b" and KEY not in current,
          before=before, selected=current.ax_target, canonical_id=current.get(KEY))
    gc.collect()
    check(report, "legacy borrowed strings remain usable", current.ax_target == "target-b"
          and bool(LEGACY_ITEMS), retained_item_lists=len(LEGACY_ITEMS))
    blend = output / "legacy.blend"
    saved = bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    check(report, "legacy Enum selection saved in blend", saved == {"FINISHED"} and blend.is_file(),
          path=str(blend), bytes=blend.stat().st_size)
    del bpy.types.Scene.ax_target
    check(report, "temporary legacy RNA registration is removed",
          not hasattr(bpy.types.Scene, "ax_target") and KEY not in current)
    report["saved_blend"] = str(blend)
    report["legacy_selected_id"] = "target-b"


def legacy_reopen(report, blend):
    prior_path = blend.parent / "report.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    check(report, "legacy preparation completed", prior["status"] == "pass"
          and prior["mode"] == "legacy-prepare", preparation_report=str(prior_path))
    check(report, "legacy reopen uses a separate process", prior["pid"] != os.getpid(),
          preparation_pid=prior["pid"], reopen_pid=os.getpid())
    check(report, "legacy modes use the same maintained implementation",
          prior["implementation_sha256"] == report["implementation_sha256"])
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    check(report, "legacy blend has no canonical ID or preinstalled RNA property",
          KEY not in scene() and not hasattr(bpy.types.Scene, "ax_target"))
    # Positive control: prove the old RNA assignment really survived on disk.
    register_legacy()
    check(report, "positive control restores saved legacy RNA target B",
          scene().ax_target == "target-b" and KEY not in scene(),
          saved_legacy_enum=scene().ax_target, canonical_id=scene().get(KEY))
    del bpy.types.Scene.ax_target
    edit.register()
    check(report, "maintained property explicitly leaves legacy selection unset",
          scene().ax_target == UNSET and KEY not in scene(),
          state=state(scene()))
    rejects(report, "legacy selection cannot resolve before explicit reselection",
            lambda: edit.selected_target(scene()))
    check(report, "legacy document still offers valid destinations",
          menu_ids(edit.target_items(scene(), None)) == [UNSET, *IDS])
    scene().ax_target = "target-b"
    assert_selected(report, "explicit RNA reselection writes canonical target B", scene(), "target-b")
    edit.unregister()
    check(report, "legacy migration unregister preserves the new canonical ID",
          not hasattr(bpy.types.Scene, "ax_target") and scene().get(KEY) == "target-b")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("exercise", "reopen", "legacy-prepare", "legacy-reopen"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--blend", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if args.mode.endswith("reopen") != bool(args.blend):
        parser.error("--blend is required only in reopen or legacy-reopen mode")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"mode": args.mode, "status": "running", "pid": os.getpid(),
              "started_at": datetime.now(timezone.utc).isoformat(),
              "blender_version": bpy.app.version_string,
              "implementation": str(ROOT / "blender_edit.py"),
              "implementation_sha256": hashlib.sha256((ROOT / "blender_edit.py").read_bytes()).hexdigest(),
              "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "historical_evidence_correction": "The original exercise-01 check named 'legacy integer requires explicit reselection' used a custom IDProperty. It does not cover Blender 5.x system Enum persistence. Use legacy-prepare and legacy-reopen results for that claim.",
              "checks": [], "limits": ["Background native RNA and Undo/Redo, not a visible popup interaction test.",
                                        "Synthetic triangle objects; no ownership geometry, appearance or export acceptance."]}
    try:
        check(report, "required Blender version", bpy.app.version == (5, 1, 2),
              actual=list(bpy.app.version))
        if args.mode == "exercise":
            exercise(report, output)
        elif args.mode == "reopen":
            reopen(report, args.blend.resolve())
        elif args.mode == "legacy-prepare":
            legacy_prepare(report, output)
        else:
            legacy_reopen(report, args.blend.resolve())
        report["status"] = "pass"
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["passed_checks"] = sum(item["status"] == "pass" for item in report["checks"])
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"mode": args.mode, "status": report["status"],
                          "passed_checks": report["passed_checks"], "report": str(output / "report.json")}))


if __name__ == "__main__":
    main()
