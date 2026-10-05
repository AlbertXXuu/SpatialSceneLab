"""Run six native Blender task chains and fresh-process reopen checks. MIT.

The prepared S0 results supply immutable B0 surface bundles. This runner never
loads physical reference labels and does not measure segmentation improvement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

TASKS = (
    ("dev-contact-chair", "contact-chair", "contact-table"),
    ("dev-same-color-dining", "same-table", "same-chair-west"),
    ("dev-curved-sofa-wall", "curved-sofa", "curved-ottoman"),
    ("dev-occluded-bookshelf", "occluded-shelf", "occluding-screen"),
    ("dev-thin-rack-noise", "thin-rack", "thin-table"),
    ("dev-combined-study", "combined-desk", "combined-chair"),
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def task_inputs(results, selected=None):
    root = Path(results).resolve()
    if selected and set(selected) - {task[0] for task in TASKS}:
        raise ValueError("requested scene is not an S1 development task")
    tasks = []
    for name, primary, secondary in TASKS:
        if selected and name not in selected:
            continue
        path = root / name / "surface" / "surface-scene.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        ids = [part["id"] for part in doc["parts"]]
        if len(set(ids)) != len(ids) or {"environment", primary, secondary} - set(ids):
            raise ValueError(f"task regions missing or duplicated: {name}")
        for part in doc["parts"]:
            source = (path.parent / part["path"]).resolve()
            if path.parent not in source.parents or sha256(source) != part["sha256"]:
                raise ValueError(f"source part escaped or changed: {name}/{part['id']}")
        tasks.append((name, primary, secondary, path))
    return tasks


def run_process(command, log, timeout):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    try:
        import psutil
    except ImportError:
        psutil = None
    peak = 0
    samples = 0
    started = time.perf_counter()
    with Path(log).open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        timed_out = False
        while process.poll() is None:
            if psutil is not None:
                try:
                    parent = psutil.Process(process.pid)
                    members = [parent, *parent.children(recursive=True)]
                    memory = 0
                    for member in members:
                        try:
                            memory += member.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    peak = max(peak, memory)
                    samples += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if time.perf_counter() - started > timeout:
                timed_out = True
                if psutil is not None:
                    try:
                        for child in psutil.Process(process.pid).children(recursive=True):
                            child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
                break
            time.sleep(.1)
        code = process.wait()
    return {"status": "pass" if code == 0 and not timed_out else "fail", "exit_code": code,
            "timed_out": timed_out, "elapsed_seconds": time.perf_counter() - started,
            "sampled_process_tree_peak_rss_bytes": peak if psutil is not None else None,
            "memory_samples": samples, "psutil": psutil.__version__ if psutil is not None else None,
            "memory_scope": "concurrent RSS of Blender parent and observed descendants, sampled every 0.1 seconds"}


def run_suite(results, output, blender, *, selected=None, render_scene="dev-combined-study", timeout=1200):
    inputs = task_inputs(results, selected)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("retain existing runs; output must be new or empty")
    root = Path(results).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("output and prepared results must not overlap")
    output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).parent / "verify_blender_edit.py"
    source_names = ("blender_edit.py", "verify_blender_edit.py", "geometry_compare.py", "blender_scene.py", "run_edit_suite.py")
    summary = {"schema": "spatial-scene-lab.edit-suite.v1", "status": "running",
               "scope": "six S0 authored development bundles; scripted ownership/edit/undo/reopen/export tasks",
               "platform": platform.platform(), "python": platform.python_version(),
               "new_paid_allocation": 0, "tasks": [],
               "code_sha256": {name: sha256(script.parent / name) for name in source_names}}
    started = time.perf_counter()
    for name, primary, secondary, scene in inputs:
        directory = output / name
        command = [str(blender), "--background", "--factory-startup", "--python-exit-code", "1",
                   "--python", str(script), "--", "--scene", str(scene), "--output", str(directory),
                   "--primary", primary, "--secondary", secondary]
        if name == render_scene:
            command.append("--render")
        measured = run_process(command, output / f"{name}.log", timeout)
        result = {"scene_id": name, "primary": primary, "secondary": secondary,
                  "input_scene_sha256": sha256(scene), "process": measured, "status": "fail"}
        report = directory / "task-results.json"
        if report.exists():
            result["native_task"] = json.loads(report.read_text(encoding="utf-8"))
        if measured["status"] == "pass" and result.get("native_task", {}).get("status") == "pass":
            reopened = run_process([str(blender), "--background", "--factory-startup", "--python-exit-code", "1",
                                    "--python", str(script), "--", "--mode", "reopen", "--output", str(directory)],
                                   output / f"{name}-reopen.log", timeout)
            result["reopen_process"] = reopened
            reopen_report = directory / "reopen-results.json"
            if reopen_report.exists():
                result["reopen"] = json.loads(reopen_report.read_text(encoding="utf-8"))
            if reopened["status"] == "pass" and result.get("reopen", {}).get("status") == "pass":
                result["status"] = "pass"
        result["assets"] = {member.name: {"sha256": sha256(member), "bytes": member.stat().st_size}
                            for member in directory.iterdir() if member.suffix in {".blend", ".glb", ".png"}
                            or member.name.endswith(".identity.json")} if directory.exists() else {}
        summary["tasks"].append(result)
        summary["elapsed_seconds"] = time.perf_counter() - started
        save_json(output / "edit-suite-results.json", summary)
        print(json.dumps({"scene": name, "status": result["status"], "seconds": measured["elapsed_seconds"]}), flush=True)
    summary["status"] = "pass" if all(task["status"] == "pass" for task in summary["tasks"]) else "fail"
    summary["completed_tasks"] = sum(task["status"] == "pass" for task in summary["tasks"])
    summary["attempted_tasks"] = len(summary["tasks"])
    summary["elapsed_seconds"] = time.perf_counter() - started
    save_json(output / "edit-suite-results.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blender", required=True)
    parser.add_argument("--scene", action="append")
    parser.add_argument("--render-scene", default="dev-combined-study")
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    result = run_suite(args.results, args.output, args.blender, selected=args.scene,
                       render_scene=args.render_scene, timeout=args.timeout)
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
