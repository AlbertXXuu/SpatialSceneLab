"""Replay predeclared stool transfer cases, retaining refusals and failed attempts.

Run with the Open3D environment; --fit-python selects the separate SciPy runtime.
Reference assets are passed only to evaluation, never to the fitting subprocess.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import scan_pipeline
from evaluate_stool_quality import sha256, write_new_report


def child_path(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"case path escapes suite: {relative}")
    return path


def run(command, log_prefix, timeout=180):
    start = time.perf_counter()
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=timeout)
        code, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as error:
        code = None
        stdout = error.stdout or b""
        stderr = error.stderr or b""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stderr += f"\nRun exceeded {timeout} seconds; attempt retained as timeout."
    for suffix, data in (("stdout", stdout), ("stderr", stderr)):
        with log_prefix.with_suffix(f".{suffix}.txt").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
    return {"returncode": code, "elapsed_seconds": time.perf_counter() - start,
            "timeout_seconds": timeout, "stdout": log_prefix.with_suffix(".stdout.txt").name,
            "stderr": log_prefix.with_suffix(".stderr.txt").name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fit-python", required=True)
    parser.add_argument("--fitter", type=Path, default=Path(__file__).with_name("fit_stool.py"))
    parser.add_argument("--points-from", type=Path,
                        help="reuse an earlier run's per-case points verbatim for a same-input comparison")
    args = parser.parse_args()
    suite, output, fitter = args.suite.resolve(), args.output.resolve(), args.fitter.resolve()
    if (output.exists() or args.output.is_symlink() or output == suite or suite in output.parents
            or output == fitter or output in fitter.parents):
        parser.error("output must be a new directory, outside the suite and fitter")
    if not fitter.is_file():
        parser.error("--fitter must be an existing Python source file")
    manifest_path = suite / "suite-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest["scenes"]
    identities = [case["scene_id"] for case in cases]
    if (not identities or len(set(identities)) != len(identities)
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", identity) for identity in identities)):
        parser.error("suite needs unique safe case IDs")
    for case in cases:
        child_path(suite, case["scan"])
        child_path(suite, case.get("reference_dir", f"{case['scene_id']}/reference"))
    if args.points_from:
        prior = json.loads((args.points_from / "run-results.json").read_text(encoding="utf-8"))
        if prior["suite_sha256"] != sha256(manifest_path):
            parser.error("points-from run must use the identical suite manifest")
        previous_cases = {item["scene_id"]: item for item in prior["cases"]}
        for identity in identities:
            previous = previous_cases[identity]["input_sha256"]
            folder = args.points_from / identity / "points"
            if any(sha256(folder / name) != previous[key]
                   for key, name in (("scene", "scene.json"), ("observations", "observations.npz"))):
                parser.error(f"points-from input contents changed: {identity}")
    output.mkdir(parents=True)
    report = {"schema": "alvenx.stool-transfer-run.v1", "suite_sha256": sha256(manifest_path),
              "fitter_sha256": sha256(fitter), "runner_sha256": sha256(__file__),
              "scan_pipeline_sha256": sha256(scan_pipeline.__file__),
              "fit_python": args.fit_python, "geometry_python": sys.executable,
              "scope": "Predeclared synthetic development transfer; not a held-out real-data study",
              "cases": []}
    project = Path(__file__).resolve().parent
    unexpected = []
    for case in cases:
        identity = case["scene_id"]
        folder = output / identity
        folder.mkdir()
        points = (args.points_from.resolve() / identity / "points") if args.points_from else folder / "points"
        record = {"scene_id": identity, "expected_class": case.get("expected_class", case.get("class")),
                  "points_directory": str(points), "result_directory": identity}
        try:
            if not args.points_from:
                scan_pipeline.reconstruct(child_path(suite, case["scan"]), points)
            scene, observations = points / "scene.json", points / "observations.npz"
            record["input_sha256"] = {"scene": sha256(scene), "observations": sha256(observations)}
            record["fit_process"] = run([args.fit_python, "-B", str(fitter), "--scene", str(scene),
                                         "--observations", str(observations), "--object-uuid", "same-stool",
                                         "--output", str(folder / "fitted")], folder / "fit")
            record["evaluation_process"] = run([sys.executable, "-B", str(project / "evaluate_stool_transfer.py"),
                "--reference", str(child_path(suite, case.get("reference_dir", f"{identity}/reference"))),
                "--scene", str(scene), "--observations", str(observations),
                "--fit-dir", str(folder / "fitted"), "--output", str(folder / "evaluation.json")], folder / "evaluation")
            record["evaluation_report"] = f"{identity}/evaluation.json"
            if record["fit_process"]["returncode"] not in (0, 2) or record["evaluation_process"]["returncode"] != 0:
                unexpected.append(identity)
        except Exception as error:
            record["execution_error"] = f"{type(error).__name__}: {error}"
            unexpected.append(identity)
        report["cases"].append(record)
        write_new_report(folder / "attempt.json", record)
        print(json.dumps(record), flush=True)
    report["status"] = "completed" if not unexpected else "completed_with_execution_errors"
    report["execution_error_cases"] = unexpected
    report["case_count"] = len(report["cases"])
    write_new_report(output / "run-results.json", report)
    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
