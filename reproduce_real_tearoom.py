"""Rebuild the local tea-room delivery from verified prepared input. MIT.

No input acquisition or dependency installation occurs here. Scan-derived files
are local; the input's terms remain independent from this code's MIT license.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
TARGET = "46C25048-47D3-45BE-BEDF-D3DF0DF758F6"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(scan, manifest_path):
    """Check the complete pinned inventory, including missing and escaped files."""
    scan = Path(scan).resolve(strict=True)
    if not scan.is_dir():
        raise ValueError("Input must be a prepared tea_room directory")
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Source manifest requires nonempty files")
    seen, total = set(), 0
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("Invalid manifest entry")
        relative = PurePosixPath(entry["path"])
        if (relative.is_absolute() or relative.parts[:1] != ("tea_room",) or
                len(relative.parts) < 2 or ".." in relative.parts or
                "\\" in entry["path"] or ":" in entry["path"] or entry["path"] in seen):
            raise ValueError("Source manifest path escapes or duplicates input")
        seen.add(entry["path"])
        member = scan.joinpath(*relative.parts[1:]).resolve(strict=True)
        if scan not in member.parents or not member.is_file():
            raise ValueError("Source manifest member must remain inside input")
        expected = entry.get("sha256")
        size = entry.get("bytes")
        if (type(size) is not int or size < 0 or not isinstance(expected, str) or
                len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError("Invalid source size or SHA256")
        if member.stat().st_size != size or sha256(member) != expected:
            raise ValueError(f"Source checksum or size changed: {entry['path']}")
        total += size
    actual = {"tea_room/" + p.relative_to(scan).as_posix() for p in scan.rglob("*") if p.is_file()}
    if actual != seen:
        raise ValueError("Source file inventory differs from pinned manifest")
    return {"schema": "alvenx.tea-room-source-check.v1", "passed": True,
            "upstream": manifest.get("upstream"), "files": len(entries),
            "bytes": total, "manifest_sha256": sha256(manifest_path)}


def validate_paths(scan, output, blender):
    scan = Path(scan).resolve(strict=True)
    output = Path(output).resolve()
    blender = Path(blender).resolve(strict=True)
    if not scan.is_dir() or not blender.is_file():
        raise ValueError("Prepared input directory and Blender executable required")
    if output == scan or scan in output.parents or output in scan.parents:
        raise ValueError("Output and source trees must be separate")
    if output == ROOT or ROOT in output.parents and not (ROOT / ".local") in output.parents:
        raise ValueError("Repository outputs must stay under its ignored .local directory")
    if output.exists():
        raise ValueError("Output directory must be new; preserve previous results")
    return scan, output, blender


def run_stage(name, command, output, stages, allowed=(0,)):
    started = time.perf_counter()
    log = output / f"{name}.log"
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace")
    record = {"name": name, "exit_code": process.returncode,
              "elapsed_seconds": time.perf_counter() - started,
              "log": log.name, "log_sha256": sha256(log)}
    stages.append(record)
    print(json.dumps(record), flush=True)
    if process.returncode not in allowed:
        raise RuntimeError(f"{name} failed ({process.returncode}); inspect {log}")
    return process.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--blender", required=True, type=Path)
    args = parser.parse_args(argv)
    scan, output, blender = validate_paths(args.input, args.output, args.blender)
    manifest = ROOT / "docs/tea-room-source-manifest.json"
    initial = verify_source(scan, manifest)
    source_hashes = {p.name: sha256(p) for p in [
        Path(__file__), ROOT / "scan_pipeline.py", ROOT / "reconstruct_mesh.py",
        ROOT / "audit_real_tabletop.py", ROOT / "audit_real_table_alignment.py",
        ROOT / "build_real_tearoom.py", ROOT / "blender_real_tearoom_scene.py",
        ROOT / "offline_run.py", ROOT / "fit_stool.py", ROOT / "geometry_compare.py",
        ROOT / "blender_scene.py", ROOT / "blender_stool_scene.py"]}
    output.mkdir(parents=True)
    (output / "source-check.json").write_text(json.dumps(initial, indent=2) + "\n", encoding="utf-8")
    python = [sys.executable, "-B"]
    stages, failure = [], None
    receipt = {"schema": "alvenx.real-tearoom-replay.v1", "completed": False,
               "source": initial, "stages": stages,
               "runtime": "Prepared local files; no acquisition or installation",
               "source_code_sha256": source_hashes}
    def guarded(name, script, arguments, allowed=(0,)):
        return run_stage(name, python + [str(ROOT / "offline_run.py"), "--script", script,
                         "--report", str(output / f"{name}-guard.json"), "--", *arguments],
                         output, stages, allowed)
    try:
        points = output / "points"
        surface = output / "surface"
        structural = output / "structural"
        native = output / "native"
        guarded("points", "scan_pipeline.py", ["reconstruct", "--input", str(scan),
                "--output", str(points), "--frame-step", "1", "--pixel-step", "2",
                "--min-confidence", "1", "--max-depth", "6", "--voxel-size", ".015",
                "--box-margin", ".025", "--structure-tolerance", ".025"])
        guarded("surface", "reconstruct_mesh.py", ["--scene", str(points / "scene.json"),
                "--output", str(surface), "--voxel-size", ".035", "--truncation", ".105"])
        baseline = output / "baseline"
        guarded("baseline", "audit_real_tabletop.py", ["--scene", str(points / "scene.json"),
                "--observations", str(points / "observations.npz"), "--object", TARGET,
                "--output", str(baseline)])
        alignment_args = ["--scene", str(points / "scene.json"),
                "--observations", str(points / "observations.npz"), "--object", TARGET,
                "--background-record", str(baseline / "background-icp.json")]
        guarded("freeze-alignment", "audit_real_table_alignment.py", alignment_args + [
                "--freeze-protocol", str(output / "alignment-protocol.json")])
        # Alignment is diagnostic: an unaccepted geometric gate does not stop the
        # separately labelled structural authoring delivery.
        alignment_code = guarded("alignment", "audit_real_table_alignment.py", alignment_args + [
                "--protocol", str(output / "alignment-protocol.json"),
                "--output", str(output / "alignment")], allowed=(0, 2))
        receipt["alignment_gate_passed"] = alignment_code == 0
        guarded("structural", "build_real_tearoom.py", ["--scene", str(points / "scene.json"),
                "--observations", str(points / "observations.npz"), "--output", str(structural)])
        native_args = ["--scene", str(points / "scene.json"), "--surface", str(surface / "surface-scene.json"),
                       "--structural", str(structural), "--output", str(native)]
        for mode in ("build", "reopen"):
            run_stage(f"native-{mode}", [str(blender), "--background", "--factory-startup",
                "--python-exit-code", "1", "--python", str(ROOT / "offline_run.py"), "--",
                "--script", str(ROOT / "blender_real_tearoom_scene.py"), "--report",
                str(output / f"native-{mode}-guard.json"), "--blender", "--", *native_args,
                "--mode", mode], output, stages)
        verify_source(scan, manifest)
        if any(sha256(ROOT / name) != digest for name, digest in source_hashes.items()):
            raise RuntimeError("Reproduction source changed during execution")
        receipt["completed"] = True
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["failure"] = failure
        (output / "replay.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    # Geometric acceptance and successful material construction are separate.
    return 0 if receipt["alignment_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
