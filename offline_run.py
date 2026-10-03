"""Run a Python entry point with outbound Python socket/DNS operations denied. MIT.

This process guard is deliberately identified separately from OS-level network
isolation: native libraries and independent child processes are outside its scope.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import socket
import sys
import time


NETWORK_EVENTS = {"socket.connect", "socket.connect_ex", "socket.getaddrinfo",
                  "socket.gethostbyname", "socket.gethostbyaddr", "socket.getnameinfo", "socket.sendto"}


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "bpy" in sys.modules and "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--blender", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    script = Path(args.script).resolve()
    report = Path(args.report).resolve()
    if not script.is_file() or script == report:
        raise ValueError("valid script and distinct report required")
    report.parent.mkdir(parents=True, exist_ok=True)
    denied = []

    def audit(event, arguments):
        if event in NETWORK_EVENTS:
            denied.append(event)
            raise PermissionError(f"offline Python guard denied {event}")

    sys.addaudithook(audit)
    try:
        socket.getaddrinfo("offline-probe.invalid", 443)
    except PermissionError:
        probe_passed = True
    else:
        raise RuntimeError("network guard probe did not block DNS")
    denied.clear()
    command_args = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    sys.argv = [str(script), *(["--"] if args.blender else []), *command_args]
    sys.path.insert(0, str(script.parent))
    started = time.perf_counter()
    status, failure = "pass", None
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            status, failure = "fail", f"SystemExit({exc.code})"
            raise
    except BaseException as exc:
        status, failure = "fail", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if denied:
            status = "fail"
        document = {"status": status, "guard": "CPython audit-hook outbound socket and DNS denial",
                    "dns_block_probe_passed": probe_passed,
                    "runtime_denied_network_events": denied,
                    "script": script.name, "elapsed_seconds": time.perf_counter() - started,
                    "failure": failure,
                    "scope": "Python network operations; not OS isolation of native code or child processes"}
        report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print("ALVENX_OFFLINE_GUARD " + json.dumps(document))
    if status != "pass":
        raise RuntimeError("offline process guard failed")


if __name__ == "__main__":
    main()
