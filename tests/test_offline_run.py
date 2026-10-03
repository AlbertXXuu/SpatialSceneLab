import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]


class OfflineGuardTests(unittest.TestCase):
    def test_local_work_passes_and_caught_network_attempt_still_fails(self):
        # Even a script that swallows PermissionError must not get a pass.
        with tempfile.TemporaryDirectory(prefix="AlvenX-guard-") as tmp:
            root = Path(tmp)
            for name, body, expected in [
                ("local", "assert sum([1, 2, 3]) == 6", "pass"),
                ("network", "import socket\ntry:\n socket.getaddrinfo('must-not-resolve.invalid', 443)\nexcept PermissionError:\n pass", "fail"),
            ]:
                script, report = root / f"{name}.py", root / f"{name}.json"
                script.write_text(body, encoding="utf-8")
                completed = subprocess.run(
                    [sys.executable, str(PROJECT / "offline_run.py"), "--script", str(script), "--report", str(report)],
                    capture_output=True, text=True, timeout=20,
                )
                result = json.loads(report.read_text(encoding="utf-8"))
                self.assertEqual(result["status"], expected)
                self.assertTrue(result["dns_block_probe_passed"])
                self.assertEqual(completed.returncode == 0, expected == "pass")
                self.assertEqual(bool(result["runtime_denied_network_events"]), expected == "fail")
