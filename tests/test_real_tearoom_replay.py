"""Pinned source validation and non-destructive replay boundaries. MIT."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import sys

from reproduce_real_tearoom import run_stage, validate_paths, verify_source


class ReplaySourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="AlvenX-tea-replay-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scan = self.root / "scan"
        self.scan.mkdir()
        self.member = self.scan / "frame.json"
        self.member.write_bytes(b'{"frame": 0}')
        self.manifest = self.root / "manifest.json"
        self.entry = {"path": "tea_room/frame.json", "bytes": self.member.stat().st_size,
                      "sha256": hashlib.sha256(self.member.read_bytes()).hexdigest()}
        self.write_manifest([self.entry])

    def write_manifest(self, entries):
        self.manifest.write_text(json.dumps({"files": entries, "upstream": {"commit": "pinned"}}),
                                 encoding="utf-8")

    def test_verified_inventory(self):
        result = verify_source(self.scan, self.manifest)
        self.assertTrue(result["passed"])
        self.assertEqual(result["files"], 1)
        self.assertEqual(result["bytes"], self.member.stat().st_size)

    def test_mutation_rejected(self):
        self.member.write_bytes(b'{"frame": 1}')
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_source(self.scan, self.manifest)

    def test_missing_member_rejected(self):
        self.member.unlink()
        with self.assertRaises(FileNotFoundError):
            verify_source(self.scan, self.manifest)

    def test_unlisted_frame_rejected(self):
        (self.scan / "frame_extra.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify_source(self.scan, self.manifest)

    def test_escaped_and_duplicate_members_rejected(self):
        for path in ("tea_room/../outside", "/tea_room/frame.json", "other/frame.json",
                     "tea_room/C:/outside", "tea_room\\frame.json"):
            with self.subTest(path=path):
                self.write_manifest([{**self.entry, "path": path}])
                with self.assertRaises(ValueError):
                    verify_source(self.scan, self.manifest)
        self.write_manifest([self.entry, self.entry])
        with self.assertRaises(ValueError):
            verify_source(self.scan, self.manifest)

    def test_bad_size_and_hash_rejected(self):
        for patch in ({"bytes": True}, {"bytes": -1}, {"sha256": "x" * 64}, {"sha256": None}):
            with self.subTest(patch=patch):
                self.write_manifest([{**self.entry, **patch}])
                with self.assertRaises(ValueError):
                    verify_source(self.scan, self.manifest)

    def test_existing_and_input_nested_outputs_rejected(self):
        blender = self.root / "blender.exe"
        blender.write_bytes(b"placeholder")
        for output in (self.scan, self.scan / "result", self.root):
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    validate_paths(self.scan, output, blender)
        existing = self.root / "prior-result"
        existing.mkdir()
        with self.assertRaisesRegex(ValueError, "must be new"):
            validate_paths(self.scan, existing, blender)
        self.assertEqual(validate_paths(self.scan, self.root / "new", blender)[1], self.root / "new")

    def test_failed_stage_keeps_log_and_stops(self):
        stages = []
        with self.assertRaisesRegex(RuntimeError, "failed"):
            run_stage("bad", [sys.executable, "-c", "print('failure evidence'); raise SystemExit(1)"],
                      self.root, stages)
        self.assertEqual(stages[0]["exit_code"], 1)
        self.assertIn("failure evidence", (self.root / "bad.log").read_text(encoding="utf-8"))

    def test_explicit_geometric_rejection_is_retained(self):
        stages = []
        self.assertEqual(run_stage("rejected", [sys.executable, "-c", "raise SystemExit(2)"],
                                   self.root, stages, allowed=(0, 2)), 2)
        self.assertEqual(stages[0]["exit_code"], 2)


if __name__ == "__main__":
    unittest.main()
