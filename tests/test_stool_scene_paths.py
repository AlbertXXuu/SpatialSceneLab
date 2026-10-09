"""Public dining integration must reject unsafe bundles and destructive output paths."""
import io
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

from blender_stool_scene import check_paths, validate_members


class ArchiveBoundaryTests(unittest.TestCase):
    def check_archive(self, entries):
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w') as archive:
            for name, content in entries:
                if isinstance(name, str):
                    entry = zipfile.ZipInfo(name)
                    # ZipInfo's Windows constructor normalizes separators; keep
                    # malicious wire names exactly as a third-party ZIP may do.
                    entry.filename = name
                    name = entry
                archive.writestr(name, content)
        data.seek(0)
        with zipfile.ZipFile(data) as archive:
            validate_members(archive)

    def test_public_archive_shape(self):
        self.check_archive([('before.blend', b'BLENDER'), ('surface/scene.json', b'{}')])

    def test_unused_unsafe_member_still_rejected(self):
        for name in ('../outside', '/absolute', 'C:/absolute', 'surface\\escape', 'surface/space. '):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.check_archive([('before.blend', b'BLENDER'), (name, b'x')])

    def test_case_insensitive_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            self.check_archive([('before.blend', b'a'), ('BEFORE.blend', b'b')])

    def test_symlink_rejected(self):
        entry = zipfile.ZipInfo('unrelated-link')
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaises(ValueError):
            self.check_archive([('before.blend', b'BLENDER'), (entry, b'../other')])

    def test_missing_scene_rejected(self):
        with self.assertRaises(ValueError):
            self.check_archive([('not-before.blend', b'BLENDER')])

    def test_output_and_temporary_extraction_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inputs = root / 'source'
            inputs.mkdir()
            archive, fitted = inputs / 'scene.zip', inputs / 'fitted.blend'
            archive.write_bytes(b'zip')
            fitted.write_bytes(b'blend')
            scratch = root / 'scratch'
            check_paths(archive, fitted, root / 'output', scratch, 'build')
            for output in (inputs, root, scratch / 'lost-on-cleanup', root / 'frozen' / 'target'):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    check_paths(archive, fitted, output, scratch, 'build')
            occupied = root / 'occupied'
            occupied.mkdir()
            keep = occupied / 'user-work.blend'
            keep.write_bytes(b'untouched')
            with self.assertRaises(ValueError):
                check_paths(archive, fitted, occupied, scratch, 'build')
            self.assertEqual(keep.read_bytes(), b'untouched')


if __name__ == '__main__':
    unittest.main()
