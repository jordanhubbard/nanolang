"""I check descriptor-relative listings whose joined entry paths exceed 1 KiB."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FilesystemArrayExports(unittest.TestCase):
    def test_listings(self):
        with tempfile.TemporaryDirectory(prefix="nano-fs-arrays-") as tmp:
            directory = Path(tmp)
            for _ in range(4):
                directory /= "d" * 200
                directory.mkdir()
            long_name = "z" * 220 + ".txt"
            self.assertGreater(len(str(directory / long_name)), 1024)
            fd = os.open(directory, os.O_RDONLY)
            try:
                for name in ("a.txt", "B.TXT", long_name):
                    file_fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=fd)
                    os.close(file_fd)
                os.mkdir("folder", dir_fd=fd)
                os.symlink("a.txt", "alias.TXT", dir_fd=fd)
                os.symlink("folder", "linked", dir_fd=fd)
                os.symlink("missing", "broken", dir_fd=fd)
            finally:
                os.close(fd)
            output = Path(tmp) / "probe"
            built = subprocess.run([
                *shlex.split(os.environ.get("CC", "cc")), "-std=c99", "-g",
                "-Wall", "-Wextra", "-Werror", "-Isrc",
                "tests/test_filesystem_array_exports.c", "modules/filesystem/filesystem.c",
                "src/runtime/dyn_array.c", "src/runtime/gc.c", "src/runtime/gc_struct.c",
                "-o", str(output)], cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            ran = subprocess.run([str(output), str(directory), long_name],
                                 capture_output=True, text=True, timeout=15)
            self.assertEqual(ran.returncode, 0, ran.stderr)


if __name__ == "__main__":
    unittest.main()
