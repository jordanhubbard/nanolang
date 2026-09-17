"""I test PEG capture snapshots and allocation failures with production code."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PegArrays(unittest.TestCase):
    def test_captures(self):
        with tempfile.TemporaryDirectory(prefix="nano-peg-arrays-") as tmp:
            output = Path(tmp) / "probe"
            built = subprocess.run([
                *shlex.split(os.environ.get("CC", "cc")), "-std=c99", "-g",
                "-Wall", "-Wextra", "-Werror", "-D_GNU_SOURCE", "-Isrc",
                "tests/test_peg_array_exports.c", "src/runtime/dyn_array.c",
                "src/runtime/gc.c", "src/runtime/gc_struct.c", "-o", str(output)],
                cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            ran = subprocess.run([str(output)], capture_output=True, text=True, timeout=15)
            self.assertEqual(ran.returncode, 0, ran.stderr)


if __name__ == "__main__":
    unittest.main()
