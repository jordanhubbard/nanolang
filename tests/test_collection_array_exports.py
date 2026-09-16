"""I check four native array declarations without certifying string ownership."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CollectionArrayExports(unittest.TestCase):
    def test_snapshots(self):
        with tempfile.TemporaryDirectory(prefix="nano-collection-arrays-") as tmp:
            output = Path(tmp) / "snapshots"
            built = subprocess.run([
                *shlex.split(os.environ.get("CC", "cc")), "-std=c99", "-g",
                "-Wall", "-Wextra", "-Werror", "-Isrc",
                "tests/test_collection_array_exports.c",
                "modules/std/collections/collections.c", "modules/std/json/json.c",
                "src/cJSON.c", "src/runtime/dyn_array.c", "src/runtime/gc.c",
                "src/runtime/gc_struct.c", "-lm", "-o", str(output)],
                cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            ran = subprocess.run([str(output)], capture_output=True, text=True, timeout=15)
            self.assertEqual(ran.returncode, 0, ran.stderr)


if __name__ == "__main__":
    unittest.main()
