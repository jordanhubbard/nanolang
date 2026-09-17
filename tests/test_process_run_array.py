"""I test shell capture without rewritten commands or reopened temp paths."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ProcessRunArray(unittest.TestCase):
    def test_capture(self):
        with tempfile.TemporaryDirectory(prefix="nano-process-run-") as tmp:
            output = Path(tmp) / "probe"
            built = subprocess.run([
                *shlex.split(os.environ.get("CC", "cc")), "-std=c99", "-g",
                "-Wall", "-Wextra", "-Werror", "-Isrc",
                "tests/test_process_run_array.c", "src/runtime/dyn_array.c",
                "src/runtime/gc.c", "src/runtime/gc_struct.c", "-o", str(output)],
                cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            ran = subprocess.run([str(output)], capture_output=True, text=True, timeout=15)
            self.assertEqual(ran.returncode, 0, ran.stderr)


if __name__ == "__main__":
    unittest.main()
