"""I check playlist validation before truncation and report stream failures."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PreferencesSave(unittest.TestCase):
    def test_boundary(self):
        with tempfile.TemporaryDirectory(prefix="nano-prefs-save-build-") as tmp:
            output = Path(tmp) / "probe"
            built = subprocess.run([
                *shlex.split(os.environ.get("CC", "cc")), "-std=c99",
                "-Wall", "-Wextra", "-Werror", "-g",
                "tests/test_preferences_save.c", "-o", str(output)],
                cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            ran = subprocess.run([str(output)], capture_output=True,
                                 text=True, timeout=15)
            self.assertEqual(ran.returncode, 0, ran.stderr)


if __name__ == "__main__":
    unittest.main()
