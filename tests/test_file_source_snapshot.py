"""I qualify immutable compiler-input companions without service execution."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class FileSourceSnapshot(unittest.TestCase):
    def test_companion_lifetime_and_failures(self):
        with tempfile.TemporaryDirectory(prefix="nano-file-snapshot-") as directory:
            executable = Path(directory) / "snapshot"
            compiler = shlex.split(os.environ.get("NANO_FILE_SNAPSHOT_CC",
                os.environ.get("NANO_NATIVE_TEST_CC", os.environ.get("CC", "cc"))))
            flags = ["-std=c11", "-D_GNU_SOURCE", "-D_DARWIN_C_SOURCE", "-Wall", "-Wextra", "-Werror", "-g", "-O1"]
            if os.environ.get("NANO_FILE_SNAPSHOT_SANITIZERS") == "1":
                flags += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all"]
            sources = [ROOT / "tests/test_file_source_snapshot.c"]
            sources += [ROOT / "src" / (name + ".c") for name in
                        ("nsi_file_binding", "nsi_file_plan", "nsi_socket_binding", "nsi_socket_plan", "nsi", "cJSON", "utf8")]
            build = subprocess.run([*compiler, *flags, "-I", str(ROOT / "src"), *sources, "-o", executable],
                                   cwd=ROOT, capture_output=True, timeout=120)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            for catalog in (1, 2):
                run = subprocess.run([executable, ROOT / "tests/fixtures/nsi_file_plan.json", ROOT / "tests/fixtures/nsi_socket_plan.json", str(catalog)], cwd=ROOT,
                                     capture_output=True, timeout=60,
                                     env={**os.environ, "ASAN_OPTIONS": "detect_leaks=1"})
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                self.assertIn(b"PASS immutable companion", run.stdout)

if __name__ == "__main__":
    unittest.main()
