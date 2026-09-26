"""I qualify private real-file ownership; no public source/import admission."""
try:
    from tests.sanitizer_options import asan_options
except ModuleNotFoundError:
    from sanitizer_options import asan_options
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LocalFileService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix="nano-local-file-"))
        print(f"I retain private file artifacts at {cls.artifacts}", flush=True)
        cls.compiler = shlex.split(os.environ.get("NANO_FILE_TEST_CC", "cc"))
        cls.flags = ["-std=c11", "-D_DEFAULT_SOURCE", "-g", "-O1", "-Wall", "-Wextra",
                     "-Werror", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        cls.flags += shlex.split(os.environ.get("NANO_FILE_TEST_CFLAGS", ""))

    def qualify(self, name, sources):
        exe = self.artifacts / name
        env = dict(os.environ, ASAN_OPTIONS=asan_options("halt_on_error=1"),
                   UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
        for label, command in (("build", [*self.compiler, *self.flags, *sources, "-o", str(exe)]),
                               ("run", [str(exe)])):
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            (self.artifacts / f"{name}-{label}.log").write_text(result.stdout + result.stderr)
            self.assertEqual(result.returncode, 0, (command, result.stdout, result.stderr))
            if label == "run":
                self.assertIn("PASS", result.stdout)
                print(result.stdout.strip(), flush=True)

    def test_instrumented_lifetimes_and_faults(self):
        self.qualify("instrumented", ["tests/test_nsi_file.c"])

    def test_ordinary_linked_adapter(self):
        self.qualify("linked", ["tests/test_nsi_file_linked.c", "src/nsi_file.c", "src/nsi_cap.c"])


if __name__ == "__main__":
    unittest.main()
