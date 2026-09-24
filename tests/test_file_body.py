"""I qualify private acyclic File bodies, never hosted authority or execution."""
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

class FileBody(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix="nano-file-body-"))
        print(f"I retain File body artifacts at {cls.artifacts}", flush=True)
        cls.compiler = shlex.split(os.environ.get("NANO_FILE_BODY_CC", "cc"))
        cls.flags = ["-std=c11", "-D_DEFAULT_SOURCE", "-g", "-O1", "-Wall", "-Wextra", "-Werror",
                     "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        cls.flags += shlex.split(os.environ.get("NANO_FILE_BODY_CFLAGS", ""))
        cls.objects = shlex.split(os.environ["FILE_BODY_OBJECTS"])
        cls.ldflags = shlex.split(os.environ.get("FILE_BODY_LDFLAGS", "-lm -lcrypto"))

    def qualify(self, name, instrument):
        exe = self.artifacts / name
        sources = ["tests/nanoisa/test_file_body.c", "src/nanoisa/service_file_nominal.c",
                   "src/nsi_file_plan.c"]
        if not instrument:
            sources += ["src/nanoisa/service_file_nominal_plan.c", "src/nanoisa/file_flow.c"]
        command = [*self.compiler, *self.flags, *(["-DFLOW_INSTRUMENT"] if instrument else []),
                   *sources, *self.objects, *self.ldflags, "-o", str(exe)]
        env = dict(os.environ, ASAN_OPTIONS=asan_options("halt_on_error=1"),
                   UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
        for label, args in (("build", command), ("run", [str(exe)])):
            (self.artifacts / f"{name}-{label}-command.txt").write_text(shlex.join(args) + "\n")
            result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
            (self.artifacts / f"{name}-{label}.log").write_text(result.stdout + result.stderr)
            self.assertEqual(result.returncode, 0, (args, result.stdout, result.stderr))
            if label == "run":
                self.assertIn("PASS", result.stdout)
                print(result.stdout.strip(), flush=True)

    def test_instrumented_body_allocation_recovery(self):
        self.qualify("instrumented", True)

    def test_linked_bodies_lifetimes_and_refusal(self):
        self.qualify("linked", False)

if __name__ == "__main__":
    unittest.main()
