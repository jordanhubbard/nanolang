"""I qualify private File CODE preparation, never certification or execution."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]

class FileCode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix="nano-file-code-"))
        print(f"I retain File CODE artifacts at {cls.artifacts}", flush=True)
        cls.compiler = shlex.split(os.environ.get("NANO_FILE_CODE_CC", "cc"))
        cls.flags = ["-std=c11", "-D_DEFAULT_SOURCE", "-g", "-O1", "-Wall", "-Wextra", "-Werror",
                     "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        cls.flags += shlex.split(os.environ.get("NANO_FILE_CODE_CFLAGS", ""))
        cls.objects = shlex.split(os.environ["FILE_CODE_OBJECTS"])
        cls.ldflags = shlex.split(os.environ.get("FILE_CODE_LDFLAGS", "-lm -lcrypto"))

    def qualify(self, name, instrument):
        exe = self.artifacts / name
        sources = ["tests/nanoisa/test_file_code.c", "src/nanoisa/service_file_nominal.c",
                   "src/nsi_file_plan.c"]
        if not instrument:
            sources += ["src/nanoisa/service_file_nominal_plan.c", "src/nanoisa/file_flow.c"]
        command = [*self.compiler, *self.flags, *(["-DFLOW_INSTRUMENT"] if instrument else []),
                   *sources, *self.objects, *self.ldflags, "-o", str(exe)]
        env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                   UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
        for label, args in (("build", command), ("run", [str(exe)])):
            (self.artifacts / f"{name}-{label}-command.txt").write_text(shlex.join(args) + "\n")
            result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
            (self.artifacts / f"{name}-{label}.log").write_text(result.stdout + result.stderr)
            self.assertEqual(result.returncode, 0, (args, result.stdout, result.stderr))
            if label == "run":
                self.assertIn("PASS", result.stdout)
                print(result.stdout.strip(), flush=True)

    def test_instrumented_preparation_allocation_recovery(self):
        self.qualify("instrumented", True)

    def test_linked_decode_graphs_lifetimes_and_refusal(self):
        self.qualify("linked", False)

if __name__ == "__main__":
    unittest.main()
