"""I qualify non-admitting cyclic File query facts; no File module executes."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FileCyclic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix="nano-file-cyclic-"))
        print(f"I retain cyclic query artifacts at {cls.artifacts}", flush=True)
        cls.compiler = shlex.split(os.environ.get("NANO_FILE_CYCLIC_CC", "cc"))
        cls.flags = ["-std=c11", "-D_DEFAULT_SOURCE", "-g", "-O1", "-Wall", "-Wextra", "-Werror"]
        cls.flags += shlex.split(os.environ.get("NANO_FILE_CYCLIC_CFLAGS", ""))
        cls.objects = shlex.split(os.environ["FILE_CYCLIC_OBJECTS"])
        cls.ldflags = shlex.split(os.environ.get("FILE_CYCLIC_LDFLAGS", "-lm -lcrypto"))

    def command(self, name, args):
        (self.artifacts / f"{name}-command.txt").write_text(shlex.join(args) + "\n")
        env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                   UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
        process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=True)
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=240)
        except subprocess.TimeoutExpired as first:
            timed_out = True
            stdout, stderr = first.output or b"", first.stderr or b""
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    stdout, stderr = process.communicate(timeout=5)
                    break
                except subprocess.TimeoutExpired as later:
                    stdout, stderr = later.output or stdout, later.stderr or stderr
        (self.artifacts / f"{name}-stdout.log").write_bytes(stdout)
        (self.artifacts / f"{name}-stderr.log").write_bytes(stderr)
        (self.artifacts / f"{name}-status.json").write_text(json.dumps({
            "timeout": timed_out, "returncode": process.returncode,
            "reaped": process.returncode is not None, "bound_seconds": 240,
        }, indent=2) + "\n")
        self.assertFalse(timed_out, f"I retained the bounded terminal at {self.artifacts}")
        self.assertEqual(process.returncode, 0, (args, stdout, stderr))
        return stdout

    def qualify(self, name, instrument):
        executable = self.artifacts / name
        sources = ["tests/nanoisa/test_file_cyclic.c", "src/nanoisa/service_file_nominal.c", "src/nsi_file_plan.c"]
        if not instrument:
            sources += ["src/nanoisa/service_file_nominal_plan.c", "src/nanoisa/file_flow.c"]
        command = [*self.compiler, *self.flags, *(["-DFLOW_INSTRUMENT"] if instrument else []),
                   *sources, *self.objects, *self.ldflags, "-o", str(executable)]
        self.command(f"{name}-build", command)
        stdout = self.command(f"{name}-run", [str(executable)])
        self.assertIn(b"private cyclic query checks; no pending module execution", stdout)
        print(stdout.decode(errors="replace").strip(), flush=True)

    def test_instrumented_cyclic_query(self):
        self.qualify("instrumented", True)

    def test_linked_cyclic_query(self):
        self.qualify("linked", False)


if __name__ == "__main__":
    unittest.main()
