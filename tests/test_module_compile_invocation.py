"""I check one compiler invocation supplies diagnostics and exit status."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = r'''import os, pathlib, sys
args = sys.argv[1:]
if "-c" in args and any("single_invocation_probe" in arg for arg in args):
    log = pathlib.Path(os.environ["PROBE_LOG"])
    first = not log.exists()
    with log.open("a") as stream:
        stream.write("compile\n")
    mode = os.environ["PROBE_MODE"]
    if mode == "flood" or (mode == "fail_first" and first):
        sys.stderr.write("I failed this compiler invocation.\n")
        if mode == "flood":
            sys.stderr.write("diagnostic\n" * 32768)
        sys.exit(7)
os.execv(os.environ["PROBE_REAL_CC"], [os.environ["PROBE_REAL_CC"]] + args)
'''


class ModuleCompileInvocation(unittest.TestCase):
    def check_compile(self, mode):
        with tempfile.TemporaryDirectory(prefix="nanolang-module-invocation-") as directory:
            path = Path(directory)
            (path / "single_invocation_probe.nano").write_text(
                "module single_invocation_probe\n"
                "pub fn answer() -> int { return 42 }\n"
                "shadow answer { assert (== (answer) 42) }\n")
            (path / "main.nano").write_text(
                'module "single_invocation_probe.nano" as probe\n'
                "fn main() -> int { assert (== (probe.answer) 42) return 0 }\n"
                "shadow main { assert (== (main) 0) }\n")
            wrapper = path / "compiler.py"
            wrapper.write_text(WRAPPER)
            env = os.environ.copy()
            env.update(NANO_CC=f"{sys.executable} {wrapper}", PROBE_LOG=str(path / "calls"),
                       PROBE_MODE=mode, PROBE_REAL_CC=shutil.which("cc"), TMPDIR=directory)
            compiler = str(Path(os.environ.get("NANOLANG_COMPILER", str(ROOT / "bin/nanoc_c"))).resolve())
            process = subprocess.Popen([compiler, "main.nano", "-o", "program"], cwd=path,
                                       env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, start_new_session=True)
            try:
                output, _ = process.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                output, _ = process.communicate()
                self.fail("I did not drain compiler diagnostics before waiting: " + output[-2000:])
            calls = (path / "calls").read_text().splitlines() if (path / "calls").exists() else []
            self.assertEqual(calls, ["compile"], output[-4000:])
            if mode == "success":
                self.assertEqual(process.returncode, 0, output[-4000:])
                subprocess.run([str(path / "program")], check=True, timeout=10)
            else:
                self.assertNotEqual(process.returncode, 0, output[-4000:])
                self.assertIn("I failed this compiler invocation.", output)
                self.assertFalse((path / "program").exists())

    def test_success_runs_compiler_once(self):
        self.check_compile("success")

    def test_first_failure_is_not_hidden_by_retry(self):
        self.check_compile("fail_first")

    def test_large_diagnostics_are_drained(self):
        self.check_compile("flood")


if __name__ == "__main__":
    unittest.main()
