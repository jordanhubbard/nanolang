"""I check one compiler invocation supplies diagnostics and exit status."""
import os
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = r'''import os, pathlib, sys, json, time
args = sys.argv[1:]
if "-c" in args and any("single_invocation_probe" in arg for arg in args):
    log = pathlib.Path(os.environ["PROBE_LOG"])
    first = not log.exists()
    with log.open("a") as stream:
        stream.write("compile\n")
    mode = os.environ["PROBE_MODE"]
    if mode == "missing":
        sys.exit(0)
    if mode == "partial":
        pathlib.Path(args[args.index("-o") + 1]).write_bytes(b"partial object")
        sys.stderr.write("I failed this compiler invocation.\n")
        sys.exit(7)
    if mode == "overlap":
        identity = os.environ["PROBE_ID"]
        pathlib.Path(identity + ".ready").write_text(json.dumps(args))
        deadline = time.monotonic() + 45
        while not pathlib.Path(identity + ".release").exists():
            if time.monotonic() > deadline:
                sys.exit(8)
            time.sleep(0.01)
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
            if mode == "overlap":
                processes = []
                try:
                    for identity in ("one", "two"):
                        processes.append(subprocess.Popen(
                            [compiler, "main.nano", "-o", "program-" + identity], cwd=path,
                            env=dict(env, PROBE_ID=identity), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True))
                    deadline = time.monotonic() + 40
                    while not all((path / (name + ".ready")).exists() for name in ("one", "two")):
                        self.assertTrue(all(p.poll() is None for p in processes), "compiler exited before barrier")
                        self.assertLess(time.monotonic(), deadline, "compiler barrier timed out")
                        time.sleep(0.01)
                    commands = [json.loads((path / (name + ".ready")).read_text()) for name in ("one", "two")]
                    sources = [path / args[-1] for args in commands]
                    objects = [path / args[args.index("-o") + 1] for args in commands]
                    self.assertNotEqual(sources[0], sources[1], "shared generated source")
                    self.assertNotEqual(objects[0], objects[1], "shared output during compilation")
                    for source in sources:
                        self.assertEqual(source.parent.stat().st_mode & 0o777, 0o700)
                    (path / "one.release").touch()
                    output, _ = processes[0].communicate(timeout=45)
                    self.assertEqual(processes[0].returncode, 0, output[-4000:])
                    self.assertFalse(sources[0].exists())
                    self.assertTrue(sources[1].exists(), "first build removed the second build's source")
                    (path / "two.release").touch()
                    output, _ = processes[1].communicate(timeout=45)
                    self.assertEqual(processes[1].returncode, 0, output[-4000:])
                    self.assertFalse(sources[1].parent.exists())
                    for identity in ("one", "two"):
                        subprocess.run([str(path / ("program-" + identity))], check=True, timeout=10)
                    self.assertEqual((path / "calls").read_text().splitlines(), ["compile", "compile"])
                finally:
                    for process in processes:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                        process.communicate()
                return
            cached = path / "obj/nano_modules/single_invocation_probe.o"
            if mode in ("partial", "missing"):
                cached.parent.mkdir(parents=True)
                cached.write_bytes(b"previous object")
                os.utime(cached, (1, 1))
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
                self.assertIn("I could not publish the module object" if mode == "missing" else
                              "I failed this compiler invocation.", output)
                self.assertFalse((path / "program").exists())
                if mode in ("partial", "missing"):
                    self.assertEqual(cached.read_bytes(), b"previous object")
                    self.assertFalse(list(cached.parent.glob("*.build.*/object.o")))

    def test_success_runs_compiler_once(self):
        self.check_compile("success")

    def test_first_failure_is_not_hidden_by_retry(self):
        self.check_compile("fail_first")

    def test_large_diagnostics_are_drained(self):
        self.check_compile("flood")

    def test_overlapping_builds_have_private_intermediates(self):
        self.check_compile("overlap")

    def test_failed_partial_object_does_not_replace_cache(self):
        self.check_compile("partial")

    def test_success_without_an_object_cannot_publish(self):
        self.check_compile("missing")


if __name__ == "__main__":
    unittest.main()
