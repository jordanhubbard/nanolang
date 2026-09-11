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
            link_arguments = []
            if mode in ("quoted_library", "shell_library"):
                library_name = "probe space's" if mode == "quoted_library" else "probe$(touch injected)"
                library_dir = path / ("library space's" if mode == "quoted_library" else "library$(touch injected)")
                library_dir.mkdir()
                (path / "library.c").write_text("#include <stdint.h>\nint64_t probe_value(void) { return 42; }\n")
                subprocess.run(["cc", "-c", str(path / "library.c"), "-o", str(path / "library.o")],
                               check=True, capture_output=True, timeout=20)
                subprocess.run(["ar", "rcs", str(library_dir / ("lib" + library_name + ".a")), str(path / "library.o")],
                               check=True, capture_output=True, timeout=20)
                link_arguments = ["-L", str(library_dir), "-l", library_name]
            output_name = "program"
            if mode == "quoted_output":
                output_name = "program space's"
            elif mode == "shell_output":
                output_name = "program$(touch injected)"
            module_relative = "single_invocation_probe.nano"
            if mode in ("quoted_filename", "shell_filename"):
                module_relative = ("single_invocation_probe space's.nano" if mode == "quoted_filename" else
                                   "single_invocation_probe$(touch injected).nano")
            if mode in ("quoted_path", "shell_path"):
                parent = "odd space's" if mode == "quoted_path" else "odd$(touch injected)"
                (path / parent).mkdir()
                module_relative = parent + "/single_invocation_probe.nano"
            (path / module_relative).write_text(
                "module single_invocation_probe\n"
                "pub fn answer() -> int { return 42 }\n"
                "shadow answer { assert (== (answer) 42) }\n")
            (path / "main.nano").write_text(
                f'module "{module_relative}" as probe\n' +
                ("extern fn probe_value() -> int\n" if link_arguments else "") +
                "fn main() -> int { assert (== (probe.answer) 42) " +
                ("unsafe { assert (== (probe_value) 42) } " if link_arguments else "") +
                "return 0 }\n" +
                ("shadow main { assert (== (probe.answer) 42) }\n" if link_arguments else
                 "shadow main { assert (== (main) 0) }\n"))
            wrapper = path / "compiler.py"
            wrapper.write_text(WRAPPER)
            env = os.environ.copy()
            env.update(NANO_CC=f"{sys.executable} {wrapper}", PROBE_LOG=str(path / "calls"),
                       PROBE_MODE=mode, PROBE_REAL_CC=shutil.which("cc"), TMPDIR=directory)
            if mode == "long_command":
                env["NANO_CC"] = " " * 5000 + env["NANO_CC"]
            if mode == "quoted_tmp":
                temporary = path / "temporary space's$(touch injected)"
                temporary.mkdir()
                env["TMPDIR"] = str(temporary)
            compiler = str(Path(os.environ.get("NANOLANG_COMPILER", str(ROOT / "bin/nanoc_c"))).resolve())
            if mode in ("quoted_root", "shell_root"):
                root = path / ("checkout space's" if mode == "quoted_root" else "checkout$(touch injected)")
                (root / "bin").mkdir(parents=True)
                shutil.copy2(compiler, root / "bin/nanoc_c")
                for child in ("src", "modules", "scripts"):
                    (root / child).symlink_to(ROOT / child, target_is_directory=True)
                compiler = str(root / "bin/nanoc_c")
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
            process = subprocess.Popen([compiler, "main.nano", "-o", output_name] + link_arguments, cwd=path,
                                       env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, start_new_session=True)
            try:
                output, _ = process.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                output, _ = process.communicate()
                self.fail("I did not drain compiler diagnostics before waiting: " + output[-2000:])
            calls = (path / "calls").read_text().splitlines() if (path / "calls").exists() else []
            self.assertFalse((path / "injected").exists(), "module path executed a shell substitution")
            if mode == "long_command":
                self.assertEqual(calls, [], output[-4000:])
                self.assertNotEqual(process.returncode, 0)
                self.assertIn("I could not represent all module compiler arguments.", output)
                return
            self.assertEqual(calls, ["compile"], output[-4000:])
            if mode in ("success", "quoted_path", "shell_path", "quoted_output", "shell_output", "quoted_tmp", "quoted_library", "shell_library", "quoted_root", "shell_root", "quoted_filename", "shell_filename"):
                self.assertEqual(process.returncode, 0, output[-4000:])
                subprocess.run([str(path / output_name)], check=True, timeout=10)
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

    def test_module_directory_with_spaces_and_quote(self):
        self.check_compile("quoted_path")

    def test_module_directory_cannot_execute_shell_substitution(self):
        self.check_compile("shell_path")

    def test_truncated_command_is_not_executed(self):
        self.check_compile("long_command")

    def test_output_path_with_spaces_and_quote(self):
        self.check_compile("quoted_output")

    def test_output_path_cannot_execute_shell_substitution(self):
        self.check_compile("shell_output")

    def test_temporary_directory_remains_literal(self):
        self.check_compile("quoted_tmp")

    def test_library_names_and_directories_with_spaces_and_quotes(self):
        self.check_compile("quoted_library")

    def test_library_arguments_cannot_execute_shell_substitution(self):
        self.check_compile("shell_library")

    def test_checkout_directory_with_spaces_and_quote(self):
        self.check_compile("quoted_root")

    def test_checkout_directory_cannot_execute_shell_substitution(self):
        self.check_compile("shell_root")

    def test_module_filename_with_spaces_and_quote(self):
        self.check_compile("quoted_filename")

    def test_module_filename_cannot_execute_shell_substitution(self):
        self.check_compile("shell_filename")


if __name__ == "__main__":
    unittest.main()
