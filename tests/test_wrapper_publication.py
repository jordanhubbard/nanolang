"""I publish complete wrappers without treating filenames as commands."""
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest

from tests.test_bytecode_shadows import ROOT


class WrapperPublication(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="nano-wrapper-safe-")
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.env = os.environ.copy()
        for key in ("NANO_CC", "CC", "NANO_BUILD_CACHE", "NANO_ALLOW_PACKAGE_INSTALL"):
            self.env.pop(key, None)
        self.env["NANO_VIRT_LIB"] = str(ROOT / "obj")
        self.program = self.directory / "program.nano"
        self.program.write_text("fn main() -> int { return 42 }\nshadow main { assert (== (main) 42) }\n")
        self.output = self.directory / "program"

    def start(self, daemon=False, env=None, output=None):
        args = [str(ROOT / "bin/nano_virt"), str(self.program), "-o", str(output or self.output)]
        if daemon:
            args.append("--daemon-wrapper")
        process = subprocess.Popen(args, cwd=self.directory, env=env or self.env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        self.addCleanup(self.stop, process)
        return process

    def stop(self, process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=10)

    def finish(self, process, success=True):
        stdout, stderr = process.communicate(timeout=25)
        if success:
            self.assertEqual(process.returncode, 0, (stdout, stderr))
        else:
            self.assertNotEqual(process.returncode, 0, (stdout, stderr))
        return stdout, stderr

    def execute(self, path=None):
        result = subprocess.run([str(path or self.output)], cwd=self.directory,
                                env=self.env, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 42, (result.stdout, result.stderr))

    def compiler(self, phase):
        wrapper = self.directory / "compiler fixture"
        observation = self.directory / "observation.json"
        release = self.directory / "release"
        wrapper.write_text(f'''#!{sys.executable}
import json, pathlib, stat, subprocess, sys, time
output = pathlib.Path(sys.argv[sys.argv.index("-o") + 1])
source = next(pathlib.Path(arg) for arg in sys.argv[1:] if arg.endswith(".c"))
pathlib.Path({str(observation)!r}).write_text(json.dumps({{"directory": str(output.parent), "source_directory": str(source.parent), "mode": stat.S_IMODE(output.parent.stat().st_mode)}}))
if {phase!r} == "wait":
    while not pathlib.Path({str(release)!r}).exists(): time.sleep(0.01)
if {phase!r} in ("partial", "wait"):
    output.write_bytes(b"partial")
    sys.exit(19)
if {phase!r} == "empty":
    output.touch()
    output.chmod(0o700)
    sys.exit(0)
if {phase!r} == "symlink":
    output.symlink_to({str(self.output)!r})
    sys.exit(0)
if {phase!r} == "hardlink":
    import os
    os.link({str(self.output)!r}, output)
    sys.exit(0)
if {phase!r} == "nonexec":
    output.write_bytes(b"not executable")
    output.chmod(0o600)
    sys.exit(0)
if {phase!r} == "missing": sys.exit(0)
sys.exit(subprocess.run([{shutil.which('cc')!r}] + sys.argv[1:]).returncode)
''')
        wrapper.chmod(0o700)
        env = self.env.copy()
        # I preserve compiler command-fragment configuration, including quoting.
        env["NANO_CC"] = shlex.quote(str(wrapper))
        return env, observation, release

    def test_literal_paths_in_both_modes(self):
        odd = self.directory / "literal ' \" ; $(touch injected) \\ \n directory"
        odd.mkdir()
        alias = self.directory / "-objects ' ; $(touch injected)"
        alias.symlink_to(ROOT / "obj", target_is_directory=True)
        self.env["NANO_VIRT_LIB"] = alias.name
        for daemon in (False, True):
            with self.subTest(daemon=daemon):
                output = odd / ("daemon" if daemon else "standalone")
                self.finish(self.start(daemon, output=output))
                self.assertTrue(os.access(output, os.X_OK))
                if not daemon:
                    self.execute(output)
                self.assertFalse((self.directory / "injected").exists())
                self.assertFalse(list(odd.glob(".nano-wrapper-*")))

    def test_long_quoted_object_alias_in_both_modes(self):
        # I keep each component portable while the complete link closure grows.
        alias = self.directory / ("objects-" + "x" * 220 + " ' ; $(touch injected)")
        self.assertLessEqual(len(os.fsencode(alias.name)), 255)
        alias.symlink_to(ROOT / "obj", target_is_directory=True)
        self.env["NANO_VIRT_LIB"] = alias.name
        for daemon in (False, True):
            with self.subTest(daemon=daemon):
                output = self.directory / ("long-daemon" if daemon else "long-standalone")
                self.finish(self.start(daemon, output=output))
                self.assertTrue(os.access(output, os.X_OK))
                if not daemon:
                    self.execute(output)
                self.assertFalse((self.directory / "injected").exists())
                self.assertFalse(list(self.directory.glob(".nano-wrapper-*")))

    def test_failed_compilers_preserve_existing_output(self):
        self.finish(self.start())
        old = self.output.read_bytes()
        for daemon in (False, True):
            for phase in ("partial", "empty", "symlink", "hardlink", "nonexec", "missing"):
                with self.subTest(daemon=daemon, phase=phase):
                    env, observation, _ = self.compiler(phase)
                    self.finish(self.start(daemon, env), success=False)
                    self.assertEqual(self.output.read_bytes(), old)
                    self.execute()
                    info = json.loads(observation.read_text())
                    self.assertEqual(info["mode"], 0o700)
                    self.assertEqual(info["directory"], info["source_directory"])
                    self.assertEqual(Path(info["directory"]).parent, self.directory.resolve())
                    self.assertFalse(Path(info["directory"]).exists())

    def test_overlapping_failure_does_not_replace_success(self):
        self.finish(self.start())
        for daemon in (False, True):
            with self.subTest(daemon=daemon):
                env, observation, release = self.compiler("wait")
                observation.unlink(missing_ok=True)
                release.unlink(missing_ok=True)
                slow = self.start(daemon, env)
                deadline = time.monotonic() + 10
                while not observation.exists() and slow.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(observation.exists())
                self.assertIsNone(slow.poll())
                self.execute()
                self.finish(self.start())
                new = self.output.read_bytes()
                release.touch()
                self.finish(slow, success=False)
                self.assertEqual(self.output.read_bytes(), new)
                self.assertFalse(list(self.directory.glob(".nano-wrapper-*")))
                self.execute()

    def test_import_path_is_a_c_string_not_source(self):
        parent = self.directory / "foreign ' \" \\ literal"
        parent.mkdir()
        module = parent / "api.nano"
        module.write_text("pub extern fn labs(value: int) -> int\n")
        self.program.write_text(f'''module {json.dumps(str(module))} as foreign
fn main() -> int {{ unsafe {{ return (foreign.labs -42) }} }}
shadow main {{ assert (== (main) 42) }}
''')
        self.finish(self.start())
        self.execute()

    def test_destination_symlink_is_replaced_not_followed(self):
        victim = self.directory / "retained"
        victim.write_bytes(b"I retain the symlink target")
        for daemon in (False, True):
            with self.subTest(daemon=daemon):
                self.output.unlink(missing_ok=True)
                self.output.symlink_to(victim)
                self.finish(self.start(daemon))
                self.assertFalse(self.output.is_symlink())
                self.assertEqual(victim.read_bytes(), b"I retain the symlink target")
                if not daemon:
                    self.execute()

    def test_failed_publication_preserves_directory(self):
        self.output.mkdir()
        sentinel = self.output / "retained"
        sentinel.write_bytes(b"retain")
        for daemon in (False, True):
            with self.subTest(daemon=daemon):
                self.finish(self.start(daemon), success=False)
                self.assertEqual(sentinel.read_bytes(), b"retain")
                self.assertFalse(list(self.directory.glob(".nano-wrapper-*")))

    def test_killed_compilation_leaves_output_and_ignores_orphan(self):
        self.finish(self.start())
        old = self.output.read_bytes()
        for daemon in (False, True):
            with self.subTest(daemon=daemon):
                env, observation, _ = self.compiler("wait")
                observation.unlink(missing_ok=True)
                slow = self.start(daemon, env)
                deadline = time.monotonic() + 10
                while not observation.exists() and slow.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(observation.exists())
                self.assertIsNone(slow.poll())
                self.stop(slow)
                self.assertEqual(self.output.read_bytes(), old)
                orphan = Path(json.loads(observation.read_text())["directory"])
                retained = (orphan / "source.c").read_bytes()
                self.finish(self.start())
                self.execute()
                self.assertEqual((orphan / "source.c").read_bytes(), retained)
                old = self.output.read_bytes()


if __name__ == "__main__":
    unittest.main()
