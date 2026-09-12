"""I protect cached foreign artifacts while compilers fail or overlap."""
import json
import os
from pathlib import Path
import select
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest

from tests import test_bytecode_shadows as shadows

ROOT = shadows.ROOT


class ModuleCachePublication(unittest.TestCase):
    def setUp(self):
        self.support = shadows.BytecodeShadows()

    def snapshot(self, cache):
        return {p.name: p.read_bytes() for p in cache.iterdir()
                if p.is_file() and p.name != ".build.lock"}

    def start(self, directory, source, env):
        directory.mkdir(exist_ok=True)
        program = directory / "program.nano"
        program.write_text(source)
        return subprocess.Popen([str(ROOT / "bin/nano_virt"), str(program), "--emit-nvm",
                                 "-o", str(directory / "program.nvm")], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)

    def stop(self, process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=10)

    def wait_ready(self, ready, process):
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(ready.exists(), "I did not reach the controlled compiler boundary")
        self.assertIsNone(process.poll())

    def test_partial_outputs_preserve_cache_and_recover(self):
        for phase in ("object", "library", "symlink"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory(prefix="nano-publish-") as tmp:
                directory = Path(tmp)
                module, source, env = self.support.foreign_build_fixture(directory)
                result, output = self.support.compile(source, directory, "--run", env=env)
                self.assertEqual(result.returncode, 42, result.stderr)
                cache = module / ".build"
                old = self.snapshot(cache)
                old_program = output.read_bytes()
                c_source = module / "answer.c"
                c_source.write_text(c_source.read_text().replace("42", "43"))
                compiler = directory / "cc-fixture"
                observed = directory / "observed.json"
                compiler.write_text(f'''#!{sys.executable}
import json, os, pathlib, stat, subprocess, sys
shared = "-dynamiclib" in sys.argv or "-shared" in sys.argv
if ({phase!r} in ("object", "symlink") and "-c" in sys.argv) or ({phase!r} == "library" and shared):
    output = pathlib.Path(sys.argv[sys.argv.index("-o") + 1])
    pathlib.Path({str(observed)!r}).write_text(json.dumps({{"directory": str(output.parent), "mode": stat.S_IMODE(output.parent.stat().st_mode)}}))
    if {phase!r} == "symlink":
        subprocess.run([{shutil.which('cc')!r}] + sys.argv[1:], check=True)
        output.unlink()
        output.symlink_to({str(cache / 'answer_native.o')!r})
        sys.exit(0)
    output.write_bytes(b"partial compiler output")
    sys.exit(24)
os.execv({shutil.which('cc')!r}, [{shutil.which('cc')!r}] + sys.argv[1:])
''')
                compiler.chmod(0o700)
                env["NANO_CC"] = str(compiler)
                result, output = self.support.compile(source.replace("42", "43"), directory, env=env)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.snapshot(cache), old)
                self.assertEqual(output.read_bytes(), old_program)
                self.assertEqual(self.support.execute(output, env=env).returncode, 42)
                observation = json.loads(observed.read_text())
                self.assertNotEqual(Path(observation["directory"]), cache)
                self.assertEqual(observation["mode"], 0o700)
                self.assertFalse(list(cache.glob(".nano-build-*")))
                env.pop("NANO_CC")
                result, output = self.support.compile(source.replace("42", "43"), directory, "--run", env=env)
                self.assertEqual(result.returncode, 43, result.stderr)
                self.assertFalse(list(cache.glob(".nano-build-*")))

    def test_multi_source_and_private_dependencies_publish(self):
        with tempfile.TemporaryDirectory(prefix="nano-publish-") as tmp:
            directory = Path(tmp)
            module, source, env = self.support.foreign_build_fixture(directory)
            (module / "answer.c").write_text("#include <stdint.h>\nint64_t helper(void); int64_t private_value(void);\nint64_t nano_build_answer(void) { return helper() + private_value(); }\n")
            (module / "helper.c").write_text("#include <stdint.h>\nint64_t helper(void) { return 40; }\n")
            (module / "private.c").write_text("#include <stdint.h>\nint64_t private_value(void) { return 2; }\n")
            (module / "module.json").write_text(json.dumps({"name": "answer_native", "c_sources": ["answer.c", "helper.c"], "shared_c_sources": ["private.c"]}))
            result, output = self.support.compile(source, directory, "--run", env=env)
            self.assertEqual(result.returncode, 42, result.stderr)
            self.assertEqual(self.support.execute(output, env=env).returncode, 42)
            cache = module / ".build"
            for name in ("answer_native.o", "answer_native_0.o", "answer_native_1.o", "answer_native_0.d", "answer_native_1.d", "__shared_0.o", "__shared_0.d", "source_hashes.json"):
                self.assertGreater((cache / name).stat().st_size, 0)
            self.assertFalse(list(cache.glob(".nano-build-*")))

    def test_failed_publication_invalidates_hash_evidence(self):
        with tempfile.TemporaryDirectory(prefix="nano-publish-") as tmp:
            directory = Path(tmp)
            module, source, env = self.support.foreign_build_fixture(directory)
            result, output = self.support.compile(source, directory, "--run", env=env)
            self.assertEqual(result.returncode, 42, result.stderr)
            old_program = output.read_bytes()
            cache = module / ".build"
            extension = "dylib" if sys.platform == "darwin" else "so"
            library = cache / f"libanswer_native.{extension}"
            library.unlink()
            library.mkdir()  # I force the library rename to fail after the object rename.
            result, output = self.support.compile(source, directory, env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"could not publish complete", result.stderr)
            self.assertEqual(output.read_bytes(), old_program)
            self.assertFalse((cache / "source_hashes.json").exists())
            library.rmdir()
            result, output = self.support.compile(source, directory, "--run", env=env)
            self.assertEqual(result.returncode, 42, result.stderr)

    def blocking_compiler(self, directory):
        compiler = directory / "cc-blocked"
        compiler.write_text(f'''#!{sys.executable}
import os, pathlib, sys, time
if "-c" in sys.argv:
    with open({str(directory / 'calls')!r}, "a") as log:
        log.write(str(os.getpid()) + "\\n")
    output = pathlib.Path(sys.argv[sys.argv.index("-o") + 1])
    output.write_bytes(b"partial compiler output")
    pathlib.Path({str(directory / 'ready')!r}).touch()
    deadline = time.monotonic() + 15
    while not pathlib.Path({str(directory / 'release')!r}).exists():
        if time.monotonic() >= deadline: sys.exit(25)
        time.sleep(0.01)
os.execv({shutil.which('cc')!r}, [{shutil.which('cc')!r}] + sys.argv[1:])
''')
        compiler.chmod(0o700)
        return compiler

    def test_interrupted_compiler_preserves_cache_and_releases_lock(self):
        with tempfile.TemporaryDirectory(prefix="nano-publish-") as tmp:
            directory = Path(tmp)
            module, source, env = self.support.foreign_build_fixture(directory)
            result, output = self.support.compile(source, directory, "--run", env=env)
            self.assertEqual(result.returncode, 42, result.stderr)
            cache = module / ".build"
            old = self.snapshot(cache)
            c_source = module / "answer.c"
            c_source.write_text(c_source.read_text().replace("42", "43"))
            env["NANO_CC"] = str(self.blocking_compiler(directory))
            process = self.start(directory / "interrupted", source.replace("42", "43"), env)
            try:
                self.wait_ready(directory / "ready", process)
            finally:
                self.stop(process)
            self.assertEqual(self.snapshot(cache), old)
            self.assertEqual(self.support.execute(output, env=env).returncode, 42)
            orphaned = list(cache.glob(".nano-build-*"))
            self.assertEqual(len(orphaned), 1)
            self.assertEqual(stat.S_IMODE(orphaned[0].stat().st_mode), 0o700)
            env.pop("NANO_CC")
            result, output = self.support.compile(source.replace("42", "43"), directory, "--run", env=env)
            self.assertEqual(result.returncode, 43, result.stderr)

    def test_overlapping_builders_wait_and_reuse(self):
        with tempfile.TemporaryDirectory(prefix="nano-publish-") as tmp:
            directory = Path(tmp)
            module, source, env = self.support.foreign_build_fixture(directory)
            env["NANO_CC"] = str(self.blocking_compiler(directory))
            env["NANO_VERBOSE_BUILD"] = "1"
            processes = []
            try:
                first = self.start(directory / "first", source, env)
                processes.append(first)
                self.wait_ready(directory / "ready", first)
                second = self.start(directory / "second", source, env)
                processes.append(second)
                observed = b""
                deadline = time.monotonic() + 10
                while b"I wait for the C-library cache lock" not in observed and time.monotonic() < deadline:
                    readable, _, _ = select.select([second.stderr], [], [], 0.1)
                    if readable:
                        chunk = os.read(second.stderr.fileno(), 4096)
                        if not chunk: break
                        observed += chunk
                self.assertIn(b"I wait for the C-library cache lock", observed)
                self.assertIsNone(first.poll())
                self.assertIsNone(second.poll())
                (directory / "release").touch()
                for process in processes:
                    _, error = process.communicate(timeout=20)
                    self.assertEqual(process.returncode, 0, error)
                self.assertEqual(len((directory / "calls").read_text().splitlines()), 1)
                for name in ("first", "second"):
                    self.assertEqual(self.support.execute(directory / name / "program.nvm", env=env).returncode, 42)
                self.assertFalse(list((module / ".build").glob(".nano-build-*")))
            finally:
                for process in processes:
                    self.stop(process)


if __name__ == "__main__":
    unittest.main()
