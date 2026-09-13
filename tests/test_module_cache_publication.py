"""I protect cached foreign artifacts while compilers fail or overlap."""
import json
import ctypes
import hashlib
import os
from pathlib import Path
import select
import shlex
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
    @classmethod
    def setUpClass(cls):
        cls.probe = ROOT / "obj/test_module_generation_probe"
        version = subprocess.run([shutil.which("cc"), "--version"], capture_output=True, timeout=10, check=True)
        cls.gcc_validation = b"Free Software Foundation" in version.stdout and b"clang version" not in version.stdout
        if not cls.probe.is_file():
            raise RuntimeError("I need make test-bytecode-shadows to build the production cache probe")

    def setUp(self):
        self.support = shadows.BytecodeShadows()

    def probe_path(self, mode, module, env):
        result = subprocess.run([str(self.probe), mode, str(module)], cwd=ROOT,
                                env=env, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        return Path(result.stdout.decode().strip())

    def library_answer(self, library):
        handle = ctypes.CDLL(str(library))
        handle.nano_build_answer.restype = ctypes.c_int64
        return handle.nano_build_answer()

    def snapshot(self, cache):
        if (cache / "current").is_symlink():
            cache = (cache / "current").resolve(strict=True)
        return {p.name: p.read_bytes() for p in cache.iterdir()
                if p.is_file() and p.name != ".build.lock"}

    def test_generation_sync_rejects_nonregular_entries(self):
        for kind in ("regular", "symlink", "directory", "fifo", "root-symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix="nano-sync-entry-") as tmp:
                directory = Path(tmp)
                stage = directory / "stage"
                stage.mkdir()
                target = directory / "external"
                target.write_bytes(b"I remain outside the generation.")
                item = stage / "entry"
                if kind == "regular": item.write_bytes(b"artifact")
                elif kind == "symlink": item.symlink_to(target)
                elif kind == "directory": item.mkdir()
                elif kind == "fifo": os.mkfifo(item)
                else:
                    alias = directory / "alias"
                    alias.symlink_to(stage, target_is_directory=True)
                    stage = alias
                result = subprocess.run([str(self.probe), "sync-generation", str(stage)],
                                        capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 0 if kind == "regular" else 1, result.stderr)
                self.assertEqual(target.read_bytes(), b"I remain outside the generation.")

    def test_linker_flag_fragments_preserve_literal_paths(self):
        with tempfile.TemporaryDirectory(prefix="nano-link-fragment-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.foreign_build_fixture(directory)
            member, obj = directory / "member.c", directory / "member.o"
            archive = directory / "space ' back\\slash\narchive.a"
            (module / "answer.c").write_text("extern long long selected(void);\n"
                "long long nano_build_answer(void) { return selected(); }\n")
            (module / "module.json").write_text(json.dumps({"name": "answer_native",
                "c_sources": ["answer.c"], "ldflags": [shlex.quote(str(archive))]}))
            for answer in (42, 43):
                member.write_text(f"long long selected(void) {{ return {answer}; }}\n")
                for command in (["cc", "-fPIC", "-c", str(member), "-o", str(obj)],
                                ["ar", "rcs", str(archive), str(obj)]):
                    result = subprocess.run(command, capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                self.probe_path("build", module, env)
                generation = self.probe_path("directory", module, env)
                self.assertEqual(self.library_answer(self.probe_path("library", module, env)), answer)
                self.probe_path("build", module, env)
                self.assertEqual(self.probe_path("directory", module, env), generation)

    def test_private_cleanup_never_follows_symlinks(self):
        for kind in ("regular", "root-symlink", "entry-symlink", "directory", "fifo", "swap"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix="nano-cleanup-") as tmp:
                directory = Path(tmp)
                external = directory / "external"
                external.mkdir()
                valuable = external / "artifact"
                valuable.write_bytes(b"I am not a compiler artifact.")
                stage = directory / "stage"
                if kind == "root-symlink":
                    stage.symlink_to(external, target_is_directory=True)
                else:
                    stage.mkdir()
                    item = stage / "artifact"
                    if kind in ("regular", "swap"): item.write_bytes(b"partial output")
                    elif kind == "entry-symlink": item.symlink_to(external, target_is_directory=True)
                    elif kind == "directory":
                        item.mkdir()
                        (item / "valuable").write_bytes(b"I need explicit cleanup.")
                    else: os.mkfifo(item)
                env = dict(os.environ)
                if kind == "swap":
                    env.update(NANO_TEST_CLEANUP_STAGE=str(stage), NANO_TEST_CLEANUP_TARGET=str(external))
                result = subprocess.run([str(self.probe), "remove-staging", str(stage)],
                                        env=env, capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(valuable.read_bytes(), b"I am not a compiler artifact.")
                if kind in ("root-symlink", "swap"):
                    self.assertTrue(stage.is_symlink())
                    self.assertIn(b"retained private build files", result.stderr)
                    if kind == "swap":
                        self.assertEqual(list(Path(str(stage) + ".moved").iterdir()), [])
                elif kind == "directory":
                    self.assertEqual((stage / "artifact/valuable").read_bytes(), b"I need explicit cleanup.")
                    self.assertIn(b"retained private build files", result.stderr)
                else:
                    self.assertFalse(stage.exists())

    def test_process_crash_at_publication_boundaries(self):
        def fresh_library_answer(library):
            result = subprocess.run([sys.executable, "-c",
                "import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); "
                "lib.nano_build_answer.restype=ctypes.c_int64; print(lib.nano_build_answer())", str(library)],
                capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            return int(result.stdout)

        def native_answer(directory, generation, expected):
            main = directory / "native-main.c"
            executable = directory / "native-main"
            main.write_text("long long nano_build_answer(void);\n"
                            "int main(void) { return (int)nano_build_answer(); }\n")
            result = subprocess.run([shutil.which("cc"), str(main), str(generation / "answer_native.o"),
                                     "-o", str(executable)], capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(executable)], capture_output=True, timeout=10)
            self.assertEqual(result.returncode, expected, result.stderr)

        for shared in (False, True):
            for replacement in (False, True):
                for boundary in ("file", "stage", "generation", "cache-1", "ancestor", "pointer", "cache-2"):
                    with self.subTest(shared=shared, replacement=replacement, boundary=boundary), \
                            tempfile.TemporaryDirectory(prefix="nano-publication-crash-") as tmp:
                        directory = Path(tmp)
                        module, _, env = self.support.foreign_build_fixture(directory)
                        if shared:
                            env["NANO_BUILD_CACHE"] = str(directory / "nested" / "shared-cache")
                        previous = previous_library = None
                        if replacement:
                            self.probe_path("build", module, env)
                            previous = self.probe_path("directory", module, env)
                            previous_library = self.probe_path("library", module, env)
                        (module / "answer.c").write_text("long long nano_build_answer(void) { return 43; }\n")
                        root = self.probe_path("root", module, env)
                        events = directory / "events"
                        env.update(NANO_TEST_SYNC_CACHE=str(root), NANO_TEST_SYNC_EVENTS=str(events),
                                   NANO_TEST_CRASH_EVENT=boundary)
                        result = subprocess.run([str(self.probe), "build", str(module)], env=env,
                                                capture_output=True, timeout=20)
                        self.assertEqual(result.returncode, -signal.SIGKILL, result.stderr)
                        self.assertEqual(events.read_text().splitlines()[-1], boundary)
                        switched = boundary in ("pointer", "cache-2")
                        visible = None
                        if switched:
                            visible = self.probe_path("directory", module, env)
                            self.assertTrue(visible.is_dir())
                            self.assertNotEqual(visible, previous)
                            self.assertEqual(fresh_library_answer(self.probe_path("library", module, env)), 43)
                            native_answer(directory, visible, 43)
                        elif previous:
                            self.assertEqual(self.probe_path("directory", module, env), previous)
                            self.assertEqual(fresh_library_answer(self.probe_path("library", module, env)), 42)
                        else:
                            self.assertFalse((root / "current").is_symlink())
                        if previous_library:
                            self.assertEqual(fresh_library_answer(previous_library), 42)
                        del env["NANO_TEST_CRASH_EVENT"]
                        # A completed retry also checks that process death released
                        # the advisory lock. Old private stages are not reused.
                        self.probe_path("build", module, env)
                        recovered = self.probe_path("directory", module, env)
                        self.assertEqual(fresh_library_answer(self.probe_path("library", module, env)), 43)
                        native_answer(directory, recovered, 43)
                        if visible:
                            self.assertEqual(recovered, visible)
                        self.probe_path("build", module, env)
                        self.assertEqual(self.probe_path("directory", module, env), recovered)
                        if previous_library:
                            self.assertEqual(fresh_library_answer(previous_library), 42)
                            native_answer(directory, previous, 42)

    def test_nested_cache_ancestor_failure_and_retry(self):
        with tempfile.TemporaryDirectory(prefix="nano-cache-ancestors-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.foreign_build_fixture(directory)
            cache_base = directory / "new" / "deep" / "cache"
            env["NANO_BUILD_CACHE"] = str(cache_base)
            root = self.probe_path("root", module, env)
            events, identities = directory / "events", directory / "identities"
            failed_parent = cache_base.parent
            env.update(NANO_TEST_SYNC_CACHE=str(root), NANO_TEST_SYNC_EVENTS=str(events),
                       NANO_TEST_SYNC_IDENTITIES=str(identities), NANO_TEST_SYNC_FAIL_PATH=str(failed_parent))
            result = subprocess.run([str(self.probe), "build", str(module)], env=env,
                                    capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertTrue(failed_parent.is_dir())
            self.assertFalse((root / "current").exists())
            self.assertNotIn("pointer", events.read_text().splitlines())
            del env["NANO_TEST_SYNC_FAIL_PATH"]
            events.write_text("")
            identities.write_text("")
            self.probe_path("build", module, env)
            observed = identities.read_text().splitlines()
            positions = []
            for parent in (root, cache_base, failed_parent, cache_base.parent.parent, directory):
                st = parent.stat()
                self.assertIn(f"{st.st_dev}:{st.st_ino}", observed)
                positions.append(observed.index(f"{st.st_dev}:{st.st_ino}"))
            self.assertEqual(positions, sorted(positions))
            generation = self.probe_path("directory", module, env)
            self.assertEqual(self.library_answer(self.probe_path("library", module, env)), 42)
            env["NANO_TEST_SYNC_FAIL_PATH"] = str(failed_parent)
            events.write_text("")
            result = subprocess.run([str(self.probe), "build", str(module)], env=env,
                                    capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertNotIn("pointer", events.read_text().splitlines())
            self.assertEqual(self.probe_path("directory", module, env), generation)
            del env["NANO_TEST_SYNC_FAIL_PATH"]
            self.probe_path("build", module, env)
            self.assertEqual(self.probe_path("directory", module, env), generation)

    def test_publication_sync_order_and_failure_recovery(self):
        self.check_publication_sync_order_and_failure_recovery(full_sync=False)

    @unittest.skipUnless(sys.platform == "darwin", "I use F_FULLFSYNC on Darwin")
    def test_device_sync_order_and_failure_recovery(self):
        self.check_publication_sync_order_and_failure_recovery(full_sync=True)

    @unittest.skipUnless(sys.platform == "darwin", "I use F_FULLFSYNC on Darwin")
    def test_unsupported_device_sync_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="nano-device-sync-") as tmp:
            stage = Path(tmp)
            (stage / "artifact").write_bytes(b"artifact")
            env = dict(os.environ, NANO_TEST_FULL_SYNC="1", NANO_TEST_SYNC_FAILURE="unsupported")
            result = subprocess.run([str(self.probe), "sync-generation", str(stage)],
                                    env=env, capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 1, result.stderr)
            del env["NANO_TEST_SYNC_FAILURE"]
            result = subprocess.run([str(self.probe), "sync-generation", str(stage)],
                                    env=env, capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)

    def check_publication_sync_order_and_failure_recovery(self, full_sync):
        for failure in ("file", "stage", "cache-1", "ancestor", "cache-2", "eintr"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(prefix="nano-sync-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.foreign_build_fixture(directory)
                if full_sync:
                    env["NANO_TEST_FULL_SYNC"] = "1"
                self.probe_path("build", module, env)
                previous = self.probe_path("directory", module, env)
                previous_library = self.probe_path("library", module, env)
                (module / "answer.c").write_text("long long nano_build_answer(void) { return 43; }\n")
                events = directory / "events"
                root = self.probe_path("root", module, env)
                env.update(NANO_TEST_SYNC_CACHE=str(root), NANO_TEST_SYNC_EVENTS=str(events),
                           NANO_TEST_SYNC_FAILURE=failure)
                result = subprocess.run([str(self.probe), "build", str(module)], env=env,
                                        capture_output=True, timeout=20)
                self.assertEqual(result.returncode, 0 if failure == "eintr" else 1, result.stderr)
                observed = events.read_text().splitlines()
                self.assertEqual(observed[0], "file")
                current = self.probe_path("directory", module, env)
                if failure in ("cache-2", "eintr"):
                    self.assertNotEqual(current, previous)
                    self.assertTrue(current.is_dir(), "I must retain the published generation")
                    self.assertEqual(self.library_answer(self.probe_path("library", module, env)), 43)
                    file_end = observed.index("stage")
                    self.assertTrue(all(event == "file" for event in observed[:file_end]))
                    self.assertEqual([event for event in observed[file_end:] if event != "ancestor"],
                                     ["stage", "generation", "cache-1", "pointer", "cache-2"])
                    self.assertTrue(all(event == "ancestor" for event in
                                        observed[observed.index("cache-1") + 1:observed.index("pointer")]))
                    self.assertIn("ancestor", observed)
                    if failure == "cache-2":
                        self.assertIn(b"could not confirm", result.stderr)
                else:
                    self.assertEqual(current, previous)
                    self.assertNotIn("pointer", observed)
                self.assertEqual(self.library_answer(previous_library), 42)
                del env["NANO_TEST_SYNC_FAILURE"]
                events.write_text("")
                self.probe_path("build", module, env)
                recovered = self.probe_path("directory", module, env)
                self.assertEqual(self.library_answer(self.probe_path("library", module, env)), 43)
                if failure in ("cache-2", "eintr"):
                    self.assertEqual(recovered, current)
                    warm_events = events.read_text().splitlines()
                    self.assertEqual(warm_events[0], "cache-1")
                    self.assertTrue(all(event == "ancestor" for event in warm_events[1:]))
                    self.assertGreater(len(warm_events), 1)
                events.write_text("")
                env["NANO_TEST_SYNC_FAILURE"] = "cache-1"
                result = subprocess.run([str(self.probe), "build", str(module)], env=env,
                                        capture_output=True, timeout=20)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(events.read_text().splitlines(), ["cache-1"])
                self.assertEqual(self.probe_path("directory", module, env), recovered)

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
        output.symlink_to({str((cache / 'current').resolve() / 'answer_native.o')!r})
        sys.exit(0)
    output.write_bytes(b"partial compiler output")
    sys.exit(24)
os.execv({shutil.which('cc')!r}, [{shutil.which('cc')!r}] + sys.argv[1:])
''')
                compiler.chmod(0o700)
                env["NANO_CC"] = str(compiler)
                result, output = self.support.compile(source.replace(" 42)", " 43)"), directory, env=env)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.snapshot(cache), old)
                self.assertEqual(output.read_bytes(), old_program)
                self.assertEqual(self.support.execute(output, env=env).returncode, 42)
                observation = json.loads(observed.read_text())
                self.assertNotEqual(Path(observation["directory"]), cache)
                self.assertEqual(observation["mode"], 0o700)
                self.assertFalse(list(cache.glob(".nano-build-*")))
                env.pop("NANO_CC")
                result, output = self.support.compile(source.replace(" 42)", " 43)"), directory, "--run", env=env)
                self.assertEqual(result.returncode, 43, result.stderr)
                self.assertFalse(list(cache.glob(".nano-build-*")))

    def test_bytecode_retains_generation_after_rebuild(self):
        for transitive in (False, True):
            with self.subTest(transitive=transitive), tempfile.TemporaryDirectory(prefix="nano-binding-") as tmp:
                directory = Path(tmp)
                module, source, env = self.support.foreign_build_fixture(directory)
                source = source.replace(str(module / "api.nano"), "foreign/api.nano")
                if transitive:
                    helper = directory / "helper.nano"
                    helper.write_text(f'''module "{module / 'api.nano'}" as foreign
pub fn answer() -> int {{ unsafe {{ return (foreign.nano_build_answer) }} }}
shadow answer {{ assert (== (answer) 42) }}
''')
                    source = f'''module "{helper}" as helper
fn main() -> int {{ return (helper.answer) }}
shadow main {{ assert (== (main) 42) }}
'''
                result, output = self.support.compile(source, directory, env=env)
                self.assertEqual(result.returncode, 0, result.stderr)
                old_output = directory / "old.nvm"
                shutil.copyfile(output, old_output)
                old_library = self.probe_path("library", module, env)
                self.assertIn(str(old_library).encode(), old_output.read_bytes())
                c_source = module / "answer.c"
                c_source.write_text(c_source.read_text().replace("42", "43"))
                result, output = self.support.compile(source.replace(" 42)", " 43)"), directory, "--run", env=env)
                self.assertEqual(result.returncode, 43, result.stderr)
                self.assertNotEqual(old_library, self.probe_path("library", module, env))
                self.assertEqual(self.support.execute(old_output, env=env).returncode, 42)
                self.assertEqual(self.support.execute(output, env=env).returncode, 43)
                cop = subprocess.run([str(ROOT / "bin/nano_vm"), "--isolate-ffi", str(old_output)],
                                     cwd=ROOT, env=env, capture_output=True, timeout=10)
                self.assertEqual(cop.returncode, 42, cop.stderr)
                old_library.rename(old_library.with_suffix(".retained"))
                execution = self.support.execute(old_output, env=env)
                self.assertEqual(execution.returncode, 1, (execution.stdout, execution.stderr))
                self.assertIn(b"nano_build_answer", execution.stdout + execution.stderr)
                cop = subprocess.run([str(ROOT / "bin/nano_vm"), "--isolate-ffi", str(old_output)],
                                     cwd=ROOT, env=env, capture_output=True, timeout=10)
                self.assertEqual(cop.returncode, 1, (cop.stdout, cop.stderr))
                self.assertEqual(self.support.execute(output, env=env).returncode, 43)

    def test_shadow_and_production_share_retained_generation(self):
        with tempfile.TemporaryDirectory(prefix="nano-binding-") as tmp:
            directory = Path(tmp)
            module, source, env = self.support.foreign_build_fixture(directory)
            c_source = module / "answer.c"
            c_source.write_text('''#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
int64_t nano_build_answer(void) {
    const char *ready = getenv("NANO_BIND_READY");
    const char *release = getenv("NANO_BIND_RELEASE");
    if (ready && release) {
        FILE *file = fopen(ready, "w");
        if (!file) return -1;
        fclose(file);
        while (access(release, F_OK) != 0) usleep(1000);
    }
    return 42;
}
''')
            ready, release = directory / "ready", directory / "release"
            env["NANO_BIND_READY"] = str(ready)
            env["NANO_BIND_RELEASE"] = str(release)
            build_dir = directory / "compile"
            process = self.start(build_dir, source, env)
            try:
                self.wait_ready(ready, process)
                old_library = self.probe_path("library", module, env)
                c_source.write_text(c_source.read_text().replace("return 42;", "return 43;"))
                self.probe_path("build", module, env)
                self.assertNotEqual(old_library, self.probe_path("library", module, env))
                release.touch()
                stdout, stderr = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, (stdout, stderr))
                output = build_dir / "program.nvm"
                self.assertIn(str(old_library).encode(), output.read_bytes())
                self.assertEqual(self.support.execute(output, env=env).returncode, 42)
            finally:
                self.stop(process)

    def test_packaged_wrapper_retains_foreign_generation(self):
        with tempfile.TemporaryDirectory(prefix="nano-wrapper-") as tmp:
            directory = Path(tmp)
            module, source, env = self.support.foreign_build_fixture(directory)
            env["NANO_VIRT_LIB"] = str(ROOT / "obj")
            program = directory / "program.nano"
            program.write_text(source)
            wrapper = directory / "program"
            result = subprocess.run([str(ROOT / "bin/nano_virt"), str(program), "-o", str(wrapper)],
                                    cwd=ROOT, env=env, capture_output=True, timeout=25)
            self.assertEqual(result.returncode, 0, (result.stdout, result.stderr))
            elsewhere = directory / "elsewhere"
            elsewhere.mkdir()
            def execute():
                return subprocess.run([str(wrapper)], cwd=elsewhere, env=env,
                                      capture_output=True, timeout=10)
            self.assertEqual(execute().returncode, 42)
            old_library = self.probe_path("library", module, env)
            c_source = module / "answer.c"
            c_source.write_text(c_source.read_text().replace("42", "43"))
            result, output = self.support.compile(source.replace(" 42)", " 43)"), directory, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.support.execute(output, env=env).returncode, 43)
            self.assertEqual(execute().returncode, 42)
            old_library.rename(old_library.with_suffix(".retained"))
            result = execute()
            self.assertEqual(result.returncode, 1, (result.stdout, result.stderr))
            self.assertIn(b"nano_build_answer", result.stderr)

    def test_foreign_compiler_paths_are_literal(self):
        for multi, special in ((False, ""), (True, ""), (False, "\\"), (True, "\\"), (True, "\n")):
            with self.subTest(multi=multi, special=special), tempfile.TemporaryDirectory(prefix="nano-foreign-path-") as tmp:
                directory = Path(tmp)
                parent = directory / f"literal ' \" {special} $(touch injected) ; directory"
                parent.mkdir()
                module, source, env = self.support.foreign_build_fixture(parent)
                raw = str(module / "api.nano")
                source = source.replace('"' + raw + '"', json.dumps(raw))
                includes = module / f"include ' \" {special} literal"
                includes.mkdir()
                (includes / "answer.h").write_text("#define ANSWER 40\n")
                name = f"answer ' \" {special} literal.c"
                (module / "answer.c").rename(module / name)
                (module / name).write_text('#include <stdint.h>\n#include "answer.h"\nint64_t private_answer(void);\nint64_t nano_build_answer(void) { return ANSWER + private_answer(); }\n')
                private = f"private ' \" {special} literal.c"
                (module / private).write_text('#include <stdint.h>\n#include "answer.h"\nint64_t private_answer(void) { return ANSWER - 38; }\n')
                metadata = {"name": "answer_native", "c_sources": [name],
                            "shared_c_sources": [private], "include_dirs": [str(includes)]}
                if multi:
                    extra = f"extra ' \" {special} literal.c"
                    (module / extra).write_text("int extra_answer(void) { return 1; }\n")
                    metadata["c_sources"].append(extra)
                (module / "module.json").write_text(json.dumps(metadata))
                env["NANO_BUILD_CACHE"] = str(directory / f"cache ' \" {special} literal")
                result, output = self.support.compile(source, directory, "--run", env=env, cwd=directory)
                self.assertEqual(result.returncode, 42, (result.stdout, result.stderr))
                self.assertEqual(self.support.execute(output, env=env).returncode, 42)
                self.assertFalse((directory / "injected").exists())
                generation = self.probe_path("directory", module, env)
                record = generation / "source_hashes.json"
                if special:
                    self.assertFalse(record.exists())
                else:
                    self.assertTrue(record.exists(), [(p.name, p.read_text()) for p in generation.glob("*.d")])
                    self.assertIn("dep:" + str(includes / "answer.h"), json.loads(record.read_text()))
                (includes / "answer.h").write_text("#define ANSWER 41\n")
                result, output = self.support.compile(source.replace(" 42)", " 44)"), directory,
                                                      "--run", env=env, cwd=directory)
                self.assertEqual(result.returncode, 44, (result.stdout, result.stderr))

    def test_transitive_system_header_invalidates_cache(self):
        with tempfile.TemporaryDirectory(prefix="nano-system-deps-") as tmp:
            directory = Path(tmp)
            module, source, env = self.support.foreign_build_fixture(directory)
            includes = directory / "system-include"
            includes.mkdir()
            (includes / "public.h").write_text('#include "private.h"\n')
            header = includes / "private.h"
            header.write_text("#define ANSWER 42\n")
            (module / "answer.c").write_text(
                '#include <stdint.h>\n#include <public.h>\n'
                'int64_t nano_build_answer(void) { return ANSWER; }\n')
            manifest = module / "module.json"
            metadata = json.loads(manifest.read_text())
            metadata["cflags"] = ["-isystem", str(includes)]
            manifest.write_text(json.dumps(metadata))
            result, output = self.support.compile(source, directory, "--run", env=env)
            self.assertEqual(result.returncode, 42, result.stderr)
            generation = self.probe_path("directory", module, env)
            result, output = self.support.compile(source, directory, "--run", env=env)
            self.assertEqual(result.returncode, 42, result.stderr)
            self.assertEqual(self.probe_path("directory", module, env), generation)
            old_stat = header.stat()
            header.write_text("#define ANSWER 43\n")
            os.utime(header, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
            result, output = self.support.compile(source.replace(" 42)", " 43)"), directory,
                                                  "--run", env=env)
            self.assertEqual(result.returncode, 43, result.stderr)
            self.assertNotEqual(self.probe_path("directory", module, env), generation)
            self.assertEqual(self.support.execute(output, env=env).returncode, 43)
            record = json.loads((generation / "source_hashes.json").read_text())
            self.assertIn("dep:" + str(header), record)
            self.assertIn("dep:" + str(includes / "public.h"), record)

    def test_transitive_path_alias_cannot_authorize_reuse(self):
        for phase in ("single", "multi", "shared"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory(prefix="nano-transitive-alias-") as tmp:
                directory = Path(tmp)
                module, source, env = self.support.foreign_build_fixture(directory)
                actual = module / "hidden\\answer.h"
                alias = module / "hidden" / "answer.h"
                alias.parent.mkdir()
                actual.write_text("#define ANSWER 42\n")
                alias.write_text("#define ANSWER 17\n")
                body = ('#include <stdint.h>\n#include "hidden\\answer.h"\n'
                        'int64_t nano_build_answer(void) { return ANSWER; }\n')
                metadata = {"name": "answer_native", "c_sources": ["answer.c"]}
                (module / "answer.c").write_text(body)
                if phase == "multi":
                    (module / "extra.c").write_text("int extra(void) { return 1; }\n")
                    metadata["c_sources"].append("extra.c")
                elif phase == "shared":
                    (module / "answer.c").write_text("int extra(void) { return 1; }\n")
                    (module / "private.c").write_text(body.replace(
                        'int64_t nano_build_answer', '__attribute__((visibility("default"))) int64_t nano_build_answer'))
                    metadata["shared_c_sources"] = ["private.c"]
                (module / "module.json").write_text(json.dumps(metadata))
                result, output = self.support.compile(source, directory, "--run", env=env)
                self.assertEqual(result.returncode, 42, result.stderr)
                generation = self.probe_path("directory", module, env)
                old_stat = actual.stat()
                actual.write_text("#define ANSWER 43\n")
                os.utime(actual, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
                result, output = self.support.compile(source.replace(" 42)", " 43)"), directory,
                                                      "--run", env=env)
                self.assertEqual(result.returncode, 43, result.stderr)
                self.assertEqual(self.support.execute(output, env=env).returncode, 43)
                record = json.loads((generation / "source_hashes.json").read_text())
                self.assertIn("dep:" + str(actual.resolve()), record)

    def test_include_trace_requires_unambiguous_paths(self):
        with tempfile.TemporaryDirectory(prefix="nano-include-record-") as tmp:
            directory = Path(tmp)
            trace = directory / "trace.includes"
            def inspect(text):
                trace.write_bytes(text)
                return subprocess.run([str(self.probe), "includes", str(trace)], cwd=ROOT,
                                      capture_output=True, timeout=10)
            self.assertEqual(inspect(b"").returncode, 0)
            for name in ("plain.h", 'quoted " header.h', "back\\slash.h", "tab\theader.h", "line\nheader.h", "octal\rheader.h"):
                header = directory / name
                header.write_text("#define VALUE 42\n")
                escaped = str(header).replace("\\", "\\\\").replace('"', '\\"').replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\015")
                result = inspect((".. " + escaped + "\n").encode())
                self.assertEqual(result.returncode, 0, (name, result.stderr))
                self.assertIn("dep:" + str(header), json.loads(result.stdout))
            raw = directory / "ambiguous\\n.h"
            decoded = directory / "ambiguous\n.h"
            raw.write_text("#define VALUE 42\n")
            decoded.symlink_to(raw)
            self.assertEqual(inspect((". " + str(raw) + "\n").encode()).returncode, 1)
            plain = directory / "plain.h"
            guard_list = f". {plain}\nMultiple include guards may be useful for:\n{plain}\n"
            self.assertEqual(inspect(guard_list.encode()).returncode, 0)
            self.assertEqual(inspect((guard_list + str(raw) + "\n").encode()).returncode, 1)
            # I reject two spellings even when they currently name one inode:
            # a later symlink retarget must not hide behind that coincidence.
            for bad in (b"warning: fixture\n", b". /missing-nano-header\n", b". incomplete",
                        b". embedded\0null\n", b". raw\tcontrol\n", b". " + b"x" * 8192 + b"\n"):
                self.assertEqual(inspect(bad).returncode, 1, bad[:80])

    def test_compiler_diagnostics_survive_include_capture(self):
        with tempfile.TemporaryDirectory(prefix="nano-include-diag-") as tmp:
            directory = Path(tmp)
            module, source, env = self.support.foreign_build_fixture(directory)
            c_source = module / "answer.c"
            original = c_source.read_text()
            c_source.write_text('#warning I_preserve_this_compiler_warning\n' + original)
            result, output = self.support.compile(source, directory, "--run", env=env)
            self.assertEqual(result.returncode, 42, result.stderr)
            self.assertIn(b"I_preserve_this_compiler_warning", result.stderr)
            old_output = output.read_bytes()
            c_source.write_text('#error I_preserve_this_compiler_error\n' + original)
            result, output = self.support.compile(source, directory, env=env)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn(b"I_preserve_this_compiler_error", result.stderr)
            self.assertEqual(output.read_bytes(), old_output)
            c_source.write_text(original)
            result, _ = self.support.compile(source, directory, "--run", env=env)
            self.assertEqual(result.returncode, 42, result.stderr)

    def test_dependency_records_reject_incomplete_evidence(self):
        with tempfile.TemporaryDirectory(prefix="nano-dep-record-") as tmp:
            directory = Path(tmp)
            header = directory / "quoted ' $ # header.h"
            header.write_text("#define ANSWER 42\n")
            escaped = str(header).replace("$", "$$").replace("#", "\\#").replace(" ", "\\ ")
            dependency = directory / "record.d"
            dependency.write_text("nano_module_dependencies: \\\n " + escaped + "\n")
            result = subprocess.run([str(self.probe), "deps", str(dependency)], cwd=ROOT,
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("dep:" + str(header), json.loads(result.stdout))
            bad = ["", "wrong_target: " + escaped, "nano_module_dependencies:",
                   "nano_module_dependencies: " + escaped + " /no-such-nano-header.h",
                   "nano_module_dependencies: " + "x" * 5000,
                   "nano_module_dependencies: " + escaped + "\x00hidden",
                   "nano_module_dependencies: $unexpanded"]
            for text in bad:
                dependency.write_text(text)
                result = subprocess.run([str(self.probe), "deps", str(dependency)], cwd=ROOT,
                                        capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 1, (text[:80], result.stderr))

    def test_oversized_foreign_command_preserves_generation(self):
        with tempfile.TemporaryDirectory(prefix="nano-command-bound-") as tmp:
            directory = Path(tmp)
            module, source, env = self.support.foreign_build_fixture(directory)
            result, output = self.support.compile(source, directory, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            old = output.read_bytes()
            generation = self.probe_path("directory", module, env)
            manifest = module / "module.json"
            metadata = json.loads(manifest.read_text())
            metadata["cflags"] = ["-DOVERSIZED=" + "x" * 10000]
            manifest.write_text(json.dumps(metadata))
            result, output = self.support.compile(source, directory, env=env)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(output.read_bytes(), old)
            self.assertEqual(self.probe_path("directory", module, env), generation)
            self.assertEqual(self.support.execute(output, env=env).returncode, 42)

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
                self.assertGreater((cache / "current" / name).stat().st_size, 0)
            self.assertFalse(list(cache.glob(".nano-build-*")))

    def test_failed_publication_preserves_old_generation(self):
        with tempfile.TemporaryDirectory(prefix="nano-publish-") as tmp:
            directory = Path(tmp)
            module, source, env = self.support.foreign_build_fixture(directory)
            result, output = self.support.compile(source, directory, "--run", env=env)
            self.assertEqual(result.returncode, 42, result.stderr)
            old_program = output.read_bytes()
            cache = module / ".build"
            pointer = cache / "current"
            old_generation = pointer.resolve(strict=True)
            old = self.snapshot(old_generation)
            pointer.unlink()
            pointer.mkdir()  # I force the final pointer rename to fail.
            c_source = module / "answer.c"
            c_source.write_text(c_source.read_text().replace("42", "43"))
            result, output = self.support.compile(source.replace(" 42)", " 43)"), directory, env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"could not publish complete", result.stderr)
            self.assertEqual(output.read_bytes(), old_program)
            self.assertEqual(self.snapshot(old_generation), old)
            self.assertEqual([p.resolve() for p in cache.glob(".nano-gen-*")], [old_generation])
            self.assertFalse(list(cache.glob(".nano-build-*")))
            pointer.rmdir()
            pointer.symlink_to(old_generation.name)
            self.assertEqual(self.support.execute(output, env=env).returncode, 42)
            result, output = self.support.compile(source.replace(" 42)", " 43)"), directory, "--run", env=env)
            self.assertEqual(result.returncode, 43, result.stderr)

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

    def counting_compiler(self, compiler, calls, answer=42):
        compiler.write_text(f'''#!{sys.executable}
import os, sys
if "-c" in sys.argv:
    with open({str(calls)!r}, "a") as log: log.write("compile\\n")
os.execv({shutil.which('cc')!r}, [{shutil.which('cc')!r}, "-DANSWER={answer}"] + sys.argv[1:])
''')
        compiler.chmod(0o700)

    def test_changed_compiler_identity_changes_output(self):
        for selection in ("NANO_CC", "CC", "PATH", "same_path"):
            with self.subTest(selection=selection), tempfile.TemporaryDirectory(prefix="nano-identity-") as tmp:
                directory = Path(tmp)
                module, source, env = self.support.foreign_build_fixture(directory)
                (module / "answer.c").write_text("#include <stdint.h>\nint64_t nano_build_answer(void) { return ANSWER; }\n")
                calls = directory / "calls"
                first = directory / "cc"
                second = directory / "cc-next"
                self.counting_compiler(first, calls, 42)
                self.counting_compiler(second, calls, 43)
                if selection == "PATH":
                    env["PATH"] = str(directory) + os.pathsep + env["PATH"]
                else:
                    env["CC" if selection == "CC" else "NANO_CC"] = str(first)
                for _ in range(2):
                    result, output = self.support.compile(source, directory, "--run", env=env)
                    self.assertEqual(result.returncode, 42, result.stderr)
                self.assertEqual(calls.read_text().splitlines(), ["compile"] * (3 if self.gcc_validation else 1))
                if selection == "PATH":
                    next_dir = directory / "next"
                    next_dir.mkdir()
                    second.rename(next_dir / "cc")
                    env["PATH"] = str(next_dir) + os.pathsep + env["PATH"]
                elif selection == "same_path":
                    stamp = first.stat()
                    self.counting_compiler(first, calls, 43)
                    os.utime(first, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                else:
                    env[selection] = str(second)
                result, output = self.support.compile(source.replace(" 42)", " 43)"), directory, "--run", env=env)
                self.assertEqual(result.returncode, 43, result.stderr)
                execution = self.support.execute(output, env=env)
                self.assertEqual(execution.returncode, 43, execution.stderr)
                self.assertEqual(calls.read_text().splitlines(), ["compile"] * (5 if self.gcc_validation else 2))

    def test_new_earlier_header_invalidates_unchanged_search_path(self):
        for phase in ("single", "multi", "shared"):
            for route in ("include_dirs", "cflags", "CPATH", "quote"):
                with self.subTest(phase=phase, route=route), tempfile.TemporaryDirectory(prefix="nano-search-") as tmp:
                    directory = Path(tmp)
                    module, source, env = self.support.foreign_build_fixture(directory)
                    early, late = directory / "early", directory / "late"
                    early.mkdir()
                    late.mkdir()
                    (late / "answer.h").write_text("#define ANSWER 42\n")
                    directive = '#include "answer.h"' if route == "quote" else '#include <answer.h>'
                    body = '#include <stdint.h>\n' + directive + '\nint64_t nano_build_answer(void) { return ANSWER; }\n'
                    metadata = {"name": "answer_native", "c_sources": ["answer.c"]}
                    if route == "include_dirs": metadata["include_dirs"] = [str(early), str(late)]
                    elif route == "cflags": metadata["cflags"] = ["-I" + str(early), "-I" + str(late)]
                    elif route == "CPATH": env["CPATH"] = str(early) + os.pathsep + str(late)
                    else:
                        metadata["include_dirs"] = [str(late)]
                        early = module
                    (module / "answer.c").write_text(body)
                    if phase == "multi":
                        (module / "extra.c").write_text("int extra(void) { return 1; }\n")
                        metadata["c_sources"].append("extra.c")
                    elif phase == "shared":
                        (module / "answer.c").write_text("int extra(void) { return 1; }\n")
                        (module / "private.c").write_text(body.replace('int64_t nano_build_answer',
                            '__attribute__((visibility("default"))) int64_t nano_build_answer'))
                        metadata["shared_c_sources"] = ["private.c"]
                    (module / "module.json").write_text(json.dumps(metadata))
                    result, output = self.support.compile(source, directory, "--run", env=env)
                    self.assertEqual(result.returncode, 42, result.stderr)
                    generation = self.probe_path("directory", module, env)
                    result, _ = self.support.compile(source, directory, "--run", env=env)
                    self.assertEqual(result.returncode, 42, result.stderr)
                    self.assertEqual(self.probe_path("directory", module, env), generation)
                    (early / "answer.h").write_text("#define ANSWER 43\n")
                    result, output = self.support.compile(source.replace(" 42)", " 43)"), directory, "--run", env=env)
                    self.assertEqual(result.returncode, 43, result.stderr)
                    self.assertEqual(self.support.execute(output, env=env).returncode, 43)

    def test_failed_preprocessing_cannot_authorize_reuse(self):
        for failure in ("partial", "empty"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(prefix="nano-probe-failure-") as tmp:
                directory = Path(tmp)
                module, source, env = self.support.foreign_build_fixture(directory)
                compiler, calls = directory / "cc", directory / "calls"
                compiler.write_text(f'''#!{sys.executable}
import os, pathlib, sys
if "-E" in sys.argv or "-S" in sys.argv:
    if {failure!r} == "partial":
        print("partial preprocessor output")
        sys.exit(23)
    sys.exit(0)
if "-c" in sys.argv:
    with open({str(calls)!r}, "a") as log: log.write("compile\\n")
os.execv({shutil.which('cc')!r}, [{shutil.which('cc')!r}] + sys.argv[1:])
''')
                compiler.chmod(0o700)
                env["NANO_CC"] = str(compiler)
                for count in (1, 2):
                    result, output = self.support.compile(source, directory, "--run", env=env)
                    self.assertEqual(result.returncode, 42, result.stderr)
                    self.assertEqual(self.support.execute(output, env=env).returncode, 42)
                    self.assertEqual(len(calls.read_text().splitlines()), count)
                    self.assertFalse((module / ".build" / "current" / "source_hashes.json").exists())

    def test_header_change_during_compilation_withholds_reuse(self):
        with tempfile.TemporaryDirectory(prefix="nano-probe-race-") as tmp:
            directory = Path(tmp)
            module, source, env = self.support.foreign_build_fixture(directory)
            header = module / "answer.h"
            header.write_text("#define ANSWER 42\n")
            (module / "answer.c").write_text('#include <stdint.h>\n#include "answer.h"\nint64_t nano_build_answer(void) { return ANSWER; }\n')
            compiler = directory / "cc"
            compiler.write_text(f'''#!{sys.executable}
import os, pathlib, subprocess, sys
if "-c" in sys.argv:
    result = subprocess.run([{shutil.which('cc')!r}] + sys.argv[1:])
    header = pathlib.Path({str(header)!r})
    stamp = header.stat()
    header.write_text("#define ANSWER 43\\n")
    os.utime(header, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    sys.exit(result.returncode)
os.execv({shutil.which('cc')!r}, [{shutil.which('cc')!r}] + sys.argv[1:])
''')
            compiler.chmod(0o700)
            env["NANO_CC"] = str(compiler)
            result, output = self.support.compile(source, directory, "--run", env=env)
            self.assertEqual(result.returncode, 42, result.stderr)
            self.assertFalse((module / ".build" / "current" / "source_hashes.json").exists())
            result, output = self.support.compile(source.replace(" 42)", " 43)"), directory, "--run", env=env)
            self.assertEqual(result.returncode, 43, result.stderr)
            self.assertEqual(self.support.execute(output, env=env).returncode, 43)
            generation = self.probe_path("directory", module, env)
            self.assertTrue((generation / "source_hashes.json").is_file())
            result, _ = self.support.compile(source.replace(" 42)", " 43)"), directory, "--run", env=env)
            self.assertEqual(result.returncode, 43, result.stderr)
            self.assertEqual(self.probe_path("directory", module, env), generation)

    def test_preprocessing_veto_preserves_original_pch_mode(self):
        compiler = shutil.which("cc")
        version = subprocess.run([compiler, "--version"], capture_output=True, timeout=10)
        if b"clang" not in version.stdout.lower():
            self.skipTest("I exercise Clang's explicit PCH mode here")
        with tempfile.TemporaryDirectory(prefix="nano-probe-pch-") as tmp:
            directory = Path(tmp)
            module, source, env = self.support.foreign_build_fixture(directory)
            header, pch = module / "precompiled.h", module / "precompiled.pch"
            header.write_text("typedef char measured_type[42];\n")
            flags = [compiler, "-fPIC"]
            if sys.platform != "darwin": flags.append("-D_POSIX_C_SOURCE=200809L")
            build = subprocess.run(flags + ["-x", "c-header", str(header), "-o", str(pch)],
                                   capture_output=True, timeout=10)
            self.assertEqual(build.returncode, 0, build.stderr)
            stamp = header.stat()
            header.write_text("typedef char measured_type[43];\n")
            os.utime(header, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            (module / "answer.c").write_text("long long nano_build_answer(void) { return sizeof(measured_type); }\n")
            (module / "module.json").write_text(json.dumps({"name": "answer_native", "c_sources": ["answer.c"],
                                                           "cflags": ["-include-pch", str(pch)]}))
            for _ in range(2):
                result, output = self.support.compile(source, directory, "--run", env=env)
                self.assertEqual(result.returncode, 42, result.stderr)
                self.assertEqual(self.support.execute(output, env=env).returncode, 42)

    @unittest.skipUnless(sys.platform == "darwin", "I have integrated Darwin linker records")
    def test_linker_archive_bytes_and_search_invalidate_reuse(self):
        from tests.characterize_linker_inputs import measure
        observed = measure(shutil.which("cc"), self.probe)
        self.assertTrue(observed["reusable_record_created"])
        self.assertTrue(observed["selected_archive_recorded"])
        self.assertTrue(observed["unchanged_generation_reused"])
        self.assertEqual(observed["unchanged_answer"], 42)
        self.assertTrue(observed["archive_changed"])
        self.assertTrue(observed["archive_size_preserved"])
        self.assertTrue(observed["archive_timestamp_preserved"])
        self.assertEqual(observed["cache_answer_after_archive_edit"], 43)
        self.assertTrue(observed["cache_generation_changed_after_archive_edit"])
        self.assertEqual(observed["cache_answer_after_earlier_library"], 44)
        self.assertTrue(observed["cache_generation_changed_after_earlier_library"])

    @unittest.skipUnless(sys.platform == "darwin", "I have integrated Darwin linker records")
    def test_linker_mutation_does_not_cache_unlinked_bytes(self):
        for mode in ("first", "second", "postprocess"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix="nano-link-mutation-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.foreign_build_fixture(directory)
                compiler = shutil.which("cc")
                archive, replacement = directory / "selected.a", directory / "replacement.a"
                for path, value in ((archive, 42), (replacement, 43)):
                    c_source, obj = directory / "member.c", directory / "member.o"
                    c_source.write_text(f"long long selected_answer(void) {{ return {value}; }}\n")
                    subprocess.run([compiler, "-fPIC", "-c", str(c_source), "-o", str(obj)], check=True,
                                   capture_output=True, timeout=10)
                    subprocess.run(["ar", "rcs", str(path), str(obj)], check=True, capture_output=True, timeout=10)
                (module / "answer.c").write_text("extern long long selected_answer(void);\n"
                                                 "long long nano_build_answer(void) { return selected_answer(); }\n")
                (module / "module.json").write_text(json.dumps({"name": "answer_native",
                    "c_sources": ["answer.c"], "ldflags": [str(archive)]}))
                counter, changed = directory / "count", directory / "changed"
                wrapper = directory / "cc-wrapper"
                wrapper.write_text(f'''#!{sys.executable}
import os, pathlib, subprocess, sys
args = sys.argv[1:]
counter, changed = pathlib.Path({str(counter)!r}), pathlib.Path({str(changed)!r})
count = int(counter.read_text()) if counter.exists() else 0
if "-dynamiclib" in args:
    count += 1
    counter.write_text(str(count))
result = subprocess.run([{compiler!r}] + args)
mode = {mode!r}
mutate = ((mode == "first" and "-dynamiclib" in args and count == 1) or
          (mode == "second" and "-dynamiclib" in args and count == 2) or
          (mode == "postprocess" and ("-E" in args or "-S" in args) and count > 0))
if result.returncode == 0 and mutate and not changed.exists():
    archive = pathlib.Path({str(archive)!r})
    stamp = archive.stat()
    archive.write_bytes(pathlib.Path({str(replacement)!r}).read_bytes())
    os.utime(archive, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    changed.touch()
sys.exit(result.returncode)
''')
                wrapper.chmod(0o700)
                env["NANO_CC"] = str(wrapper)
                self.probe_path("build", module, env)
                value = self.library_answer(self.probe_path("library", module, env))
                generation = self.probe_path("directory", module, env)
                self.assertTrue(changed.exists())
                self.assertIn(value, (42, 43))
                if value == 42:
                    self.assertFalse((generation / "source_hashes.json").exists(),
                                     "I must not cache old code under the replacement archive hash")
                self.probe_path("build", module, env)
                self.assertEqual(self.library_answer(self.probe_path("library", module, env)), 43)
                recovered = self.probe_path("directory", module, env)
                self.assertTrue((recovered / "source_hashes.json").is_file())
                self.probe_path("build", module, env)
                self.assertEqual(self.probe_path("directory", module, env), recovered)

    @unittest.skipUnless(sys.platform == "darwin", "I have integrated Darwin linker records")
    def test_failed_final_link_preserves_previous_generation(self):
        with tempfile.TemporaryDirectory(prefix="nano-final-link-failure-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.foreign_build_fixture(directory)
            compiler = shutil.which("cc")
            counter, control = directory / "count", directory / "mode"
            control.write_text("ok")
            wrapper = directory / "cc-wrapper"
            wrapper.write_text(f'''#!{sys.executable}
import os, pathlib, sys
args = sys.argv[1:]
counter, control = pathlib.Path({str(counter)!r}), pathlib.Path({str(control)!r})
if "-dynamiclib" in args:
    count = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(count))
    if control.read_text() == "fail" and count == 2:
        pathlib.Path(args[args.index("-o") + 1]).write_bytes(b"partial library")
        sys.exit(23)
os.execv({compiler!r}, [{compiler!r}] + args)
''')
            wrapper.chmod(0o700)
            env["NANO_CC"] = str(wrapper)
            self.probe_path("build", module, env)
            previous = self.probe_path("directory", module, env)
            self.assertEqual(counter.read_text(), "2")
            c_source = module / "answer.c"
            c_source.write_text(c_source.read_text() + "\n/* I force another build. */\n")
            counter.write_text("0")
            control.write_text("fail")
            result = subprocess.run([str(self.probe), "build", str(module)], env=env,
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(counter.read_text(), "2", "I must not retry past a failed final link")
            self.assertEqual(self.probe_path("directory", module, env), previous)
            self.assertEqual(self.library_answer(self.probe_path("library", module, env)), 42)
            control.write_text("ok")
            self.probe_path("build", module, env)
            recovered = self.probe_path("directory", module, env)
            self.assertNotEqual(recovered, previous)
            self.assertEqual(self.library_answer(self.probe_path("library", module, env)), 42)
            self.probe_path("build", module, env)
            self.assertEqual(self.probe_path("directory", module, env), recovered)

    @unittest.skipUnless(sys.platform == "darwin", "I have integrated Darwin linker records")
    def test_linker_record_boundaries(self):
        with tempfile.TemporaryDirectory(prefix="nano-link-record-") as tmp:
            directory = Path(tmp)
            stage = directory / "stage"
            stage.mkdir()
            obj, output = stage / "input.o", stage / "library.dylib"
            obj.write_bytes(b"object")
            external = directory / "space ' back\\slash\nline.a"
            external.write_bytes(b"archive")
            missing = directory / "absent.a"
            record = directory / "record"

            def field(tag, path):
                return bytes([tag]) + os.fsencode(path) + b"\0"

            version = field(0, "@(#)PROGRAM:ld PROJECT:ld-1267\n")
            body = field(16, obj) + field(16, external) + field(17, missing)
            end = field(64, output)
            valid = version + body + end
            cases = [("literal", valid, 0), ("empty", b"", 1),
                     ("no-version", body + end, 1), ("no-output", version + body, 1),
                     ("unknown-tag", version + field(99, external) + body + end, 1),
                     ("truncated", valid[:-1], 1), ("empty-path", version + field(16, "") + body + end, 1),
                     ("no-internal-input", version + field(16, external) + end, 1),
                     ("wrong-output", version + body + field(64, missing), 1),
                     ("trailing-record", valid + field(16, external), 1),
                     ("existing-negative", version + body + field(17, external) + end, 1),
                     ("missing-input", version + body + field(16, missing) + end, 1),
                     ("directory-input", version + field(16, directory) + body + end, 1),
                     ("oversized-path", version + field(16, "x" * 8192) + body + end, 1)]
            for name, data, expected in cases:
                with self.subTest(name=name):
                    record.write_bytes(data)
                    result = subprocess.run([str(self.probe), "link-inputs", str(record), str(stage), str(output)],
                                            capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, expected, result.stderr)
                    if not expected:
                        inputs = json.loads(result.stdout)
                        self.assertIn(str(external), inputs)
                        self.assertEqual(inputs[str(missing)], "missing")
                        self.assertNotIn(str(obj), inputs)

    @unittest.skipUnless(sys.platform == "darwin", "I have integrated Darwin linker records")
    def test_linker_capture_fallback_and_response_file(self):
        with tempfile.TemporaryDirectory(prefix="nano-link-fallback-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.foreign_build_fixture(directory)
            wrapper = directory / "cc-wrapper"
            control = directory / "mode"
            control.write_text("ok")
            compiler = shutil.which("cc")
            wrapper.write_text(f'''#!{sys.executable}
import os, pathlib, sys
args = sys.argv[1:]
mode = pathlib.Path({str(control)!r}).read_text()
if "-dependency_info" in args:
    path = pathlib.Path(args[args.index("-dependency_info") + 2])
    if mode == "unsupported":
        path.write_bytes(b"partial")
        sys.exit(23)
    if mode == "malformed":
        import subprocess
        result = subprocess.run([{compiler!r}] + args)
        path.write_bytes(b"partial")
        sys.exit(result.returncode)
os.execv({compiler!r}, [{compiler!r}] + args)
''')
            wrapper.chmod(0o700)
            env["NANO_CC"] = str(wrapper)
            previous = None
            for mode in ("unsupported", "malformed", "ok", "ok"):
                control.write_text(mode)
                self.probe_path("build", module, env)
                generation = self.probe_path("directory", module, env)
                self.assertEqual(self.library_answer(self.probe_path("library", module, env)), 42)
                self.assertEqual((generation / "source_hashes.json").is_file(), mode == "ok")
                if previous and mode == "ok":
                    self.assertEqual(previous, generation)
                if mode == "ok": previous = generation
            response = directory / "flags.rsp"
            response.write_text("-lm\n")
            manifest = module / "module.json"
            metadata = json.loads(manifest.read_text())
            metadata["ldflags"] = ["-Xlinker", shlex.quote("@" + str(response))]
            manifest.write_text(json.dumps(metadata))
            self.probe_path("build", module, env)
            generation = self.probe_path("directory", module, env)
            self.assertFalse((generation / "source_hashes.json").exists())
            self.assertEqual(self.library_answer(self.probe_path("library", module, env)), 42)

    def test_pkg_config_query_status_and_recovery(self):
        with tempfile.TemporaryDirectory(prefix="nano-pkg-status-") as tmp:
            directory = Path(tmp)
            module, source, env = self.support.foreign_build_fixture(directory)
            tools = directory / "tools"
            tools.mkdir()
            control = directory / "query-mode"
            control.write_text("ok")
            pkg = tools / "pkg-config"
            pkg.write_text(f'''#!{sys.executable}
import pathlib, sys
if "--version" in sys.argv: print("1.0"); sys.exit(0)
if "--exists" in sys.argv: sys.exit(0)
mode = pathlib.Path({str(control)!r}).read_text()
query = "cflags" if "--cflags" in sys.argv else "libs"
if mode == query:
    print("-DQUERY_PARTIAL=1" if query == "cflags" else "-lm")
    sys.exit(23)
print("   ")
''')
            pkg.chmod(0o700)
            env["PKG_CONFIG"] = str(pkg)
            env["PATH"] = str(tools) + os.pathsep + env["PATH"]
            manifest = module / "module.json"
            metadata = json.loads(manifest.read_text())
            metadata["pkg_config"] = ["fixture"]
            manifest.write_text(json.dumps(metadata))
            result, output = self.support.compile(source, directory, "--run", env=env)
            self.assertEqual(result.returncode, 42, result.stderr)
            previous = output.read_bytes()
            old_generation = self.probe_path("directory", module, env)
            for query in ("cflags", "libs"):
                control.write_text(query)
                result, output = self.support.compile(source, directory, env=env)
                self.assertEqual(result.returncode, 1, (query, result.stderr))
                self.assertEqual(output.read_bytes(), previous)
                self.assertEqual(self.probe_path("directory", module, env), old_generation)
                control.write_text("ok")
                result, output = self.support.compile(source, directory, "--run", env=env)
                self.assertEqual(result.returncode, 42, result.stderr)
            for query in ("cflags", "libs"):
                control.write_text(query)
                c_source = module / "answer.c"
                c_source.write_text(c_source.read_text() + "\n/* I force a cold build. */\n")
                result, output = self.support.compile(source, directory, env=env)
                self.assertEqual(result.returncode, 1, (query, result.stderr))
                self.assertEqual(output.read_bytes(), previous)
                self.assertEqual(self.probe_path("directory", module, env), old_generation)
            no_source = directory / "source-free"
            no_source.mkdir()
            (no_source / "module.json").write_text(json.dumps({"name": "source_free", "pkg_config": ["fixture"]}))
            for mode, expected in (("ok", 0), ("cflags", 1), ("libs", 1), ("ok", 0)):
                control.write_text(mode)
                result = subprocess.run([str(self.probe), "build-info", str(no_source)], env=env,
                                        capture_output=True, timeout=10)
                self.assertEqual(result.returncode, expected, (mode, result.stderr))
            result, _ = self.support.compile(source, directory, "--run", env=env)
            self.assertEqual(result.returncode, 42, result.stderr)

    def test_pkg_config_snapshot_and_source_free_flags(self):
        with tempfile.TemporaryDirectory(prefix="nano-pkg-snapshot-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.foreign_build_fixture(directory)
            counts = directory / "counts.json"
            pkg = directory / "pkg-config"
            pkg.write_text(f'''#!{sys.executable}
import json, os, pathlib, sys
if "--version" in sys.argv: print("1.0"); sys.exit(0)
if "--exists" in sys.argv: sys.exit(0)
path = pathlib.Path({str(counts)!r})
counts = json.loads(path.read_text()) if path.exists() else {{}}
query = "cflags" if "--cflags" in sys.argv else "libs"
counts[query] = counts.get(query, 0) + 1
path.write_text(json.dumps(counts))
if os.environ.get("NANO_TEST_PKG_POST_FAILURE") == query and counts[query] > 1:
    print("-DPARTIAL=1")
    sys.exit(23)
print("-D" + ("VALUE" if query == "cflags" else "LINK_VALUE") + "=" +
      ("42" if counts[query] == 1 else "43"))
''')
            pkg.chmod(0o700)
            env["PKG_CONFIG"] = str(pkg)
            (module / "answer.c").write_text("long long nano_build_answer(void) { return VALUE; }\n")
            (module / "shared.c").write_text(
                "#ifndef LINK_VALUE\n#define LINK_VALUE VALUE\n#endif\n"
                '__attribute__((visibility("default"))) long long snapshot_link_answer(void) { return LINK_VALUE; }\n'
                '__attribute__((visibility("default"))) long long snapshot_compile_answer(void) { return VALUE; }\n')
            (module / "module.json").write_text(json.dumps({
                "name": "answer_native", "c_sources": ["answer.c"],
                "shared_c_sources": ["shared.c"], "pkg_config": ["fixture"]}))

            def build(target):
                result = subprocess.run([str(self.probe), "build-info", str(target)],
                                        env=env | {"NANO_VERBOSE_BUILD": "1"},
                                        capture_output=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                return result.stdout.decode().splitlines()

            lines = build(module)
            self.assertIn("compile:-DVALUE=42", lines)
            self.assertIn("link:-DLINK_VALUE=42", lines)
            shared_links = [line for line in lines if "Building shared library:" in line]
            self.assertEqual(len(shared_links), 1)
            self.assertIn(" -DVALUE=42", shared_links[0])
            self.assertIn(" -DLINK_VALUE=42", shared_links[0])
            self.assertNotIn("=43", shared_links[0])
            generation = self.probe_path("directory", module, env)
            self.assertFalse((generation / "source_hashes.json").exists())
            self.assertEqual(json.loads(counts.read_text()), {"cflags": 2, "libs": 2})
            library = self.probe_path("library", module, env)
            self.assertEqual(self.library_answer(library), 42)
            handle = ctypes.CDLL(str(library))
            for symbol in ("snapshot_link_answer", "snapshot_compile_answer"):
                function = getattr(handle, symbol)
                function.restype = ctypes.c_int64
                self.assertEqual(function(), 42)
            lines = build(module)
            self.assertIn("compile:-DVALUE=43", lines)
            self.assertIn("link:-DLINK_VALUE=43", lines)
            recovered = self.probe_path("directory", module, env)
            self.assertNotEqual(recovered, generation)
            self.assertTrue((recovered / "source_hashes.json").is_file())
            build(module)
            self.assertEqual(self.probe_path("directory", module, env), recovered)

            for query in ("cflags", "libs"):
                counts.write_text("{}")
                env["NANO_TEST_PKG_POST_FAILURE"] = query
                c_source = module / "answer.c"
                c_source.write_text(c_source.read_text() + "\n/* I force another cold build. */\n")
                lines = build(module)
                self.assertIn("compile:-DVALUE=42", lines)
                self.assertIn("link:-DLINK_VALUE=42", lines)
                generation = self.probe_path("directory", module, env)
                self.assertFalse((generation / "source_hashes.json").exists())
                self.assertEqual(self.library_answer(self.probe_path("library", module, env)), 42)
                del env["NANO_TEST_PKG_POST_FAILURE"]
                build(module)
                recovered = self.probe_path("directory", module, env)
                self.assertTrue((recovered / "source_hashes.json").is_file())

            no_source = directory / "source-free"
            no_source.mkdir()
            (no_source / "module.json").write_text(json.dumps({
                "name": "source_free", "pkg_config": ["fixture"]}))
            counts.write_text("{}")
            lines = build(no_source)
            self.assertIn("no object", lines)
            self.assertIn("compile:-DVALUE=42", lines)
            self.assertIn("link:-DLINK_VALUE=42", lines)
            self.assertEqual(json.loads(counts.read_text()), {"cflags": 1, "libs": 1})

    def test_pkg_config_link_only_change_invalidates_cache(self):
        with tempfile.TemporaryDirectory(prefix="nano-pkg-link-change-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.foreign_build_fixture(directory)
            response = directory / "link-response"
            response.write_text("-DLINK_SELECTION=42")
            pkg = directory / "pkg-config"
            pkg.write_text(f'''#!{sys.executable}
import pathlib, sys
if "--version" in sys.argv: print("1.0")
elif "--libs" in sys.argv: print(pathlib.Path({str(response)!r}).read_text())
''')
            pkg.chmod(0o700)
            env["PKG_CONFIG"] = str(pkg)
            manifest = module / "module.json"
            metadata = json.loads(manifest.read_text())
            metadata["pkg_config"] = ["fixture"]
            manifest.write_text(json.dumps(metadata))
            self.probe_path("build", module, env)
            first = self.probe_path("directory", module, env)
            self.probe_path("build", module, env)
            self.assertEqual(self.probe_path("directory", module, env), first)
            response.write_text("-DLINK_SELECTION=43")
            self.probe_path("build", module, env)
            second = self.probe_path("directory", module, env)
            self.assertNotEqual(second, first)
            self.probe_path("build", module, env)
            self.assertEqual(self.probe_path("directory", module, env), second)

    def test_pkg_config_output_and_literal_arguments(self):
        with tempfile.TemporaryDirectory(prefix="nano-pkg-boundary-") as tmp:
            directory = Path(tmp)
            pkg = directory / "pkg ' literal"
            pkg.write_text(f'''#!{sys.executable}
import json, os, signal, sys
mode = os.environ["NANO_QUERY_FIXTURE"]
if mode == "empty": print(" \\t\\r"); sys.exit(0)
if mode == "partial": print("-DIGNORED=1"); sys.exit(23)
if mode == "signal": os.kill(os.getpid(), signal.SIGTERM)
if mode == "nul": sys.stdout.buffer.write(b"-DOK=1\\0-DHIDDEN=1"); sys.exit(0)
if mode == "large": print("x" * 65537); sys.exit(0)
if mode == "limit": sys.stdout.write("x" * 65536); sys.exit(0)
print(json.dumps({{"args": sys.argv[1:], "path": os.environ.get("PKG_CONFIG_PATH", "")}}))
''')
            pkg.chmod(0o700)
            env = os.environ.copy()
            env["PKG_CONFIG"] = str(pkg)
            env["NANO_ALLOW_PACKAGE_INSTALL"] = "0"
            env["PKG_CONFIG_PATH"] = "literal ' $(touch injected-env) ; path"
            package = "-literal ' $(touch injected-package) ; package"
            for mode, expected in (("empty", 0), ("partial", 1), ("signal", 1), ("nul", 1), ("large", 1), ("limit", 0), ("literal", 0)):
                env["NANO_QUERY_FIXTURE"] = mode
                result = subprocess.run([str(self.probe), "pkgflags", package], cwd=directory,
                                        env=env, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, expected, (mode, result.stderr))
                if mode == "empty": self.assertEqual(result.stdout, b"\n")
                elif mode == "limit": self.assertEqual(len(result.stdout), 65537)
                elif mode == "literal":
                    observed = json.loads(result.stdout)
                    self.assertEqual(observed["args"], ["--cflags", "--", package])
                    self.assertTrue(observed["path"].endswith(env["PKG_CONFIG_PATH"]))
                self.assertFalse((directory / "injected-env").exists())
                self.assertFalse((directory / "injected-package").exists())

    def test_changed_include_search_path_changes_output(self):
        with tempfile.TemporaryDirectory(prefix="nano-identity-") as tmp:
            directory = Path(tmp)
            module, source, env = self.support.foreign_build_fixture(directory)
            (module / "answer.c").write_text("#include <stdint.h>\n#include <answer.h>\nint64_t nano_build_answer(void) { return ANSWER; }\n")
            for answer in (42, 43):
                include = directory / str(answer)
                include.mkdir()
                (include / "answer.h").write_text(f"#define ANSWER {answer}\n")
                env["CPATH"] = str(include)
                result, output = self.support.compile(source.replace(" 42)", f" {answer})"), directory, "--run", env=env)
                self.assertEqual(result.returncode, answer, result.stderr)
                execution = self.support.execute(output, env=env)
                self.assertEqual(execution.returncode, answer, execution.stderr)

    def test_toolchain_stamp_and_unresolved_compiler(self):
        with tempfile.TemporaryDirectory(prefix="nano-identity-") as tmp:
            directory = Path(tmp)
            module, source, env = self.support.foreign_build_fixture(directory)
            calls = directory / "calls"
            compiler = directory / "cc"
            self.counting_compiler(compiler, calls)
            env["NANO_CC"] = str(compiler)
            expected = (("first", 2), ("first", 3), ("second", 5)) if self.gcc_validation else (("first", 1), ("first", 1), ("second", 2))
            for stamp, count in expected:
                env["NANO_TOOLCHAIN_ID"] = stamp
                result, _ = self.support.compile(source, directory, "--run", env=env)
                self.assertEqual(result.returncode, 42, result.stderr)
                self.assertEqual(len(calls.read_text().splitlines()), count)
            old = self.snapshot(module / ".build")
            env["NANO_CC"] = str(directory / "missing-compiler")
            result, _ = self.support.compile(source, directory, env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(self.snapshot(module / ".build"), old)
            env["NANO_CC"] = str(compiler) + " -DFIXTURE=1"
            for count in ((6, 7) if self.gcc_validation else (3, 4)):
                result, _ = self.support.compile(source, directory, "--run", env=env)
                self.assertEqual(result.returncode, 42, result.stderr)
                self.assertEqual(len(calls.read_text().splitlines()), count)
                self.assertFalse((module / ".build" / "current" / "source_hashes.json").exists())

    def test_working_directory_changes_relative_include_resolution(self):
        with tempfile.TemporaryDirectory(prefix="nano-identity-42-") as tmp:
            directory = Path(tmp)
            module, source, env = self.support.foreign_build_fixture(directory)
            (module / "answer.c").write_text("#include <stdint.h>\n#include <answer.h>\nint64_t nano_build_answer(void) { return ANSWER; }\n")
            (module / "module.json").write_text(json.dumps({"name": "answer_native", "c_sources": ["answer.c"], "include_dirs": ["headers"]}))
            for answer in (42, 43):
                cwd = directory / str(answer)
                include = cwd / "headers"
                include.mkdir(parents=True)
                (include / "answer.h").write_text(f"#define ANSWER {answer}\n")
                result, output = self.support.compile(source.replace(" 42)", f" {answer})"), directory, "--run", env=env, cwd=cwd)
                self.assertEqual(result.returncode, answer, result.stderr)
                execution = self.support.execute(output, env=env)
                self.assertEqual(execution.returncode, answer, execution.stderr)

    def test_driver_change_during_build_withholds_reuse_evidence(self):
        with tempfile.TemporaryDirectory(prefix="nano-identity-") as tmp:
            directory = Path(tmp)
            module, source, env = self.support.foreign_build_fixture(directory)
            calls = directory / "calls"
            compiler = directory / "cc"
            self.counting_compiler(compiler, calls)
            text = compiler.read_text().replace("import os, sys", "import os, sys, pathlib")
            text = text.replace("os.execv(", f'''marker = pathlib.Path({str(directory / 'changed')!r})
if not marker.exists():
    marker.touch()
    with open(__file__, "a") as script: script.write("\\n# changed driver bytes\\n")
os.execv(''')
            compiler.write_text(text)
            env["NANO_CC"] = str(compiler)
            expected = ((2, False), (4, True), (5, True)) if self.gcc_validation else ((1, False), (2, True), (2, True))
            for count, reusable in expected:
                result, _ = self.support.compile(source, directory, "--run", env=env)
                self.assertEqual(result.returncode, 42, result.stderr)
                self.assertEqual(len(calls.read_text().splitlines()), count)
                self.assertEqual((module / ".build" / "current" / "source_hashes.json").exists(), reusable)

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
            process = self.start(directory / "interrupted", source.replace(" 42)", " 43)"), env)
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
            result, output = self.support.compile(source.replace(" 42)", " 43)"), directory, "--run", env=env)
            self.assertEqual(result.returncode, 43, result.stderr)

    def test_surviving_compiler_stage_is_not_abandoned(self):
        for shared in (False, True):
            with self.subTest(shared=shared), tempfile.TemporaryDirectory(prefix="nano-orphan-child-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.foreign_build_fixture(directory)
                if shared: env["NANO_BUILD_CACHE"] = str(directory / "shared")
                self.probe_path("build", module, env)
                previous_library = self.probe_path("library", module, env)
                previous = self.probe_path("directory", module, env)
                previous_bytes = self.snapshot(previous)
                (module / "answer.c").write_text("long long nano_build_answer(void) { return 43; }\n")
                compiler = directory / "cc-survivor"
                compiler.write_text(f'''#!{sys.executable}
import json, os, pathlib, subprocess, sys, time
control = pathlib.Path({str(directory)!r})
if "-c" in sys.argv:
    output = pathlib.Path(sys.argv[sys.argv.index("-o") + 1])
    output.write_bytes(b"partial compiler output")
    (control / "ready").write_text(str(output))
    deadline = time.monotonic() + 20
    while not (control / "release").exists():
        if (control / "ping").exists(): (control / "ack").touch()
        if time.monotonic() >= deadline: sys.exit(25)
        time.sleep(0.01)
    result = subprocess.run([{shutil.which('cc')!r}] + sys.argv[1:])
    (control / "done").write_text(json.dumps({{"status": result.returncode, "size": output.stat().st_size}}))
    sys.exit(result.returncode)
os.execv({shutil.which('cc')!r}, [{shutil.which('cc')!r}] + sys.argv[1:])
''')
                compiler.chmod(0o700)
                child_env = dict(env, NANO_CC=str(compiler))
                process = subprocess.Popen([str(self.probe), "build", str(module)], cwd=ROOT,
                                           env=child_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           start_new_session=True)
                def await_file(name):
                    path = directory / name
                    deadline = time.monotonic() + 10
                    while not path.exists() and time.monotonic() < deadline: time.sleep(0.01)
                    self.assertTrue(path.exists(), f"I did not observe {name}")
                    return path
                try:
                    self.wait_ready(directory / "ready", process)
                    private_output = Path((directory / "ready").read_text())
                    self.assertEqual(private_output.parent.parent, self.probe_path("root", module, env).resolve())
                    self.assertTrue(private_output.parent.name.startswith(".nano-build-"))
                    process.kill()
                    self.assertEqual(process.wait(timeout=5), -signal.SIGKILL)
                    # I require new evidence from the child after its parent died.
                    (directory / "ping").touch()
                    await_file("ack")
                    self.assertEqual(private_output.read_bytes(), b"partial compiler output")
                    (module / "answer.c").write_text("long long nano_build_answer(void) { return 44; }\n")
                    self.probe_path("build", module, env)
                    current = self.probe_path("directory", module, env)
                    self.assertNotEqual(current, previous)
                    published = self.snapshot(current)
                    self.assertEqual(private_output.read_bytes(), b"partial compiler output")
                    (directory / "release").touch()
                    _, error = process.communicate(timeout=10)
                    done = json.loads(await_file("done").read_text())
                    self.assertEqual(done["status"], 0, error)
                    self.assertGreater(done["size"], len(b"partial compiler output"))
                    self.assertTrue(private_output.is_file())
                    self.assertEqual(self.snapshot(current), published)
                    self.assertEqual(self.snapshot(previous), previous_bytes)
                    self.assertEqual(self.library_answer(previous_library), 42)
                    self.assertEqual(self.library_answer(self.probe_path("library", module, env)), 44)
                    self.probe_path("build", module, env)
                    self.assertEqual(self.probe_path("directory", module, env), current)
                finally:
                    if process.poll() is None or not (directory / "done").exists():
                        try: os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError: pass
                    process.communicate(timeout=10)

    def test_make_clean_preserves_foreign_generations(self):
        from tests.test_clean_cache_retention import fixture
        with tempfile.TemporaryDirectory(prefix="nano-retained-library-") as tmp:
            directory = Path(tmp)
            fixture(directory)
            module, _, env = self.support.foreign_build_fixture(directory)
            env["NANO_BUILD_CACHE"] = str(directory / "obj/module_cache")
            self.probe_path("build", module, env)
            generation = self.probe_path("directory", module, env)
            library = self.probe_path("library", module, env)
            retained = self.snapshot(generation)
            result = subprocess.run(["make", "-f", "Makefile.gnu", "clean"], cwd=directory,
                                    env=env, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.snapshot(generation), retained)
            result = subprocess.run([sys.executable, "-c",
                "import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); "
                "lib.nano_build_answer.restype=ctypes.c_int64; print(lib.nano_build_answer())", str(library)],
                capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), b"42")
            self.probe_path("build", module, env)
            self.assertEqual(self.probe_path("directory", module, env), generation)

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
                self.assertEqual(len((directory / "calls").read_text().splitlines()), 3 if self.gcc_validation else 1)
                for name in ("first", "second"):
                    self.assertEqual(self.support.execute(directory / name / "program.nvm", env=env).returncode, 42)
                self.assertFalse(list((module / ".build").glob(".nano-build-*")))
            finally:
                for process in processes:
                    self.stop(process)

    def test_old_object_and_library_paths_survive_new_publication(self):
        with tempfile.TemporaryDirectory(prefix="nano-generation-") as tmp:
            directory = Path(tmp)
            module, source, env = self.support.foreign_build_fixture(directory)
            old_object = self.probe_path("build", module, env)
            old_library = self.probe_path("library", module, env)
            self.assertIn(".nano-gen-", old_object.parent.name)
            self.assertEqual(old_object.parent, old_library.parent)
            old = self.snapshot(old_object.parent)
            changed = module / "answer.c"
            changed.write_text(changed.read_text().replace("42", "43"))
            result, _ = self.support.compile(source.replace(" 42)", " 43)"), directory, "--run", env=env)
            self.assertEqual(result.returncode, 43, result.stderr)
            new_object = self.probe_path("build", module, env)
            new_library = self.probe_path("library", module, env)
            self.assertNotEqual(old_object.parent, new_object.parent)
            self.assertEqual(new_object.parent, new_library.parent)
            self.assertEqual(self.snapshot(old_object.parent), old)
            self.assertEqual(self.library_answer(old_library), 42)
            self.assertEqual(self.library_answer(new_library), 43)
            # I link the previously returned object only after replacement.
            main = directory / "main.c"
            main.write_text("#include <stdint.h>\nint64_t nano_build_answer(void);\nint main(void) { return (int)nano_build_answer(); }\n")
            executable = directory / "old-native"
            subprocess.run([shutil.which("cc"), str(main), str(old_object), "-o", str(executable)], check=True, capture_output=True, timeout=10)
            self.assertEqual(subprocess.run([str(executable)], timeout=10).returncode, 42)

    def test_reader_keeps_complete_generation_during_build(self):
        with tempfile.TemporaryDirectory(prefix="nano-generation-") as tmp:
            directory = Path(tmp)
            module, source, env = self.support.foreign_build_fixture(directory)
            old_object = self.probe_path("build", module, env)
            old_library = self.probe_path("library", module, env)
            changed = module / "answer.c"
            changed.write_text(changed.read_text().replace("42", "43"))
            env["NANO_CC"] = str(self.blocking_compiler(directory))
            process = self.start(directory / "writer", source.replace(" 42)", " 43)"), env)
            try:
                self.wait_ready(directory / "ready", process)
                for _ in range(3):
                    self.assertEqual(self.probe_path("library", module, env), old_library)
                    self.assertEqual(self.library_answer(old_library), 42)
                    self.assertTrue((old_object.parent / "source_hashes.json").is_file())
                    self.assertIsNone(process.poll())
                (directory / "release").touch()
                _, error = process.communicate(timeout=20)
                self.assertEqual(process.returncode, 0, error)
                new_library = self.probe_path("library", module, env)
                self.assertNotEqual(new_library, old_library)
                self.assertEqual(self.library_answer(new_library), 43)
                self.assertEqual(self.library_answer(old_library), 42)
            finally:
                self.stop(process)

    def test_invalid_generation_pointer_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="nano-generation-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.foreign_build_fixture(directory)
            cache = module / ".build"
            cache.mkdir()
            self.assertEqual(self.probe_path("directory", module, env), cache)
            pointer = cache / "current"
            for target in ("../outside", ".nano-gen-unknown", ".nano-gen-ABC123/../outside", ".nano-gen-ABC123"):
                with self.subTest(target=target):
                    pointer.symlink_to(target)
                    result = subprocess.run([str(self.probe), "directory", str(module)], env=env, capture_output=True, timeout=10)
                    self.assertNotEqual(result.returncode, 0)
                    pointer.unlink()
            (cache / ".nano-gen-ABC123").symlink_to(directory, target_is_directory=True)
            pointer.symlink_to(".nano-gen-ABC123")
            result = subprocess.run([str(self.probe), "directory", str(module)], env=env, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)

    def test_pointer_rename_failure_keeps_valid_old_pointer(self):
        with tempfile.TemporaryDirectory(prefix="nano-generation-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.foreign_build_fixture(directory)
            old_object = self.probe_path("build", module, env)
            old_library = self.probe_path("library", module, env)
            old = self.snapshot(old_object.parent)
            changed = module / "answer.c"
            changed.write_text(changed.read_text().replace("42", "43"))
            env["NANO_TEST_POINTER_FAILURE"] = "1"
            result = subprocess.run([str(self.probe), "build", str(module)], cwd=ROOT,
                                    env=env, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"could not publish complete", result.stderr)
            self.assertEqual(self.probe_path("library", module, env), old_library)
            self.assertEqual(self.library_answer(old_library), 42)
            self.assertEqual(self.snapshot(old_object.parent), old)
            self.assertEqual(len(list((module / ".build").glob(".nano-gen-*"))), 1)
            self.assertFalse(list((module / ".build").glob(".nano-build-*")))
            env.pop("NANO_TEST_POINTER_FAILURE")
            self.assertNotEqual(self.probe_path("build", module, env), old_object)
            self.assertEqual(self.library_answer(self.probe_path("library", module, env)), 43)

    def test_damaged_cached_artifact_rebuilds_new_generation(self):
        for artifact in ("object", "library"):
            for damage in ("empty", "symlink"):
                with self.subTest(artifact=artifact, damage=damage), tempfile.TemporaryDirectory(prefix="nano-generation-") as tmp:
                    directory = Path(tmp)
                    module, _, env = self.support.foreign_build_fixture(directory)
                    old_object = self.probe_path("build", module, env)
                    path = old_object if artifact == "object" else self.probe_path("library", module, env)
                    if damage == "empty":
                        path.write_bytes(b"")
                    else:
                        saved = directory / path.name
                        path.rename(saved)
                        path.symlink_to(saved)
                    new_object = self.probe_path("build", module, env)
                    self.assertNotEqual(old_object.parent, new_object.parent)
                    self.assertGreater(new_object.stat().st_size, 0)
                    self.assertEqual(self.library_answer(self.probe_path("library", module, env)), 42)
                    if damage == "empty": self.assertEqual(path.stat().st_size, 0)
                    else: self.assertTrue(path.is_symlink())

    def test_distinct_module_paths_do_not_share_cache(self):
        with tempfile.TemporaryDirectory(prefix="nano-namespace-") as tmp:
            directory = Path(tmp)
            objects = []
            for parent, answer in ((directory / "a_b", 42), (directory / "a" / "b", 43)):
                parent.mkdir(parents=True)
                module, source, env = self.support.foreign_build_fixture(parent)
                env["NANO_BUILD_CACHE"] = str(directory / "cache")
                (module / "answer.c").write_text('#include <stdint.h>\n#include "answer.h"\nint64_t nano_build_answer(void) { return ANSWER; }\n')
                (module / "answer.h").write_text(f"#define ANSWER {answer}\n")
                result, output = self.support.compile(source.replace(" 42)", f" {answer})"), parent, "--run", env=env)
                self.assertEqual(result.returncode, answer, result.stderr)
                objects.append(self.probe_path("build", module, env))
                self.assertEqual(self.library_answer(self.probe_path("library", module, env)), answer)
                self.assertEqual(self.support.execute(output, env=env).returncode, answer)
            self.assertNotEqual(objects[0].parent.parent, objects[1].parent.parent)

    def test_namespace_is_canonical_bounded_and_versioned(self):
        with tempfile.TemporaryDirectory(prefix="nano-namespace-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.foreign_build_fixture(directory)
            cache = directory / "cache"
            env["NANO_BUILD_CACHE"] = str(cache)
            alias = directory / "alias"
            alias.symlink_to(module, target_is_directory=True)
            expected = cache / ("v2-" + hashlib.sha256(os.fsencode(module.resolve())).hexdigest())
            for spelling in (module, module.resolve(), Path(os.path.relpath(module, ROOT)), alias, module / "."):
                self.assertEqual(self.probe_path("root", spelling, env), expected)
            size = len(os.fsencode(expected)) + 1
            for capacity, succeeds in ((0, False), (1, False), (size - 1, False), (size, True)):
                result = subprocess.run([str(self.probe), "root", str(module), str(capacity)], cwd=ROOT, env=env, capture_output=True, timeout=10)
                self.assertEqual(result.returncode == 0, succeeds, result.stderr)
            for invalid in ("", str(directory / "missing"), str(module / "answer.c"), str(directory / ("z" * 1500))):
                result = subprocess.run([str(self.probe), "root", invalid], cwd=ROOT, env=env, capture_output=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)

    def test_long_directory_uses_fixed_length_namespace(self):
        with tempfile.TemporaryDirectory(prefix="nano-namespace-") as tmp:
            directory = Path(tmp)
            parent = directory
            for _ in range(6): parent = parent / ("long" * 15)
            parent.mkdir(parents=True)
            module, _, env = self.support.foreign_build_fixture(parent)
            env["NANO_BUILD_CACHE"] = str(directory / "cache")
            root = self.probe_path("root", module, env)
            self.assertEqual(len(root.name), 67)
            self.probe_path("build", module, env)
            self.assertEqual(self.library_answer(self.probe_path("library", module, env)), 42)

    def test_alias_metadata_uses_physical_module_directory(self):
        with tempfile.TemporaryDirectory(prefix="nano-namespace-") as tmp:
            directory = Path(tmp)
            physical = directory / "physical"
            physical.mkdir()
            module, _, env = self.support.foreign_build_fixture(physical)
            env["NANO_BUILD_CACHE"] = str(directory / "cache")
            (module / "answer.c").write_text("#include <stdint.h>\n#include <answer.h>\nint64_t nano_build_answer(void) { return ANSWER; }\n")
            (module / "module.json").write_text(json.dumps({"name": "answer_native", "c_sources": ["answer.c"], "include_dirs": ["headers"]}))
            alternate = directory / "alternate"
            for parent, answer in ((physical, 42), (alternate, 43)):
                headers = parent / "headers"
                headers.mkdir(parents=True)
                (headers / "answer.h").write_text(f"#define ANSWER {answer}\n")
            alias = alternate / "foreign"
            alias.symlink_to(module, target_is_directory=True)
            aliased_object = self.probe_path("build", alias, env)
            self.assertEqual(self.library_answer(self.probe_path("library", alias, env)), 42)
            self.assertEqual(self.probe_path("build", module, env), aliased_object)

    def test_ambiguous_legacy_cache_is_not_reused_or_modified(self):
        with tempfile.TemporaryDirectory(prefix="nano-namespace-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.foreign_build_fixture(directory)
            cache = directory / "cache"
            env["NANO_BUILD_CACHE"] = str(cache)
            legacy = cache / str(module).replace("/", "_")
            legacy.mkdir(parents=True)
            extension = "dylib" if sys.platform == "darwin" else "so"
            (legacy / f"libanswer_native.{extension}").write_bytes(b"untrusted legacy artifact")
            before = self.snapshot(legacy)
            result = subprocess.run([str(self.probe), "library", str(module)], cwd=ROOT, env=env, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            current = self.probe_path("build", module, env)
            self.assertNotEqual(current.parent.parent, legacy)
            self.assertEqual(self.library_answer(self.probe_path("library", module, env)), 42)
            self.assertEqual(self.snapshot(legacy), before)
            self.assertEqual(len(list(legacy.iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
