"""I protect cached foreign artifacts while compilers fail or overlap."""
import json
import ctypes
import hashlib
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
    @classmethod
    def setUpClass(cls):
        cls.probe = ROOT / "obj/test_module_generation_probe"
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
                self.assertEqual(calls.read_text().splitlines(), ["compile"])
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
                self.assertEqual(calls.read_text().splitlines(), ["compile", "compile"])

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
if "-E" in sys.argv:
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
            for stamp, count in (("first", 1), ("first", 1), ("second", 2)):
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
            for count in (3, 4):
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
            for count, reusable in ((1, False), (2, True), (2, True)):
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
