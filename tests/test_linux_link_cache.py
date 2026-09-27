"""I require current Linux link results without inferring GNU input inventories."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests import test_module_cache_publication as cache_support
from tests.characterize_linker_inputs import measure

ROOT = cache_support.ROOT


@unittest.skipUnless(sys.platform == "linux", "I exercise the Linux warm-link path")
class LinuxLinkCache(unittest.TestCase):
    probe = ROOT / "obj/test_module_generation_probe"

    def setUp(self):
        self.cache = cache_support.ModuleCachePublication()
        self.cache.probe = self.probe
        self.cache.setUp()

    def run_command(self, argv, env=None):
        result = subprocess.run([str(value) for value in argv], cwd=ROOT, env=env,
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def answer(self, library):
        result = self.run_command([sys.executable, "-c",
            "import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); "
            "lib.nano_build_answer.restype=ctypes.c_int64; print(lib.nano_build_answer())", library])
        return int(result.stdout)

    def test_archive_and_search_results(self):
        report = measure(shutil.which("cc"), self.probe)
        self.assertTrue(report["unchanged_generation_reused"])
        self.assertTrue(report["archive_size_preserved"])
        self.assertTrue(report["archive_timestamp_preserved"])
        self.assertEqual(report["cache_answer_after_archive_edit"], 43)
        self.assertEqual(report["cache_answer_after_earlier_library"], 44)

    def test_thin_response_and_unusual_archive_inputs(self):
        for mode in ("thin", "response", "unusual"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix="nano-linux-link-") as tmp:
                directory = Path(tmp)
                module, _, env = self.cache.support.foreign_build_fixture(directory)
                member, obj = directory / "member.c", directory / "member.o"
                archive = directory / ("space ' back\\slash\narchive.a" if mode == "unusual" else "selected.a")
                def compile_member(value):
                    member.write_text(f"long long selected(void) {{ return {value}; }}\n")
                    self.run_command(["cc", "-fPIC", "-c", member, "-o", obj])
                compile_member(42)
                self.run_command(["ar", "rcsT" if mode == "thin" else "rcs", archive, obj])
                original_archive = archive.read_bytes()
                if mode == "thin": self.assertTrue(original_archive.startswith(b"!<thin>\n"))
                response = directory / "input.rsp"
                response.write_text(str(archive) + "\n")
                flags = ["-Xlinker", "@" + str(response)] if mode == "response" else [str(archive)]
                (module / "answer.c").write_text("extern long long selected(void);\n"
                    "long long nano_build_answer(void) { return selected(); }\n")
                (module / "module.json").write_text(json.dumps({"name": "answer_native",
                    "c_sources": ["answer.c"], "ldflags": [shlex.quote(flag) for flag in flags]}))
                self.cache.probe_path("build", module, env)
                previous = self.cache.probe_path("directory", module, env)
                old_library = self.cache.probe_path("library", module, env)
                self.assertEqual(self.answer(old_library), 42)
                self.cache.probe_path("build", module, env)
                self.assertEqual(self.cache.probe_path("directory", module, env), previous)
                stamp = obj.stat()
                compile_member(43)
                os.utime(obj, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                if mode == "thin":
                    self.assertEqual(archive.read_bytes(), original_archive)
                elif mode == "response":
                    replacement = directory / "replacement.a"
                    self.run_command(["ar", "rcs", replacement, obj])
                    response.write_text(str(replacement) + "\n")
                else:
                    self.run_command(["ar", "rcs", archive, obj])
                self.cache.probe_path("build", module, env)
                current = self.cache.probe_path("directory", module, env)
                self.assertNotEqual(current, previous)
                self.assertEqual(self.answer(self.cache.probe_path("library", module, env)), 43)
                self.assertEqual(self.answer(old_library), 42)
                self.cache.probe_path("build", module, env)
                self.assertEqual(self.cache.probe_path("directory", module, env), current)

    def test_shared_objects_reuse_and_failed_link_recovery(self):
        version = subprocess.run([shutil.which("cc"), "--version"], capture_output=True, check=True, timeout=10)
        gcc_validation = b"Free Software Foundation" in version.stdout and b"clang version" not in version.stdout
        with tempfile.TemporaryDirectory(prefix="nano-linux-warm-") as tmp:
            directory = Path(tmp)
            module, _, env = self.cache.support.foreign_build_fixture(directory)
            (module / "answer.c").write_text("extern long long private_answer(void), extra(void);\n"
                "long long nano_build_answer(void) { return private_answer() + extra(); }\n")
            (module / "extra.c").write_text("long long extra(void) { return 2; }\n")
            (module / "private.c").write_text("long long private_answer(void) { return 40; }\n")
            (module / "module.json").write_text(json.dumps({"name": "answer_native",
                "c_sources": ["answer.c", "extra.c"], "shared_c_sources": ["private.c"]}))
            calls, fail = directory / "calls", directory / "fail"
            compiler = directory / "cc-controlled"
            compiler.write_text(f'''#!{sys.executable}
import os, pathlib, sys
if "-c" in sys.argv or "-shared" in sys.argv:
    with open({str(calls)!r}, "a") as log: log.write("C\\n" if "-c" in sys.argv else "L\\n")
if "-shared" in sys.argv and pathlib.Path({str(fail)!r}).exists():
    pathlib.Path(sys.argv[sys.argv.index("-o") + 1]).write_bytes(b"partial link")
    sys.exit(23)
os.execv({shutil.which('cc')!r}, [{shutil.which('cc')!r}] + sys.argv[1:])
''')
            compiler.chmod(0o700)
            env["NANO_CC"] = str(compiler)
            self.cache.probe_path("build", module, env)
            previous = self.cache.probe_path("directory", module, env)
            retained = self.cache.snapshot(previous)
            self.cache.probe_path("build", module, env)
            self.assertEqual(self.cache.probe_path("directory", module, env), previous)
            self.assertEqual(calls.read_text().splitlines().count("C"), 9 if gcc_validation else 3)
            self.assertEqual(calls.read_text().splitlines().count("L"), 2)
            fail.touch()
            result = subprocess.run([str(self.probe), "build", str(module)], cwd=ROOT,
                                    env=env, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn(b"could not validate", result.stderr)
            self.assertEqual(calls.read_text().splitlines().count("L"), 3)
            self.assertEqual(self.cache.snapshot(previous), retained)
            self.assertEqual(self.cache.probe_path("directory", module, env), previous)
            fail.unlink()
            self.cache.probe_path("build", module, env)
            self.assertEqual(self.cache.probe_path("directory", module, env), previous)
            self.assertEqual(calls.read_text().splitlines().count("C"), 15 if gcc_validation else 3)
            self.assertEqual(self.answer(self.cache.probe_path("library", module, env)), 42)

    def test_warm_validation_accepts_complete_long_link_command(self):
        with tempfile.TemporaryDirectory(prefix="nano-linux-long-link-") as tmp:
            directory = Path(tmp) / ("long-path-" * 18)
            directory.mkdir()
            module, _, env = self.cache.support.foreign_build_fixture(directory)
            sources = []
            for index in range(18):
                name = f"private_{index}.c"
                (module / name).write_text(f"long long private_{index}(void) {{ return {index}; }}\n")
                sources.append(name)
            declarations = ", ".join(f"private_{index}(void)" for index in range(18))
            expression = " + ".join(f"private_{index}()" for index in range(18))
            (module / "answer.c").write_text(
                f"extern long long {declarations};\n"
                f"long long nano_build_answer(void) {{ return {expression}; }}\n")
            (module / "module.json").write_text(json.dumps({
                "name": "answer_native", "c_sources": ["answer.c"],
                "shared_c_sources": sources}))
            self.cache.probe_path("build", module, env)
            previous = self.cache.probe_path("directory", module, env)
            self.cache.probe_path("build", module, env)
            self.assertEqual(self.cache.probe_path("directory", module, env), previous)
            self.assertEqual(self.answer(self.cache.probe_path("library", module, env)), 153)

    def test_library_comparison_boundaries(self):
        for kind in ("equal", "different", "short", "empty", "missing", "symlink", "fifo"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix="nano-link-compare-") as tmp:
                directory = Path(tmp)
                left, right = directory / "left", directory / "right"
                left.write_bytes(b"a" * 9000)
                if kind == "equal": right.write_bytes(left.read_bytes())
                elif kind == "different": right.write_bytes(b"a" * 8999 + b"b")
                elif kind == "short": right.write_bytes(b"a")
                elif kind == "empty": right.touch()
                elif kind == "symlink": right.symlink_to(left)
                elif kind == "fifo": os.mkfifo(right)
                result = subprocess.run([str(self.probe), "equal-libraries", str(left), str(right)],
                                        capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 0 if kind == "equal" else 1, result.stderr)


if __name__ == "__main__":
    unittest.main()
