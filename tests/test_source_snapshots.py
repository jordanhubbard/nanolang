"""I compile retained translation units and keep failed replacements private."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests import test_module_cache_publication as cache
from tests.characterize_source_snapshot import measure


class SourceSnapshots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run([shutil.which("cc"), "--version"], capture_output=True, timeout=10)
        if result.returncode or b"clang version" not in result.stdout.splitlines()[0]:
            raise unittest.SkipTest("I have verified retained translation units for ordinary Clang C")

    def setUp(self):
        self.support = cache.ModuleCachePublication()
        self.support.probe = cache.ROOT / "obj/test_module_generation_probe"
        self.support.setUp()

    def answer(self, library):
        result = subprocess.run([sys.executable, "-c",
            "import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); "
            "lib.nano_build_answer.restype=ctypes.c_int64; print(lib.nano_build_answer())",
            str(library)], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        return int(result.stdout)

    def test_restored_source_and_header_changes(self):
        observed = measure(shutil.which("cc"))
        self.assertEqual(len(observed["cases"]), 4)
        for case in observed["cases"]:
            with self.subTest(case=case):
                self.assertEqual((case["cold_answer"], case["warm_answer"], case["fresh_answer"]), (42, 42, 42))
                for key in ("bytes_restored", "size_preserved", "mtime_preserved", "reuse_record", "generation_reused"):
                    self.assertTrue(case[key], key)
                self.assertEqual(case["total_compilations"], 1)

    def test_multiple_and_shared_only_sources(self):
        with tempfile.TemporaryDirectory(prefix="nano-retained-multiple-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.support.foreign_build_fixture(directory)
            (module / "module.json").write_text(json.dumps({"name": "answer_native",
                "c_sources": ["answer.c", "extra.c"], "shared_c_sources": ["private.c"]}))
            (module / "answer.c").write_text("long long extra(void); long long private_value(void);\n"
                "long long nano_build_answer(void) { return 20 + extra() + private_value(); }\n")
            (module / "extra.c").write_text("long long extra(void) { return 10; }\n")
            (module / "private.c").write_text("long long private_value(void) { return 12; }\n")
            wrapper, calls = directory / "cc", directory / "calls"
            wrapper.write_text(f'''#!{sys.executable}
import os, pathlib, subprocess, sys
if "-c" in sys.argv:
    with open({str(calls)!r}, "a") as log: log.write("C\\n")
    paths = list(pathlib.Path({str(module)!r}).glob("*.c"))
    original = [(p, p.read_bytes(), p.stat()) for p in paths]
    try:
        for p, data, stamp in original: p.write_bytes(data.replace(b"return 1", b"return 2"))
        result = subprocess.run([{shutil.which('cc')!r}] + sys.argv[1:])
    finally:
        for p, data, stamp in original:
            p.write_bytes(data)
            os.utime(p, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    sys.exit(result.returncode)
os.execv({shutil.which('cc')!r}, [{shutil.which('cc')!r}] + sys.argv[1:])
''')
            wrapper.chmod(0o700)
            env["NANO_CC"] = str(wrapper)
            self.support.probe_path("build", module, env)
            generation = self.support.probe_path("directory", module, env)
            self.assertEqual(len(list(generation.glob("__snapshot_*.i"))), 3)
            self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)
            self.support.probe_path("build", module, env)
            self.assertEqual(self.support.probe_path("directory", module, env), generation)
            self.assertEqual(len(calls.read_text().splitlines()), 3)
            self.assertTrue((generation / "source_hashes.json").is_file())

    def test_compile_diagnostics_preserve_previous_generation(self):
        with tempfile.TemporaryDirectory(prefix="nano-retained-failure-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.support.foreign_build_fixture(directory)
            self.support.probe_path("build", module, env)
            generation = self.support.probe_path("directory", module, env)
            library = self.support.probe_path("library", module, env)
            original = self.support.snapshot(generation)
            source = module / "answer.c"
            source.write_text("long long nano_build_answer(void) { return missing_value; }\n")
            result = subprocess.run([str(self.support.probe), "build", str(module)],
                                    env=env, capture_output=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn((str(source) + ":1:").encode(), result.stderr)
            self.assertEqual(self.support.probe_path("directory", module, env), generation)
            self.assertEqual(self.support.snapshot(generation), original)
            self.assertEqual(self.answer(library), 42)
            source.write_text("long long nano_build_answer(void) { return 44; }\n")
            self.support.probe_path("build", module, env)
            recovered = self.support.probe_path("directory", module, env)
            self.assertNotEqual(recovered, generation)
            self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 44)
            self.support.probe_path("build", module, env)
            self.assertEqual(self.support.probe_path("directory", module, env), recovered)

    def test_restored_edit_during_capture_cannot_authorize_reuse(self):
        with tempfile.TemporaryDirectory(prefix="nano-retained-capture-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.support.foreign_build_fixture(directory)
            source, wrapper, marker = module / "answer.c", directory / "cc", directory / "captured"
            wrapper.write_text(f'''#!{sys.executable}
import os, pathlib, subprocess, sys
marker = pathlib.Path({str(marker)!r})
if "-E" in sys.argv and not marker.exists():
    source = pathlib.Path({str(source)!r})
    data, stamp = source.read_bytes(), source.stat()
    try:
        source.write_bytes(data.replace(b"42", b"43"))
        result = subprocess.run([{shutil.which('cc')!r}] + sys.argv[1:])
    finally:
        source.write_bytes(data)
        os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    marker.touch()
    sys.exit(result.returncode)
os.execv({shutil.which('cc')!r}, [{shutil.which('cc')!r}] + sys.argv[1:])
''')
            wrapper.chmod(0o700)
            env["NANO_CC"] = str(wrapper)
            self.support.probe_path("build", module, env)
            first = self.support.probe_path("directory", module, env)
            self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 43)
            self.assertFalse((first / "source_hashes.json").exists())
            self.support.probe_path("build", module, env)
            recovered = self.support.probe_path("directory", module, env)
            self.assertNotEqual(recovered, first)
            self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)
            self.support.probe_path("build", module, env)
            self.assertEqual(self.support.probe_path("directory", module, env), recovered)


if __name__ == "__main__":
    unittest.main()
