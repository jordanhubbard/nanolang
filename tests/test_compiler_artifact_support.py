"""I bind foreign support to the immutable generation returned by its build."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CompilerArtifactSupport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="nano-artifact-driver-")
        directory = Path(cls.temporary.name)
        source = directory / "driver.nano"
        source.write_text('from "modules/compiler_support/compiler_support.nano" import module_artifact\n'
                          'extern fn get_argv(index: int) -> string\n'
                          'fn main() -> int { unsafe { let path: string = (module_artifact (get_argv 1)) '
                          'let cleared: string = (module_artifact "") assert (== cleared "") '
                          '(println path) } return 0 }\n'
                          'shadow main { assert true }\n')
        cls.driver = directory / "driver"
        result = subprocess.run([ROOT / "bin/nanoc_c", source, "-o", cls.driver], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def invoke(self, source, cache):
        env = os.environ.copy()
        env["NANO_BUILD_CACHE"] = str(cache)
        # A moved compiler receives its installed capture helper explicitly.
        env["NANO_AS_CAPTURE_HELPER"] = str(ROOT / "bin/nano_as_capture.so")
        result = subprocess.run([self.driver, source], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout.strip(), result.stderr

    def fixture(self, directory):
        module = directory / "foreign space ' owner"
        module.mkdir()
        source = module / "api.nano"
        source.write_text("extern fn answer() -> int\n")
        (module / "module.json").write_text(json.dumps({"name": "answer", "c_sources": ["answer.c"]}))
        c_source = module / "answer.c"
        c_source.write_text("long long answer(void) { return 37; }\n")
        return source, c_source

    def answer(self, path):
        library = ctypes.CDLL(str(path))
        library.answer.argtypes = []
        library.answer.restype = ctypes.c_longlong
        return library.answer()

    def test_artifact_reuse_and_new_generation_preserve_old_bytes(self):
        with tempfile.TemporaryDirectory(prefix="nano-artifact-generation-") as tmp:
            directory = Path(tmp)
            source, c_source = self.fixture(directory)
            first, error = self.invoke(source, directory / "cache")
            self.assertTrue(Path(first).is_absolute(), error)
            self.assertNotIn("/current/", first)
            self.assertTrue(Path(first).is_file())
            self.assertEqual(self.answer(first), 37)
            repeated, _ = self.invoke(source, directory / "cache")
            self.assertEqual(repeated, first)
            original = Path(first).read_bytes()
            c_source.write_text("long long answer(void) { return 42; }\n")
            second, error = self.invoke(source, directory / "cache")
            self.assertNotEqual(second, first, error)
            self.assertEqual(self.answer(second), 42)
            self.assertEqual(Path(first).read_bytes(), original)
            self.assertEqual(self.answer(first), 37)

    def test_failed_build_returns_empty_and_preserves_previous_artifact(self):
        with tempfile.TemporaryDirectory(prefix="nano-artifact-failure-") as tmp:
            directory = Path(tmp)
            source, c_source = self.fixture(directory)
            first, _ = self.invoke(source, directory / "cache")
            original = Path(first).read_bytes()
            c_source.write_text("I am not C.\n")
            failed, error = self.invoke(source, directory / "cache")
            self.assertEqual(failed, "", error)
            self.assertEqual(Path(first).read_bytes(), original)
            self.assertEqual(self.answer(first), 37)

    def test_missing_source_manifest_and_non_file_are_refused(self):
        with tempfile.TemporaryDirectory(prefix="nano-artifact-missing-") as tmp:
            directory = Path(tmp)
            source = directory / "api.nano"
            source.write_text("fn answer() -> int { return 0 }\n")
            for path in ("", str(directory / "missing.nano"), str(directory), str(source)):
                with self.subTest(path=path):
                    value, error = self.invoke(path, directory / "cache")
                    self.assertEqual(value, "", error)


if __name__ == "__main__":
    unittest.main()
