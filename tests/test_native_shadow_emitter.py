"""I compile and execute my self-hosted shadow emitter's C output."""
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeShadowEmitter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix="nano-native-shadows-")
        cls.directory = Path(cls.workspace.name)
        cls.addClassCleanup(cls.workspace.cleanup)
        cls.emitters = []
        for compiler in ("nanoc_c", "nanoc_stage2"):
            emitter = cls.directory / compiler
            result = cls.run_command([ROOT / "bin" / compiler,
                                      ROOT / "tests/selfhost_shadow_emitter.nano", "-o", emitter], 300)
            if result.returncode:
                raise AssertionError(f"{compiler} exited {result.returncode}: {result.stdout}{result.stderr}")
            cls.emitters.append(emitter)

    @classmethod
    def run_command(cls, args, timeout=30):
        command = list(map(str, args))
        process = subprocess.Popen(command, cwd=ROOT,
                                   env=dict(os.environ, TMPDIR=str(cls.directory)),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

    def check(self, source, first=0, expected=0, stdout="", ndebug=False):
        for emitter in self.emitters:
            with self.subTest(compiler=emitter.name):
                path = self.directory / "fixture.nano"
                output = self.directory / "fixture.c"
                binary = self.directory / "fixture"
                path.write_text(source)
                result = self.run_command([emitter, path, output, first])
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                code = output.read_text()
                self.assertEqual("__nano_shadow_" in code, first >= 0 and "shadow " in source)
                result = self.run_command(["cc", "-std=gnu11", "-I", ROOT / "src", "-I", ROOT / "modules/std",
                                           *(["-DNDEBUG"] if ndebug else []), output, "-lm", "-o", binary])
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                result = self.run_command([binary], 5)
                if expected is None:
                    self.assertNotEqual(result.returncode, 0)
                else:
                    self.assertEqual(result.returncode, expected, result.stderr)
                self.assertEqual(result.stdout, stdout)

    def test_failing_assertion_with_ndebug(self):
        self.check("fn f() -> int { return 7 } shadow f { assert false }", expected=None, ndebug=True)

    def test_main_remains_callable(self):
        self.check('fn main() -> int { (println "product") return 7 } shadow main { assert (== (main) 7) }', stdout="product\n")

    def test_product_does_not_run_shadows(self):
        self.check('fn main() -> int { (println "product") return 7 } shadow main { assert false }', first=-1, expected=7, stdout="product\n")

    def test_main_is_not_implicitly_a_test(self):
        self.check("fn main() -> int { assert false return 7 } shadow main { assert true }")

    def test_local_scopes_and_order(self):
        self.check('let answer: int = 7\nfn f() -> int { return answer }\n'
                   'shadow f { let x: int = 7 assert (== (f) x) (println "first") }\n'
                   'shadow f { let x: string = "seven" assert (== (str_length x) 5) (println "second") }',
                   stdout="first\nsecond\n")

    def test_explicit_selection(self):
        self.check("fn f() -> int { return 7 } shadow f { assert false } shadow f { assert (== (f) 7) }", first=1)

    def test_helper_assertions_are_enabled(self):
        self.check("fn f() -> int { assert false return 7 } shadow f { assert (== (f) 7) }", expected=None, ndebug=True)

    def test_invalid_selection_does_not_emit(self):
        for emitter in self.emitters:
            with self.subTest(compiler=emitter.name):
                path = self.directory / "invalid.nano"
                output = self.directory / "preserved.c"
                path.write_text("fn f() -> int { return 1 } shadow f { assert true }")
                output.write_text("preserve me")
                result = self.run_command([emitter, path, output, 2])
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output.read_text(), "preserve me")

    def test_record_string_survives_callee_locals(self):
        for compiler in ("nanoc_c", "nanoc_stage2"):
            with self.subTest(compiler=compiler):
                binary = self.directory / "string-lifetime"
                result = self.run_command([ROOT / "bin" / compiler,
                                          ROOT / "tests/nl_shadow_struct_string_lifetime.nano", "-o", binary], 60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                result = self.run_command([binary], 5)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
