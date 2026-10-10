"""I execute my self-hosted NanoISA shadow modules in VM and native products."""
import os
from pathlib import Path
import signal
import sys
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
                output = self.directory / "fixture.nasm"
                module = self.directory / "fixture.nvm"
                c_source = self.directory / "fixture.c"
                binary = self.directory / "fixture"
                path.write_text(source)
                result = self.run_command([emitter, path, output, first])
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                assembly = output.read_text()
                self.assertEqual("CALL __nanoisa_shadow_" in assembly, first >= 0 and "shadow " in source)
                commands = [
                    [ROOT / "bin/nanoisa", "asm", output, "-o", module],
                    [ROOT / "bin/nano_vm", "--verify-only", module],
                    [ROOT / "bin/nvm2c", module, "-o", c_source],
                    ["cc", "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror",
                     *(["-DNDEBUG"] if ndebug else []), c_source,
                     ROOT / "bin/nano_aot_runtime.o", "-lm",
                     *(["-Wl,--export-dynamic", "-ldl"] if sys.platform.startswith("linux") else []),
                     "-o", binary],
                ]
                for command in commands:
                    result = self.run_command(command)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                for backend, command in (("vm", [ROOT / "bin/nano_vm", module]), ("native", [binary])):
                    with self.subTest(backend=backend):
                        result = self.run_command(command, 5)
                        if expected is None:
                            self.assertNotEqual(result.returncode, 0)
                        else:
                            self.assertEqual(result.returncode, expected, result.stderr)
                        self.assertEqual(result.stdout, stdout)

    def test_compiler_drivers_isolate_shadow_output(self):
        source = self.directory / "isolated.nano"
        source.write_text('fn main()->int { (println "product") return 7 } '
                          'shadow main { assert (== (main) 7) }')
        for compiler in ("nanoc_c", "nanoc_stage2"):
            with self.subTest(compiler=compiler):
                binary = self.directory / ("isolated-" + compiler)
                compiled = self.run_command([ROOT / "bin" / compiler, source, "-o", binary], 90)
                self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                self.assertEqual(compiled.stdout, "")
                if compiler == "nanoc_stage2":
                    self.assertIn("product\n", compiled.stderr)
                else:
                    # I suppress passing interpreter shadow output unless verbose.
                    self.assertEqual(compiled.stderr, "")
                executed = self.run_command([binary], 5)
                self.assertEqual(executed.returncode, 7, executed.stderr)
                self.assertEqual(executed.stdout, "product\n")
                previous = binary.read_bytes()
                source.write_text('fn main()->int { return 7 } shadow main { assert false }')
                refused = self.run_command([ROOT / "bin" / compiler, source, "-o", binary], 90)
                self.assertNotEqual(refused.returncode, 0)
                self.assertEqual(binary.read_bytes(), previous)
                source.write_text('fn main()->int { (println "product") return 7 } '
                                  'shadow main { assert (== (main) 7) }')

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

    def test_opaque_null_arguments_compile(self):
        self.check("opaque type SDL_Window\n"
                   "fn is_null(value: SDL_Window) -> bool { return (== value 0) }\n"
                   "shadow is_null { assert (is_null 0) }\n"
                   "fn relay(value: SDL_Window) -> bool { return (is_null value) }\n"
                   "shadow relay { assert (relay 0) }\n"
                   "fn main() -> int { assert (is_null 0) return 0 }\n"
                   "shadow main { assert (== (main) 0) }")

    def test_opaque_arguments_reject_nonzero_and_wrong_declared_handles(self):
        fixtures = [
            'opaque type Handle fn accept(h:Handle)->void{} shadow accept { (accept 1) }',
            'opaque type Handle fn accept(h:Handle)->void{} shadow accept { (accept true) }',
            'opaque type First opaque type Second fn accept(h:First)->void{} '
            'fn relay(h:Second)->void{ (accept h) } shadow relay { (relay 0) }',
        ]
        for emitter in self.emitters:
            for text in fixtures:
                with self.subTest(compiler=emitter.name, source=text):
                    source = self.directory / "refused.nano"
                    output = self.directory / "refused.nasm"
                    source.write_text(text)
                    output.write_text("preserve me")
                    result = self.run_command([emitter, source, output, 0])
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(output.read_text(), "preserve me")

    def test_invalid_selection_does_not_emit(self):
        for emitter in self.emitters:
            with self.subTest(compiler=emitter.name):
                path = self.directory / "invalid.nano"
                output = self.directory / "preserved.nasm"
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
