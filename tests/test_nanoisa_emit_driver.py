"""I publish deterministic verified modules through my C-seed-hosted driver."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "bin/nanoisa_emit"
SOURCE = '''fn add(a: int, b: int) -> int { return (+ a b) }
shadow add { assert (== (add 2 3) 5) }
fn main() -> int {
 assert (== (add 2 3) 5)
 (println "module-ok")
 return 0
}
shadow main { assert true }
'''

UNSUPPORTED_RESULT = '''struct Point { x: int }
fn main() -> array<array<Point>> { return [[Point { x: 1 }]] }
shadow main { assert true }
'''


class NanoisaEmitDriver(unittest.TestCase):
    def run_command(self, args, expected=0):
        result = subprocess.run([str(x) for x in args], cwd=ROOT,
                                capture_output=True, timeout=90)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_repeatable_module_executes_in_vm_and_native(self):
        with tempfile.TemporaryDirectory(prefix="nano-driver-") as tmp:
            directory = Path(tmp)
            source, first, second = (directory / name for name in ("source.nano", "a.nvm", "b.nvm"))
            source.write_text(SOURCE)
            for output in (first, second):
                self.run_command([DRIVER, source, "--emit-nvm", "-o", output])
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first.read_bytes()[:4], b"NVM\x02")
            self.assertEqual(self.run_command([ROOT / "bin/nano_vm", first]).stdout, b"module-ok\n")
            generated, binary = directory / "module.c", directory / "native"
            self.run_command([ROOT / "bin/nvm2c", first, "-o", generated])
            self.assertNotIn("nano_vm", generated.read_text())
            self.run_command(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", generated, "-lm", "-o", binary])
            self.assertEqual(self.run_command([binary]).stdout, b"module-ok\n")
            self.assertEqual(list(directory.glob("*.tmp.*")), [])

    def test_assembly_mode_stays_equivalent(self):
        with tempfile.TemporaryDirectory(prefix="nano-driver-text-") as tmp:
            directory = Path(tmp)
            source, assembly, module, direct = (directory / name for name in
                                                ("source.nano", "source.nasm", "from-text.nvm", "direct.nvm"))
            source.write_text(SOURCE)
            stdout = self.run_command([DRIVER, source]).stdout
            self.run_command([DRIVER, source, "-o", assembly])
            self.assertEqual(assembly.read_bytes(), stdout)
            self.run_command([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
            self.run_command([DRIVER, source, "--emit-nvm", "-o", direct])
            self.assertEqual(module.read_bytes(), direct.read_bytes())

    def test_output_aliases_preserve_source(self):
        for mode in ([], ["--emit-nvm"]):
            for alias in ("same", "relative", "symlink", "hardlink"):
                with self.subTest(mode=mode, alias=alias), tempfile.TemporaryDirectory(prefix="nano-source-guard-") as tmp:
                    directory = Path(tmp)
                    source = directory / "source.nano"
                    source.write_text(SOURCE)
                    output = source
                    if alias == "relative":
                        output = directory / "child" / ".." / "source.nano"
                        (directory / "child").mkdir()
                    elif alias == "symlink":
                        output = directory / "alias.nano"
                        output.symlink_to(source)
                    elif alias == "hardlink":
                        output = directory / "alias.nano"
                        output.hardlink_to(source)
                    result = self.run_command([DRIVER, source, *mode, "-o", output], expected=1)
                    self.assertIn(b"aliases my source", result.stdout + result.stderr)
                    self.assertEqual(source.read_text(), SOURCE)
                    self.assertEqual(output.read_text(), SOURCE)
                    if alias == "symlink":
                        self.assertTrue(output.is_symlink())
                    self.assertEqual(list(directory.glob("*.tmp.*")), [])

    def test_unresolved_output_identity_preserves_source(self):
        with tempfile.TemporaryDirectory(prefix="nano-source-identity-") as tmp:
            directory = Path(tmp)
            source, output = directory / "source.nano", directory / "loop"
            source.write_text(SOURCE)
            output.symlink_to(output.name)
            for mode in ([], ["--emit-nvm"]):
                result = self.run_command([DRIVER, source, *mode, "-o", output], expected=1)
                self.assertIn(b"unresolved identity", result.stdout + result.stderr)
                self.assertEqual(source.read_text(), SOURCE)
                self.assertTrue(output.is_symlink())

    def test_rejection_preserves_previous_output(self):
        with tempfile.TemporaryDirectory(prefix="nano-driver-errors-") as tmp:
            directory = Path(tmp)
            source, output = directory / "bad.nano", directory / "prior.nvm"
            for body in ("fn broken(", UNSUPPORTED_RESULT):
                source.write_text(body)
                output.write_bytes(b"previous bytes")
                self.run_command([DRIVER, source, "--emit-nvm", "-o", output], expected=1)
                self.assertEqual(output.read_bytes(), b"previous bytes")
            source.write_text(SOURCE)
            self.run_command([DRIVER, source, "--emit-nvm"], expected=2)
            blocked = directory / "blocked"
            blocked.mkdir()
            self.run_command([DRIVER, source, "--emit-nvm", "-o", blocked], expected=1)
            self.assertEqual(output.read_bytes(), b"previous bytes")
            self.assertEqual(list(directory.glob("*.tmp.*")), [])

    def test_supported_float_boolean_and_nested_array_results(self):
        with tempfile.TemporaryDirectory(prefix="nano-supported-results-") as tmp:
            directory=Path(tmp); source=directory/"supported.nano"; output=directory/"supported.nvm"
            source.write_text('fn number() -> float { return 1.5 }\nshadow number { assert true }\n'
                              'fn flags() -> array<bool> { return [true] }\nshadow flags { assert true }\n'
                              'fn grid() -> array<array<float>> { return [[1.5, 2.5], [3.5]] }\n'
                              'shadow grid { assert (== (at (at (grid) 1) 0) 3.5) }\n'
                              'fn main() -> int { assert (== (number) 1.5) assert (at (flags) 0) '
                              'assert (== (at (at (grid) 0) 1) 2.5) return 0 }\n'
                              'shadow main { assert true }\n')
            self.run_command([DRIVER,source,"--emit-nvm","-o",output])
            self.run_command([ROOT/"bin/nano_vm","--verify-only",output])
            self.run_command([ROOT/"bin/nano_vm",output])

    def test_lowering_refusal_reports_the_exact_boundary(self):
        with tempfile.TemporaryDirectory(prefix="nano-driver-diagnostic-") as tmp:
            directory = Path(tmp)
            source, output = directory / "unsupported.nano", directory / "unsupported.nasm"
            source.write_text(UNSUPPORTED_RESULT)
            result = self.run_command([DRIVER, source, "-o", output], expected=1)
            self.assertIn(b"I refused that program: unsupported result type array<array<Point>>", result.stdout)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
