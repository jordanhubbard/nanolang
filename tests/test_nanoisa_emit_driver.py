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

UNSUPPORTED_RESULT = '''union Choice { Item { x: int } }
fn main() -> array<Choice> { return [Choice.Item { x: 1 }] }
shadow main { assert true }
'''


class NanoisaEmitDriver(unittest.TestCase):
    def run_command(self, args, expected=0):
        result = subprocess.run([str(x) for x in args], cwd=ROOT,
                                capture_output=True, timeout=90)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_scalar_boolean_casts_agree_in_vm_and_native(self):
        from tests.native_toolchain import native_cc, native_link_flags

        source = ROOT / "tests/nanoisa/fixtures/selfhost_native_u8.nano"
        with tempfile.TemporaryDirectory(prefix="nano-bool-casts-") as tmp:
            directory = Path(tmp)
            module = directory / "program.nvm"
            generated = directory / "program.c"
            binary = directory / "program"
            self.run_command([DRIVER, source, "--emit-nvm", "-o", module])
            self.run_command([ROOT / "bin/nano_vm", module])
            self.run_command([ROOT / "bin/nvm2c", module, "-o", generated])
            self.run_command([*native_cc(), "-std=c11", "-Wall", "-Wextra",
                              "-Werror", generated, "-lm", *native_link_flags(),
                              "-o", binary])
            self.run_command([binary])

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

    def test_selected_patterns_retain_concrete_payload_identity(self):
        from tests.test_generic_selected_patterns import GenericSelectedPatterns
        fixtures = GenericSelectedPatterns()
        cases = [
            (fixtures.ordinary('let Box.Some { value } = payload'), True),
            (fixtures.ordinary('let Box.Some { value } = payload', 'string',
                               '"kept"', 'assert (== value "kept") return 7'), True),
            (fixtures.ordinary('let Box.Some {} = payload', body='return 0'), False),
            (fixtures.ordinary('let Box.Some { value, value } = payload'), False),
            (fixtures.ordinary('let Box.Other { value } = payload'), False),
            (fixtures.ordinary('let Box.Some { missing } = payload', body='return 0'), False),
        ]
        pair = '''union Pair<T> { Both { left: T, right: T } }
fn read(pair: Pair<int>) -> int { match pair {
 Both(payload) => { PATTERN return (+ left right) }
} }
fn main() -> int { let pair: Pair<int> = Pair.Both { left: 3, right: 4 }
 return (- (read pair) 7) }
'''
        cases.extend([
            (pair.replace("PATTERN", "let Pair.Both { right, left } = payload"), True),
            (pair.replace("PATTERN", "let Pair.Both { left, left } = payload"), False),
            (pair.replace("PATTERN", "let Pair.Both { left, right } = pair"), False),
        ])
        for index, (source, accepted) in enumerate(cases):
            with self.subTest(case=index), tempfile.TemporaryDirectory(prefix="nano-selected-") as tmp:
                root = Path(tmp)
                program, module = root / "main.nano", root / "main.nvm"
                program.write_text(source)
                module.write_bytes(b"prior artifact")
                self.run_command([DRIVER, program, "--emit-nvm", "-o", module],
                                 expected=0 if accepted else 1)
                if not accepted:
                    self.assertEqual(module.read_bytes(), b"prior artifact")
                    continue
                self.run_command([ROOT / "bin/nano_vm", module])
                generated, binary = root / "main.c", root / "native"
                self.run_command([ROOT / "bin/nvm2c", module, "-o", generated])
                self.run_command(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                                  generated, "-lm", "-o", binary])
                self.run_command([binary])

    def test_nested_union_payloads_and_phantom_resource_arguments(self):
        nested = (ROOT / "tests/unit/test_native_nested_generics.nano").read_text()
        phantom = '''resource struct Handle { fd: int }
union Marker<T> { Mark { number: int } }
union Box<T> { Some { value: T }, None {} }
fn read(value: Box<Marker<Handle>>) -> int { match value {
 Some(payload) => { let Box.Some { value } = payload match value {
  Mark(inner) => { return inner.number }
 } }
 None(payload) => { return 0 }
} }
fn main() -> int {
 let marker: Marker<Handle> = Marker.Mark { number: 7 }
 let boxed: Box<Marker<Handle>> = Box.Some { value: marker }
 let copy: Box<Marker<Handle>> = boxed
 return (- (+ (read boxed) (read copy)) 14)
}
'''
        for source in (nested, phantom):
            with self.subTest(source=source), tempfile.TemporaryDirectory(prefix="nano-nested-union-") as tmp:
                root = Path(tmp)
                program, module = root / "main.nano", root / "main.nvm"
                program.write_text(source)
                self.run_command([DRIVER, program, "--emit-nvm", "-o", module])
                self.run_command([ROOT / "bin/nano_vm", module])
                generated, binary = root / "main.c", root / "native"
                self.run_command([ROOT / "bin/nvm2c", module, "-o", generated])
                self.run_command(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                                  generated, "-lm", "-o", binary])
                self.run_command([binary])

    def test_nested_union_cycles_and_unknown_arguments_preserve_output(self):
        cases = [
            'union Cycle { Next { value: Cycle } } fn main() -> int { let value: Cycle = 0 return 0 }',
            'union Grow<T> { Next { value: Grow<Grow<T>> } } fn main() -> int { let value: Grow<int> = 0 return 0 }',
            'union Marker<T> { Mark { value: int } } fn main() -> int { let value: Marker<Unknown> = Marker.Mark { value: 7 } return 0 }',
        ]
        for source in cases:
            with self.subTest(source=source), tempfile.TemporaryDirectory(prefix="nano-union-boundary-") as tmp:
                program, module = Path(tmp) / "main.nano", Path(tmp) / "prior.nvm"
                program.write_text(source)
                module.write_bytes(b"prior artifact")
                self.run_command([DRIVER, program, "--emit-nvm", "-o", module], expected=1)
                self.assertEqual(module.read_bytes(), b"prior artifact")

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
            self.assertIn(b"I refused that program: unsupported result type array<Choice>", result.stdout)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
