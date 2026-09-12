"""I execute shadows before publishing bytecode, not from production main."""
from pathlib import Path
import os
import signal
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BytecodeShadows(unittest.TestCase):
    def compile(self, source, directory, *options):
        path = directory / "program.nano"
        path.write_text(source)
        output = directory / "program.nvm"
        args = [str(ROOT / "bin/nano_virt"), str(path), "--emit-nvm", "-o", str(output), *options]
        process = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=25)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        result = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
        return result, output

    def execute(self, output):
        return subprocess.run([str(ROOT / "bin/nano_vm"), str(output)], cwd=ROOT,
                              capture_output=True, timeout=10)

    def test_failed_shadow_preserves_output(self):
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            directory = Path(tmp)
            output = directory / "program.nvm"
            output.write_bytes(b"preserve existing artifact")
            result, output = self.compile("fn f() -> int { return 42 }\nshadow f { assert false }\n"
                                          "fn main() -> int { return 0 }\nshadow main { assert true }\n", directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"shadow", result.stderr.lower())
            self.assertEqual(output.read_bytes(), b"preserve existing artifact")

    def test_main_shadow_does_not_recurse_or_leak_into_product(self):
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            result, output = self.compile('fn main() -> int { (println "product") return 7 }\n'
                                          'shadow main { assert (== (main) 7) }\n', Path(tmp), "--run")
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertEqual(result.stdout, b"product\n")
            self.assertNotIn(b"$shadow_", output.read_bytes())
            execution = self.execute(output)
            self.assertEqual(execution.returncode, 7, execution.stderr)
            self.assertEqual(execution.stdout, b"product\n")

    def test_production_main_is_not_automatically_a_test(self):
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            result, output = self.compile("fn f() -> int { return 42 }\nshadow f { assert (== (f) 42) }\n"
                                          "fn main() -> int { assert false return 0 }\nshadow main { assert true }\n", Path(tmp))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotEqual(self.execute(output).returncode, 0)

    def test_shadow_only_source(self):
        for assertion, success in (("(== (f) 42)", True), ("false", False)):
            with self.subTest(assertion=assertion), tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
                result, output = self.compile(f"fn f() -> int {{ return 42 }}\nshadow f {{ assert {assertion} }}\n", Path(tmp))
                self.assertEqual(result.returncode == 0, success, result.stderr)
                self.assertEqual(output.exists(), success)

    def test_shadow_locals_and_globals_are_not_product_state(self):
        source = '''let mut count: int = 0
fn f() -> int { return count }
shadow f { let x: int = 7 set count x assert (== (f) 7) }
fn g() -> int { return 5 }
shadow g { let x: string = "hello" assert (== (str_length x) (g)) }
fn main() -> int { return count }
shadow main { assert true }
'''
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            result, output = self.compile(source, Path(tmp))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.execute(output).returncode, 0)

    def test_runtime_trap_blocks_publication(self):
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            result, output = self.compile("fn f() -> int { return 0 }\n"
                                          "shadow f { let a: array<int> = [1] assert (== (at a 2) 0) }\n", Path(tmp))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"shadow", result.stderr.lower())
            self.assertFalse(output.exists())

    def test_nonterminating_shadow_is_bounded(self):
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            result, output = self.compile("fn f() -> int { return 0 }\nshadow f { while true {} }\n", Path(tmp))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"after 10 seconds", result.stderr)
            self.assertFalse(output.exists())

    def test_root_shadow_calls_imported_helper(self):
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            directory = Path(tmp)
            module = directory / "helper.nano"
            module.write_text("pub fn twice(x: int) -> int { return (* x 2) }\nshadow twice { assert true }\n")
            for expected in (6, 7):
                with self.subTest(expected=expected):
                    source = f'module "{module}" as helper\nfn main() -> int {{ return 0 }}\nshadow main {{ assert (== (helper.twice 3) {expected}) }}\n'
                    result, output = self.compile(source, directory)
                    self.assertEqual(result.returncode == 0, expected == 6, result.stderr)

    def test_foreign_code_cannot_cancel_parent_deadline(self):
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            source = "extern fn alarm(seconds: int) -> int\nfn f() -> int { return 0 }\nshadow f { unsafe { (alarm 0) } while true {} }\n"
            result, output = self.compile(source, Path(tmp))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"after 10 seconds", result.stderr)
            self.assertFalse(output.exists())

    def test_entry_exit_values(self):
        for value, expected in ((0, 0), (7, 7), (-1, 255), (256, 0)):
            with self.subTest(value=value), tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
                result, output = self.compile(f"fn main() -> int {{ return {value} }}\nshadow main {{ assert true }}\n", Path(tmp), "--run")
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertEqual(self.execute(output).returncode, expected)

    def test_shadow_array_inference(self):
        source = '''fn first(words: array<string>) -> string { return (at words 0) }
shadow first {
    let words = ["nano", "lang"]
    assert (== (first words) "nano")
    let empty: array<string> = []
    assert (== (array_length empty) 0)
    let values = [-1, 2, 3]
    assert (== (+ (at values 0) (at values 1)) 1)
}
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            result, output = self.compile(source, Path(tmp))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.execute(output).returncode, 0)

    def test_invalid_shadow_types_preserve_output(self):
        for body in ('assert 7', 'let x: int = "wrong" assert true',
                     'unsafe { assert 7 }', 'return 7', 'break',
                     'let a: array<int> = ["wrong"] assert true'):
            with self.subTest(body=body), tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
                directory = Path(tmp)
                (directory / "program.nvm").write_bytes(b"preserve")
                result, output = self.compile('fn main() -> int { return 0 }\n'
                                              f'shadow main {{ {body} }}\n', directory)
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertIn(b"type check failed", result.stderr)
                self.assertEqual(output.read_bytes(), b"preserve")

    def test_absolute_value_preserves_numeric_kind(self):
        source = '''fn magnitude(value: float) -> float { return (abs value) }
shadow magnitude {
    assert (== (magnitude -3.5) 3.5)
    assert (== (magnitude 3.5) 3.5)
    assert (== (magnitude 0.0) 0.0)
}
fn integer(value: int) -> int { return (abs value) }
shadow integer {
    assert (== (integer -7) 7)
    assert (== (integer 7) 7)
    assert (== (integer 0) 0)
    let minimum: int = (- (- 0 9223372036854775807) 1)
    assert (== (integer minimum) minimum)
}
fn main() -> int {
    assert (== (abs -3.14) 3.14)
    assert (== (magnitude -9.5) 9.5)
    assert (== (integer -9) 9)
    return 0
}
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            result, output = self.compile(source, Path(tmp))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.execute(output).returncode, 0)

    def test_array_literal_annotations_in_functions(self):
        bodies = ('let a: array<int> = ["wrong"]',
                  'let mut a: array<string> = [] set a [1]',
                  'let a: array<float> = [true]',
                  'let a: array<string> = [1, 2]')
        for compiler in ("nanoc_c", "nano_virt"):
            for body in bodies:
                with self.subTest(compiler=compiler, body=body), tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
                    directory = Path(tmp)
                    source = directory / "input.nano"
                    source.write_text(f"fn main() -> int {{ {body} return 0 }}\nshadow main {{ assert true }}\n")
                    output = directory / "output"
                    output.write_bytes(b"preserve")
                    args = [str(ROOT / "bin" / compiler), str(source), "-o", str(output)]
                    if compiler == "nano_virt":
                        args.append("--emit-nvm")
                    result = subprocess.run(args, cwd=ROOT, capture_output=True, timeout=25)
                    self.assertNotEqual(result.returncode, 0, result.stderr)
                    self.assertIn(b"array elements", result.stderr)
                    self.assertEqual(output.read_bytes(), b"preserve")

    def test_matching_and_empty_array_literals(self):
        source = '''fn main() -> int {
    let mut words: array<string> = []
    set words ["nano"]
    assert (== (at words 0) "nano")
    let values: array<float> = [1.5, 2.5]
    assert (== (at values 1) 2.5)
    return 0
}
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            directory = Path(tmp)
            result, output = self.compile(source, directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.execute(output).returncode, 0)
            native = directory / "program.native"
            result = subprocess.run([str(ROOT / "bin/nanoc_c"), str(directory / "program.nano"),
                                     "-o", str(native)], cwd=ROOT, capture_output=True, timeout=25)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(subprocess.run([str(native)], capture_output=True, timeout=10).returncode, 0)

    def test_min_max_values_and_evaluation_order(self):
        checks = []
        for left, right in ((2, 7), (7, 2), (2, 2), (-7, -2), (-2, -7),
                            (2.5, 7.8), (7.8, 2.5), (2.5, 2.5), (-7.8, -2.5), (-2.5, -7.8)):
            for operation in (min, max):
                checks.append(f"assert (== ({operation.__name__} {left} {right}) {operation(left, right)})")
        source = '''let mut calls: int = 0
fn next(value: int) -> int { set calls (+ (* calls 10) value) return value }
shadow next { set calls 0 assert (== (next 2) 2) assert (== calls 2) set calls 0 }
fn main() -> int {
''' + '\n'.join(checks) + '''
    set calls 0
    assert (== (min (next 2) (next 1)) 1)
    assert (== calls 21)
    set calls 0
    assert (== (max (next 1) (next 2)) 2)
    assert (== calls 12)
    return 0
}
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            result, output = self.compile(source, Path(tmp))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.execute(output).returncode, 0)

    def test_shadow_local_type_overrides_target_parameter_metadata(self):
        source = '''fn f(value: int) -> int { return value }
shadow f {
    let value: float = -3.5
    assert (== (abs value) 3.5)
}
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            result, output = self.compile(source, Path(tmp))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.execute(output).returncode, 0)

    def test_nested_function_factory_signatures(self):
        source = '''fn add(a: int, b: int) -> int { return (+ a b) }
shadow add { assert (== (add 2 3) 5) }
fn factory() -> fn(int, int) -> int { return add }
shadow factory { let op: fn(int, int) -> int = (factory) assert (== (op 2 3) 5) }
fn apply(build: fn() -> fn(int, int) -> int) -> int {
    let op: fn(int, int) -> int = (build)
    return (op 2 3)
}
shadow apply { assert (== (apply factory) 5) }
fn main() -> int { return (apply factory) }
shadow main { assert (== (main) 5) }
'''
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            for attempt in range(5):
                with self.subTest(attempt=attempt):
                    result, output = self.compile(source, Path(tmp))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(self.execute(output).returncode, 5)

    def test_bad_function_signatures_reject_without_crashing(self):
        mismatch = '''fn add(a: int, b: int) -> int { return (+ a b) }
fn factory() -> fn(int, int) -> int { return add }
fn accept(build: fn() -> fn(int, int) -> float) -> int { return 0 }
fn main() -> int { return (accept factory) }
shadow main { assert true }
'''
        for source in (mismatch, "fn f(value: fn(int) ->", "fn f(value: fn(int"):
            with self.subTest(source=source), tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
                directory = Path(tmp)
                (directory / "program.nvm").write_bytes(b"preserve")
                result, output = self.compile(source, directory)
                self.assertGreater(result.returncode, 0, result.stderr)
                self.assertEqual(output.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
