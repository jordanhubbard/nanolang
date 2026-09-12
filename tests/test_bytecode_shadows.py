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


if __name__ == "__main__":
    unittest.main()
