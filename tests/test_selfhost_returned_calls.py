"""I test computed callees with explicit Stage2 execution and rejection checks."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get("NANOLANG_SELFHOST_COMPILER", ROOT / "bin/nanoc_stage2")).resolve()


class ReturnedCalls(unittest.TestCase):
    def compile(self, source, directory):
        artifact = directory / "program"
        result = subprocess.run([str(COMPILER), str(source), "-o", str(artifact)],
                                cwd=ROOT, capture_output=True, timeout=120)
        return result, artifact

    def test_branch_fixtures(self):
        for name, valid in (("calls", True), ("arg_type_error", False), ("arity_error", False)):
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix="nano-returned-") as d:
                source = ROOT / "tests/selfhost" / f"test_returned_function_{name}.nano"
                result, artifact = self.compile(source, Path(d))
                if valid:
                    self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
                    run = subprocess.run([str(artifact)], capture_output=True, timeout=10)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    self.assertEqual(run.stdout, b"callee\nargument\n")
                else:
                    self.assertGreater(result.returncode, 0, result.stderr)
                    self.assertIn(b"function expression", result.stdout + result.stderr)
                    self.assertFalse(artifact.exists())

    def test_nested_signature_and_printed_result(self):
        source_text = '''
fn identity(n: int) -> int { return n }
shadow identity { assert (== (identity 7) 7) }
fn apply(f: fn(int) -> int, n: int) -> int { return (f n) }
shadow apply { assert (== (apply identity 7) 7) }
fn choose() -> fn(fn(int) -> int, int) -> int { return apply }
shadow choose { assert (== ((choose) identity 7) 7) }
fn main() -> int {
    (println ((choose) identity 42))
    return 0
}
shadow main { assert (== ((choose) identity 9) 9) }
'''
        with tempfile.TemporaryDirectory(prefix="nano-returned-") as d:
            directory = Path(d)
            source = directory / "nested.nano"
            source.write_text(source_text)
            result, artifact = self.compile(source, directory)
            self.assertEqual(result.returncode, 0, (result.stdout + result.stderr).decode(errors="replace"))
            run = subprocess.run([str(artifact)], capture_output=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(run.stdout, b"42\n")

    def test_multiple_arguments_execute_once_in_order(self):
        source_text = '''
fn pack(a: int, b: int) -> int { return (+ (* a 10) b) }
shadow pack { assert (== (pack 1 2) 12) }
fn select() -> fn(int, int) -> int { (println "callee") return pack }
shadow select { assert (== ((select) 1 2) 12) }
fn first() -> int { (println "first") return 1 }
shadow first { assert (== (first) 1) }
fn second() -> int { (println "second") return 2 }
shadow second { assert (== (second) 2) }
fn main() -> int {
    assert (== ((select) (first) (second)) 12)
    return 0
}
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix="nano-returned-") as d:
            directory = Path(d)
            source = directory / "order.nano"
            source.write_text(source_text)
            result, artifact = self.compile(source, directory)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run = subprocess.run([str(artifact)], capture_output=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(run.stdout, b"callee\nfirst\nsecond\n")


if __name__ == "__main__":
    unittest.main()
