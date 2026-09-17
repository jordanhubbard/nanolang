"""I evaluate filled-array operands once before rejecting negative counts."""
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
NANOC = Path(os.environ.get("NANOLANG_COMPILER", ROOT / "bin/nanoc_c")).resolve()
NANO = Path(os.environ.get("NANOLANG_INTERPRETER", ROOT / "bin/nano")).resolve()

PROGRAM = '''
fn count(trace: array<int>, size: int) -> int {
    (array_set trace 0 (+ (* (at trace 0) 10) 1))
    return size
}
shadow count { let trace: array<int> = [0] assert (== (count trace 2) 2) }
fn fill(trace: array<int>) -> int {
    (array_set trace 0 (+ (* (at trace 0) 10) 2))
    return 7
}
shadow fill { let trace: array<int> = [0] assert (== (fill trace) 7) }
fn main() -> int {
    let trace: array<int> = [0]
    let empty: array<int> = (array_new (count trace 0) (fill trace))
    assert (== (array_length empty) 0)
    assert (== (at trace 0) 12)
    (array_set trace 0 0)
    let xs: array<int> = (array_new (count trace 3) (fill trace))
    assert (== (at trace 0) 12)
    assert (== (array_length xs) 3)
    assert (== (at xs 2) 7)
    let _size: int = 2
    let _arr: int = 3
    let _i: int = 8
    let __nl_arg_0_0: int = 9
    let nested: array<int> = (array_new _size (array_length (array_new _arr _i)))
    assert (== (at nested 1) 3)
    assert (== __nl_arg_0_0 9)
    return 0
}
shadow main { assert true }
'''


class ArrayNewEvaluation(unittest.TestCase):
    def run_ok(self, argv):
        result = subprocess.run([str(x) for x in argv], cwd=ROOT, capture_output=True,
                                text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_native_and_interpreter_evaluate_count_then_fill_once(self):
        with tempfile.TemporaryDirectory(prefix="nano-filled-order-") as tmp:
            source, executable = Path(tmp) / "main.nano", Path(tmp) / "main"
            source.write_text(PROGRAM)
            self.run_ok([NANO, source])
            self.run_ok([NANOC, source, "-o", executable])
            self.run_ok([executable])

    def test_native_preserves_supported_fill_types(self):
        program = '''struct Item { number: int, text: string }
fn item() -> Item { return Item { number: 9, text: "kept" } }
shadow item { let value: Item = (item) assert (== value.number 9) }
fn main() -> int {
 let strings: array<string> = (array_new 2 (+ "ke" "pt"))
 let booleans: array<bool> = (array_new 2 true)
 let floats: array<float> = (array_new 2 1.25)
 let records: array<Item> = (array_new 2 (item))
 let nested: array<array<int>> = (array_new 2 [5])
 assert (== (at strings 1) "kept")
 assert (at booleans 1)
 assert (== (at floats 1) 1.25)
 let record: Item = (at records 1)
 assert (== record.number 9)
 assert (== record.text "kept")
 let inner: array<int> = (at nested 1)
 assert (== (at inner 0) 5)
 return 0
}
shadow main { assert true }
'''
        with tempfile.TemporaryDirectory(prefix="nano-filled-types-") as tmp:
            source, executable = Path(tmp) / "main.nano", Path(tmp) / "main"
            source.write_text(program)
            self.run_ok([NANOC, source, "-o", executable])
            self.run_ok([executable])

    def test_native_rejects_negative_after_both_operands(self):
        with tempfile.TemporaryDirectory(prefix="nano-filled-negative-") as tmp:
            source, executable = Path(tmp) / "main.nano", Path(tmp) / "main"
            marker = Path(tmp) / "fill.txt"
            program = PROGRAM.replace('    return 7\n',
                '    assert (== (at trace 0) 12)\n'
                '    unsafe { assert (== (file_write "' + str(marker) + '" "fill") 0) }\n    return 7\n')
            # The negative path is exercised by main, not during compilation shadows.
            program = program.replace('shadow fill { let trace: array<int> = [0] assert (== (fill trace) 7) }',
                                      'shadow fill { assert true }')
            program = program[:program.index('fn main()')] + '''fn main() -> int {
 let trace: array<int> = [0]
 let xs: array<int> = (array_new (count trace -1) (fill trace))
 return 0
}
shadow main { assert true }
'''
            source.write_text("extern fn file_write(path: string, text: string) -> int\n" + program)
            self.run_ok([NANOC, source, "-o", executable])
            result = subprocess.run([executable], cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, -signal.SIGABRT, result.stdout + result.stderr)
            self.assertIn("I require a non-negative array count", result.stderr)
            self.assertEqual(marker.read_text(), "fill")


if __name__ == "__main__":
    unittest.main()
