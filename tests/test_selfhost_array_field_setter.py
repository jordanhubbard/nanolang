"""I retain array receiver types through native mutation lowering."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ArrayFieldSetter(unittest.TestCase):
    def run_program(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'fields.nano'
            source.write_text(text)
            for name in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
                with self.subTest(compiler=name):
                    output = Path(tmp) / name
                    result = subprocess.run([ROOT / 'bin' / name, source, '-o', output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=180)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    result = subprocess.run([output], capture_output=True, text=True, timeout=20)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_scalar_fields(self):
        self.run_program('''struct Box { flags: array<bool>, words: array<string>, numbers: array<float> }
fn main() -> int {
 let box: Box = Box { flags: [false], words: ["before"], numbers: [1.25] }
 (array_set box.flags 0 true)
 (array_set box.words 0 "after")
 (array_set box.numbers 0 2.5)
 assert (at box.flags 0)
 assert (== (at box.words 0) "after")
 assert (== (at box.numbers 0) 2.5)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_nested_record_field(self):
        self.run_program('''struct Inner { flags: array<bool> }
struct Outer { inner: Inner }
fn main() -> int {
 let outer: Outer = Outer { inner: Inner { flags: [false] } }
 (array_set outer.inner.flags 0 true)
 assert (at outer.inner.flags 0)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_returned_array(self):
        self.run_program('''struct Box { words: array<string> }
fn view(box: Box) -> array<string> { return box.words }
shadow view { assert (== (array_length (view Box { words: ["before"] })) 1) }
fn main() -> int {
 let box: Box = Box { words: ["before"] }
 (array_set (view box) 0 "after")
 assert (== (at box.words 0) "after")
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_nested_array_replacement_context(self):
        self.run_program('''struct Box { rows: array<array<string>> }
fn main() -> int {
 let rows: array<array<string>> = [["before"]]
 let box: Box = Box { rows: rows }
 (array_set box.rows 0 [])
 let row: array<string> = (at box.rows 0)
 let filled: array<string> = (array_push row "after")
 assert (== (at filled 0) "after")
 return 0
}
shadow main { assert (== (main) 0) }
''')


if __name__ == '__main__':
    unittest.main()
