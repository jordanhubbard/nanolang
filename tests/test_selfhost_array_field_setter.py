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


    def test_empty_replacement_element_types(self):
        for kind, value in [('int', '7'), ('float', '2.5'), ('bool', 'true'), ('string', '"after"')]:
            with self.subTest(kind=kind):
                self.run_program(f'''struct Box {{ rows: array<array<{kind}>> }}
fn main() -> int {{
 let box: Box = Box {{ rows: [[{value}]] }}
 (array_set box.rows 0 [])
 let row: array<{kind}> = (at box.rows 0)
 let filled: array<{kind}> = (array_push row {value})
 assert (== (array_length filled) 1)
 assert (== (at filled 0) {value})
 return 0
}}
shadow main {{ assert (== (main) 0) }}
''')

    def test_scalar_and_record_accesses_evaluate_operands_once_in_order(self):
        for kind, before, after in (
            ('int', '1', '7'), ('bool', 'false', 'true'),
            ('string', '"before"', '"after"'), ('float', '1.25', '2.5'),
            ('Cell', 'Cell { value: 1 }', 'Cell { value: 9 }'),
        ):
            with self.subTest(element=kind):
                check = '(== value.value 9)' if kind == 'Cell' else f'(== value {after})'
                self.run_program(f'''struct Cell {{ value: int }}
let mut order: int = 0
fn receiver(values: array<{kind}>) -> array<{kind}> {{
 set order (+ (* order 10) 1)
 return values
}}
shadow receiver {{
 set order 0
 assert (== (array_length (receiver [{before}])) 1)
 assert (== order 1)
}}
fn offset() -> int {{ set order (+ (* order 10) 2) return 0 }}
shadow offset {{ set order 0 assert (== (offset) 0) assert (== order 2) }}
fn replacement() -> {kind} {{ set order (+ (* order 10) 3) return {after} }}
shadow replacement {{
 set order 0
 let value: {kind} = (replacement)
 assert {check}
 assert (== order 3)
}}
fn main() -> int {{
 let values: array<{kind}> = [{before}]
 set order 0
 (array_set (receiver values) (offset) (replacement))
 assert (== order 123)
 set order 0
 let value: {kind} = (at (receiver values) (offset))
 assert (== order 12)
 assert {check}
 set order 0
 let second: {kind} = (array_get (receiver values) (offset))
 assert (== order 12)
 assert {check.replace("value", "second", 1)}
 return 0
}}
shadow main {{ assert (== (main) 0) }}
''')

    def test_setter_evaluates_arguments_once_in_order(self):
        self.run_program('''struct Box { rows: array<array<string>> }
let mut order: int = 0
fn receiver(box: Box) -> array<array<string>> {
 set order (+ (* order 10) 1)
 return box.rows
}
shadow receiver {
 set order 0
 assert (== (array_length (receiver Box { rows: [["before"]] })) 1)
 assert (== order 1)
}
fn offset() -> int { set order (+ (* order 10) 2) return 0 }
shadow offset { set order 0 assert (== (offset) 0) assert (== order 2) }
fn replacement() -> array<string> { set order (+ (* order 10) 3) return [] }
shadow replacement { set order 0 assert (== (array_length (replacement)) 0) assert (== order 3) }
fn main() -> int {
 set order 0
 let box: Box = Box { rows: [["before"]] }
 (array_set (receiver box) (offset) (replacement))
 assert (== order 123)
 assert (== (array_length (at box.rows 0)) 0)
 return 0
}
shadow main { assert (== (main) 0) }
''')


if __name__ == '__main__':
    unittest.main()
