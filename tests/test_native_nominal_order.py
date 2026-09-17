"""I order complete native record/union layouts before their consumers."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER_ROOT = Path(os.environ.get('NANOLANG_NOMINAL_COMPILER_ROOT', ROOT/'bin'))
COMPILERS = os.environ.get('NANOLANG_NOMINAL_COMPILERS', 'nanoc_stage1,nanoc_stage2').split(',')
MAIN = 'fn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n'

class NativeNominalOrder(unittest.TestCase):
    def check(self, source, accepted=True):
        for compiler in COMPILERS:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-nominal-order-') as directory:
                source_path = Path(directory)/'main.nano'
                output = Path(directory)/'program'
                source_path.write_text(source)
                output.write_bytes(b'prior artifact')
                result = subprocess.run([str(COMPILER_ROOT/compiler), str(source_path), '-o', str(output)], cwd=ROOT, capture_output=True, text=True, timeout=90)
                if accepted:
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    run = subprocess.run([str(output)], capture_output=True, text=True, timeout=10)
                    self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
                else:
                    self.assertGreater(result.returncode, 0, result.stdout+result.stderr)
                    self.assertIn('cyclic by-value', result.stdout+result.stderr)
                    self.assertEqual(output.read_bytes(), b'prior artifact')

    def test_union_contains_record(self):
        self.check('''struct Plain { value: int }
union Choice { Some { item: Plain }, None {} }
fn read(value: Choice) -> int { match value { Some(payload) => { return payload.item.value } None(empty) => { return 0 } } }
shadow read { let value: Choice = Choice.Some { item: Plain { value: 7 } } assert (== (read value) 7) }
fn main() -> int { let value: Choice = Choice.Some { item: Plain { value: 7 } } return (- (read value) 7) }
shadow main { assert (== (main) 0) }
''')

    def test_record_contains_union(self):
        self.check('union Choice { Some { value: int }, None {} }\nstruct Boxed { item: Choice }\n'+MAIN)

    def test_alternating_chain(self):
        self.check('struct Leaf { value: int }\nunion Inner { Some { leaf: Leaf } }\nstruct Middle { inner: Inner }\nunion Outer { Some { middle: Middle } }\nstruct Top { outer: Outer }\n'+MAIN)

    def test_reverse_record_dependencies(self):
        self.check('struct Outer { inner: Inner }\nstruct Inner { value: int }\n'+MAIN)

    def test_pointer_backed_array_is_not_a_layout_cycle(self):
        self.check('struct Node { children: array<Node>, value: int }\n'+MAIN)

    def test_pointer_backed_list_is_not_a_layout_cycle(self):
        self.check('struct Node { children: List<Node>, value: int }\n'+MAIN)

    def test_record_cycle_preserves_previous_output(self):
        self.check('struct Left { right: Right }\nstruct Right { left: Left }\n'+MAIN, False)

    def test_mixed_cycle_preserves_previous_output(self):
        self.check('struct Loop { choice: Choice }\nunion Choice { Some { item: Loop } }\n'+MAIN, False)

if __name__ == '__main__':
    unittest.main()
