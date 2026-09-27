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

CYCLIC_LAYOUTS = (
    'struct Left { right: Right } struct Right { left: Left }',
    'struct Loop { choice: Choice } union Choice { Some { item: Loop } }',
    'union Box<T> { Some { value: T } } struct Loop { value: Box<Loop> }',
    'union Grow<T> { More { value: Grow<array<T>> } }',
    'union Wrap<T> { Some { value: Box<T> } } union Box<T> { Some { value: T } } struct Loop { value: Wrap<Loop> }',
    'union Box<T> { Some { value: T } } struct Loop { value: Box<(int,Loop)> }',
    'struct List { next: List }',
    'union List<T> { More { next: List<T> } }',
)
FINITE_LAYOUTS = (
    'union Marker<T> { Mark { value: int } } struct Node { value: Marker<Node> }',
    'union Box<T> { Some { value: T } } struct Node { value: Box<Box<int>> }',
    'union Box<T> { Some { value: T } } struct Node { children: array<Box<Node>>, siblings: List<Node> }',
    'union Box<T> { Some { value: T } } struct T { value: int }',
    'union Box<T> { Some { value: T } } union Marker<T> { Mark { value: Box<int> } } struct Node { value: Marker<Node> }',
    'struct List { value: int } struct Holder { value: List }',
)

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

    def test_primitive_list_union_fields_keep_runtime_typedefs(self):
        for element, value in [('int', '7'), ('string', '"kept"')]:
            with self.subTest(element=element):
                self.check(f'''union Choice {{ Some {{ values: List<{element}> }} }}
fn size(value: Choice) -> int {{ match value {{ Some(payload) => {{ return (list_{element}_length payload.values) }} }} }}
shadow size {{ let values: List<{element}> = (list_{element}_new) (list_{element}_push values {value}) let value: Choice = Choice.Some {{ values: values }} assert (== (size value) 1) }}
fn main() -> int {{ let values: List<{element}> = (list_{element}_new) (list_{element}_push values {value}) let value: Choice = Choice.Some {{ values: values }} return (- (size value) 1) }}
shadow main {{ assert (== (main) 0) }}
''')

    def test_primitive_list_record_fields_keep_runtime_typedefs(self):
        for element, first, second in [('int', '7', '9'), ('string', '"first"', '"second"')]:
            with self.subTest(element=element):
                self.check(f'''struct Holder {{ values: List<{element}>, tail: int }}
union Choice {{ Some {{ item: Holder }} }}
fn read(holder: Holder) -> {element} {{ return (list_{element}_get holder.values 1) }}
shadow read {{ let values: List<{element}> = (list_{element}_new) (list_{element}_push values {first}) (list_{element}_push values {second}) assert (== (read Holder {{ values: values, tail: 5 }}) {second}) }}
fn choice_size(value: Choice) -> int {{ match value {{ Some(payload) => {{ return (+ (list_{element}_length payload.item.values) payload.item.tail) }} }} }}
shadow choice_size {{ let values: List<{element}> = (list_{element}_new) let value: Choice = Choice.Some {{ item: Holder {{ values: values, tail: 5 }} }} assert (== (choice_size value) 5) }}
fn main() -> int {{
 let values: List<{element}> = (list_{element}_new)
 let holder: Holder = Holder {{ values: values, tail: 5 }}
 (list_{element}_push holder.values {first})
 (list_{element}_push holder.values {second})
 assert (== (read holder) {second})
 assert (== (list_{element}_length values) 2)
 let choice: Choice = Choice.Some {{ item: holder }}
 assert (== (choice_size choice) 7)
 return 0
}}
shadow main {{ assert (== (main) 0) }}
''')

    def test_record_cycle_preserves_previous_output(self):
        self.check('struct Left { right: Right }\nstruct Right { left: Left }\n'+MAIN, False)

    def test_mixed_cycle_preserves_previous_output(self):
        self.check('struct Loop { choice: Choice }\nunion Choice { Some { item: Loop } }\n'+MAIN, False)

    def test_generic_by_value_cycles_preserve_previous_output(self):
        for source in CYCLIC_LAYOUTS[2:]:
            with self.subTest(source=source):
                self.check(source + '\n' + MAIN, False)

    def test_finite_generic_and_pointer_layouts(self):
        for source in FINITE_LAYOUTS:
            with self.subTest(source=source):
                self.check(source + '\n' + MAIN)

    def test_long_finite_declaration_chain(self):
        source = ''.join(f'struct Node{i} {{ next: Node{i+1} }}\n' for i in range(150))
        self.check(source + 'struct Node150 { value: int }\n' + MAIN)

if __name__ == '__main__':
    unittest.main()
