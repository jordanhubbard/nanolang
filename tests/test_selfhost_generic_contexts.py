"""I retain concrete native union payloads and ordered global initialization."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER_ROOT = Path(os.environ.get('NANOLANG_CONTEXT_COMPILER_ROOT', ROOT/'bin'))

class SelfhostGenericContexts(unittest.TestCase):
    def check(self, source):
        for stage in ('nanoc_stage1', 'nanoc_stage2'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory(prefix='nano-union-context-') as directory:
                directory = Path(directory)
                path, output = directory/'input.nano', directory/'program'
                path.write_text(source)
                result = subprocess.run([COMPILER_ROOT/stage, path, '-o', output], cwd=ROOT,
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                result = subprocess.run([output], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_function_value_concrete_union_identity(self):
        # I exercise typechecking and C publication here. Native callback
        # lowering and ownership remain separate acceptance criteria.
        source = """union Box<T> { Some { value: T } }
fn make_box() -> Box<int> { return Box.Some { value: 7 } }
shadow make_box { let box: Box<int> = (make_box) match box { Some(payload) => { assert (== payload.value 7) } } }
fn apply(make: fn() -> Box<int>) -> int {
 let box: Box<int> = (make)
 match box { Some(payload) => { return payload.value } }
}
shadow apply { assert (== (apply make_box) 7) }
fn main() -> int { return (- (apply make_box) 7) }
shadow main { assert (== (main) 0) }
"""
        for stage in ('nanoc_stage1', 'nanoc_stage2'):
            for mismatch in (False, True):
                with self.subTest(stage=stage, mismatch=mismatch), tempfile.TemporaryDirectory(prefix='nano-callback-identity-') as directory:
                    directory = Path(directory)
                    path, output = directory/'input.nano', directory/'output.c'
                    text = source.replace('fn() -> Box<int>', 'fn() -> Box<bool>') if mismatch else source
                    path.write_text(text)
                    output.write_text('previous artifact')
                    result = subprocess.run([COMPILER_ROOT/stage, path, '--target', 'c', '-o', output], cwd=ROOT,
                                            capture_output=True, text=True, timeout=120)
                    diagnostic = result.stdout + result.stderr
                    if mismatch:
                        self.assertNotEqual(result.returncode, 0, diagnostic)
                        self.assertIn("Argument 1 to 'apply'", diagnostic)
                        self.assertIn('fn()->Box<int>', diagnostic)
                        self.assertEqual(output.read_text(), 'previous artifact')
                    else:
                        self.assertEqual(result.returncode, 0, diagnostic)
                        self.assertNotIn('NSType checking failed', diagnostic)
                        self.assertNotEqual(output.read_text(), 'previous artifact')

    def test_constructor_contexts_and_global_array_payload(self):
        self.check((ROOT/'tests/unit/test_selfhost_generic_contexts.nano').read_text())

    def test_nested_specialization_with_declared_inner_context(self):
        self.check('''union Inner<T> { Value { value: T } }
union Outer<T> { Value { value: Inner<T> } }
fn read(value: Inner<int>) -> int { match value { Value(payload) => { return payload.value } } }
shadow read { let value: Inner<int> = Inner.Value { value: 42 } assert (== (read value) 42) }
fn main() -> int {
 let outer: Outer<int> = Outer.Value { value: Inner.Value { value: 42 } }
 match outer { Value(payload) => { assert (== (read payload.value) 42) } }
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_union_globals_share_source_order_with_guarded_globals(self):
        self.check('''union Box<T> { Value { value: T } }
let mut trace: int = 0
fn make(number: int) -> Box<int> { set trace (+ (* trace 10) number) return Box.Value { value: number } }
shadow make { let before: int = trace let value: Box<int> = (make 3) match value { Value(payload) => { assert (== payload.value 3) } } set trace before }
let first: Box<int> = (make 1)
let mut marker: int = 7
fn make_second() -> Box<int> { assert (== marker 7) return (make 2) }
shadow make_second { let before: int = trace let value: Box<int> = (make_second) match value { Value(payload) => { assert (== payload.value 2) } } set trace before }
let second: Box<int> = (make_second)
let copied: Box<int> = first
fn main() -> int {
 assert (== trace 12)
 match first { Value(payload) => { assert (== payload.value 1) } }
 match second { Value(payload) => { assert (== payload.value 2) } }
 match copied { Value(payload) => { assert (== payload.value 1) } }
 return 0
}
shadow main { assert (== (main) 0) }
''')

if __name__ == '__main__': unittest.main()
