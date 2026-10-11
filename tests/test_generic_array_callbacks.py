"""I infer concrete functional-array callback signatures from full input types."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from tests import test_native_callable_arrays as arrays

PRELUDE = arrays.PRELUDE + '''fn keep(value:T)->bool{return true}
shadow keep {assert (keep 7)}
fn latest(previous:T, value:T)->T{return value}
shadow latest {assert (== (latest 1 2) 2)}
'''


class GenericArrayCallbacks(unittest.TestCase):
    def execute(self, body):
        arrays.harness.NativeGlobalTypes().execute(PRELUDE + body)

    def test_map_multiple_scalar_instances(self):
        self.execute('''fn main()->int {
let numbers:array<int> = (map [7,9] identity)
let flags:array<bool> = (map [true,false] identity)
assert (== (at numbers 1) 9) assert (at flags 0) assert (not (at flags 1)) return 0}''')

    def test_map_callable_identity(self):
        self.execute('''fn main()->int {let original:array<fn(int)->int> = [increment]
let copied:array<fn(int)->int> = (map original identity)
(array_set original 0 identity)
let callback:fn(int)->int=(at copied 0) assert (== (callback 7) 8) return 0}''')

    def test_filter_generic_predicate(self):
        self.execute('''fn main()->int {let callbacks:array<fn(int)->int> = [increment,identity]
let kept:array<fn(int)->int> = (filter callbacks keep)
assert (== (array_length kept) 2)
let callback:fn(int)->int=(at kept 0) assert (== (callback 7) 8) return 0}''')

    def test_reduce_generic_callable_accumulator(self):
        self.execute('''fn main()->int {let callbacks:array<fn(int)->int> = [identity,increment]
let callback:fn(int)->int=(reduce callbacks identity latest)
assert (== (callback 7) 8) return 0}''')

    def test_reduce_independent_type_variables(self):
        self.execute('''fn retain(accumulator:A, value:T)->A{return accumulator}
shadow retain {assert (== (retain 7 true) 7)}
fn main()->int {let callback:fn(int)->int=(reduce [true,false] increment retain)
assert (== (callback 7) 8) return 0}''')

    def test_typed_empty_inputs(self):
        self.execute('''fn main()->int {let callbacks:array<fn(int)->int> = []
let copied:array<fn(int)->int> = (map callbacks identity)
let kept:array<fn(int)->int> = (filter callbacks keep)
let callback:fn(int)->int=(reduce callbacks identity latest)
assert (== (array_length copied) 0) assert (== (array_length kept) 0)
assert (== (callback 7) 7) return 0}''')

    def test_inline_callable_inputs(self):
        self.execute('''fn main()->int {
let copied:array<fn(int)->int> = (map [increment] identity)
let kept:array<fn(int)->int> = (filter [increment] keep)
let callback=(reduce [increment] increment latest)
let first:fn(int)->int=(at copied 0) let second:fn(int)->int=(at kept 0)
assert (== (first 7) 8) assert (== (second 7) 8)
assert (== (callback 7) 8) return 0}''')

    def test_incompatible_contexts_preserve_output(self):
        for expression in ('(filter [7] identity)', '(reduce [true] 7 latest)',
                           'let bad:array<fn(int)->bool> = (filter callbacks keep)',
                           'let bad:array<fn(int)->bool> = (map callbacks identity)'):
            with self.subTest(expression=expression), tempfile.TemporaryDirectory() as tmp:
                source, output = Path(tmp)/'main.nano', Path(tmp)/'main'
                source.write_text(PRELUDE +
                    'fn unused()->int {let callbacks:array<fn(int)->int> = [increment] '
                    + expression + ' return 0}\nshadow unused {assert true}\n'
                    'fn main()->int{return 0}\nshadow main {assert true}\n')
                output.write_bytes(b'prior executable')
                result=subprocess.run(list(map(str,[arrays.harness.COMPILER,source,'-o',output])),
                    cwd=arrays.harness.ROOT,capture_output=True,text=True,timeout=120)
                self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('E001 TYPE MISMATCH',result.stderr)
                self.assertEqual(output.read_bytes(),b'prior executable')
