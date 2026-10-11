"""I retain callable array identities across native storage and evaluator mutation."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from tests import test_native_global_types as harness

PRELUDE = '''fn increment(value:int)->int{return (+ value 1)}
shadow increment {assert (== (increment 7) 8)}
fn identity(value:T)->T{return value}
shadow identity {assert (== (identity 7) 7)}
'''


class NativeCallableArrays(unittest.TestCase):
    def execute(self, source):
        harness.NativeGlobalTypes().execute(PRELUDE + source)

    def test_copy_replace_slice_and_pop(self):
        self.execute('''fn main()->int {
let callbacks:array<fn(int)->int> = [identity]
let saved:fn(int)->int=(at callbacks 0)
let alias=callbacks
(array_set alias 0 increment)
let changed:fn(int)->int=(at callbacks 0)
assert (== (saved 7) 7) assert (== (changed 7) 8)
let sliced:array<fn(int)->int> = (array_slice callbacks 0 1)
(array_set callbacks 0 identity)
let popped:fn(int)->int=(array_pop sliced)
assert (== (popped 7) 8) assert (== (array_length sliced) 0)
let retained:fn(int)->int=(at callbacks 0)
assert (== (retained 7) 7) return 0}''')

    def test_empty_push_and_remove(self):
        self.execute('''fn main()->int {
let callbacks:array<fn(int)->int> = []
(array_push callbacks increment)
(array_push callbacks identity)
(array_remove_at callbacks 0)
let callback:fn(int)->int=(at callbacks 0)
assert (== (callback 7) 7) assert (== (array_length callbacks) 1) return 0}''')

    def test_repeated_initializer(self):
        self.execute('''fn main()->int {
let callbacks:array<fn(int)->int> = (array_new 3 identity)
let first:fn(int)->int=(at callbacks 0)
let last:fn(int)->int=(at callbacks 2)
assert (== (first 7) 7) assert (== (last 9) 9) return 0}''')

    def test_returned_array(self):
        self.execute('''fn make()->array<fn(int)->int>{return [identity]}
shadow make {let callbacks=(make) assert (== (array_length callbacks) 1)}
fn main()->int {let callbacks=(make) let callback:fn(int)->int=(at callbacks 0)
assert (== (callback 7) 7) return 0}''')

    def test_record_global_and_parameter(self):
        self.execute('''struct Holder {callbacks:array<fn(int)->int>}
let holder:Holder=Holder {callbacks:[identity]}
fn apply(callbacks:array<fn(int)->int>)->int {let callback:fn(int)->int=(at callbacks 0) return (callback 7)}
shadow apply {assert (== (apply [identity]) 7)}
fn main()->int {assert (== (apply holder.callbacks) 7) return 0}''')

    def test_incompatible_mutation_preserves_output(self):
        for operation in ('(array_set callbacks 0 bad)', '(array_push callbacks bad)'):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as tmp:
                source, output = Path(tmp)/'main.nano', Path(tmp)/'main'
                source.write_text(PRELUDE + 'fn bad(value:bool)->bool{return value}\nshadow bad {assert (bad true)}\n'
                    + 'fn unused()->int {let callbacks:array<fn(int)->int> = [increment] '
                    + operation + ' return 0}\nshadow unused {assert true}\n'
                    + 'fn main()->int{return 0}\nshadow main {assert true}\n')
                output.write_bytes(b'prior executable')
                result = subprocess.run(list(map(str,[harness.COMPILER,source,'-o',output])),
                    cwd=harness.ROOT,capture_output=True,text=True,timeout=120)
                self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertEqual(output.read_bytes(),b'prior executable')
