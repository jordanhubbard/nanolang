"""I execute callable array elements through map, filter and reduce."""
import unittest
import tempfile
import subprocess
from pathlib import Path
from tests import test_native_callable_arrays as arrays

PRELUDE = arrays.PRELUDE + '''fn apply(callback:fn(int)->int)->int{return (callback 7)}
shadow apply {assert (== (apply increment) 8)}
fn selected(callback:fn(int)->int)->bool{return (== (callback 7) 8)}
shadow selected {assert (selected increment)}
fn sum_callback(total:int, callback:fn(int)->int)->int{return (+ total (callback 7))}
shadow sum_callback {assert (== (sum_callback 1 increment) 9)}
'''


class NativeCallableOperations(unittest.TestCase):
    def execute(self, source):
        arrays.harness.NativeGlobalTypes().execute(PRELUDE + source)

    def test_map_callable_elements(self):
        self.execute('''fn main()->int {
let callbacks:array<fn(int)->int> = [increment, identity]
let values:array<int> =(map callbacks apply)
assert (== (at values 0) 8) assert (== (at values 1) 7) return 0}''')

    def test_map_callable_results(self):
        self.execute('''fn choose(value:int)->fn(int)->int{return increment}
shadow choose {let callback:fn(int)->int=(choose 0) assert (== (callback 7) 8)}
fn main()->int {let callbacks:array<fn(int)->int> =(map [1,2] choose)
let callback:fn(int)->int=(at callbacks 1) assert (== (callback 7) 8) return 0}''')

    def test_filter_callable_elements(self):
        self.execute('''fn main()->int {let callbacks:array<fn(int)->int> =[identity,increment]
let kept:array<fn(int)->int> =(filter callbacks selected)
assert (== (array_length kept) 1)
let callback:fn(int)->int=(at kept 0) assert (== (callback 7) 8) return 0}''')

    def test_filter_retains_value_before_predicate_mutation(self):
        self.execute('''let callbacks:array<fn(int)->int> =[increment,increment]
fn replace(callback:fn(int)->int)->bool {
(array_set callbacks 0 identity) (array_set callbacks 1 identity)
return (== (callback 7) 8)}
shadow replace {assert (replace increment)}
fn main()->int {
(array_set callbacks 0 increment) (array_set callbacks 1 increment)
let kept:array<fn(int)->int> =(filter callbacks replace)
assert (== (array_length kept) 1)
let callback:fn(int)->int=(at kept 0) assert (== (callback 7) 8) return 0}''')

    def test_reduce_callable_elements(self):
        self.execute('''fn main()->int {let callbacks:array<fn(int)->int> =[increment,identity]
assert (== (reduce callbacks 1 sum_callback) 16) return 0}''')

    def test_reduce_callable_accumulator(self):
        self.execute('''fn choose(previous:fn(int)->int, next:fn(int)->int)->fn(int)->int{return next}
shadow choose {let callback:fn(int)->int=(choose identity increment) assert (== (callback 7) 8)}
fn main()->int {let callbacks:array<fn(int)->int> =[identity,increment]
let callback:fn(int)->int=(reduce callbacks identity choose)
assert (== (callback 7) 8) return 0}''')

    def test_empty_callable_operations(self):
        self.execute('''fn main()->int {let callbacks:array<fn(int)->int> =[]
let values:array<int> =(map callbacks apply)
let kept:array<fn(int)->int> =(filter callbacks selected)
assert (== (array_length values) 0) assert (== (array_length kept) 0)
assert (== (reduce callbacks 9 sum_callback) 9) return 0}''')

    def test_map_retains_callable_argument_result(self):
        self.execute('''fn retain(callback:fn(int)->int)->fn(int)->int{return callback}
shadow retain {let callback:fn(int)->int=(retain increment) assert (== (callback 7) 8)}
fn main()->int {let original:array<fn(int)->int> = [increment]
let copied:array<fn(int)->int> = (map original retain)
(array_set original 0 identity)
let callback:fn(int)->int=(at copied 0) assert (== (callback 7) 8) return 0}''')

    def test_reduce_operand_order(self):
        self.execute('''let mut order:int=0
fn input()->array<fn(int)->int>{set order (+ (* order 10) 1) return [increment]}
shadow input {let callbacks=(input) assert (== (array_length callbacks) 1)}
fn initial()->int {set order (+ (* order 10) 2) return 1}
shadow initial {assert (== (initial) 1)}
fn reducer()->fn(int,fn(int)->int)->int {set order (+ (* order 10) 3) return sum_callback}
shadow reducer {let callback=(reducer) assert (== (callback 1 increment) 9)}
fn main()->int {set order 0 assert (== (reduce (input) (initial) (reducer)) 9)
assert (== order 123) return 0}''')

    def test_incompatible_callbacks_preserve_output(self):
        cases = [
            ('fn wrong(callback:fn(bool)->bool)->int{return 0}\nshadow wrong {assert (== (wrong bool_identity) 0)}', '(map callbacks wrong)'),
            ('fn wrong(callback:fn(bool)->bool)->bool{return true}\nshadow wrong {assert (wrong bool_identity)}', '(filter callbacks wrong)'),
            ('fn wrong(total:int, callback:fn(bool)->bool)->int{return total}\nshadow wrong {assert (== (wrong 1 bool_identity) 1)}', '(reduce callbacks 0 wrong)'),
        ]
        for declaration, operation in cases:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as tmp:
                source, output = Path(tmp)/'main.nano', Path(tmp)/'main'
                source.write_text(PRELUDE +
                    'fn bool_identity(value:bool)->bool{return value}\nshadow bool_identity {assert (bool_identity true)}\n'
                    + declaration + '\nfn unused()->int {let callbacks:array<fn(int)->int> = [increment] '
                    + operation + ' return 0}\nshadow unused {assert true}\nfn main()->int{return 0}\nshadow main {assert true}\n')
                output.write_bytes(b'prior executable')
                result = subprocess.run(list(map(str,[arrays.harness.COMPILER,source,'-o',output])),
                    cwd=arrays.harness.ROOT,capture_output=True,text=True,timeout=120)
                self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('E001 TYPE MISMATCH',result.stderr)
                self.assertEqual(output.read_bytes(),b'prior executable')
