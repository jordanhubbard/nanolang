"""I specialize generic function values from complete callable contexts."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from tests import test_cseed_generic_functions as shared

PRELUDE = 'fn identity(value:T)->T{return value}\nshadow identity {assert (== (identity 1) 1)}\n'
SHADOW = '\nshadow main {assert (== (main) 0)}\n'


class ContextualGenericFunctions(unittest.TestCase):
    def execute(self, source, files=None):
        shared.CseedGenericFunctions().execute(source, files=files)

    def test_local_multiple_instances(self):
        self.execute(PRELUDE+'''fn main()->int {
 let number:fn(int)->int = identity
 let flag:fn(bool)->bool = identity
 assert (== (number 7) 7)
 assert (flag true)
 assert (== (identity 8) 8)
 return 0
}'''+SHADOW)

    def test_structural_array_context(self):
        self.execute(PRELUDE+'''fn main()->int {
 let callback:fn(array<int>)->array<int> = identity
 let result = (callback [7])
 assert (== (at result 0) 7)
 return 0
}'''+SHADOW)

    def test_nominal_and_tuple_context(self):
        self.execute(PRELUDE+'''struct Box {value:int}
fn main()->int {
 let callback:fn((Box,bool))->(Box,bool) = identity
 let result = (callback (Box {value:7},true))
 assert (== result.0.value 7)
 assert result.1
 return 0
}'''+SHADOW)

    def test_returned_callback(self):
        self.execute(PRELUDE+'''fn choose()->fn(int)->int{return identity}
shadow choose {assert (== ((choose) 7) 7)}
fn main()->int{assert (== ((choose) 7) 7) return 0}'''+SHADOW)

    def test_callback_argument(self):
        self.execute(PRELUDE+'''fn apply(f:fn(int)->int)->int{return (f 7)}
shadow apply {assert (== (apply identity) 7)}
fn main()->int{assert (== (apply identity) 7) return 0}'''+SHADOW)

    def test_tail_and_indirect_callback_arguments(self):
        self.execute(PRELUDE+'''fn apply(f:fn(int)->int)->int{return (f 7)}
shadow apply {assert (== (apply identity) 7)}
fn forward()->int{return (apply identity)}
shadow forward {assert (== (forward) 7)}
fn main()->int {
 let indirect:fn(fn(int)->int)->int = apply
 assert (== (indirect identity) 7)
 assert (== (forward) 7)
 return 0
}'''+SHADOW)

    def test_reassigned_callback(self):
        self.execute(PRELUDE+'''fn increment(value:int)->int{return (+ value 1)}
shadow increment {assert (== (increment 7) 8)}
fn main()->int {
 let mut callback:fn(int)->int = increment
 set callback identity
 assert (== (callback 7) 7)
 return 0
}'''+SHADOW)

    def test_imported_alias_callback(self):
        self.execute('from "identity.nano" import identity as same\n'+'''fn main()->int {
 let callback:fn(int)->int = same
 assert (== (callback 7) 7)
 return 0
}'''+SHADOW, files={'identity.nano':PRELUDE.replace('fn identity','pub fn identity',1)})

    def test_inconsistent_context_preserves_output(self):
        for body in [
            'fn main()->int{let f:fn(int)->bool=identity return 0}',
            'fn choose()->fn(int)->bool{return identity}\nshadow choose {assert true}\nfn main()->int{return 0}',
            'fn apply(f:fn(int)->bool)->int{return 0}\nshadow apply {assert true}\nfn main()->int{return (apply identity)}',
            'fn main()->int{let mut f:fn(int)->bool=positive set f identity return 0}',
            'fn choose()->fn(int)->int{return positive}\nshadow choose {assert true}\nfn main()->int{return 0}',
        ]:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as tmp:
                source, output = Path(tmp)/'main.nano', Path(tmp)/'main.nvm'
                source.write_text(PRELUDE+'fn positive(value:int)->bool{return (> value 0)}\nshadow positive {assert (positive 1)}\n'+body+SHADOW)
                output.write_bytes(b'prior')
                result=subprocess.run([str(shared.COMPILER),str(source),'--emit-nvm','-o',str(output)],
                                      cwd=shared.ROOT,capture_output=True,text=True,timeout=120)
                self.assertGreater(result.returncode,0,result.stdout+result.stderr)
                self.assertEqual(output.read_bytes(),b'prior')
