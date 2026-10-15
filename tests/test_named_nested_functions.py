"""I preserve named lexical closures, recursive environments and refusal boundaries."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_local_binding_metadata as metadata
DRIVERS, ROOT = metadata.DRIVERS, metadata.ROOT

class NamedNestedFunctions(unittest.TestCase):
    run_checked = metadata.LocalBindingMetadata.run_checked
    produce = metadata.LocalBindingMetadata.produce

    def test_lexical_closures_and_recursive_values(self):
        cases = [
            ('basic', 'fn main()->int {fn helper(value:int)->int{return (+ value 1)} return (helper 6)}', 7),
            ('escape', 'fn make(offset:int)->fn(int)->int {fn helper(value:int)->int{return (+ value offset)} return helper} shadow make {assert true} fn main()->int {let f:fn(int)->int=(make 4) return (f 3)}', 7),
            ('recursive', 'fn main()->int {fn sum(value:int)->int{if (== value 0) {return 0} return (+ value (sum (- value 1)))} return (sum 3)}', 6),
            ('escaped_recursive', 'fn make(offset:int)->fn(int)->int{fn sum(value:int)->int{if (== value 0){return offset} return (+ value (sum (- value 1)))} return sum} shadow make {assert true} fn main()->int{let f:fn(int)->int=(make 1) return (f 3)}', 7),
            ('tail_capture', 'fn main()->int{let offset:int=7 fn walk(n:int)->int{if (== n 0){return offset} return (walk (- n 1))} return (walk 20)}', 7),
            ('scopes', 'fn main()->int {fn helper()->int{return 7} if true {fn helper()->int{return 3} assert (== (helper) 3)} return (helper)}', 7),
            ('grandparent', 'fn make(offset:int)->fn()->fn()->int{fn middle()->fn()->int{fn inner()->int{return offset} return inner} return middle} shadow make {assert true} fn main()->int{let first:fn()->fn()->int=(make 7) let second:fn()->int=(first) return (second)}', 7),
            ('parameter_shadow', 'fn main()->int{fn helper(helper:int)->int{return helper} return (helper 7)}', 7),
        ]
        for name, source, expected in cases:
            with self.subTest(case=name):
                self.produce(source+'\nshadow main {assert true}\n', expected, lambda bindings: None)

    def test_imported_lexical_function_shadows_module_function(self):
        source = 'import "helpers.nano" as Helpers\nfn main()->int{let f:fn(int)->int=(Helpers.make 4) return (f 3)}\nshadow main{assert (== (main) 7)}\n'
        helper = 'fn helper(value:bool)->bool{return value}\nshadow helper{assert true}\npub fn make(offset:int)->fn(int)->int{fn helper(value:int)->int{return (+ value offset)} return helper}\nshadow make{assert true}\n'
        self.produce(source,7,lambda bindings:None,files={'helpers.nano':helper})

    def test_invalid_lexical_declarations_preserve_output(self):
        sources = [
            'fn main()->int {return (helper)} fn other()->int{fn helper()->int{return 7} return (helper)} shadow other {assert true}',
            'fn main()->int {if true {fn helper()->int{return 7}} return (helper)}',
            'fn main()->int {let answer:int=(helper) fn helper()->int{return 7} return answer}',
            'fn main()->int {fn helper()->int{return true} return (helper)}',
            'fn main()->int {fn helper(value:int)->int{return value} return (helper true)}',
        ]
        for driver in DRIVERS:
            for source_text in sources:
                with self.subTest(driver=driver,source=source_text), tempfile.TemporaryDirectory() as name:
                    path=Path(name);source=path/'case.nano';output=path/'case.nvm'
                    source.write_text(source_text+'\nshadow main {assert true}\n')
                    output.write_bytes(b'previous artifact')
                    result=subprocess.run([driver,source,'--emit-nvm','-o',output],cwd=ROOT,capture_output=True,text=True,timeout=120)
                    self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                    self.assertEqual(output.read_bytes(),b'previous artifact')

if __name__=='__main__': unittest.main()
