"""I execute canonical scalar callbacks with exact types and source order."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]

class FunctionalArrays(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch=tempfile.TemporaryDirectory(prefix='nano-functional-driver-')
        cls.driver=Path(cls.scratch.name)/'shadows'
        source=Path(cls.scratch.name)/'driver.nano'
        source.write_text((ROOT/'tests/nanoisa/fixtures/shadow_module_driver.nano.txt').read_text())
        cls.run_checked([ROOT/'bin/nanoc_c',source,'-o',cls.driver])
    @classmethod
    def tearDownClass(cls): cls.scratch.cleanup()
    @staticmethod
    def run_checked(args):
        r=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True,timeout=180,
            env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'})
        if r.returncode: raise AssertionError(f'{args}: {r.returncode}\n{r.stdout}\n{r.stderr}')
        return r
    def execute_module(self,module,folder):
        self.run_checked([ROOT/'bin/nano_vm','--verify-only',module])
        vm=self.run_checked([ROOT/'bin/nano_vm',module])
        generated=folder/'output.c';binary=folder/'native'
        self.run_checked([os.environ.get('NANO_FUNCTIONAL_NVM2C', str(ROOT/'bin/nvm2c')),module,'-o',generated])
        self.run_checked([os.environ.get('CC','cc'),'-std=c11','-O2','-Wall','-Wextra','-Werror',
            '-fsanitize=address,undefined','-fno-sanitize-recover=all',generated,'-lm','-o',binary])
        self.assertEqual(self.run_checked([binary]).stdout,vm.stdout)
    def paired(self,text,mode='raw'):
        with tempfile.TemporaryDirectory(prefix='nano-functional-') as tmp:
            folder=Path(tmp);source=folder/'input.nano';module=folder/'input.nvm';assembly=folder/'shadows.nasm'
            source.write_text(text)
            if mode=='raw':
                self.run_checked([ROOT/'bin/nanoisa_emit',source,'--emit-nvm','-o',module])
                self.execute_module(module,folder)
            result=self.run_checked([self.driver,source,0,mode])
            assembly.write_text(result.stdout)
            self.run_checked([ROOT/'bin/nanoisa','asm',assembly,'-o',module])
            self.execute_module(module,folder)
            return result.stdout
    def test_existing_core_examples(self):
        for name in ('nl_syntax_operators','nl_functions_map_reduce','nl_functions_filter','nl_functions_array_param'):
            with self.subTest(name=name): self.paired((ROOT/'tests'/f'{name}.nano').read_text())
    def test_source_order_and_captured_length(self):
        self.paired('''let mut trace: int = 0
let mut calls: int = 0
let values: array<int> = [1,2]
fn source()->array<int>{set trace (+ (* trace 10) 1) return [2,3]}
shadow source { assert true }
fn initial()->int{set trace (+ (* trace 10) 2) return 10}
shadow initial { assert true }
fn combine(a:int,b:int)->int{set trace (+ (* trace 10) 3) return (+ a b)}
shadow combine { assert true }
fn keep(x:int)->bool{set calls (+ calls 1) let grown:array<int> = (array_push values 9) return (> x 0)}
shadow keep { assert true }
fn main()->int{
 set trace 0
 assert (== (reduce (source) (initial) combine) 15)
 assert (== trace 1233)
 let result:array<int> = (filter values keep)
 assert (== calls 2)
 assert (== (array_length result) 2)
 assert (== (array_length values) 4)
 return 0
}
shadow main { assert (== (main) 0) }
''')
    def test_reduce_initializer_mutation_precedes_length_snapshot(self):
        text = """let values: array<int> = [2,3]
fn initial() -> int { let grown: array<int> = (array_push values 4) return 10 }
shadow initial { assert true }
fn combine(a: int, b: int) -> int { return (+ a b) }
shadow combine { assert true }
fn main() -> int { assert (== (reduce values (initial) combine) 19) return 0 }
shadow main { assert (== (main) 0) }
"""
        self.paired(text)
        with tempfile.TemporaryDirectory(prefix='nano-functional-seed-order-') as tmp:
            folder=Path(tmp);source=folder/'input.nano';module=folder/'seed.nvm'
            source.write_text(text)
            self.run_checked([ROOT/'bin/nano_virt',source,'--emit-nvm','-o',module])
            self.run_checked([ROOT/'bin/nano_vm','--verify-only',module])
            self.run_checked([ROOT/'bin/nano_vm',module])
    def test_typed_empty_fresh_outputs_and_changed_map_type(self):
        self.paired('''fn render(x:int)->string{return (int_to_string x)}
shadow render { assert true }
fn never(x:float)->bool{return false}
shadow never { assert true }
fn fraction(x:int)->float{if (== x 1) {return 1.5} return 2.5}
shadow fraction { assert true }
fn main()->int{
 let first:array<float> = (filter [] never)
 let second:array<float> = (filter [] never)
 let grown:array<float> = (array_push first 3.5)
 assert (== (array_length second) 0)
 assert (== (at grown 0) 3.5)
 let empty:array<string> = (map [] render)
 let filled:array<string> = (array_push empty "kept")
 assert (== (at filled 0) "kept")
 let words:array<string> = (map [1,2] render)
 assert (== (at words 0) "1")
 assert (== (at words 1) "2")
 let floats:array<float> = (map [1,2] fraction)
 assert (== (at floats 1) 2.5)
 assert (== (abs -4) 4)
 assert (== (abs -1.5) 1.5)
 assert (== (float_to_string (abs -0.0)) "-0.0")
 return 0
}
shadow main { assert (== (main) 0) }
''')
    def test_scalar_reductions_and_element_read_timing(self):
        self.paired("""let values: array<int> = [1,2]
let mut calls: int = 0
fn changing(x: int) -> bool {
 (array_set values 0 99)
 (array_set values 1 7)
 return true
}
shadow changing { assert true }
fn join(a: string, b: string) -> string { return (+ a b) }
shadow join { assert true }
fn both(a: bool, b: bool) -> bool { return (and a b) }
shadow both { assert true }
fn add_float(a: float, b: float) -> float { return (+ a b) }
shadow add_float { assert true }
fn negative() -> int { set calls (+ calls 1) return -7 }
shadow negative { assert true }
fn main() -> int {
 let saved: array<int> = (filter values changing)
 assert (== (at saved 0) 1)
 assert (== (at saved 1) 7)
 assert (== (at values 0) 99)
 assert (== (reduce ["a","b"] "start" join) "startab")
 assert (== (reduce [] "empty" join) "empty")
 assert (not (reduce [true,false] true both))
 assert (== (reduce [1.5,2.5] 0.5 add_float) 4.5)
 assert (== (abs (negative)) 7)
 assert (== calls 1)
 return 0
}
shadow main { assert (== (main) 0) }
""")
    def test_declared_builtin_names_keep_identity(self):
        self.paired('''fn map(xs:array<int>,value:int)->int{return value}
shadow map { assert true }
fn abs(value:int)->int{return 91}
shadow abs { assert true }
fn main()->int{assert (== (map [1] 7) 7) assert (== (abs -2) 91) return 0}
shadow main { assert (== (main) 0) }
''')
    def test_bound_callback_owners_and_selected_shadow_dependencies(self):
        assembly=self.paired('''fn dep_a_value(n:int)->int{return (+ n 10)}
shadow value { let xs:array<int> = (map [1] value) assert (== (at xs 0) 11) }

fn dep_b_value(n:int)->int{return (+ n 20)}
shadow value { let xs:array<int> = (map [1] value) assert (== (at xs 0) 21) }

fn main()->int{return 0}
shadow main { assert true }
''','bound')
        self.assertIn('CALL dep_a_value',assembly)
        self.assertIn('CALL dep_b_value',assembly)
    def test_signature_and_shadowing_refusals_preserve_output(self):
        declarations='''fn integer(x:int)->int{return x}
shadow integer { assert true }
fn floating(x:float)->float{return x}
shadow floating { assert true }
fn add(a:int,b:int)->int{return (+ a b)}
shadow add { assert true }
fn container(x:int)->array<int>{return [x]}
shadow container { assert true }
'''
        expressions=('(map [1] add)','(filter [1] integer)','(map [1] floating)',
            '(reduce [1] "wrong" add)','(map [1] container)','(map [1])','(abs true)',
            '(map [1] missing)')
        with tempfile.TemporaryDirectory(prefix='nano-functional-refusal-') as tmp:
            folder=Path(tmp);source=folder/'input.nano';output=folder/'output.nvm'
            for expression in expressions:
                with self.subTest(expression=expression):
                    source.write_text(declarations+'fn main()->int{ let result: array<int> = '+expression+' return 0 }\nshadow main { assert true }\n')
                    output.write_bytes(b'previous')
                    result=subprocess.run([ROOT/'bin/nanoisa_emit',source,'--emit-nvm','-o',output],cwd=ROOT,capture_output=True,text=True,timeout=60)
                    self.assertEqual(result.returncode,1,result.stdout+result.stderr)
                    self.assertNotIn('parse failed', result.stdout)
                    self.assertEqual(output.read_bytes(),b'previous')
            source.write_text(declarations+'fn main()->int{ let integer: int = 7\n let result: array<int> = (map [1] integer) return 0 }\n')
            output.write_bytes(b'previous')
            result=subprocess.run([ROOT/'bin/nanoisa_emit',source,'--emit-nvm','-o',output],cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,1,result.stdout+result.stderr)
            self.assertIn('indirect',result.stdout)
            self.assertEqual(output.read_bytes(),b'previous')

if __name__=='__main__': unittest.main()
