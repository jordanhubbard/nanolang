"""I qualify exact reduce source signatures separately from native FUNCREF."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ScalarReduce(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix='nano-reduce-source-'))
        print('I retain reduce source evidence at', self.work, flush=True)

    def command(self, *args, success=True):
        command = list(map(str, args))
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=240)
        message = f'tool={command[0]} command={command!r}\n{result.stdout}\n{result.stderr}'
        (self.assertEqual if success else self.assertNotEqual)(result.returncode, 0, message)
        return result

    def source(self, text):
        path = self.work / 'ordinary.nano'
        path.write_text(text)
        return path

    def legacy(self, source, names=('nanoc_c', 'nanoc_stage1', 'nanoc_stage2')):
        for name in names:
            with self.subTest(legacy=name):
                output = self.work / (name + '-legacy')
                self.command(ROOT/'bin'/name, source, '-o', output)
                self.command(output)

    def canonical(self, source):
        for name in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
            with self.subTest(canonical=name):
                module = self.work / (name + '.nvm')
                self.command(ROOT/'bin'/name, source, '--emit-nvm', '-o', module)
                self.command(ROOT/'bin/nano_vm', '--verify-only', module)
                self.command(ROOT/'bin/nano_vm', module)
                output = self.work / (name + '.c')
                if name == 'nano_virt':
                    output.write_text('previous output')
                    refused = self.command(ROOT/'bin/nvm2c', module, '-o', output, success=False)
                    self.assertIn('unsupported opcode FUNCREF', refused.stderr)
                    self.assertEqual(output.read_text(), 'previous output')
                    continue
                self.command(ROOT/'bin/nvm2c', module, '-o', output)
                executable = self.work / (name + '-native')
                self.command(os.environ.get('CC', 'cc'), '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined', '-fno-sanitize-recover=all', output, '-lm', '-o', executable)
                self.command(executable)

    def test_float_contexts_and_empty(self):
        source = self.source(FLOAT)
        self.command(ROOT/'bin/nano', source)
        self.legacy(source)
        self.canonical(source)

    def test_ordered_operands_and_post_initializer_length(self):
        source = self.source(ORDER)
        self.legacy(source, ('nanoc_stage1', 'nanoc_stage2'))

    def test_neighbor_helpers(self):
        source = self.source(NEIGHBORS)
        self.command(ROOT/'bin/nano', source)
        self.legacy(source)

    def test_bound_reduce_function_and_local(self):
        source = self.source(BOUND)
        self.legacy(source, ('nanoc_stage1', 'nanoc_stage2'))

    def test_checked_signature_refusals_preserve_output(self):
        cases = {
            'arity': 'fn fold(a:float,b:float)->float{return (+ a b)} fn main()->int{let x=(reduce [1.0] 0.0) return 0}',
            'array': 'fn fold(a:float,b:float)->float{return (+ a b)} fn main()->int{let x=(reduce 1.0 0.0 fold) return 0}',
            'parameter': 'fn fold(a:int,b:float)->int{return a} fn main()->int{let x=(reduce [1.0] 0.0 fold) return 0}',
            'result': 'fn fold(a:float,b:float)->int{return 0} fn main()->int{let x=(reduce [1.0] 0.0 fold) return 0}',
        }
        for label, text in cases.items():
            source = self.source(text)
            for name in ('nanoc_stage1', 'nanoc_stage2'):
                for flags in (('--target', 'c'), ('--emit-nvm',)):
                    with self.subTest(case=label, producer=name, flags=flags):
                        output = self.work/'prior'
                        output.write_bytes(b'previous output')
                        result = self.command(ROOT/'bin'/name, source, *flags, '-o', output, success=False)
                        self.assertIn('reduce', result.stdout + result.stderr)
                        self.assertEqual(output.read_bytes(), b'previous output')

    def test_supported_checker_unsupported_legacy_abi(self):
        source = self.source('fn fold(a:bool,b:bool)->bool{return (and a b)} shadow fold{assert (fold true true)} fn main()->int{let x:bool=(reduce [true] true fold) assert x return 0} shadow main{assert (== (main) 0)}')
        for name in ('nanoc_stage1', 'nanoc_stage2'):
            for flags in (('--target', 'c'), ()):
                output = self.work/'prior'
                output.write_bytes(b'previous output')
                result = self.command(ROOT/'bin'/name, source, *flags, '-o', output, success=False)
                self.assertIn('exact homogeneous', result.stdout + result.stderr)
                self.assertEqual(output.read_bytes(), b'previous output')

FLOAT = '''fn fold(a:float,b:float)->float{return (+ a b)}
shadow fold{assert (== (fold 1.0 2.0) 3.0)}
fn total(xs:array<float>)->float{return (reduce xs 0.0 fold)}
shadow total{assert (== (total [2.0,3.0]) 5.0)}
fn main()->int{
 let xs:array<float> = [1.0,2.0]
 let empty:array<float> = []
 let stored:float=(reduce xs 4.0 fold)
 assert (== (float_to_bits stored) 4619567317775286272)
 assert (== (float_to_bits (reduce xs 0.0 fold)) 4613937818241073152)
 assert (== (float_to_bits (total xs)) 4613937818241073152)
 assert (== (float_to_bits (reduce xs (reduce empty 2.0 fold) fold)) 4617315517961601024)
 assert (== (float_to_bits (reduce empty (float_from_bits -9223372036854775808) fold)) -9223372036854775808)
 assert (== (float_to_bits (at xs 0)) 4607182418800017408)
 return 0
}
shadow main{assert (== (main) 0)}
'''
NEIGHBORS = '''fn sum(a:int,b:int)->int{return (+ a b)}
shadow sum{assert (== (sum 2 3) 5)}
fn join(a:string,b:string)->string{return (+ a b)}
shadow join{assert (== (join "a" "b") "ab")}
fn main()->int{
 assert (== (reduce [2,3] 4 sum) 9)
 assert (== (reduce ["a","b"] "x" join) "xab")
 return 0
}
shadow main{assert (== (main) 0)}
'''
ORDER = '''let mut events:int=0
let mut values:array<float> = []
fn source()->array<float>{set events (+ (* events 10) 1) return values}
shadow source{let saved:int=events let xs:array<float>=(source) set events saved}
fn initial()->float{set events (+ (* events 10) 2) set values (array_push values 3.0) return 4.0}
shadow initial{let saved:int=events let old:array<float>=values set values [] assert (== (initial) 4.0) set values old set events saved}
fn combine(a:float,b:float)->float{set events (+ (* events 10) 4) return (+ a b)}
shadow combine{let saved:int=events assert (== (combine 1.0 2.0) 3.0) set events saved}
fn callback()->fn(float,float)->float{set events (+ (* events 10) 3) return combine}
shadow callback{let saved:int=events let f:fn(float,float)->float=(callback) set events saved}
fn main()->int{
 set events 0
 set values [2.0]
 let result:float=(reduce (source) (initial) (callback))
 assert (== (float_to_bits result) 4621256167635550208)
 assert (== events 12344)
 assert (== (array_length values) 2)
 return 0
}
shadow main{let saved:int=events let old:array<float>=values assert (== (main) 0) set values old set events saved}
'''.replace('array<float>=', 'array<float> =')
BOUND = '''fn reduce(x:float)->float{return (+ x 2.0)}
shadow reduce{assert (== (reduce 1.0) 3.0)}
fn identity(x:int)->int{return x}
shadow identity{assert (== (identity 4) 4)}
fn main()->int{
 assert (== (float_to_bits (reduce 1.0)) 4613937818241073152)
 let reduce:fn(int)->int=identity
 assert (== (reduce 9) 9)
 return 0
}
shadow main{assert (== (main) 0)}
'''
if __name__ == '__main__':
    unittest.main()
