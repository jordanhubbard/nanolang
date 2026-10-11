"""I qualify exact reduce source signatures and named canonical native calls."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
# I retain installed-stage defaults and allow a fresh product to be qualified explicitly.
STAGES = [Path(path) for path in os.environ.get(
    'NANOLANG_REDUCE_COMPILERS',
    os.pathsep.join(str(ROOT/'bin'/name) for name in ('nanoc_stage1', 'nanoc_stage2'))
).split(os.pathsep) if path]
if not STAGES:
    raise ValueError('I require at least one reduce compiler product')


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

    def legacy(self, source, names=None):
        for name in ([ROOT/'bin/nanoc_c', *STAGES] if names is None else names):
            with self.subTest(legacy=name):
                output = self.work / (Path(name).name + '-legacy')
                self.command(name, source, '-o', output)
                self.command(output)

    def canonical(self, source):
        for name in [ROOT/'bin/nano_virt', *STAGES]:
            with self.subTest(canonical=name):
                module = self.work / (Path(name).name + '.nvm')
                self.command(name, source, '--emit-nvm', '-o', module)
                self.command(ROOT/'bin/nano_vm', '--verify-only', module)
                self.command(ROOT/'bin/nano_vm', module)
                output = self.work / (Path(name).name + '.c')
                self.command(ROOT/'bin/nvm2c', module, '-o', output)
                executable = self.work / (Path(name).name + '-native')
                self.command(os.environ.get('CC', 'cc'), '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined', '-fno-sanitize-recover=all', output, '-lm', '-o', executable)
                self.command(executable)

    def test_float_contexts_and_empty(self):
        source = self.source(FLOAT)
        self.command(ROOT/'bin/nano', source)
        self.legacy(source)
        self.canonical(source)

    def test_ordered_operands_and_post_initializer_length(self):
        source = self.source(ORDER)
        self.legacy(source)
        self.canonical(source)

    def test_neighbor_helpers(self):
        source = self.source(NEIGHBORS)
        self.command(ROOT/'bin/nano', source)
        self.legacy(source)

    def test_bound_reduce_function_and_local(self):
        source = self.source(BOUND)
        self.legacy(source, STAGES)

    def test_checked_signature_refusals_preserve_output(self):
        cases = {
            'arity': 'fn fold(a:float,b:float)->float{return (+ a b)} fn main()->int{let x=(reduce [1.0] 0.0) return 0}',
            'array': 'fn fold(a:float,b:float)->float{return (+ a b)} fn main()->int{let x=(reduce 1.0 0.0 fold) return 0}',
            'parameter': 'fn fold(a:int,b:float)->int{return a} fn main()->int{let x=(reduce [1.0] 0.0 fold) return 0}',
            'result': 'fn fold(a:float,b:float)->int{return 0} fn main()->int{let x=(reduce [1.0] 0.0 fold) return 0}',
        }
        for label, text in cases.items():
            source = self.source(text)
            for name in STAGES:
                for flags in (('--target', 'c'), ('--emit-nvm',)):
                    with self.subTest(case=label, producer=name, flags=flags):
                        output = self.work/'prior'
                        output.write_bytes(b'previous output')
                        result = self.command(name, source, *flags, '-o', output, success=False)
                        self.assertIn('reduce', result.stdout + result.stderr)
                        self.assertEqual(output.read_bytes(), b'previous output')

    def test_conflicting_global_array_results_preserve_output(self):
        assembly = self.work/'conflicting.nasm'
        module = self.work/'conflicting.nvm'
        output = self.work/'prior.c'
        assembly.write_text('''.entry main
.function read 0 0 0 array 1
LOAD_GLOBAL 0
RET
.end
.function main 0 0 0 int 1
ARR_NEW 1
STORE_GLOBAL 0
CALL read
POP
ARR_NEW 3
STORE_GLOBAL 0
CALL read
POP
PUSH_I64 0
RET
.end
''')
        self.command(ROOT/'bin/nanoisa', 'asm', assembly, '-o', module)
        self.command(ROOT/'bin/nano_vm', '--verify-only', module)
        self.command(ROOT/'bin/nano_vm', module)
        output.write_bytes(b'previous output')
        self.command(ROOT/'bin/nvm2c', module, '-o', output, success=False)
        self.assertEqual(output.read_bytes(), b'previous output')

    def test_global_array_mutations_preserve_result_storage(self):
        for element, first, second in [('int', '7', '9'), ('float', '2.0', '3.0'),
                                        ('bool', 'true', 'false'), ('string', '"old"', '"new"')]:
            with self.subTest(element=element):
                source = self.source(f'''let mut values:array<{element}> = []
fn read()->array<{element}> {{return values}}
shadow read {{assert (>= (array_length (read)) 0)}}
fn main()->int {{
 set values []
 assert (== (array_length (read)) 0)
 set values (array_push values {first})
 assert (== (at (read) 0) {first})
 (array_set values 0 {second})
 assert (== (at (read) 0) {second})
 return 0
}}
shadow main {{assert (== (main) 0)}}
''')
                self.legacy(source)
                self.canonical(source)

    def test_boolean_results_and_empty(self):
        source = self.source(BOOLEAN)
        self.command(ROOT/'bin/nano', source)
        self.legacy(source)
        self.canonical(source)
        for name in STAGES:
            with self.subTest(c_source=name):
                output = self.work/'boolean.c'
                self.command(name, source, '--target', 'c', '-o', output)
                executable = self.work/'boolean-c'
                self.command(os.environ.get('CC', 'cc'), '-std=c11', '-Wall', '-Wextra', '-Werror',
                             '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                             output, '-lm', '-o', executable)
                self.command(executable)

BOOLEAN = '''fn fold(a:bool,b:bool)->bool{return (and a b)}
shadow fold{assert (fold true true) assert (not (fold true false))}
fn main()->int{
 let empty:array<bool> = []
 assert (reduce [true,true] true fold)
 assert (not (reduce [true,false] true fold))
 assert (not (reduce [true] false fold))
 assert (reduce empty true fold)
 assert (not (reduce empty false fold))
 return 0
}
shadow main{assert (== (main) 0)}
'''

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
