"""I qualify direct scalar calls and retain unsupported callback boundaries."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class NamedScalarCallbacks(unittest.TestCase):
    @classmethod
    def command(cls, *args, ok=True):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=900)
        if (result.returncode == 0) != ok:
            raise AssertionError(f'{args}: {result.returncode}\n{result.stdout}\n{result.stderr}')
        return result

    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='nano-named-callbacks-'))
        print('I retain named callback artifacts at', cls.work, flush=True)
        cls.emitters = []
        for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            emitter = cls.work / (compiler + '-emit')
            cls.command(ROOT/'bin'/compiler, ROOT/'src_nano/nanoisa_emit.nano', '-o', emitter)
            cls.emitters.append(emitter)

    def source(self, text):
        path = self.work / (self._testMethodName + '.nano')
        path.write_text(text)
        return path

    def seed(self, source):
        module = source.with_suffix('.nvm')
        self.command(ROOT/'bin/nano_virt', source, '--emit-nvm', '-o', module)
        return module

    def execute(self, module, direct=True, refusal=None):
        self.command(ROOT/'bin/nano_vm', '--verify-only', module)
        vm = self.command(ROOT/'bin/nano_vm', module)
        dump = self.command(ROOT/'bin/nanoisa', 'dump', module).stdout
        if direct:
            self.assertNotIn('FUNCREF', dump)
            self.assertNotIn('CALL_INDIRECT', dump)
            self.assertIn('CALL ', dump)
        output = module.with_suffix('.c')
        if not direct:
            self.assertIn('FUNCREF', dump)
            self.assertIn('CALL_INDIRECT', dump)
            output.write_text('previous')
            refused = self.command(ROOT/'bin/nvm2c', module, '-o', output, ok=False)
            self.assertIsNotNone(refusal, 'I require the specific unsupported boundary')
            self.assertIn(refusal, refused.stderr)
            self.assertEqual(output.read_text(), 'previous')
            return
        self.command(ROOT/'bin/nvm2c', module, '-o', output)
        executable = module.with_suffix('.native')
        self.command(os.environ.get('CC', 'cc'), '-std=c11', '-O2', '-Wall', '-Wextra',
                     '-Werror', '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                     output, '-lm', '-o', executable)
        self.assertEqual(self.command(executable).stdout, vm.stdout)

    def paired(self, text, legacy=False):
        source = self.source(text)
        self.execute(self.seed(source))
        for i, emitter in enumerate(self.emitters):
            assembly = self.work / f'{self._testMethodName}-{i}.nasm'
            module = assembly.with_suffix('.nvm')
            self.command(emitter, source, '-o', assembly)
            self.command(ROOT/'bin/nanoisa', 'asm', assembly, '-o', module)
            self.execute(module)
        if legacy:
            self.command(ROOT/'bin/nano', source)
            for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
                output = self.work / (self._testMethodName + '-' + compiler)
                self.command(ROOT/'bin'/compiler, source, '-o', output)
                self.command(output)

    def test_exact_binary64_results_and_unchanged_inputs(self):
        self.paired(BITS, legacy=True)

    def test_int_bool_results_and_typed_empty(self):
        self.paired(SCALARS)

    def test_source_initializer_order_and_alias_length(self):
        self.paired(ORDER)

    def test_user_builtin_names_keep_route_specific_binding(self):
        source = self.source('''fn map(x:int)->int{return (+ x 2)}
shadow map{assert (== (map 3) 5)}
fn reduce(x:int)->int{return (+ x 4)}
shadow reduce{assert (== (reduce 3) 7)}
fn main()->int{assert (== (map 1) 3) assert (== (reduce 1) 5) return 0}
shadow main{assert (== (main) 0)}
''')
        prior = self.work/'reserved-name.nvm'; prior.write_bytes(b'previous')
        refused = self.command(ROOT/'bin/nano_virt', source, '--emit-nvm', '-o', prior, ok=False)
        self.assertIn('Cannot redefine built-in function', refused.stderr)
        self.assertEqual(prior.read_bytes(), b'previous')
        for i, emitter in enumerate(self.emitters):
            assembly = self.work/f'reserved-{i}.nasm'; module = assembly.with_suffix('.nvm')
            self.command(emitter, source, '-o', assembly)
            self.command(ROOT/'bin/nanoisa', 'asm', assembly, '-o', module)
            self.execute(module)

    def test_local_and_computed_callbacks_keep_indirect_boundary(self):
        for expression, binding in [('fold', 'let fold:fn(float,float)->float=difference'),
                                    ('(choose)', '')]:
            source = self.source(ALIASES + f'''fn main()->int{{
 {binding}
 assert (== (reduce [3.0] 10.0 {expression}) 7.0)
 return 0
}}
shadow main{{assert (== (main) 0)}}
''')
            self.execute(self.seed(source), direct=False,
                         refusal='CALL_INDIRECT has no exact scalar target')

    def test_global_alias_after_initializer_mutation_stays_indirect(self):
        self.execute(self.seed(self.source(GLOBAL_ALIAS)), direct=False,
                     refusal='cannot yet store an aggregate or unresolved global')

    def test_u8_keeps_existing_indirect_boundary(self):
        self.execute(self.seed(self.source('''fn byte_identity(x:u8)->u8{return x}
shadow byte_identity{let mut input:array<u8> = [] set input (array_push input 2) let result:u8=(byte_identity (array_get input 0)) let mut output:array<u8> = [] set output (array_push output result) assert (== (array_get output 0) (array_get input 0))}
fn main()->int{let input:array<u8> = [] let result:array<u8> = (map input byte_identity) assert (== (array_length result) 0) return 0}
shadow main{assert (== (main) 0)}
''')), direct=False,
                     refusal='I support int, bool, float, string, array or struct elements in ARR_NEW')

    def test_wrong_signatures_preserve_previous_output(self):
        for text in (
            'fn fold(a:int,b:float)->int{return a} fn main()->int{let x=(reduce [1.0] 0.0 fold) return 0}',
            'fn fold(a:float,b:float)->int{return 0} fn main()->int{let x=(reduce [1.0] 0.0 fold) return 0}',
            'fn change(a:int)->int{return a} fn main()->int{let x=(map [1.0] change) return 0}',
        ):
            source = self.source(text)
            for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
                output = self.work/'previous.nvm'; output.write_bytes(b'previous')
                self.command(ROOT/'bin'/compiler, source, '--emit-nvm', '-o', output, ok=False)
                self.assertEqual(output.read_bytes(), b'previous')

BITS = '''fn normalize(x:float)->float{return (+ x 0.0)}
shadow normalize{assert (== (normalize 2.0) 2.0)}
fn zero_divisor(x:float)->float{return (/ x -0.0)}
shadow zero_divisor{assert (== (float_to_bits (zero_divisor 1.0)) 0)}
fn fold(a:float,b:float)->float{return (+ a b)}
shadow fold{assert (== (fold 2.0 3.0) 5.0)}
fn separate(x:float)->float{return (- (* x (float_from_bits 4607182418800017406)) 1.0)}
shadow separate{assert (== (float_to_bits (separate (float_from_bits 4607182418800017409))) 0)}
fn main()->int{
 let positive:float=(float_from_bits 9218868437227405313)
 let negative:float=(float_from_bits -2251799813685247)
 let values:array<float> = [positive, negative]
 let normalized:array<float> = (map values normalize)
 assert (== (float_to_bits (array_get normalized 0)) 9221120237041090560)
 assert (== (float_to_bits (array_get normalized 1)) 9221120237041090560)
 let zeros:array<float> = (map values zero_divisor)
 assert (== (float_to_bits (array_get zeros 0)) 0)
 assert (== (float_to_bits (array_get zeros 1)) 0)
 assert (== (float_to_bits (reduce values 0.0 fold)) 9221120237041090560)
 assert (== (float_to_bits (array_get values 0)) 9218868437227405313)
 assert (== (float_to_bits (array_get values 1)) -2251799813685247)
 let rounded:array<float> = (map [(float_from_bits 4607182418800017409)] separate)
 assert (== (float_to_bits (array_get rounded 0)) 0)
 return 0
}
shadow main{assert (== (main) 0)}
'''
SCALARS = '''fn increment(x:int)->int{return (+ x 1)}
shadow increment{assert (== (increment 1) 2)}
fn add(a:int,b:int)->int{return (+ a b)}
shadow add{assert (== (add 1 2) 3)}
fn positive(x:int)->bool{return (> x 0)}
shadow positive{assert (positive 1)}
fn both(a:bool,b:bool)->bool{return (and a b)}
shadow both{assert (both true true)}
fn main()->int{
 let changed:array<int> = (map [1, 2] increment)
 assert (== (reduce changed 0 add) 5)
 let flags:array<bool> = (map [1, 2] positive)
 assert (reduce flags true both)
 let empty:array<int> = []
 assert (== (reduce empty 19 add) 19)
 assert (== (array_length (map empty increment)) 0)
 return 0
}
shadow main{assert (== (main) 0)}
'''
ORDER = '''let mut events:int=0
fn source(values:array<float>)->array<float>{set events (+ (* events 10) 1) return values}
shadow source{let saved:int=events assert (== (array_length (source [1.0])) 1) set events saved}
fn initial(values:array<float>)->float{set events (+ (* events 10) 2) let grown:array<float> = (array_push values 3.0) return 4.0}
shadow initial{let saved:int=events let values:array<float> = [1.0] assert (== (initial values) 4.0) assert (== (array_length values) 2) set events saved}
fn combine(a:float,b:float)->float{set events (+ (* events 10) 4) return (+ a b)}
shadow combine{let saved:int=events assert (== (combine 1.0 2.0) 3.0) set events saved}
fn main()->int{
 set events 0
 let values:array<float> = [2.0]
 assert (== (reduce (source values) (initial values) combine) 9.0)
 assert (== events 1244)
 assert (== (array_length values) 2)
 return 0
}
shadow main{let saved:int=events assert (== (main) 0) set events saved}
'''
ALIASES = '''fn fold(a:float,b:float)->float{return (+ a b)}
shadow fold{assert (== (fold 1.0 2.0) 3.0)}
fn difference(a:float,b:float)->float{return (- a b)}
shadow difference{assert (== (difference 3.0 2.0) 1.0)}
fn choose()->fn(float,float)->float{return difference}
shadow choose{let f:fn(float,float)->float=(choose) assert (== (f 3.0 2.0) 1.0)}
'''
GLOBAL_ALIAS = ALIASES + '''fn selected(a:float,b:float)->float{return (+ a b)}
shadow selected{assert (== (selected 1.0 2.0) 3.0)}
let mut selected:fn(float,float)->float=fold
fn initial()->float{set selected difference return 10.0}
shadow initial{set selected fold assert (== (initial) 10.0) set selected fold}
fn main()->int{
 set selected fold
 assert (== (reduce [3.0] (initial) selected) 7.0)
 return 0
}
shadow main{assert (== (main) 0)}
'''
if __name__ == '__main__':
    unittest.main()
