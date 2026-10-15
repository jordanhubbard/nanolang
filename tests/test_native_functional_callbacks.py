"""I execute functional callable arrays through NanoISA and native C AOT."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_native_callables as harness

ROOT = harness.ROOT
TRANSLATOR = Path(os.environ.get('NANO_DEAD_CALLABLE_TRANSLATOR', ROOT/'bin/nvm2c'))


class NativeFunctionalCallbacks(unittest.TestCase):
    def execute(self, source, work, product=True):
        module, c = work/'input.nvm', work/'input.c'
        runner = harness.NativeCallables()
        compiler = Path(os.environ.get('NANO_FUNCTIONAL_COMPILER', ROOT/'bin/nano_virt'))
        checker = os.environ.get('NANO_FUNCTIONAL_CHECKER')
        emitter = os.environ.get('NANO_FUNCTIONAL_EMITTER')
        if checker:
            runner.checked([checker, source])
        producers = (['product'] if product else []) + (['whole', 'program', 'raw'] if emitter else [])
        for producer in producers:
            with self.subTest(producer=producer):
                if producer == 'product':
                    runner.checked([compiler, source, '--emit-nvm', '-o', module])
                else:
                    assembly = work/'input.nasm'
                    assembly.write_text(runner.checked([emitter, source, '0', producer]).stdout)
                    runner.checked([ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
                runner.checked([ROOT/'bin/nano_vm', '--verify-only', module])
                vm = runner.checked([ROOT/'bin/nano_vm', module])
                runner.checked([TRANSLATOR, module, '-o', c])
                native = runner.sanitized(c, work/'native')
                self.assertEqual(native.stdout, vm.stdout)

    def test_retained_source_operations(self):
        for evidence, count in (('callable-array-operations-20261010', 9),
                                ('generic-array-callbacks-20261010', 7)):
            sources = sorted((ROOT/'docs/evidence'/evidence).glob('test_*.nano'))
            self.assertEqual(len(sources), count)
            for source in sources:
                with self.subTest(source=source.name), tempfile.TemporaryDirectory() as tmp:
                    self.execute(source, Path(tmp))

    def test_callback_expression_order_and_length_snapshot(self):
        source_text = """let values:array<int> = [2,3]
let mut trace:int=0
fn source()->array<int>{set trace (+ (* trace 10) 1) return values}
shadow source {assert true}
fn initial()->int{set trace (+ (* trace 10) 2) return 10}
shadow initial {assert true}
fn combine(a:int,b:int)->int{return (+ a b)}
shadow combine {assert (== (combine 1 2) 3)}
fn callback()->fn(int,int)->int{
set trace (+ (* trace 10) 3) (array_push values 4) return combine}
shadow callback {assert true}
fn main()->int{
set trace 0
assert (== (reduce (source) (initial) (callback)) 19)
assert (== trace 123)
return 0}
shadow main {assert (== (main) 0)}
"""
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source = work/'source.nano'
            source.write_text(source_text)
            self.execute(source, work)

    def test_callback_variable_is_captured_once(self):
        source_text = """fn plus(x:int)->int{return (+ x 1)}
shadow plus {assert (== (plus 1) 2)}
let mut chosen:fn(int)->int=plus
fn switch(x:int)->int{set chosen plus return (+ x 10)}
shadow switch {assert true}
fn main()->int{
set chosen switch
let result:array<int> = (map [1,2] chosen)
assert (== (at result 0) 11) assert (== (at result 1) 12)
assert (== (chosen 1) 2) return 0}
shadow main {assert (== (main) 0)}
"""
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source = work/'source.nano'
            source.write_text(source_text)
            self.execute(source, work)

    def test_declared_map_keeps_its_signature(self):
        if not any(os.environ.get(key) for key in ('NANO_FUNCTIONAL_COMPILER', 'NANO_FUNCTIONAL_CHECKER', 'NANO_FUNCTIONAL_EMITTER')):
            self.skipTest('I require a self-hosted route for its declared builtin-name contract')
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source = work/'source.nano'
            source.write_text('fn map(xs:array<int>,value:int)->int{return value}\n'
                'shadow map {assert true}\n'
                'fn main()->int{assert (== (map [1] 7) 7) return 0}\n'
                'shadow main {assert (== (main) 0)}\n')
            self.execute(source, work, product=bool(os.environ.get('NANO_FUNCTIONAL_COMPILER')))

    def test_inline_generic_elements_take_callback_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source = work/'source.nano'
            source.write_text('fn identity(x:T)->T{return x}\n'
                'shadow identity {assert (== (identity 1) 1)}\n'
                'fn apply(f:fn(bool)->bool)->bool{return (f true)}\n'
                'shadow apply {assert (apply identity)}\n'
                'fn main()->int{let values:array<bool> = (map [identity] apply) '
                'assert (at values 0) return 0}\nshadow main {assert (== (main) 0)}\n')
            self.execute(source, work)

    def test_returned_global_array_storage(self):
        source_text = """fn plus(x:int)->int{return (+ x 1)}
shadow plus {assert (== (plus 1) 2)}
let numbers:array<int> = [7]
let words:array<string> = ["kept"]
let flags:array<bool> = [true]
let fractions:array<float> = [1.5]
let callbacks:array<fn(int)->int> = [plus]
fn read_numbers()->array<int>{return numbers}
shadow read_numbers {assert true}
fn read_words()->array<string>{return words}
shadow read_words {assert true}
fn read_flags()->array<bool>{return flags}
shadow read_flags {assert true}
fn read_fractions()->array<float>{return fractions}
shadow read_fractions {assert true}
fn read_callbacks()->array<fn(int)->int>{return callbacks}
shadow read_callbacks {assert true}
fn main()->int{
assert (== (at (read_numbers) 0) 7)
assert (== (at (read_words) 0) "kept")
assert (at (read_flags) 0)
assert (== (at (read_fractions) 0) 1.5)
let callback:fn(int)->int=(at (read_callbacks) 0)
assert (== (callback 7) 8) return 0}
shadow main {assert (== (main) 0)}
"""
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source = work/'source.nano'
            source.write_text(source_text)
            self.execute(source, work)

    def test_contextual_generic_array_mutations(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source = work/'source.nano'
            source.write_text('fn identity(x:T)->T{return x}\n'
                'shadow identity {assert (== (identity 1) 1)}\n'
                'fn main()->int{let callbacks:array<fn(int)->int> = [identity] '
                '(array_set callbacks 0 identity) (array_push callbacks identity) '
                'assert (== (array_length callbacks) 2) '
                'let callback:fn(int)->int=(at callbacks 1) '
                'assert (== (callback 7) 7) return 0}\nshadow main {assert (== (main) 0)}\n')
            self.execute(source, work)

    def test_callable_for_loop_bindings(self):
        sources = sorted((ROOT/'docs/evidence/selfhost-functional-callbacks-20261010').glob('test_iteration_*.nano'))
        self.assertEqual(len(sources), 3)
        for source in sources:
            with self.subTest(source=source.name), tempfile.TemporaryDirectory() as tmp:
                self.execute(source, Path(tmp))

    def test_enum_and_byte_array_storage_conversions(self):
        source_text = """enum Kind {One = 7}
fn number()->int{return 300}
shadow number {assert (== (number) 300)}
fn main()->int{
let values:array<int> = []
(array_push values Kind.One) (array_set values 0 Kind.One)
assert (== (at values 0) 7)
let bytes:array<u8> = []
(array_push bytes (number)) (array_set bytes 0 (number))
let value:u8=(at bytes 0) assert (== value 44) return 0}
shadow main {assert (== (main) 0)}
"""
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source = work/'source.nano'
            source.write_text(source_text)
            self.execute(source, work)

    def test_invalid_unused_callbacks_preserve_output(self):
        declarations = """fn identity(x:T)->T{return x}
shadow identity {assert (== (identity 1) 1)}
fn keep(x:T)->bool{return true}
shadow keep {assert (keep 1)}
fn latest(a:T,b:T)->T{return b}
shadow latest {assert (== (latest 1 2) 2)}
fn wrong(x:fn(bool)->bool)->int{return 0}
shadow wrong {assert true}
"""
        expressions = ['(filter [7] identity)', '(reduce [true] 7 latest)',
            '(map callbacks wrong)',
            'let bad:array<fn(int)->bool> = (filter callbacks keep)',
            'let bad:array<fn(int)->bool> = (map callbacks identity)',
            '(map [1] 7)', '(map [1])',
            '(array_set callbacks 0 keep)', '(array_push callbacks keep)',
            'for callback in callbacks {(callback true)}']
        for expression in expressions:
            with self.subTest(expression=expression), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                source, output = work/'source.nano', work/'output.nvm'
                source.write_text(declarations +
                    'fn unused()->int{let callbacks:array<fn(int)->int> = [identity] ' +
                    expression + ' return 0}\nshadow unused {assert true}\n' +
                    'fn main()->int{return 0}\nshadow main {assert true}\n')
                checker = os.environ.get('NANO_FUNCTIONAL_CHECKER')
                if checker:
                    result = subprocess.run([checker, source], capture_output=True, text=True, timeout=90)
                    self.assertEqual(result.returncode, 1, result.stdout+result.stderr)
                output.write_bytes(b'prior artifact')
                compiler = os.environ.get('NANO_FUNCTIONAL_COMPILER', str(ROOT/'bin/nano_virt'))
                result = subprocess.run([compiler, source, '--emit-nvm', '-o', output],
                    capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 1, result.stdout+result.stderr)
                self.assertEqual(output.read_bytes(), b'prior artifact')

    def test_incompatible_tagged_parameter_preserves_output(self):
        text = '''.entry main
.function leaf 0 0 0 int 1
PUSH_I64 7
RET
.end
.function retain 1 1 0 function 1
LOAD_LOCAL 0
RET
.end
.function main 0 0 0 int 1
FUNCREF leaf
CALL retain
POP
PUSH_I64 7
ARR_LITERAL 1 1
PUSH_I64 0
ARR_GET
CALL retain
POP
PUSH_I64 0
RET
.end
'''
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            assembly, module, output = work/'input.nasm', work/'input.nvm', work/'output.c'
            assembly.write_text(text)
            harness.NativeCallables().checked([ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
            output.write_bytes(b'prior accepted output')
            result = subprocess.run([TRANSLATOR, module, '-o', output], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(output.read_bytes(), b'prior accepted output')
