"""I execute concrete generic bodies with mandatory source shadows."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get("NANO_GENERIC_CSEED_COMPILER", ROOT / "bin/nano_virt"))


class CseedGenericFunctions(unittest.TestCase):
    def run_command(self, args):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def execute(self, source, files=None):
        with tempfile.TemporaryDirectory(prefix='nano-generic-') as tmp:
            work = Path(tmp)
            for name, contents in (files or {}).items():
                (work / name).write_text(contents)
            path, module, c, binary = [work / name for name in ('main.nano', 'main.nvm', 'main.c', 'main')]
            path.write_text(source)
            self.run_command([COMPILER, path, '--emit-nvm', '-o', module])
            self.run_command([ROOT / 'bin/nano_vm', '--verify-only', module])
            self.run_command([ROOT / 'bin/nano_vm', module])
            self.run_command([ROOT / 'bin/nvm2c', module, '-o', c])
            cc = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc'))
            self.run_command(cc + ['-std=c11', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                c, ROOT / 'bin/nano_aot_runtime.o', '-lm', '-o', binary])
            self.run_command([binary])

    def test_builtin_tail_call_does_not_enter_generic_specialization(self):
        self.execute('fn size(text: string) -> int { return (str_length text) }\nshadow size { assert (== (size "abc") 3) }\nfn main() -> int { assert (== (size "four") 4) return 0 }\nshadow main { assert (== (main) 0) }\n')

    def test_structural_array_parameters(self):
        self.execute((ROOT/'tests/nanoisa/fixtures/structural_generic_arrays.nano.txt').read_text())

    def test_structural_callable_parameters(self):
        self.execute((ROOT/'tests/nanoisa/fixtures/structural_generic_callable.nano.txt').read_text())

    def test_structural_array_minimum(self):
        self.execute((ROOT/'docs/evidence/structural-generic-baseline-20261010/array-parameter.nano').read_text())

    def test_imported_structural_generic_parameters(self):
        self.execute('module "a.nano" as a\nfrom "a.nano" import first as head\n'
            'fn wrapped(values:array<int>)->int{return (a.first values)}\n'
            'shadow wrapped {assert (== (wrapped [3]) 3)}\n'
            'fn main()->int{assert (== (wrapped [7]) 7) '
            'assert (== (head [true]) true) assert (== (at (a.first [[9]]) 0) 9) return 0}\n'
            'shadow main {assert (== (main) 0)}\n', files={'a.nano':
            'pub fn first(values:array<T>)->T{return (at values 0)}\n'
            'shadow first {assert (== (first [1]) 1)}\n'})

    def test_structural_generic_refusals_preserve_output(self):
        bodies = ['return (combine [1] true)', 'return (combine true 1)',
                  'return (combine [1] [1])', 'return (apply 1 positive)']
        for body in bodies:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as tmp:
                path, module = Path(tmp)/'main.nano', Path(tmp)/'main.nvm'
                path.write_text('fn combine(values:array<T>,value:T)->T{return value}\n'
                    'shadow combine {assert (== (combine [1] 2) 2)}\n'
                    'fn positive(value:int)->bool{return (> value 0)}\nshadow positive {assert (positive 1)}\n'
                    'fn apply(value:T,callback:fn(T)->T)->T{return (callback value)}\n'
                    'shadow apply {assert true}\nfn main()->int{'+body+'}\n'
                    'shadow main {assert (== (main) 0)}\n')
                module.write_bytes(b'prior-output')
                result = subprocess.run([str(COMPILER),str(path),'--emit-nvm','-o',str(module)],
                    cwd=ROOT,capture_output=True,text=True,timeout=120)
                self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('one concrete identity',result.stdout+result.stderr)
                self.assertEqual(module.read_bytes(),b'prior-output')

    def test_structural_callable_local_and_nominal_letter(self):
        self.execute('struct A {value:int}\n'
            'fn fixed(value:A)->A{return value}\n'
            'shadow fixed {assert (== (fixed A {value:3}).value 3)}\n'
            'fn positive(value:int)->bool{return (> value 0)}\nshadow positive {assert (positive 1)}\n'
            'fn apply(value:T,callback:fn(T)->E)->E{let saved:fn(T)->E=callback return (saved value)}\n'
            'shadow apply {assert (apply 1 positive)}\n'
            'fn main()->int{assert (apply 2 positive) assert (== (fixed A {value:7}).value 7) return 0}\n'
            'shadow main {assert (== (main) 0)}\n')

    def test_unbound_structural_result_preserves_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, module = Path(tmp)/'main.nano', Path(tmp)/'main.nvm'
            path.write_text('fn unbound(value:T)->array<E>{return []}\nshadow unbound {assert true}\n'
                'fn main()->int{let values:array<int> = (unbound 1) return (array_length values)}\n'
                'shadow main {assert (== (main) 0)}\n')
            module.write_bytes(b'prior-output')
            result = subprocess.run([str(COMPILER),str(path),'--emit-nvm','-o',str(module)],
                cwd=ROOT,capture_output=True,text=True,timeout=120)
            self.assertGreater(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('one concrete identity',result.stdout+result.stderr)
            self.assertEqual(module.read_bytes(),b'prior-output')

    def test_retained_record_result_regression(self):
        self.execute((ROOT / 'docs/evidence/generic-results-global-audit-20261010/generic-producer-repro.nano').read_text())

    def test_primitive_generic_fixture(self):
        self.execute((ROOT / 'tests/unit/test_generics.nano').read_text())

    def test_typed_locals_transitive_calls_and_recursion(self):
        self.execute('''struct Box { value: int }
fn identity(value: T) -> T { let copy: T = value return copy }
shadow identity { assert (== (identity 7) 7) }
fn forward(value: T) -> T { let copy: T = (identity value) return (identity copy) }
shadow forward { assert (== (forward true) true) }
fn repeat(value: T, count: int) -> T {
 if (== count 0) { return value }
 return (repeat (forward value) (- count 1))
}
shadow repeat { assert (== (repeat 5 3) 5) }
fn main() -> int {
 assert (== (repeat 17 5000) 17)
 assert (== (repeat 2.5 3) 2.5)
 assert (== (forward false) false)
 assert (== (repeat "text" 2) "text")
 let record: Box = (repeat Box { value: 29 } 3)
 assert (== record.value 29)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_distinct_record_specializations(self):
        self.execute('''struct First { value: int }
struct Second { text: string }
fn identity(value: T) -> T { let copy: T = value return copy }
shadow identity { assert (== (identity 7) 7) }
fn main() -> int {
 let first: First = (identity First { value: 31 })
 let second: Second = (identity Second { text: "second" })
 assert (== first.value 31)
 assert (== second.text "second")
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_repeated_type_variable_rejects_distinct_records(self):
        with tempfile.TemporaryDirectory(prefix='nano-generic-refusal-') as tmp:
            path, module = Path(tmp) / 'main.nano', Path(tmp) / 'main.nvm'
            path.write_text('''struct First { value: int }
struct Second { value: int }
fn choose(a: T, b: T) -> T { return a }
shadow choose { assert (== (choose 1 2) 1) }
fn main() -> int {
 let result: First = (choose First { value: 1 } Second { value: 2 })
 return result.value
}
shadow main { assert true }
''')
            module.write_bytes(b'prior-output')
            result = subprocess.run([str(COMPILER), str(path),
                '--emit-nvm', '-o', str(module)], cwd=ROOT, capture_output=True,
                text=True, timeout=120)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('one concrete identity', result.stdout + result.stderr)
            self.assertEqual(module.read_bytes(), b'prior-output')

    def test_array_binding_parity(self):
        self.execute((ROOT / 'docs/evidence/generic-specialization-budget-20261010/array-parity.nano').read_text())

    def test_nested_array_calls_and_record_elements(self):
        self.execute('''struct First { value:int }
struct Second { text:string }
fn identity(value:T)->T{let copy:T = value return copy}
shadow identity {assert (== (identity 1) 1)}
fn relay(value:T)->T{return (identity value)}
shadow relay {assert (== (relay true) true)}
fn make()->array<float>{return [1.25,2.5]}
shadow make {assert (== (at (make) 1) 2.5)}
fn main()->int{
 let nested:array<array<int>> = (relay (identity [[3,5],[7,9]]))
 assert (== (at (at nested 1) 0) 7)
 assert (== (at (at (identity [[11,13]]) 0) 1) 13)
 let records:array<First> = (identity [First {value:17}])
 let others:array<Second> = (relay [Second {text:"other"}])
 assert (== (at records 0).value 17)
 assert (== (at others 0).text "other")
 let floats:array<float> = (identity (make))
 (array_set floats 0 3.5)
 assert (== (at floats 0) 3.5)
 let strings:array<string> = (identity ["a","b"])
 assert (== (at strings 1) "b")
 return 0
}
shadow main {assert (== (main) 0)}
''')

    def test_generic_array_mutation_preserves_aliases_and_empty_types(self):
        self.execute('''fn identity(value:T)->T{let copy:T = value return copy}
shadow identity {assert (== (identity 1) 1)}
fn main()->int{
 let values:array<int> = [1,2]
 let alias:array<int> = (identity values)
 (array_set alias 0 9)
 assert (== (at values 0) 9)
 let empty:array<bool> = []
 let copied:array<bool> = (identity empty)
 assert (== (array_length copied) 0)
 return 0
}
shadow main {assert (== (main) 0)}
''')

    def test_repeated_generic_array_types_must_match(self):
        for left, right, result_type in [('[1]', '[true]', 'array<int>'),
                ('[[1]]', '[[true]]', 'array<array<int>>'),
                ('[First {value:1}]', '[Second {value:2}]', 'array<First>')]:
            with self.subTest(left=left, right=right), tempfile.TemporaryDirectory() as tmp:
                source, module = Path(tmp)/'main.nano', Path(tmp)/'main.nvm'
                source.write_text('struct First {value:int}\nstruct Second {value:int}\n'
                    'fn choose(a:T,b:T)->T{return a}\nshadow choose {assert (== (choose 1 2) 1)}\n'
                    'fn main()->int{let result:' + result_type + ' = (choose ' + left + ' ' + right + ') return 0}\n'
                    'shadow main {assert (== (main) 0)}\n')
                module.write_bytes(b'prior-output')
                result = subprocess.run([str(COMPILER), str(source), '--emit-nvm', '-o', str(module)],
                    cwd=ROOT, capture_output=True, text=True, timeout=120)
                self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                self.assertIn('one concrete identity', result.stdout+result.stderr)
                self.assertEqual(module.read_bytes(), b'prior-output')

    def test_generic_array_contexts_reject_wrong_elements(self):
        for body in ['let flags:array<bool> = (identity [1]) return 0',
                     '(consume (identity [1])) return 0',
                     'let value:array<int> = (fixed [1] true) return 0']:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as tmp:
                path, module = Path(tmp)/'main.nano', Path(tmp)/'main.nvm'
                path.write_text('fn identity(value:T)->T{return value}\n'
                    'shadow identity {assert (== (identity 1) 1)}\n'
                    'fn consume(value:array<bool>)->void{}\nshadow consume {(consume [true])}\n'
                    'fn fixed(value:T, count:int)->T{return value}\nshadow fixed {assert (== (fixed 1 0) 1)}\n'
                    'fn main()->int{' + body + '}\nshadow main {assert (== (main) 0)}\n')
                module.write_bytes(b'prior-output')
                result = subprocess.run([str(COMPILER),str(path),'--emit-nvm','-o',str(module)],
                    cwd=ROOT,capture_output=True,text=True,timeout=120)
                self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertEqual(module.read_bytes(),b'prior-output')

    def test_growing_array_specializations_preserve_output(self):
        source = ROOT/'docs/evidence/generic-specialization-budget-20261010/growing.nano'
        with tempfile.TemporaryDirectory() as tmp:
            module = Path(tmp)/'prior.nvm'
            module.write_bytes(b'prior-output')
            result = subprocess.run([str(COMPILER),str(source),'--emit-nvm','-o',str(module)],
                cwd=ROOT,capture_output=True,text=True,timeout=120)
            self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('generic specialization budget',result.stdout+result.stderr)
            self.assertEqual(module.read_bytes(),b'prior-output')

    def test_imported_generic_array_results_keep_complete_types(self):
        self.execute('''module "a.nano" as source
from "a.nano" import identity as copied
struct Item {value:int}
fn identity(value:T)->T{return value}
shadow identity {assert (== (identity 7) 7)}
fn main()->int{
 let nested:array<array<int>> = (identity (source.identity [[3,5]]))
 assert (== (at (at nested 0) 1) 5)
 let records:array<Item> = (source.identity [Item {value:19}])
 assert (== (at records 0).value 19)
 let flags:array<bool> = (copied [true,false])
 assert (== (at flags 1) false)
 return 0
}
shadow main {assert (== (main) 0)}
''', {'a.nano':'pub fn identity(value:T)->T{let copy:T = value return copy}\nshadow identity {assert (== (identity 1) 1)}\n'})

    def test_imported_qualified_and_selective_owners(self):
        self.execute('''module "a.nano" as first
from "b.nano" import choose as second
fn main() -> int {
 assert (== (first.choose 11 12) 11)
 assert (== (second 11 12) 12)
 assert (== (first.choose "left" "right") "left")
 return 0
}
shadow main { assert (== (main) 0) }
''', {'a.nano': 'pub fn choose(a: T, b: T) -> T { return a }\nshadow choose { assert (== (choose 1 2) 1) }\n',
      'b.nano': 'pub fn choose(a: T, b: T) -> T { return b }\nshadow choose { assert (== (choose 1 2) 2) }\n'})


if __name__ == '__main__':
    unittest.main()
