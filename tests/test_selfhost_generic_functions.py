"""I check source generics and execute the self-hosted shadow emitter's modules."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from tests import test_cseed_generic_functions as cases

ROOT = Path(__file__).resolve().parents[1]


class SelfhostGenericFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory(prefix='nano-generic-drivers-')
        cls.drivers = {}
        for name, fixture in [('checker', 'generic_checker_driver'), ('emitter', 'generic_emitter_driver')]:
            override = os.environ.get('NANO_GENERIC_' + name.upper() + '_DRIVER')
            if override:
                cls.drivers[name] = Path(override)
            else:
                source = Path(cls.scratch.name) / (name + '.nano')
                source.write_text((ROOT / ('tests/nanoisa/fixtures/' + fixture + '.nano.txt')).read_text())
                binary = Path(cls.scratch.name) / name
                result = subprocess.run([str(ROOT / 'bin/nanoc_c'), str(source), '-o', str(binary)],
                    cwd=ROOT, capture_output=True, text=True, timeout=900)
                if result.returncode:
                    raise AssertionError(result.stdout + result.stderr)
                cls.drivers[name] = binary

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def command(self, args):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def execute(self, source, files=None):
        self.assertFalse(files, 'I require the product compiler for module-file loading.')
        with tempfile.TemporaryDirectory(prefix='nano-selfhost-generic-') as tmp:
            work = Path(tmp)
            path, assembly, module, c, binary = [work / name for name in
                ('input.nano', 'input.nasm', 'input.nvm', 'input.c', 'input')]
            path.write_text(source)
            self.command([self.drivers['checker'], path])
            for mode in ('whole', 'program', 'raw'):
                assembly.write_text(self.command([self.drivers['emitter'], path, '0', mode]).stdout)
                self.command([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
                self.command([ROOT / 'bin/nano_vm', '--verify-only', module])
                self.command([ROOT / 'bin/nano_vm', module])
                self.command([ROOT / 'bin/nvm2c', module, '-o', c])
                cc = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc'))
                self.command(cc + ['-std=c11', '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                    c, ROOT / 'bin/nano_aot_runtime.o', '-lm', '-o', binary])
                self.command([binary])

    test_retained_record_result_regression = cases.CseedGenericFunctions.test_retained_record_result_regression
    test_primitive_generic_fixture = cases.CseedGenericFunctions.test_primitive_generic_fixture
    test_typed_locals_transitive_calls_and_recursion = cases.CseedGenericFunctions.test_typed_locals_transitive_calls_and_recursion
    test_distinct_record_specializations = cases.CseedGenericFunctions.test_distinct_record_specializations
    test_nested_array_calls_and_record_elements = cases.CseedGenericFunctions.test_nested_array_calls_and_record_elements
    test_generic_array_mutation_preserves_aliases_and_empty_types = cases.CseedGenericFunctions.test_generic_array_mutation_preserves_aliases_and_empty_types

    def test_generated_names_do_not_collide(self):
        self.execute('''fn __nanoisa_generic_0() -> int { return 13 }
shadow __nanoisa_generic_0 { assert (== (__nanoisa_generic_0) 13) }
fn identity(value:T) -> T { return value }
shadow identity { assert (== (identity 7) 7) }
fn main() -> int {
 assert (== (identity 19) 19)
 assert (== (__nanoisa_generic_0) 13)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_declared_single_letter_record_is_nominal(self):
        self.execute('''struct T { value: int }
fn identity(value:T) -> T { return value }
shadow identity { assert (== (identity T { value: 7 }).value 7) }
fn main() -> int {
 let value:T = (identity T { value: 19 })
 assert (== value.value 19)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_direct_array_type_binding(self):
        self.execute('''fn identity(value:T) -> T { let copy:T = value return copy }
shadow identity { assert (== (identity 7) 7) }
fn main() -> int {
 let ints:array<int> = (identity [3,5])
 let bools:array<bool> = (identity [true,false])
 assert (== (at ints 1) 5)
 assert (== (at bools 1) false)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_finite_polymorphic_recursion(self):
        self.execute('''fn visit(value:T, remaining:int) -> int {
 if (> remaining 0) { return (visit false (- remaining 1)) }
 return remaining
}
shadow visit { assert (== (visit 13 3) 0) }
fn main() -> int { return (visit "start" 5000) }
shadow main { assert (== (main) 0) }
''')

    def test_many_independent_bindings_are_not_recursive_depth(self):
        records = ''.join('struct Record' + str(i) + ' { value:int }\n' for i in range(70))
        checks = ''.join('assert (== (identity Record' + str(i) + ' { value:' + str(i) + ' }).value ' + str(i) + ')\n' for i in range(70))
        self.execute(records + 'fn identity(value:T)->T{return value}\n'
            'shadow identity {assert (== (identity 1) 1)}\n'
            'fn main()->int{\n' + checks + 'return 0}\n'
            'shadow main {assert (== (main) 0)}\n')

    def test_growing_polymorphic_recursion_is_bounded(self):
        source = '''fn grow(value:T, remaining:int) -> void {
 if (> remaining 0) { (grow [value] (- remaining 1)) }
}
shadow grow { (grow 1 0) }
fn main() -> int { (grow 1 0) return 0 }
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix='nano-generic-growth-') as tmp:
            path = Path(tmp) / 'input.nano'
            mutual = source.replace('(grow [value]', '(relay [value]') + (
                'fn relay(value:U, remaining:int)->void{(grow value remaining)}\n'
                'shadow relay {(relay 1 0)}\n')
            for name, text in [('direct', source), ('mutual', mutual)]:
                path.write_text(text)
                self.command([self.drivers['checker'], path])
                for mode in ('whole', 'program', 'raw'):
                    with self.subTest(recursion=name, mode=mode):
                        result = subprocess.run([str(self.drivers['emitter']), str(path), '0', mode],
                            cwd=ROOT, capture_output=True, text=True, timeout=60)
                        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                        self.assertIn('generic specialization budget', result.stdout + result.stderr)
                        self.assertNotIn('.function ', result.stdout)

    def test_repeated_type_variable_refusals(self):
        for expected, arguments in [('int', '1 true'), ('First', 'First { value: 1 } Second { value: 2 }')]:
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'input.nano'
                path.write_text('struct First { value:int }\nstruct Second { value:int }\n'
                    'fn choose(a:T,b:T)->T{return a}\nshadow choose { assert (== (choose 1 2) 1) }\n'
                    'fn main()->int{let result:' + expected + ' = (choose ' + arguments + ') return 0}\n'
                    'shadow main {assert (== (main) 0)}\n')
                for command in ([self.drivers['checker'], path],
                                [self.drivers['emitter'], path, '0', 'raw']):
                    result = subprocess.run(list(map(str, command)), cwd=ROOT,
                        capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    if len(command) == 4:
                        self.assertIn('one concrete identity', result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
