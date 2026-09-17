"""I retain source graph IDs and stable scalar execution across both producers."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/nanoisa/fixtures/passive_flow.nano'


class PassiveFlowFrontends(unittest.TestCase):
    def command(self, *args):
        return subprocess.run([str(a) for a in args], cwd=ROOT, capture_output=True,
                              text=True, timeout=120)

    def checked(self, *args):
        result = self.command(*args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_native_three_stages_and_contextual_name(self):
        with tempfile.TemporaryDirectory(prefix='nano-flow-native-') as tmp:
            for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
                output = Path(tmp) / compiler
                self.checked(ROOT/'bin'/compiler, FIXTURE, '-o', output)
                self.assertEqual(self.checked(output).stdout, '15\n')
                source = Path(tmp)/'name.nano'
                source.write_text('fn main() -> int { let flow: int = 7 return (- flow 7) }\n'
                                  'shadow main { assert (== (main) 0) }\n')
                self.checked(ROOT/'bin'/compiler, source, '-o', output)
                self.checked(output)

    def test_inferred_dependencies_and_outer_binding_shadow(self):
        sources = (
            'fn main() -> int { flow { let a = (+ b 1) let b = 6 } return (- a 7) }',
            'fn main() -> int { let base: int = 99 if true { flow { let value: int = (+ base 1) let base: int = 3 } assert (== value 4) } assert (== base 99) return 0 }',
        )
        with tempfile.TemporaryDirectory(prefix='nano-flow-types-') as tmp:
            source, output = Path(tmp)/'input.nano', Path(tmp)/'out'
            for text in sources:
                source.write_text(text+'\nshadow main { assert (== (main) 0) }\n')
                for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
                    with self.subTest(source=text, compiler=compiler):
                        self.checked(ROOT/'bin'/compiler, source, '-o', output)
                        self.checked(output)

    def test_cycles_effects_duplicates_and_ordinary_forward_reference_refuse(self):
        cases = {
            'cycle': 'flow { let a: int = b let b: int = a }',
            'self': 'flow { let a: int = a }',
            'duplicate': 'flow { let a: int = 1 let a: int = 2 }',
            'mutable': 'flow { let mut a: int = 1 }',
            'statement': 'flow { assert true }',
            'nested': 'flow { flow { let a: int = 1 } }',
            'effect': 'flow { let a: int = (side) }',
            'scalar callee shadow': 'flow { let a: int = (value) let value: int = 1 }',
            'array': 'flow { let a: array<int> = [1] }',
            'empty': 'flow { }',
            'wrong type': 'flow { let a: bool = b let b: int = 1 }',
            'ordinary forward': 'let a: int = b let b: int = 1',
        }
        with tempfile.TemporaryDirectory(prefix='nano-flow-refusals-') as tmp:
            source, output = Path(tmp)/'input.nano', Path(tmp)/'prior'
            for name, body in cases.items():
                source.write_text('fn side() -> int { (println "effect") return 1 }\n'
                                  'shadow side { assert true }\nfn value() -> int { return 7 } shadow value { assert true }\nfn main() -> int { '+body+' return 0 }\n'
                                  'shadow main { assert true }\n')
                for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2', 'nanoisa_emit'):
                    with self.subTest(case=name, compiler=compiler):
                        output.write_text('prior output')
                        result = self.command(ROOT/'bin'/compiler, source, '-o', output)
                        self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                        self.assertEqual(output.read_text(), 'prior output')

    def test_unproved_external_local_remains_refused_by_bytecode_producers(self):
        with tempfile.TemporaryDirectory(prefix='nano-flow-local-') as tmp:
            source = Path(tmp)/'local.nano'
            source.write_text('fn main() -> int { let outside: int = 7 '
                              'flow { let next: int = (+ prior 1) let prior: int = outside } return 0 }\n'
                              'shadow main { assert true }\n')
            for compiler in ('nanoisa_emit', 'nano_virt'):
                output = Path(tmp)/compiler
                output.write_text('prior output')
                result = self.command(ROOT/'bin'/compiler, source, '-o', output)
                self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                self.assertIn('guarded parameters', result.stdout+result.stderr)
                self.assertEqual(output.read_text(), 'prior output')

    def test_serial_owner_calls_execute_between_blocks_in_both_producers(self):
        with tempfile.TemporaryDirectory(prefix='nano-flow-owner-') as tmp:
            source = Path(tmp)/'owner.nano'
            source.write_text(
                'fn effect(x: int) -> int { let scratch: int = 99 (println x) return scratch }\n'
                'shadow effect { assert (== (effect 0) 99) }\n'
                'fn owner(x: int) -> int { assert (== (effect 10) 99) '
                'par { let a: int = (+ x 1) } assert (== (effect 20) 99) '
                'flow { let b: int = (+ c 3) let c: int = (+ x 2) } '
                'assert (== (effect 30) 99) assert (== x 4) '
                'assert (== a 5) assert (== c 6) return b }\n'
                'shadow owner { assert (== (owner 4) 9) }\n'
                'fn main() -> int { (println (owner 4)) return 0 }\n'
                'shadow main { assert (== (main) 0) }\n')
            assembly = Path(tmp)/'owner.nasm'
            self.checked(ROOT/'bin/nanoisa_emit', source, '-o', assembly)
            for producer in ('seed', 'self'):
                module = Path(tmp)/(producer+'.nvm')
                if producer == 'seed':
                    self.checked(ROOT/'bin/nano_virt', source, '--emit-nvm', '-o', module)
                else:
                    self.checked(ROOT/'bin/nanoisa', 'asm', assembly, '-o', module)
                self.checked(ROOT/'bin/nano_vm', '--verify-only', module)
                self.assertEqual(self.checked(ROOT/'bin/nano_vm', module).stdout, '10\n20\n30\n9\n')
                generated, native = Path(tmp)/(producer+'.c'), Path(tmp)/producer
                self.checked(ROOT/'bin/nvm2c', module, '-o', generated)
                self.checked('cc', '-std=c11', '-Wall', '-Wextra', '-Werror', generated, '-lm', '-o', native)
                self.assertEqual(self.checked(native).stdout, '10\n20\n30\n9\n')

    def test_bound_module_owners_and_forward_dependencies(self):
        with tempfile.TemporaryDirectory(prefix='nano-flow-owners-') as tmp:
            directory = Path(tmp)
            for name, value in (('left', 37), ('right', 12)):
                (directory/(name+'.nano')).write_text(
                    f'fn base() -> int {{ return {value} }}\nshadow base {{ assert (== (base) {value}) }}\n'
                    f'pub fn value() -> int {{ return (base) }}\nshadow value {{ assert (== (value) {value}) }}\n')
            source = directory/'main.nano'
            source.write_text('module "left.nano" as left\nmodule "right.nano" as right\n'
                              'fn main() -> int { flow { let total: int = (+ a b) '
                              'let a: int = (left.value) let b: int = (right.value) } '
                              '(println total) return 0 }\nshadow main { assert (== (main) 0) }\n')
            for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
                native = directory/compiler
                self.checked(ROOT/'bin'/compiler, source, '-o', native)
                self.assertEqual(self.checked(native).stdout, '49\n')
                if compiler != 'nanoc_c':
                    module = directory/(compiler+'.nvm')
                    self.checked(ROOT/'bin'/compiler, source, '--emit-nvm', '-o', module)
                    self.checked(ROOT/'bin/nano_vm', '--verify-only', module)
                    self.assertEqual(self.checked(ROOT/'bin/nano_vm', module).stdout, '49\n')


if __name__ == '__main__':
    unittest.main()
