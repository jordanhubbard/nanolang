"""I exercise exact scalar union source construction and scoped statement matches."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
NVM2C = Path(os.environ.get('NVM2C', str(ROOT/'bin/nvm2c')))


class ScalarUnionEmission(unittest.TestCase):
    @staticmethod
    def command(*args, expected=0, timeout=180):
        result = subprocess.run([str(a) for a in args], cwd=ROOT, text=True,
                                capture_output=True, timeout=timeout)
        if result.returncode != expected:
            raise AssertionError(f'{args}: {result.returncode}\n{result.stdout}\n{result.stderr}')
        return result

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='scalar-unions-')
        cls.work = Path(cls.temp.name)
        cls.raw = [ROOT/'bin/nanoisa_emit', cls.work/'stage2-emit']
        # I build a compiler here; canonical qualification measured 440.639s.
        # Ordinary program checks retain their separate 180-second deadline.
        cls.command(ROOT/'bin/nanoc_stage2', ROOT/'src_nano/nanoisa_emit.nano', '-o', cls.raw[1], timeout=900)
        shadow_source = cls.work/'shadows.nano'
        shadow_source.write_text((ROOT/'tests/nanoisa/fixtures/shadow_module_driver.nano.txt').read_text())
        cls.shadows = cls.work/'shadows'
        cls.command(ROOT/'bin/nanoc_c', shadow_source, '-o', cls.shadows)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def execute(self, module, expected=''):
        self.command(ROOT/'bin/nano_vm', '--verify-only', module)
        vm = self.command(ROOT/'bin/nano_vm', module)
        source, binary = self.work/'native.c', self.work/'native'
        self.command(NVM2C, module, '-o', source)
        self.command(os.environ.get('CC', 'cc'), '-std=c11', '-Wall', '-Wextra', '-Werror',
                     '-fsanitize=address,undefined', '-fno-omit-frame-pointer', source, '-lm', '-o', binary)
        native = self.command(binary)
        self.assertEqual(vm.stdout, expected)
        self.assertEqual(native.stdout, expected)

    def test_unchanged_core_examples_and_selected_shadows(self):
        for name in ('nl_control_flow', 'nl_control_match'):
            fixture = ROOT/'tests'/f'{name}.nano'
            expected = (f'{name}: All control-flow tests passed!\n' if name == 'nl_control_flow'
                        else f'{name}: All match arm binding tests passed!\n')
            for compiler in ('nanoc_stage1', 'nanoc_stage2'):
                with self.subTest(fixture=name, compiler=compiler):
                    module = self.work/f'{name}-{compiler}.nvm'
                    self.command(ROOT/'bin'/compiler, fixture, '--emit-nvm', '-o', module)
                    self.execute(module, expected)
            assembly, module = self.work/'shadows.nasm', self.work/'shadows.nvm'
            assembly.write_text(self.command(self.shadows, fixture, '0', 'raw').stdout)
            self.command(ROOT/'bin/nanoisa', 'asm', assembly, '-o', module)
            self.execute(module, expected)

    def test_scalar_payloads_calls_returns_and_lexical_scope(self):
        source = self.work/'values.nano'
        source.write_text('''union Value { Data { number: int, text: string, flag: bool, real: float }, Empty {} }
fn fresh() -> Value { return Value.Data { number: 7, text: "yes", flag: true, real: 1.5 } }
shadow fresh { let v: Value = (fresh) assert (== (read v) 7) }
fn read(v: Value) -> int {
 let result: int = 99
 match v {
  Data(result) => { assert (== result.text "yes") assert result.flag assert (== result.real 1.5) return result.number }
  Empty(empty) => { assert true }
 }
 return result
}
shadow read { assert (== (read (fresh)) 7) assert (== (read Value.Empty {}) 99) }
fn main() -> int { assert (== (read (fresh)) 7) assert (== (read Value.Empty {}) 99) return 0 }
shadow main { assert (== (main) 0) }
''')
        for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
            module = self.work/f'{compiler}.nvm'
            result = self.command(ROOT/'bin'/compiler, source, '--emit-nvm', '-o', module)
            self.assertNotIn('E001 TYPE MISMATCH', result.stderr)
            self.execute(module)
        for emitter in self.raw:
            assembly, module = self.work/'raw.nasm', self.work/'raw.nvm'
            self.command(emitter, source, '-o', assembly)
            self.command(ROOT/'bin/nanoisa', 'asm', assembly, '-o', module)
            self.execute(module)

    def test_refusals_preserve_output(self):
        prefix = 'union One { Pair { first: int, second: int }, Empty {} } union Two { Pair { first: int, second: int }, Empty {} } '
        cases = {
            'duplicate': 'let value: One = One.Pair { first: 1, first: 2 }',
            'duplicate_declaration': 'let value: Bad = Bad.Item { same: 1, same: 2 }',
            'missing': 'let value: One = One.Pair { first: 1 }',
            'unknown': 'let value: One = One.Pair { first: 1, extra: 2 }',
            'wrong_field_type': 'let value: One = One.Pair { first: true, second: 2 }',
            'wrong_nominal': 'let value: One = Two.Pair { first: 1, second: 2 }',
            'wrong_call': 'let value: int = (accept Two.Empty {})',
            'missing_arm': 'let value: One = One.Empty {} match value { Empty(e) => { assert true } }',
            'unknown_arm': 'let value: One = One.Empty {} match value { Empty(e) => { assert true } Other(o) => { assert true } }',
            'escaped_binding': 'let value: One = One.Empty {} match value { Pair(p) => { assert true } Empty(e) => { assert true } } let x: int = p.first',
        }
        source, output = self.work/'bad.nano', self.work/'keep.nasm'
        for name, body in cases.items():
            extra = 'union Bad { Item { same: int, same: int } } ' if name == 'duplicate_declaration' else ''
            source.write_text(prefix+extra+'fn accept(value: One) -> int { return 0 } shadow accept { assert true } fn main() -> int { '+body+' return 0 } shadow main { assert true }')
            for emitter in self.raw:
                with self.subTest(case=name, emitter=emitter.name):
                    output.write_text('retained')
                    result = subprocess.run([emitter, source, '-o', output], cwd=ROOT,
                                            text=True, capture_output=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertEqual(output.read_text(), 'retained')
            for compiler in ('nanoc_stage1', 'nanoc_stage2'):
                with self.subTest(case=name, compiler=compiler):
                    output.write_text('retained')
                    result = subprocess.run([ROOT/'bin'/compiler, source, '--emit-nvm', '-o', output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertEqual(output.read_text(), 'retained')

    def test_reordered_constructor_evaluates_once_in_source_order(self):
        source = self.work/'order.nano'
        source.write_text('''union Pair { Item { first: int, second: int } }
fn mark(value: int) -> int { (println value) return value }
shadow mark { assert (== (mark 1) 1) }
fn check(value: Pair) -> int { match value { Item(p) => { assert (== p.first 3) assert (== p.second 8) return 0 } } }
shadow check { assert (== (check Pair.Item { first: 3, second: 8 }) 0) }
fn main() -> int { let pair: Pair = Pair.Item { second: (mark 8), first: (mark 3) } return (check pair) }
shadow main { assert true }
''')
        for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
            module = self.work/'order.nvm'
            self.command(ROOT/'bin'/compiler, source, '--emit-nvm', '-o', module)
            self.execute(module, '8\n3\n')
        for emitter in self.raw:
            assembly, module = self.work/'order.nasm', self.work/'order.nvm'
            self.command(emitter, source, '-o', assembly)
            self.command(ROOT/'bin/nanoisa', 'asm', assembly, '-o', module)
            self.execute(module, '8\n3\n')

    def test_unsupported_payloads_and_failed_shadows_refuse(self):
        cases = {
            'resource': 'resource struct Item { value: int } union Box { Some { value: Item } } fn main() -> int { let x: Box = Box.Some { value: Item { value: 1 } } return 0 } shadow main { assert true }',
            'wrong_nested': 'union Inner { Empty {} } union Other { Empty {} } union Box { Some { value: Inner } } fn main() -> int { let x: Box = Box.Some { value: Other.Empty {} } return 0 } shadow main { assert true }',
            'failed_shadow': 'union Choice { Empty {} } fn main() -> int { let x: Choice = Choice.Empty {} return 0 } shadow main { assert false }',
            'wrong_return': 'union One { Empty {} } union Two { Empty {} } fn wrong() -> One { return Two.Empty {} } shadow wrong { assert true } fn main() -> int { let x: One = (wrong) return 0 } shadow main { assert true }',
        }
        for name, text in cases.items():
            source, output = self.work/'reject.nano', self.work/'keep.nvm'
            source.write_text(text)
            for compiler in ('nanoc_stage1', 'nanoc_stage2'):
                with self.subTest(case=name, compiler=compiler):
                    output.write_text('retained')
                    result = subprocess.run([ROOT/'bin'/compiler, source, '--emit-nvm', '-o', output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertEqual(output.read_text(), 'retained')

    def test_c_seed_resource_payload_constructor_remains_checked(self):
        source = self.work/'resource.nano'
        source.write_text((ROOT/'tests/nanoisa/fixtures/union_resource_constructor_order.nano').read_text())
        module = self.work/'resource.nvm'
        result = self.command(ROOT/'bin/nano_virt', source, '--emit-nvm', '-o', module)
        self.assertNotIn('E001 TYPE MISMATCH', result.stderr)
        self.execute(module)

    def test_nominal_payload_offsets_do_not_follow_shared_field_names(self):
        source = self.work/'nominal.nano'
        source.write_text('''union First { Item { value: int, other: int } }
union Second { Item { other: int, value: int } }
fn read_first(value: First) -> int { match value { Item(p) => { return p.value } } }
shadow read_first { assert (== (read_first First.Item { value: 3, other: 8 }) 3) }
fn read_second(value: Second) -> int { match value { Item(p) => { return p.value } } }
shadow read_second { assert (== (read_second Second.Item { other: 3, value: 8 }) 8) }
fn main() -> int { assert (== (read_first First.Item { value: 3, other: 8 }) 3) assert (== (read_second Second.Item { other: 3, value: 8 }) 8) return 0 }
shadow main { assert (== (main) 0) }
''')
        for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
            module = self.work/'nominal.nvm'
            result = self.command(ROOT/'bin'/compiler, source, '--emit-nvm', '-o', module)
            self.assertNotIn('E001 TYPE MISMATCH', result.stderr)
            self.execute(module)
