"""I admit a closed source-reference profile only with executable ownership."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/nanoisa/fixtures'


def native_test_compiler():
    return os.environ.get('NANO_NATIVE_TEST_CC') or os.environ.get('CC', 'cc')


class NativeTestCompilerSelection(unittest.TestCase):
    def test_native_override_does_not_replace_driver_environment(self):
        with mock.patch.dict(os.environ, {'CC': 'driver-cc'}, clear=True):
            self.assertEqual(native_test_compiler(), 'driver-cc')
        with mock.patch.dict(os.environ, {
            'CC': 'driver-cc',
            'NANO_NATIVE_TEST_CC': 'native-cc',
        }, clear=True):
            self.assertEqual(native_test_compiler(), 'native-cc')
            self.assertEqual(os.environ['CC'], 'driver-cc')
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(native_test_compiler(), 'cc')


class SourceBorrowEmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='nano-source-borrows-')
        cls.work = Path(cls.temporary.name)
        cls.emitters = [ROOT / 'bin/nanoisa_emit']
        cls.shadow_tools = []
        shadow_source = cls.work / 'shadow_driver.nano'
        shadow_source.write_text((FIXTURES / 'shadow_module_driver.nano.txt').read_text())
        for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            if compiler != 'nanoc_c':
                emitter = cls.work / (compiler + '-emit')
                cls.command(ROOT / 'bin' / compiler, ROOT / 'src_nano/nanoisa_emit.nano', '-o', emitter, timeout=900)
                cls.emitters.append(emitter)
            shadow_tool = cls.work / (compiler + '-shadows')
            cls.command(ROOT / 'bin' / compiler, shadow_source, '-o', shadow_tool, timeout=900)
            cls.shadow_tools.append(shadow_tool)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def command(*args, expected=0, timeout=180):
        result = subprocess.run([str(arg) for arg in args], cwd=ROOT, capture_output=True,
                                text=True, timeout=timeout)
        if result.returncode != expected:
            raise AssertionError(f'{args}: {result.returncode}\n{result.stdout}\n{result.stderr}')
        return result

    def execute_pair(self, module, expected=0, expected_output=None):
        self.command(ROOT / 'bin/nano_vm', '--verify-only', module)
        vm = subprocess.run([ROOT / 'bin/nano_vm', module], cwd=ROOT, capture_output=True, timeout=180)
        self.assertEqual(vm.returncode, expected, vm.stdout + vm.stderr)
        if expected_output is not None:
            self.assertEqual(vm.stdout, expected_output)
        source, native = self.work / 'native.c', self.work / 'native'
        self.command(ROOT / 'bin/nvm2c', module, '-o', source)
        self.command(native_test_compiler(), '-std=c11', '-Wall', '-Wextra', '-Werror',
                     '-fsanitize=address,undefined', '-fno-omit-frame-pointer', source, '-o', native)
        result = subprocess.run([native], cwd=ROOT, capture_output=True, timeout=30,
                                env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1'})
        # My standalone wrapper maps internal assertion status 2 to exit status 1.
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        self.assertNotIn(b'Sanitizer', result.stderr)
        if expected_output is not None:
            self.assertEqual(result.stdout, expected_output)
        if expected == 0:
            self.assertEqual(vm.stdout, result.stdout)

    def test_both_producers_and_native_stages_preserve_exact_contracts(self):
        for fixture in ('source_borrow_shared.nano', 'source_borrow_exclusive.nano',
                        'source_borrow_order.nano', 'source_borrow_bool.nano'):
            source = FIXTURES / fixture
            seed = self.work / 'seed.nvm'
            self.command(ROOT / 'bin/nano_virt', source, '--emit-nvm', '--strip-debug', '-o', seed)
            baseline = self.command(ROOT / 'bin/nanoisa', 'dump', seed).stdout
            self.assertIn('CALL_REF', baseline)
            self.assertIn('OWN_UNPACK_LOCAL', baseline)
            self.assertIn('.ownership', baseline)
            self.assertIn('.layouts', baseline)
            self.execute_pair(seed)
            for emitter in self.emitters:
                with self.subTest(fixture=fixture, emitter=emitter.name):
                    assembly, module = self.work / 'self.nasm', self.work / 'self.nvm'
                    self.command(emitter, source, '-o', assembly)
                    self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                    actual = self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout
                    self.assertEqual(actual, baseline)
                    self.execute_pair(module)
            for compiler in ('nanoc_stage1', 'nanoc_stage2'):
                module = self.work / (compiler + '.nvm')
                self.command(ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', module)
                self.execute_pair(module)

    def names_and_strip(self, module, expected_output=None):
        # My existing codec probe checks every PC boundary, metadata and exact
        # wire preservation through two canonical text cycles.
        records = self.command(ROOT / 'obj/test_local_bindings', module).stdout.splitlines()
        text = self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout
        self.assertIn('.metadata', text)
        stripped_text = '\n'.join(line for line in text.splitlines()
                                  if not line.startswith('.metadata')) + '\n'
        assembly, stripped = self.work / 'without-names.nasm', self.work / 'without-names.nvm'
        assembly.write_text(stripped_text)
        self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', stripped)
        self.assertEqual([line for line in self.command(ROOT / 'bin/nanoisa', 'dump', stripped).stdout.splitlines() if line],
                         [line for line in stripped_text.splitlines() if line])
        self.execute_pair(module, expected_output=expected_output)
        self.execute_pair(stripped, expected_output=expected_output)
        return [tuple(line.split()) for line in records]

    def test_borrowed_names_and_stripped_execution(self):
        for fixture in ('source_borrow_shared.nano', 'source_borrow_exclusive.nano'):
            source = self.work / ('names-' + fixture)
            text = (FIXTURES / fixture).read_text()
            text = text.replace('return view.value', 'let observed: int = view.value return observed')
            source.write_text(text)
            baseline = None
            for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
                module = self.work / (compiler + '-names.nvm')
                self.command(ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', module)
                records = self.names_and_strip(module)
                if baseline is None:
                    baseline = records
                self.assertEqual(records, baseline)
                self.assertFalse(any('__' in row[4] for row in records))
                helper = [row for row in records if row[0] != 'main']
                self.assertEqual(helper[0][1:3], ('0', '0'))
                self.assertEqual(helper[1][4], 'observed')
                self.assertGreater(int(helper[1][2]), 0)
                owners = [row for row in records if row[0] == 'main' and row[4] == 'owner']
                self.assertEqual(len(owners), 1)
                self.assertGreater(int(owners[0][2]), 0)
                self.assertEqual(int(owners[0][1]), 2)  # two unnamed field temporaries
                if fixture == 'source_borrow_shared.nano':
                    self.assertEqual([r[4] for r in records if r[0] == 'main'],
                                     ['owner', 'value', 'active'])

    def test_shadow_names_close_before_next_selected_scope(self):
        source = self.work / 'named-shadows.nano'
        text = (FIXTURES / 'source_borrow_shared.nano').read_text()
        body = text.split('shadow read {', 1)[1].split('\n}', 1)[0]
        source.write_text(text.replace('shadow main { assert true }', 'shadow main {' + body + '\n}'))
        baseline = None
        for tool in [ROOT / 'obj/borrow_shadow_names', *self.shadow_tools]:
            assembly, module = self.work / 'named-shadows.nasm', self.work / 'named-shadows.nvm'
            args = (source,) if tool.name == 'borrow_shadow_names' else (source, 0, 'raw')
            assembly.write_text(self.command(tool, *args).stdout)
            self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
            records = self.names_and_strip(module)
            if baseline is None:
                baseline = records
            self.assertEqual(records, baseline)
            owners = [row for row in records if row[4] == 'owner']
            self.assertEqual(len(owners), 2)
            self.assertNotEqual(owners[0][1], owners[1][1])
            self.assertLessEqual(int(owners[0][3]), int(owners[1][2]))
            self.assertFalse(any('__' in row[4] for row in records))

    def test_multiple_parameters_preserve_callers_and_metadata(self):
        for fixture, arity in (('source_borrow_multi.nano', 3), ('source_borrow_eight.nano', 8)):
            source = FIXTURES / fixture
            seed = self.work / 'multi-seed.nvm'
            self.command(ROOT / 'bin/nano_virt', source, '--emit-nvm', '--strip-debug', '-o', seed)
            baseline = self.command(ROOT / 'bin/nanoisa', 'dump', seed).stdout
            self.assertIn('.parameters 1' + ' struct' * arity, baseline)
            self.assertIn('REF_GET ' + str(arity - 1), baseline)
            names = self.names_and_strip(seed)
            formals = [r for r in names if r[0] != 'main' and r[2] == '0']
            self.assertEqual([int(r[1]) for r in formals], list(range(arity)))
            for emitter in self.emitters:
                assembly, module = self.work / 'multi-self.nasm', self.work / 'multi-self.nvm'
                self.command(emitter, source, '-o', assembly)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                self.execute_pair(module)
            for compiler in ('nanoc_stage1', 'nanoc_stage2'):
                module = self.work / (compiler + '-multi.nvm')
                self.command(ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', module)
                self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                self.execute_pair(module)
            shadow_dump = None
            for tool in [ROOT / 'obj/borrow_shadow_names', *self.shadow_tools]:
                args = (source,) if tool.name == 'borrow_shadow_names' else (source, 0, 'raw')
                assembly = self.work / 'multi-shadows.nasm'
                module = self.work / 'multi-shadows.nvm'
                assembly.write_text(self.command(tool, *args).stdout)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                current = self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout
                if shadow_dump is None:
                    shadow_dump = current
                self.assertEqual(current, shadow_dump)
                self.execute_pair(module)

    def test_multiple_parameter_refusals_preserve_publication(self):
        multi = (FIXTURES / 'source_borrow_multi.nano').read_text()
        eight = (FIXTURES / 'source_borrow_eight.nano').read_text()
        shared_first = """resource struct Item { value: int, active: bool }
fn inspect(first: &Item, second: &mut Item) -> int { return (+ first.value second.value) }
shadow inspect { let mut owner: Item = Item { value: 1, active: true }
 assert (== (inspect &owner &mut owner) 2)
 let Item { value, active } = owner }
fn main() -> int { let mut owner: Item = Item { value: 1, active: true }
 assert (== (inspect &owner &mut owner) 2)
 let Item { value, active } = owner return 0 }
shadow main { assert true }
"""
        duplicate = shared_first.replace('first: &Item, second: &mut Item',
                                         'first: &Item, first: &Item')
        duplicate = duplicate.replace('second.value', 'first.value').replace('&mut owner)', '&owner)')
        cases = {
            'shared_then_exclusive_alias': shared_first,
            'duplicate_formals': duplicate,
            'arity': multi.replace('&mut one &middle_owner &mut three', '&mut one &middle_owner'),
            'mode': multi.replace('&mut one &middle_owner &mut three', '&one &middle_owner &mut three'),
            'nominal': multi.replace('&mut one &middle_owner &mut three', '&mut one &one &mut three'),
            'exclusive_alias': multi.replace('&mut one &middle_owner &mut three', '&mut one &middle_owner &mut one'),
            'mixed_alias': multi.replace('middle: &Right', 'middle: &Left').replace('&middle_owner', '&one'),
            'value_formal': eight.replace('p7: &Counter', 'p7: int'),
            'nine': eight.replace('p7: &Counter', 'p7: &Counter, extra: &Counter'),
            'failed_shadow': multi.replace('shadow main { assert true }', 'shadow main { assert false }'),
        }
        for name, text in cases.items():
            source = self.work / (name + '.nano')
            source.write_text(text)
            for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
                output = self.work / 'preserved.nvm'
                output.write_bytes(b'accepted-output')
                result = subprocess.run([ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', output],
                                        cwd=ROOT, capture_output=True, text=True, timeout=60)
                self.assertGreater(result.returncode, 0, (name, compiler, result.stderr))
                self.assertEqual(output.read_bytes(), b'accepted-output')
            if name != 'failed_shadow':
                for emitter in self.emitters:
                    output = self.work / 'preserved.nasm'
                    output.write_text('accepted-output')
                    result = subprocess.run([emitter, source, '-o', output], cwd=ROOT,
                                            capture_output=True, text=True, timeout=60)
                    self.assertGreater(result.returncode, 0, (name, emitter, result.stderr))
                    self.assertEqual(output.read_text(), 'accepted-output')

    def test_control_flow_preserves_exact_joins_and_effects(self):
        for fixture in ('source_borrow_control_flow.nano', 'source_borrow_control_shared.nano'):
            source = FIXTURES / fixture
            seed = self.work / 'nested-seed.nvm'
            self.command(ROOT / 'bin/nano_virt', source, '--emit-nvm', '--strip-debug', '-o', seed)
            baseline = self.command(ROOT / 'bin/nanoisa', 'dump', seed).stdout
            self.assertIn('.ownership "02000000', baseline)
            self.assertIn('JMP_FALSE', baseline)
            self.assertIn('JMP ', baseline)
            expected = 'BORROW_PATH_SHARED' if fixture.endswith('_shared.nano') else 'BORROW_PATH_EXCLUSIVE'
            self.assertIn(expected, baseline)
            self.names_and_strip(seed)
            for emitter in self.emitters:
                assembly, module = self.work / 'nested.nasm', self.work / 'nested.nvm'
                self.command(emitter, source, '-o', assembly)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                self.execute_pair(module)
            for compiler in ('nanoc_stage1', 'nanoc_stage2'):
                module = self.work / (compiler + '-nested.nvm')
                self.command(ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', module)
                self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                self.execute_pair(module)
            shadow_dump = None
            for tool in [ROOT / 'obj/borrow_shadow_names', *self.shadow_tools]:
                args = (source,) if tool.name == 'borrow_shadow_names' else (source, 0, 'raw')
                assembly, module = self.work / 'nested-shadow.nasm', self.work / 'nested-shadow.nvm'
                assembly.write_text(self.command(tool, *args).stdout)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                current = self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout
                if shadow_dump is None:
                    shadow_dump = current
                self.assertEqual(current, shadow_dump)
                self.execute_pair(module)

    def test_explicit_resource_paths_preserve_exact_joins(self):
        text = (FIXTURES / 'source_borrow_resource_paths.nano').read_text()
        consume = 'let Leaf { value, active } = tail assert (== value 3) assert (not active) return 0'
        retained = 'assert (== (bump &mut tail) 4)'
        after = 'let Leaf { value, active } = tail assert (== value 4) assert (not active)'
        tail = 'let mut tail: Leaf = Leaf { value: 3, active: false } '
        variants = {'normal': text, 'arms_reversed': text.replace('if true {', 'if false {')}
        for choice in ('true', 'false'):
            variants['return_then_' + choice] = text.replace('# RETURN_POINT',
                tail + 'if ' + choice + ' { ' + consume + ' } else { ' + retained + ' } ' + after)
            variants['return_else_' + choice] = text.replace('# RETURN_POINT',
                tail + 'if ' + choice + ' { ' + retained + ' } else { ' + consume + ' } ' + after)
            variants['return_loop_' + choice] = text.replace('# RETURN_POINT',
                tail + 'while ' + choice + ' { ' + consume + ' } ' + consume)
        for fixture, content in variants.items():
            source = self.work / ('resource-path-' + fixture + '.nano')
            source.write_text(content)
            seed = self.work / 'nested-seed.nvm'
            self.command(ROOT / 'bin/nano_virt', source, '--emit-nvm', '--strip-debug', '-o', seed)
            baseline = self.command(ROOT / 'bin/nanoisa', 'dump', seed).stdout
            self.assertIn('.ownership "02000000', baseline)
            self.assertIn('JMP_FALSE', baseline)
            self.assertIn('JMP ', baseline)
            expected = 'BORROW_PATH_EXCLUSIVE'
            self.assertIn(expected, baseline)
            self.names_and_strip(seed)
            for emitter in self.emitters:
                assembly, module = self.work / 'nested.nasm', self.work / 'nested.nvm'
                self.command(emitter, source, '-o', assembly)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                self.execute_pair(module)
            for compiler in ('nanoc_stage1', 'nanoc_stage2'):
                module = self.work / (compiler + '-nested.nvm')
                self.command(ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', module)
                self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                self.execute_pair(module)
            shadow_dump = None
            for tool in [ROOT / 'obj/borrow_shadow_names', *self.shadow_tools]:
                args = (source,) if tool.name == 'borrow_shadow_names' else (source, 0, 'raw')
                assembly, module = self.work / 'nested-shadow.nasm', self.work / 'nested-shadow.nvm'
                assembly.write_text(self.command(tool, *args).stdout)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                current = self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout
                if shadow_dump is None:
                    shadow_dump = current
                self.assertEqual(current, shadow_dump)
                self.execute_pair(module)

    def test_loop_exits_preserve_exact_owners_and_innermost_targets(self):
        text = (FIXTURES / 'source_borrow_loop_exits.nano').read_text()
        variants = {'break': text, 'continue': text.replace('if true {', 'if false {').replace('assert (== j 1)', 'assert (== j 3)')}
        for fixture, content in variants.items():
            source = self.work / ('resource-path-' + fixture + '.nano')
            source.write_text(content)
            seed = self.work / 'nested-seed.nvm'
            self.command(ROOT / 'bin/nano_virt', source, '--emit-nvm', '--strip-debug', '-o', seed)
            baseline = self.command(ROOT / 'bin/nanoisa', 'dump', seed).stdout
            self.assertIn('.ownership', baseline)
            self.assertIn('JMP_FALSE', baseline)
            self.assertIn('JMP ', baseline)
            expected = 'BORROW_LOCAL_EXCLUSIVE'
            self.assertIn(expected, baseline)
            self.names_and_strip(seed)
            for emitter in self.emitters:
                assembly, module = self.work / 'nested.nasm', self.work / 'nested.nvm'
                self.command(emitter, source, '-o', assembly)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                self.execute_pair(module)
            for compiler in ('nanoc_stage1', 'nanoc_stage2'):
                module = self.work / (compiler + '-nested.nvm')
                self.command(ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', module)
                self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                self.execute_pair(module)
            shadow_dump = None
            for tool in [ROOT / 'obj/borrow_shadow_names', *self.shadow_tools]:
                args = (source,) if tool.name == 'borrow_shadow_names' else (source, 0, 'raw')
                assembly, module = self.work / 'nested-shadow.nasm', self.work / 'nested-shadow.nvm'
                assembly.write_text(self.command(tool, *args).stdout)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                current = self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout
                if shadow_dump is None:
                    shadow_dump = current
                self.assertEqual(current, shadow_dump)
                self.execute_pair(module)

    def test_range_for_preserves_bounds_scope_and_exact_edges(self):
        text = (FIXTURES / 'source_borrow_range_for.nano').read_text()
        variants = {'entered_return': text, 'empty_return': text.replace('for tail in (range 0 1)', 'for tail in (range 0 0)')}
        for fixture, content in variants.items():
            source = self.work / ('resource-path-' + fixture + '.nano')
            source.write_text(content)
            seed = self.work / 'nested-seed.nvm'
            self.command(ROOT / 'bin/nano_virt', source, '--emit-nvm', '--strip-debug', '-o', seed)
            baseline = self.command(ROOT / 'bin/nanoisa', 'dump', seed).stdout
            self.assertIn('.ownership', baseline)
            self.assertIn('JMP_FALSE', baseline)
            self.assertIn('JMP ', baseline)
            expected = 'BORROW_LOCAL_EXCLUSIVE'
            self.assertIn(expected, baseline)
            self.names_and_strip(seed)
            for emitter in self.emitters:
                assembly, module = self.work / 'nested.nasm', self.work / 'nested.nvm'
                self.command(emitter, source, '-o', assembly)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                self.execute_pair(module)
            for compiler in ('nanoc_stage1', 'nanoc_stage2'):
                module = self.work / (compiler + '-nested.nvm')
                self.command(ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', module)
                self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                self.execute_pair(module)
            shadow_dump = None
            for tool in [ROOT / 'obj/borrow_shadow_names', *self.shadow_tools]:
                args = (source,) if tool.name == 'borrow_shadow_names' else (source, 0, 'raw')
                assembly, module = self.work / 'nested-shadow.nasm', self.work / 'nested-shadow.nvm'
                assembly.write_text(self.command(tool, *args).stdout)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                current = self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout
                if shadow_dump is None:
                    shadow_dump = current
                self.assertEqual(current, shadow_dump)
                self.execute_pair(module)

    def test_range_for_refusals_preserve_publication(self):
        text = (FIXTURES / 'source_borrow_range_for.nano').read_text()
        cases = {
            'one_bound': text.replace('(range 0 4)', '(range 4)'),
            'array_iterable': text.replace('(range 0 4)', '[0, 1]'),
            'float_bound': text.replace('(range 0 4)', '(range 0 4.0)'),
            'local_leak': text.replace('if (== index 1) { continue }', 'if (== index 1) { let leaked: Counter = Counter { value: 1, active: true } continue }'),
            'changed_owner': text.replace('if (== index 3) { break }', 'if (== index 3) { let Counter { value, active } = owner break }'),
            'escaped_index': text.replace('return view.value', 'return turn'),
        }
        for name, content in cases.items():
            source = self.work / ('range-refusal-' + name + '.nano')
            source.write_text(content)
            for compiler in [ROOT / 'bin' / name for name in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2')] + self.emitters:
                output = self.work / 'range-preserved.output'
                output.write_text('accepted-output')
                args = [compiler, source]
                if compiler not in self.emitters:
                    args.append('--emit-nvm')
                result = subprocess.run([*args, '-o', output], cwd=ROOT, capture_output=True, text=True, timeout=60)
                self.assertGreater(result.returncode, 0, (name, compiler, result.stderr))
                self.assertNotIn('parse error', result.stderr.lower())
                self.assertEqual(output.read_text(), 'accepted-output')

    def test_consumed_owner_reassignment_preserves_paths_and_names(self):
        text = (FIXTURES / 'source_borrow_resource_paths.nano').read_text()
        text = text.replace('let tree: Pair =', 'let mut tree: Pair =')
        text = text.replace('let mut moved: Pair = tree',
                            'let mut moved: Pair = tree set tree moved set moved tree')
        text = text.replace('let leaf: Leaf =', 'let mut leaf: Leaf =')
        text = text.replace('let mut moved: Leaf = leaf',
                            'let mut moved: Leaf = leaf set leaf moved set moved leaf')
        # I restore an incoming owner on each backedge; zero iterations retain it.
        carry = """let mut owner: Leaf = Leaf { value: 3, active: false }
 let Leaf { value, active } = owner
 assert (== value 3) assert (not active)
 let replacement: Leaf = Leaf { value: 8, active: true }
 set owner replacement
 let mut turns: int = 0
 while (< turns LIMIT) {
  let carried: Leaf = owner
  set owner carried
  assert (== (bump &mut owner) (+ 9 turns))
  set turns (+ turns 1)
 }
 let Leaf { value, active } = owner
 assert (== value (+ 8 LIMIT)) assert active
"""
        for choice, limit in (('true', '0'), ('false', '2')):
            source = self.work / ('owner-reassignment-' + choice + '.nano')
            source.write_text(text.replace('if true {', 'if ' + choice + ' {')
                              .replace('# RETURN_POINT', carry.replace('LIMIT', limit)))
            seed = self.work / 'reassignment-seed.nvm'
            self.command(ROOT / 'bin/nano_virt', source, '--emit-nvm', '--strip-debug', '-o', seed)
            baseline = self.command(ROOT / 'bin/nanoisa', 'dump', seed).stdout
            records = self.names_and_strip(seed)
            self.assertEqual(len([row for row in records if row[0] == 'main' and row[4] == 'owner']), 1)
            for emitter in self.emitters:
                assembly, module = self.work / 'owner.nasm', self.work / 'owner.nvm'
                self.command(emitter, source, '-o', assembly)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                self.execute_pair(module)
            for compiler in ('nanoc_stage1', 'nanoc_stage2'):
                module = self.work / (compiler + '-owner.nvm')
                self.command(ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', module)
                self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                self.execute_pair(module)
            shadow_dump = None
            for tool in [ROOT / 'obj/borrow_shadow_names', *self.shadow_tools]:
                args = (source,) if tool.name == 'borrow_shadow_names' else (source, 0, 'raw')
                assembly, module = self.work / 'owner-shadow.nasm', self.work / 'owner-shadow.nvm'
                assembly.write_text(self.command(tool, *args).stdout)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                current = self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout
                if shadow_dump is None:
                    shadow_dump = current
                self.assertEqual(current, shadow_dump)
                self.execute_pair(module)

    def test_owner_reassignment_refusals_preserve_publication(self):
        text = (FIXTURES / 'source_borrow_shared.nano').read_text()
        start = text.index('fn main()')
        text = text[:start] + """fn main() -> int {
 let mut owner: Counter = Counter { value: 1, active: true }
 let moved: Counter = owner
 set owner moved
 let Counter { value, active } = owner
 assert active return 0
}
shadow main { assert true }
"""
        cases = {
            'immutable': text.replace('let mut owner:', 'let owner:'),
            'moved_source': text.replace('set owner moved', 'set owner owner'),
            'join_then': text.replace('set owner moved', 'if true { set owner moved }'),
            'join_else': text.replace('set owner moved', 'if true { assert true } else { set owner moved }'),
            'loop_changed': text.replace('set owner moved', 'while false { set owner moved }'),
            'borrowed_destination': text.replace('return view.value', 'set view view return view.value'),
            'live_overwrite': text.replace('set owner moved', 'set owner moved set owner owner'),
            'wrong_nominal': text.replace('let moved: Counter = owner', 'let Counter { value, active } = owner let moved: Other = Other { value: 2, active: false }'),
            'constructor': text.replace('set owner moved', 'set owner Counter { value: 2, active: false }'),
        }
        for name, content in cases.items():
            source = self.work / ('owner-refusal-' + name + '.nano')
            source.write_text(content)
            for compiler in [ROOT / 'bin' / item for item in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2')] + self.emitters:
                output = self.work / 'owner-preserved.output'
                output.write_text('accepted-output')
                args = [compiler, source]
                if compiler not in self.emitters:
                    args.append('--emit-nvm')
                result = subprocess.run([*args, '-o', output], cwd=ROOT, capture_output=True, text=True, timeout=60)
                self.assertGreater(result.returncode, 0, (name, compiler, result.stderr))
                self.assertEqual(output.read_text(), 'accepted-output')
                self.assertNotRegex(result.stdout + result.stderr, r'(?i)parse error')
                self.assertRegex(result.stdout + result.stderr,
                                 r'(?i)(immutable|mutable|moved|live resource|borrowed|consumed|exact.*owner|ownership|resource.*(?:branch|loop)|(?:branch|loop).*resource|type mismatch|expected.*Counter|cannot assign)')

    def test_return_paths_preserve_ownership_and_fallthrough(self):
        text = (FIXTURES / 'source_borrow_returns.nano').read_text()
        ending = 'if true { let code: int = 0 return code } else { return 1 }'
        variants = {
            'then': text,
            'else': text.replace(ending, 'if false { return 1 } else { let code: int = 0 return code }'),
            'loop': text.replace(ending, 'while true { let code: int = 0 return code } return 1'),
            'zero': text.replace(ending, 'while false { return 1 } return 0'),
            'single': text.replace(ending, 'if false { return 1 } return 0'),
            'then_only': text.replace(ending, 'if true { return 0 } return 1'),
            'bool': (FIXTURES / 'source_borrow_bool.nano').read_text().replace(
                'return view.active', 'if view.active { return true } else { return false }'),
        }
        for fixture, content in variants.items():
            source = self.work / ('returns-' + fixture + '.nano')
            source.write_text(content)
            seed = self.work / 'nested-seed.nvm'
            self.command(ROOT / 'bin/nano_virt', source, '--emit-nvm', '--strip-debug', '-o', seed)
            baseline = self.command(ROOT / 'bin/nanoisa', 'dump', seed).stdout
            self.assertIn('.ownership', baseline)
            self.assertIn('JMP_FALSE', baseline)
            expected = 'BORROW_LOCAL_SHARED' if fixture == 'bool' else 'BORROW_LOCAL_EXCLUSIVE'
            self.assertIn(expected, baseline)
            self.names_and_strip(seed)
            for emitter in self.emitters:
                assembly, module = self.work / 'nested.nasm', self.work / 'nested.nvm'
                self.command(emitter, source, '-o', assembly)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                self.execute_pair(module)
            for compiler in ('nanoc_stage1', 'nanoc_stage2'):
                module = self.work / (compiler + '-nested.nvm')
                self.command(ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', module)
                self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                self.execute_pair(module)
            shadow_dump = None
            for tool in [ROOT / 'obj/borrow_shadow_names', *self.shadow_tools]:
                args = (source,) if tool.name == 'borrow_shadow_names' else (source, 0, 'raw')
                assembly, module = self.work / 'nested-shadow.nasm', self.work / 'nested-shadow.nvm'
                assembly.write_text(self.command(tool, *args).stdout)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                current = self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout
                if shadow_dump is None:
                    shadow_dump = current
                self.assertEqual(current, shadow_dump)
                self.execute_pair(module)

    def test_lexical_scalars_restore_outer_bindings_and_names(self):
        for fixture in ('source_borrow_lexical_scalars.nano',):
            source = FIXTURES / fixture
            seed = self.work / 'nested-seed.nvm'
            self.command(ROOT / 'bin/nano_virt', source, '--emit-nvm', '--strip-debug', '-o', seed)
            baseline = self.command(ROOT / 'bin/nanoisa', 'dump', seed).stdout
            self.assertIn('.ownership', baseline)
            self.assertIn('JMP_FALSE', baseline)
            self.assertIn('JMP ', baseline)
            expected = 'BORROW_LOCAL_SHARED'
            self.assertIn(expected, baseline)
            records = self.names_and_strip(seed)
            values = [row for row in records if row[0] == 'read' and row[4] == 'value']
            self.assertEqual(len(values), 3)
            outer, first, second = values
            self.assertEqual(len({row[1] for row in values}), 3)
            self.assertLess(int(outer[2]), int(first[2]))
            self.assertLessEqual(int(first[3]), int(second[2]))
            self.assertLess(int(second[3]), int(outer[3]))
            unused = [row for row in records if row[0] == 'main' and row[4] == 'unused']
            self.assertEqual(len(unused), 1)
            self.assertLess(int(unused[0][2]), int(unused[0][3]))
            indexes = [row for row in records if row[0] == 'main' and row[4] == 'index']
            self.assertEqual(len(indexes), 2)
            self.assertNotEqual(indexes[0][1], indexes[1][1])
            self.assertLess(int(indexes[0][2]), int(indexes[1][2]))
            self.assertLess(int(indexes[1][3]), int(indexes[0][3]))
            self.assertFalse(any('__' in row[4] for row in records))
            for emitter in self.emitters:
                assembly, module = self.work / 'nested.nasm', self.work / 'nested.nvm'
                self.command(emitter, source, '-o', assembly)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                self.execute_pair(module)
            for compiler in ('nanoc_stage1', 'nanoc_stage2'):
                module = self.work / (compiler + '-nested.nvm')
                self.command(ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', module)
                self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                self.execute_pair(module)
            shadow_dump = None
            for tool in [ROOT / 'obj/borrow_shadow_names', *self.shadow_tools]:
                args = (source,) if tool.name == 'borrow_shadow_names' else (source, 0, 'raw')
                assembly, module = self.work / 'nested-shadow.nasm', self.work / 'nested-shadow.nvm'
                assembly.write_text(self.command(tool, *args).stdout)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                current = self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout
                if shadow_dump is None:
                    shadow_dump = current
                self.assertEqual(current, shadow_dump)
                self.execute_pair(module)

    def test_nested_paths_preserve_trees_and_callers(self):
        for fixture in ('source_borrow_nested.nano', 'source_borrow_nested_shared.nano'):
            source = FIXTURES / fixture
            seed = self.work / 'nested-seed.nvm'
            self.command(ROOT / 'bin/nano_virt', source, '--emit-nvm', '--strip-debug', '-o', seed)
            baseline = self.command(ROOT / 'bin/nanoisa', 'dump', seed).stdout
            self.assertIn('.ownership "02000000', baseline)
            self.assertIn('BORROW_PATH_SHARED', baseline)
            if fixture == 'source_borrow_nested.nano':
                self.assertIn('BORROW_PATH_EXCLUSIVE', baseline)
            self.names_and_strip(seed)
            for emitter in self.emitters:
                assembly, module = self.work / 'nested.nasm', self.work / 'nested.nvm'
                self.command(emitter, source, '-o', assembly)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                self.execute_pair(module)
            for compiler in ('nanoc_stage1', 'nanoc_stage2'):
                module = self.work / (compiler + '-nested.nvm')
                self.command(ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', module)
                self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                self.execute_pair(module)
            shadow_dump = None
            for tool in [ROOT / 'obj/borrow_shadow_names', *self.shadow_tools]:
                args = (source,) if tool.name == 'borrow_shadow_names' else (source, 0, 'raw')
                assembly, module = self.work / 'nested-shadow.nasm', self.work / 'nested-shadow.nvm'
                assembly.write_text(self.command(tool, *args).stdout)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                current = self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout
                if shadow_dump is None:
                    shadow_dump = current
                self.assertEqual(current, shadow_dump)
                self.execute_pair(module)

    def test_control_flow_refusals_preserve_publication(self):
        text = (FIXTURES / 'source_borrow_control_flow.nano').read_text()
        cases = {
            'branch_mixed_float': text.replace('set total (+ total 2)', 'let value: float = (+ 2.0 1)'),
            'loop_string': text.replace('set j 0', 'let value: string = "unsupported"'),
            'branch_owner': text.replace('set total (+ total 2)', 'let moved: Pair = root'),
            'branch_destructure': text.replace('set total (+ total 2)', 'let Pair { left, right } = root'),
            'loop_destructure': text.replace('set j 0', 'let Pair { left, right } = root'),
            'branch_return': text.replace('set total (+ total 2)', 'return 0'),
            'loop_return': text.replace('set j 0', 'return 0'),
            'break_changed_owner': text.replace('set j 0', 'let moved: Pair = root break'),
            'continue_changed_owner': text.replace('set j 0', 'let moved: Pair = root continue'),
            'integer_condition': text.replace('while (< i 3)', 'while 1'),
            'changed_scalar_type': text.replace('set j 0', 'set j true'),
            'failed_shadow': text.replace('shadow main { if true { assert true }', 'shadow main { if true { assert false }'),
        }
        for name, content in cases.items():
            source = self.work / (name + '.nano')
            source.write_text(content)
            for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
                output = self.work / 'control-preserved.nvm'
                output.write_bytes(b'accepted-output')
                result = subprocess.run([ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', output],
                                        cwd=ROOT, capture_output=True, text=True, timeout=60)
                self.assertGreater(result.returncode, 0, (name, compiler, result.stderr))
                self.assertEqual(output.read_bytes(), b'accepted-output')
            if name != 'failed_shadow':
                for emitter in self.emitters:
                    output = self.work / 'control-preserved.nasm'
                    output.write_text('accepted-output')
                    result = subprocess.run([emitter, source, '-o', output], cwd=ROOT,
                                            capture_output=True, text=True, timeout=60)
                    self.assertGreater(result.returncode, 0, (name, emitter, result.stderr))
                    self.assertEqual(output.read_text(), 'accepted-output')

    def test_lexical_scalar_visibility_refusals(self):
        text = (FIXTURES / 'source_borrow_lexical_scalars.nano').read_text()
        cases = {
            'branch_escape': text.replace('let Leaf { value, enabled } = left', 'assert sibling let Leaf { value, enabled } = left'),
            'loop_escape': text.replace('assert (== index 2)', 'assert (== delta 14) assert (== index 2)'),
            'initializer_self': text.replace('let increment: int = 1', 'let increment: int = increment'),
        }
        for name, content in cases.items():
            source = self.work / (name + '.nano')
            source.write_text(content)
            for compiler in [ROOT / 'bin' / name for name in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2')] + self.emitters:
                output = self.work / 'visibility-preserved.output'
                output.write_text('accepted-output')
                args = [compiler, source]
                if compiler not in self.emitters:
                    args.append('--emit-nvm')
                result = subprocess.run([*args, '-o', output], cwd=ROOT, capture_output=True, text=True, timeout=60)
                self.assertGreater(result.returncode, 0, (name, compiler, result.stderr))
                self.assertEqual(output.read_text(), 'accepted-output')

    def test_return_path_refusals_preserve_publication(self):
        text = (FIXTURES / 'source_borrow_returns.nano').read_text()
        cases = {
            'shadow_return': text.replace('let checked: bool = true assert checked', 'return 0'),
            'live_owner': text.replace('assert (== (read &mut first) 3)', 'if true { return 0 }'),
            'missing_return': text.replace(' return 1\n}', '\n}'),
            'owner_move': text.replace('let answer: int = view.value', 'let moved: Leaf = view'),
            'incomplete_pattern': text.replace('let Leaf { value, enabled } = first', 'let Leaf { value } = first'),
            'duplicate_pattern': text.replace('let Leaf { value, enabled } = first', 'let Leaf { value, value } = first'),
        }
        for name, content in cases.items():
            source = self.work / ('return-refusal-' + name + '.nano')
            source.write_text(content)
            for compiler in [ROOT / 'bin' / name for name in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2')] + self.emitters:
                output = self.work / 'return-preserved.output'
                output.write_text('accepted-output')
                args = [compiler, source]
                if compiler not in self.emitters:
                    args.append('--emit-nvm')
                # Raw program emitters do not select shadows; use the shadow
                # producer below for its independent nested-return refusal.
                if name == 'shadow_return' and compiler in self.emitters:
                    continue
                result = subprocess.run([*args, '-o', output], cwd=ROOT, capture_output=True, text=True, timeout=60)
                self.assertGreater(result.returncode, 0, (name, compiler, result.stderr))
                self.assertEqual(output.read_text(), 'accepted-output')
            if name == 'shadow_return':
                for tool in self.shadow_tools:
                    result = subprocess.run([tool, source, '0', 'raw'], cwd=ROOT,
                                            capture_output=True, text=True, timeout=60)
                    self.assertGreater(result.returncode, 0, (tool, result.stdout, result.stderr))

    def test_helper_owned_locals_preserve_formals_and_shadows(self):
        text = (FIXTURES / 'source_borrow_helper_owners.nano').read_text()
        prior = (FIXTURES / 'source_borrow_resource_paths.nano').read_text().replace(
            'set view.value (+ view.value 1)',
            'let owner: Leaf = Leaf { value: 1, active: true } let Leaf { value, active } = owner set view.value (+ view.value 1)')
        variants = {'normal': text, 'alternate': text.replace('if true {', 'if false {').replace('while (< index 2)', 'while (< index 0)'),
                    'former_refusal': prior}
        for name, content in variants.items():
            source = self.work / ('helper-owners-' + name + '.nano')
            source.write_text(content)
            baseline = None
            for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
                module = self.work / (compiler + '-helper.nvm')
                self.command(ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', module)
                actual = self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout
                if baseline is None:
                    baseline = actual
                self.assertEqual(actual, baseline)
                records = self.names_and_strip(module)
                if name != 'former_refusal':
                    self.assertIn('BORROW_PATH_SHARED 2 ', actual)
                    helper = [row for row in records if row[0] == 'combine']
                    self.assertEqual([row[4] for row in helper[:2]], ['target', 'source'])
                    self.assertGreaterEqual(sum(row[4] == 'source' for row in helper), 3)
            for emitter in self.emitters:
                assembly, module = self.work / 'helper.nasm', self.work / 'helper.nvm'
                self.command(emitter, source, '-o', assembly)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                self.execute_pair(module)
            shadow_dump = None
            for tool in [ROOT / 'obj/borrow_shadow_names', *self.shadow_tools]:
                args = (source,) if tool.name == 'borrow_shadow_names' else (source, 0, 'raw')
                assembly, module = self.work / 'helper-shadow.nasm', self.work / 'helper-shadow.nvm'
                assembly.write_text(self.command(tool, *args).stdout)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                current = self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout
                if shadow_dump is None:
                    shadow_dump = current
                self.assertEqual(current, shadow_dump)
                self.execute_pair(module)

    def test_helper_owner_refusals_preserve_publication(self):
        text = (FIXTURES / 'source_borrow_helper_owners.nano').read_text()
        cases = {
            'unconsumed': text.replace('return target.value', 'let extra: Leaf = Leaf { value: 1, active: true } return target.value'),
            'formal_move': text.replace('let first: Leaf = Leaf { value: 100, active: true }', 'let first: Leaf = source'),
            'formal_write': text.replace('set target.value (+ target.value source.value)', 'set source.value 9'),
            'deeper_call': text.replace('return target.value', 'return (combine target source)'),
            'wrong_nominal': text.replace('let tree: Pair = Pair { right: second, left: first }', 'let tree: Pair = first'),
            'hidden_owner': text.replace('let first: Leaf = Leaf { value: 100, active: true }', 'let first: Leaf = (cond (true Leaf { value: 100, active: true }) (else Leaf { value: 100, active: true }))'),
            'local_write': text.replace('assert (== first.value 100)', 'set first.value 100', 1).replace('let first: Leaf', 'let mut first: Leaf', 1),
        }
        for name, content in cases.items():
            source = self.work / ('helper-refusal-' + name + '.nano')
            source.write_text(content)
            for compiler in [ROOT / 'bin' / name for name in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2')] + self.emitters:
                output = self.work / 'helper-preserved.output'
                output.write_text('accepted-output')
                args = [compiler, source]
                if compiler not in self.emitters:
                    args.append('--emit-nvm')
                result = subprocess.run([*args, '-o', output], cwd=ROOT, capture_output=True, text=True, timeout=60)
                self.assertGreater(result.returncode, 0, (name, compiler, result.stderr))
                self.assertEqual(output.read_text(), 'accepted-output')
                self.assertNotRegex(result.stderr + result.stdout, r'(?i)parse (?:error|failed)|unexpected token')
                self.assertRegex(result.stderr + result.stdout,
                                 r'(?i)borrow|owner|resource|consum|nominal|constructor|type mismatch|expected|helper|exclusive|mutable')

    def test_resource_path_refusals_preserve_publication(self):
        text = (FIXTURES / 'source_borrow_resource_paths.nano').read_text()
        cases = {
            'unconsumed_local': text.replace('let Leaf { value, active } = leaf', 'let value: int = 4 let active: bool = true'),
            'join_left': text.replace('let Leaf { value, active } = outer', 'let value: int = 5 let active: bool = true'),
            'join_right': text.replace('let moved: Leaf = outer', 'let moved: Leaf = Leaf { value: 5, active: true }'),
            'loop_outer': text.replace('if true {\n  let moved: Leaf = outer', 'while true {\n  let moved: Leaf = outer').replace(' } else {\n  let Leaf { value, active } = outer\n  assert (== value 5) assert active\n }', ' }'),
            'moved_use': text.replace('let mut moved: Pair = tree', 'let mut moved: Pair = tree let again: Pair = tree'),
            'assignment': text.replace('let mut moved: Leaf = leaf', 'let mut moved: Leaf = leaf set moved Leaf { value: 1, active: true }'),
            'partial_move': text.replace('let Pair { right, left } = moved', 'let right: Leaf = moved.right let left: Leaf = moved.left'),
            'wrong_nominal': text.replace('let mut moved: Pair = tree', 'let moved: Leaf = tree'),
        }
        for name, content in cases.items():
            source = self.work / ('resource-refusal-' + name + '.nano')
            source.write_text(content)
            for compiler in [ROOT / 'bin' / name for name in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2')] + self.emitters:
                output = self.work / 'resource-preserved.output'
                output.write_text('accepted-output')
                args = [compiler, source]
                if compiler not in self.emitters:
                    args.append('--emit-nvm')
                result = subprocess.run([*args, '-o', output], cwd=ROOT, capture_output=True, text=True, timeout=60)
                self.assertGreater(result.returncode, 0, (name, compiler, result.stderr))
                self.assertEqual(output.read_text(), 'accepted-output')

    def test_control_flow_depth_boundary(self):
        text = (FIXTURES / 'source_borrow_control_flow.nano').read_text()
        for depth in (32, 33):
            nested = 'if true { ' * depth + 'assert true' + ' }' * depth
            source = self.work / f'control-depth{depth}.nano'
            source.write_text(text.replace('if true { assert (== root.left.value 6) }', nested))
            baseline = None
            for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
                output = self.work / f'control-depth{depth}-{compiler}.nvm'
                output.write_bytes(b'accepted-output')
                args = [ROOT / 'bin' / compiler, source, '--emit-nvm']
                if compiler == 'nano_virt':
                    args.append('--strip-debug')
                result = subprocess.run([*args, '-o', output], cwd=ROOT,
                                        capture_output=True, text=True, timeout=60)
                if depth == 33:
                    self.assertGreater(result.returncode, 0, (compiler, result.stderr))
                    self.assertEqual(output.read_bytes(), b'accepted-output')
                else:
                    self.assertEqual(result.returncode, 0, (compiler, result.stderr))
                    dump = self.command(ROOT / 'bin/nanoisa', 'dump', output).stdout
                    if baseline is None:
                        baseline = dump
                    self.assertEqual(dump, baseline)
                    self.execute_pair(output)

    def test_nested_source_refusals_preserve_publication(self):
        text = (FIXTURES / 'source_borrow_nested.nano').read_text()
        shared = (FIXTURES / 'source_borrow_nested_shared.nano').read_text()
        shared = shared.replace('let root: Tree', 'let mut root: Tree')
        shared_first = shared.replace('second: &Leaf', 'second: &mut Leaf')
        shared_first = shared_first.replace('&root.branch.left &root.branch.left',
                                            '&root.branch.left &mut root.branch.left')
        exclusive_first = shared.replace('first: &Leaf', 'first: &mut Leaf')
        exclusive_first = exclusive_first.replace('&root.branch.left &root.branch.left',
                                                   '&mut root.branch.left &root.branch.left')
        pair = 'resource struct Pair { left: Leaf, right: Leaf }\n'
        disposal_start = text.index(' let Tree { count, other, branch } = root')
        disposal = text[disposal_start:text.index('\n}', disposal_start)]
        cases = {
            'child_nominal': text.replace('right: right, left: left', 'right: right, left: amount'),
            'child_twice': text.replace('right: right, left: left', 'right: left, left: left'),
            'moved_parent': text.replace('assert (== (change', 'assert (== pair.left.value 10) assert (== (change'),
            'moved_destructured_parent': text.replace('assert (== count 7)', 'assert (== root.branch.left.value 10) assert (== count 7)'),
            'live_tree_exit': text.replace(disposal, ''),
            'moved_child': text.replace('let mut root: Tree', 'assert (== left.value 10) let mut root: Tree'),
            'argument_nominal': text.replace('&mut root.branch.left &mut root.branch.right', '&mut root.other &mut root.branch.right'),
            'equal_exclusive': text.replace('&mut root.branch.left &mut root.branch.right', '&mut root.branch.left &mut root.branch.left'),
            'shared_then_exclusive': shared_first,
            'exclusive_then_shared': exclusive_first,
            'wrong_field': text.replace('&mut root.branch.left', '&mut root.branch.missing'),
            'scalar_endpoint': text.replace('&mut root.branch.left', '&mut root.branch.left.value'),
            'ancestor_formal': text.replace('first: &mut Leaf', 'first: &mut Pair'),
            'incomplete_pattern': text.replace('Tree { count, other, branch } = root', 'Tree { branch } = root'),
            'inline_child': text.replace('right: right, left: left', 'right: right, left: Leaf { value: 10, enabled: true }'),
            'forward_layout': pair + text.replace(pair, ''),
            'failed_shadow': text.replace('shadow main { assert true }', 'shadow main { assert false }'),
        }
        for name, content in cases.items():
            source = self.work / (name + '.nano')
            source.write_text(content)
            for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
                output = self.work / 'nested-preserved.nvm'
                output.write_bytes(b'accepted-output')
                result = subprocess.run([ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', output],
                                        cwd=ROOT, capture_output=True, text=True, timeout=60)
                self.assertGreater(result.returncode, 0, (name, compiler, result.stderr))
                self.assertEqual(output.read_bytes(), b'accepted-output')
            # Raw lowering does not execute shadows or perform the source
            # ownership check that forbids implicit disposal of live trees.
            if name != 'failed_shadow':
                for emitter in self.emitters:
                    output = self.work / 'nested-preserved.nasm'
                    output.write_text('accepted-output')
                    result = subprocess.run([emitter, source, '-o', output], cwd=ROOT,
                                            capture_output=True, text=True, timeout=60)
                    self.assertGreater(result.returncode, 0, (name, emitter, result.stderr))
                    self.assertEqual(output.read_text(), 'accepted-output')

    def test_nested_depth_boundary(self):
        def source_at_depth(depth):
            records = 'resource struct Layer0 { value: int }\n'
            records += ''.join(f'resource struct Layer{i} {{ child: Layer{i-1} }}\n'
                               for i in range(1, depth + 1))
            body = 'let owner0: Layer0 = Layer0 { value: 9 }\n'
            body += ''.join(f'let owner{i}: Layer{i} = Layer{i} {{ child: owner{i-1} }}\n'
                            for i in range(1, depth + 1))
            body += f'assert (== (read &owner{depth}' + '.child' * depth + ') 9)\n'
            body += f'let Layer{depth} {{ child }} = owner{depth}\n'
            body += ''.join(f'let Layer{i} {{ child }} = child\n' for i in range(depth - 1, 0, -1))
            body += 'let Layer0 { value } = child assert (== value 9)\n'
            return (records + 'fn read(view: &Layer0) -> int { return view.value }\n'
                    + 'shadow read {\n' + body + '}\nfn main() -> int {\n' + body
                    + 'return 0 }\nshadow main { assert true }\n')
        for depth in (32, 33):
            source = self.work / f'depth{depth}.nano'
            source.write_text(source_at_depth(depth))
            baseline = None
            for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
                module = self.work / f'depth{depth}-{compiler}.nvm'
                module.write_bytes(b'accepted-output')
                result = subprocess.run([ROOT / 'bin' / compiler, source, '--emit-nvm', *(['--strip-debug'] if compiler == 'nano_virt' else []), '-o', module],
                                        cwd=ROOT, capture_output=True, text=True, timeout=60)
                if depth == 33:
                    self.assertGreater(result.returncode, 0, (compiler, result.stderr))
                    self.assertEqual(module.read_bytes(), b'accepted-output')
                    continue
                self.assertEqual(result.returncode, 0, result.stderr)
                current = self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout
                if baseline is None:
                    baseline = current
                self.assertEqual(current, baseline)
                self.execute_pair(module)

    def test_selected_shadows_execute_and_preserve_suffix_identity(self):
        source = self.work / 'selected.nano'
        source.write_text((FIXTURES / 'source_borrow_exclusive.nano').read_text())
        passing = None
        for tool in self.shadow_tools:
            text = self.command(tool, source, 0, 'raw').stdout
            if passing is None:
                passing = text
            self.assertEqual(text, passing)
            assembly, module = self.work / 'shadow.nasm', self.work / 'shadow.nvm'
            assembly.write_text(text)
            self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
            self.command(ROOT / 'bin/nano_vm', '--check-shadows', module)
            self.execute_pair(module)
        source.write_text(source.read_text().replace('shadow main { assert true }',
                                                   'shadow main { assert false }'))
        for tool in self.shadow_tools:
            text = self.command(tool, source, 0, 'raw').stdout
            assembly.write_text(text)
            self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
            failed = subprocess.run([ROOT / 'bin/nano_vm', '--check-shadows', module],
                                    cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(failed.returncode, 0)
            self.execute_pair(module, expected=1)
            # My empty selected suffix has a proved owner-free scalar entry.
            assembly.write_text(self.command(tool, source, 2, 'raw').stdout)
            self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
            self.command(ROOT / 'bin/nano_vm', '--check-shadows', module)
            self.execute_pair(module)

    def test_false_borrowed_helper_cleans_actual_caller_owners(self):
        source = self.work / 'helper-assertion.nano'
        text = (FIXTURES / 'source_borrow_exclusive.nano').read_text()
        source.write_text(text.replace('return view.value', 'assert false return view.value'))
        for tool in self.shadow_tools:
            assembly, module = self.work / 'failed-helper.nasm', self.work / 'failed-helper.nvm'
            assembly.write_text(self.command(tool, source, 0, 'raw').stdout)
            self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
            self.execute_pair(module, expected=1)

    def test_owned_suffix_and_synthetic_name_collision(self):
        text = (FIXTURES / 'source_borrow_shared.nano').read_text()
        text = text.replace('read', '__nanoisa_shadow_entry')
        # The first shadow fails; the selected second shadow still owns a record.
        start = text.index('shadow __nanoisa_shadow_entry {')
        end = text.index('fn main()')
        shadow = text[start:end]
        text = text[:start] + shadow.replace('assert active', 'assert false') + shadow + text[end:]
        source = self.work / 'collision.nano'
        source.write_text(text)
        baseline = None
        for tool in self.shadow_tools:
            emitted = self.command(tool, source, 1, 'raw').stdout
            self.assertIn('.function __nanoisa_shadow_entry_ 0', emitted)
            self.assertIn('.function __nanoisa_shadow_entry 1', emitted)
            if baseline is None:
                baseline = emitted
            self.assertEqual(emitted, baseline)
            assembly, module = self.work / 'suffix.nasm', self.work / 'suffix.nvm'
            assembly.write_text(emitted)
            self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
            self.command(ROOT / 'bin/nano_vm', '--check-shadows', module)
            self.execute_pair(module)
        # C-seed and canonical publication retain the same helper spelling too.
        source.write_text(text.replace('assert false', 'assert active'))
        for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
            module = self.work / (compiler + '-collision.nvm')
            self.command(ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', module)
            self.execute_pair(module)

    def test_false_shadow_and_unsupported_graph_preserve_publication(self):
        text = (FIXTURES / 'source_borrow_shared.nano').read_text()
        cases = {
            'false helper': text.replace('return view.value', 'assert false return view.value'),
            'scalar-only selected shadows': text[:text.index('shadow read {')] + '\n' + text[text.index('fn main()'):],
            'false shadow': text.replace('shadow main { assert true }', 'shadow main { assert false }'),
            'shadow calls main': text.replace('shadow main { assert true }', 'shadow main { assert (== (main) 0) }'),
            'extra function': text + '\nfn extra() -> int { return 1 } shadow extra { assert true }\n',
            'loop resource declaration': text.replace('return 0', 'while false { let extra: Counter = Counter { value: 1, active: true } } return 0'),
            'heap field': text.replace('resource struct Other { value: int, active: bool }',
                                       'resource struct Other { label: string }'),
            'wrong nominal': text.replace('let owner: Counter = Counter { value: 12, active: false }',
                                          'let owner: Other = Other { value: 12, active: false }'),
        }
        source, output = self.work / 'refusal.nano', self.work / 'prior.nvm'
        for label, body in cases.items():
            source.write_text(body)
            for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
                with self.subTest(case=label, compiler=compiler):
                    output.write_bytes(b'previous verified publication')
                    result = subprocess.run([ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    if label == 'scalar-only selected shadows' and compiler != 'nano_virt':
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertNotEqual(output.read_bytes(), b'previous verified publication')
                        self.execute_pair(output)
                        continue
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertGreater(result.returncode, 0, 'I require an ordinary reported refusal')
                    self.assertEqual(output.read_bytes(), b'previous verified publication')
                    if label in ('false shadow', 'false helper'):
                        self.assertIn('shadow', (result.stdout + result.stderr).lower())

    def test_consuming_source_calls_and_exact_metadata(self):
        for fixture in ('source_consuming_leaf.nano', 'source_consuming_nested.nano',
                        'source_consuming_bool.nano'):
            source = FIXTURES / fixture
            seed = self.work / 'consume-seed.nvm'
            self.command(ROOT / 'bin/nano_virt', source, '--emit-nvm', '--strip-debug', '-o', seed)
            baseline = self.command(ROOT / 'bin/nanoisa', 'dump', seed).stdout
            self.assertIn('CALL 1', baseline)
            self.assertNotIn('CALL_REF', baseline)
            self.assertIn('.ownership', baseline)
            self.assertIn('.layouts', baseline)
            names = self.names_and_strip(seed)
            self.assertTrue(any(row[0] == 'take' and row[1] == '0' and row[4] == 'owner' for row in names))
            for emitter in self.emitters:
                with self.subTest(fixture=fixture, emitter=emitter):
                    assembly, module = self.work / 'consume.nasm', self.work / 'consume.nvm'
                    self.command(emitter, source, '-o', assembly)
                    self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                    self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                    self.assertEqual(self.names_and_strip(module), names)
            for compiler in ('nanoc_stage1', 'nanoc_stage2'):
                module = self.work / (compiler + '-consume.nvm')
                self.command(ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', module)
                self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
                self.execute_pair(module)

    def test_consuming_selected_shadows_and_terminal_cleanup(self):
        source = self.work / 'consuming-shadows.nano'
        original = (FIXTURES / 'source_consuming_nested.nano').read_text()
        for failure in (False, True):
            source.write_text(original.replace('assert owner.right.active',
                                               'assert false' if failure else 'assert owner.right.active'))
            baseline = None
            for tool in self.shadow_tools:
                text = self.command(tool, source, 0, 'raw').stdout
                if baseline is None:
                    baseline = text
                self.assertEqual(text, baseline)
                self.assertIn('CALL 1', text)
                assembly, module = self.work / 'consume-shadow.nasm', self.work / 'consume-shadow.nvm'
                assembly.write_text(text)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                self.command(ROOT / 'bin/nano_vm', '--check-shadows', module, expected=1 if failure else 0)
                self.execute_pair(module, expected=1 if failure else 0)
            if failure:
                for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
                    output = self.work / 'consume-prior.nvm'
                    output.write_bytes(b'previous verified publication')
                    result = subprocess.run([ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=90)
                    self.assertGreater(result.returncode, 0, result.stderr)
                    self.assertIn('shadow', (result.stdout + result.stderr).lower())
                    self.assertEqual(output.read_bytes(), b'previous verified publication')

    def test_consuming_source_refusals_preserve_publication(self):
        base = '''resource struct Leaf { value: int, active: bool }
resource struct Other { value: int, active: bool }
fn take(owner: Leaf) -> int { let Leaf { value, active } = owner return value }
shadow take { let item: Leaf = Leaf { value: 7, active: true } assert (== (take item) 7) }
fn main() -> int { let item: Leaf = Leaf { value: 7, active: true } assert (== (take item) 7) return 0 }
shadow main { assert true }
'''
        cases = {
            'moved': base.replace('return 0', 'assert (== (take item) 7) return 0'),
            'nominal': base.replace('let item: Leaf = Leaf {', 'let item: Other = Other {'),
            'unconsumed': base.replace('let Leaf { value, active } = owner return value', 'return owner.value'),
            'mixed': base.replace('take(owner: Leaf)', 'take(owner: Leaf, view: &Leaf)')
                .replace('assert (== (take item) 7)', 'let view: Leaf = Leaf { value: 1, active: true } assert (== (take item &view) 7) let Leaf { value, active } = view'),
            'projected': base.replace('fn take', 'resource struct Pair { left: Leaf, right: Leaf }\nfn take', 1)
                .replace('assert (== (take item) 7)', 'let second: Leaf = Leaf { value: 1, active: true } let pair: Pair = Pair { left: item, right: second } assert (== (take pair.left) 7)'),
        }
        positives = {
            'constructed': base.replace('let item: Leaf = Leaf { value: 7, active: true } assert (== (take item) 7)',
                                       'assert (== (take Leaf { value: 7, active: true }) 7)'),
            'deeper': base.replace('let Leaf { value, active } = owner return value', 'return (other owner)')
                + 'fn other(owner: Leaf) -> int { let Leaf { value, active } = owner return value } shadow other { assert true }\n',
            'two_owned': base.replace('take(owner: Leaf)', 'take(owner: Leaf, second: Leaf)')
                .replace('let Leaf { value, active } = owner', 'let Leaf { value, active } = second let Leaf { value, active } = owner')
                .replace('assert (== (take item) 7)', 'let second: Leaf = Leaf { value: 1, active: true } assert (== (take item second) 7)'),
        }
        for name, text in positives.items():
            self.graph_positive('former-' + name, text)
        for name, text in cases.items():
            source = self.work / ('consume-refusal-' + name + '.nano')
            source.write_text(text)
            for compiler in [ROOT / 'bin' / x for x in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2')] + self.emitters:
                with self.subTest(case=name, compiler=compiler):
                    output = self.work / 'consume-refused.nvm'
                    output.write_bytes(b'previous verified publication')
                    args = [compiler, source]
                    if compiler not in self.emitters:
                        args.append('--emit-nvm')
                    result = subprocess.run([*args, '-o', output], cwd=ROOT, capture_output=True, text=True, timeout=90)
                    self.assertGreater(result.returncode, 0, result.stderr)
                    self.assertEqual(output.read_bytes(), b'previous verified publication')
                    self.assertNotRegex(result.stdout + result.stderr, r'(?i)parse (?:error|failed)|unexpected token')
                    self.assertRegex(result.stdout + result.stderr,
                                     r'(?i)owner|resource|consum|nominal|borrow|helper|parameter|live|type mismatch|expected|named')

    def graph_positive(self, name, text, expected_output=None, expected_shadow_output=None):
        source = self.work / ('graph-' + name + '.nano')
        source.write_text(text)
        baseline = None
        for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
            module = self.work / (compiler + '-graph.nvm')
            self.command(ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', module)
            actual = self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout
            if baseline is None:
                baseline = actual
            self.assertEqual(actual, baseline)
            self.names_and_strip(module, expected_output=expected_output)
        for emitter in self.emitters:
            assembly, module = self.work / 'graph.nasm', self.work / 'graph.nvm'
            self.command(emitter, source, '-o', assembly)
            self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
            self.assertEqual(self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout, baseline)
            self.execute_pair(module, expected_output=expected_output)
        shadow_dump = None
        for tool in [ROOT / 'obj/borrow_shadow_names', *self.shadow_tools]:
            args = (source,) if tool.name == 'borrow_shadow_names' else (source, 0, 'raw')
            assembly, module = self.work / 'graph-shadow.nasm', self.work / 'graph-shadow.nvm'
            assembly.write_text(self.command(tool, *args).stdout)
            self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
            actual = self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout
            if shadow_dump is None:
                shadow_dump = actual
            self.assertEqual(actual, shadow_dump)
            self.execute_pair(module, expected_output=expected_shadow_output)
        return baseline, shadow_dump

    def test_owned_value_graph_results_and_shadow_main(self):
        text = (FIXTURES / 'source_owned_value_graph.nano').read_text()
        baseline, shadow = self.graph_positive('factories', text)
        self.assertIn('.function make_handle 0', baseline)
        self.assertIn('struct 1', baseline)
        self.assertIn('void 0', baseline)
        self.assertIn('.function main 0', shadow)
        self.assertIn('.function __nanoisa_shadow_entry 0', shadow)
        for choice in ('true', 'false'):
            changed = text.replace('return h', 'if ' + choice + ' { return h } else { (consume h) return (make_handle) }')
            self.graph_positive('return-' + choice, changed)
        # I retain every shadow, including a failure in an ordinary main call.
        source = self.work / 'graph-failed-shadow.nano'
        source.write_text(text.replace('assert (== (main) 0)', 'assert (== (main) 1)'))
        for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
            output = self.work / 'graph-prior.nvm'
            output.write_bytes(b'previous verified publication')
            result = subprocess.run([ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', output],
                                    cwd=ROOT, capture_output=True, text=True, timeout=180)
            self.assertGreater(result.returncode, 0, result.stderr)
            self.assertRegex(result.stdout + result.stderr, r'(?i)shadow|assert')
            self.assertEqual(output.read_bytes(), b'previous verified publication')

    def test_owned_value_graph_eight_frames(self):
        text = 'resource struct Leaf { value: int }\n'
        for index in range(1, 6):
            target = 'factory' if index == 5 else 'relay' + str(index + 1)
            text += f'fn relay{index}() -> Leaf {{ return ({target}) }} shadow relay{index} {{ assert true }}\n'
        text += 'fn factory() -> Leaf { return Leaf { value: 42 } } shadow factory { assert true }\n'
        text += 'fn main() -> int { let owner: Leaf = (relay1) let Leaf { value } = owner assert (== value 42) return 0 } shadow main { assert (== (main) 0) }\n'
        _, shadow = self.graph_positive('eight-frames', text)
        self.assertEqual(shadow.count('.function '), 8)

    def test_owned_value_graph_eight_positional_arguments(self):
        for swapped in (False, True):
            left, right = ('Right', 'Left') if swapped else ('Left', 'Right')
            text = 'resource struct Left { value: int } resource struct Right { value: int }\n'
            text += f'fn consume(a: int, one: {left}, yes: bool, two: {right}, b: int, three: {left}, no: bool, four: {right}) -> void {{ '
            text += 'assert (== a 7) assert yes assert (== b 9) assert (not no) '
            for typ, name, expected in ((left, 'one', 1), (right, 'two', 2), (left, 'three', 3), (right, 'four', 4)):
                text += f'let {typ} {{ value }} = {name} assert (== value {expected}) '
            text += '} shadow consume { assert true }\nfn main() -> int { '
            for typ, name, value in ((left, 'one', 1), (right, 'two', 2), (left, 'three', 3), (right, 'four', 4)):
                text += f'let {name}: {typ} = {typ} {{ value: {value} }} '
            text += '(consume (+ one.value 6) one true two 9 three false four) return 0 } shadow main { assert (== (main) 0) }\n'
            baseline, _ = self.graph_positive('eight-args-' + str(swapped), text)
            self.assertIn('.parameters 1 int struct bool struct int struct bool struct', baseline)

    def test_owned_value_graph_refusals_preserve_output(self):
        base = (FIXTURES / 'source_owned_value_graph.nano').read_text()
        cases = {
            'late_observation': base.replace('(forward 7 h true)', '(forward h.value h (== h.value 42))'),
            'duplicate_move': base.replace('yes: bool', 'other: Handle').replace('assert yes', '(consume other)').replace('(forward 7 h true)', '(forward 7 h h)'),
            'discard_result': base.replace('let k: Handle = (forward 7 h true)\n    (consume k)', '(forward 7 h true)'),
            'wrong_return_nominal': base.replace('resource struct Handle', 'resource struct Other { value: int }\nresource struct Handle', 1).replace('return Handle { value: 42 }', 'return Other { value: 42 }'),
            'unconsumed': base.replace('return h', 'return Handle { value: 42 }'),
            'constructed_actual': base.replace('(forward 7 h true)', '(forward 7 Handle { value: 42 } true)'),
            'missing_result': base.replace('return h', 'let Handle { value } = h'),
            'scalar_result': base.replace('return h', 'let Handle { value } = h return value'),
            'cycle': base.replace('return Handle { value: 42 }', 'return (make_handle)'),
            'function_shadow': base.replace('(consume k)\n    return 0', 'let consume: int = 1 (consume k)\n    return 0'),
            'too_many_functions': base + ''.join(f'fn extra{i}() -> int {{ return 1 }} shadow extra{i} {{ assert true }}\n' for i in range(5)),
        }
        for name, text in cases.items():
            source = self.work / ('graph-refused-' + name + '.nano')
            source.write_text(text)
            for compiler in [ROOT / 'bin' / x for x in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2')] + self.emitters:
                with self.subTest(case=name, compiler=compiler):
                    output = self.work / 'graph-refused.output'
                    output.write_bytes(b'previous verified publication')
                    args = [compiler, source]
                    if compiler not in self.emitters:
                        args.append('--emit-nvm')
                    result = subprocess.run([*args, '-o', output], cwd=ROOT, capture_output=True, text=True, timeout=180)
                    self.assertGreater(result.returncode, 0, result.stderr)
                    self.assertEqual(output.read_bytes(), b'previous verified publication')
                    self.assertNotRegex(result.stdout + result.stderr, r'(?i)parse (?:error|failed)|unexpected token')
                    self.assertRegex(result.stdout + result.stderr, r'(?i)owner|resource|consum|nominal|result|return|live|type|expected|named|function|graph|shadow|call|exact constructor')

    def test_owned_local_routing_and_shadow_only_admission(self):
        local = 'resource struct Leaf { value: int }\nfn main() -> int { let owner: Leaf = Leaf { value: 3 } let Leaf { value } = owner assert (== value 3) return 0 }\nshadow main { let owner: Leaf = Leaf { value: 7 } let Leaf { value } = owner assert (== value 7) }\n'
        baseline, _ = self.graph_positive('local-only', local)
        self.assertIn('OWN_PACK', baseline)
        self.assertIn('.ownership', baseline)
        shadow_only = local.replace('let owner: Leaf = Leaf { value: 3 } let Leaf { value } = owner assert (== value 3) ', '')
        source = self.work / 'shadow-only-owner.nano'
        source.write_text(shadow_only)
        shadow_dump = None
        for tool in [ROOT / 'obj/borrow_shadow_names', *self.shadow_tools]:
            args = (source,) if tool.name == 'borrow_shadow_names' else (source, 0, 'raw')
            assembly, module = self.work / 'only-shadow.nasm', self.work / 'only-shadow.nvm'
            assembly.write_text(self.command(tool, *args).stdout)
            self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
            actual = self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout
            self.assertIn('OWN_PACK', actual)
            if shadow_dump is None:
                shadow_dump = actual
            self.assertEqual(actual, shadow_dump)
            self.execute_pair(module)
        for label, text in [('no-transfer', shadow_only), ('global', local + '\nlet global_owner: Leaf = Leaf { value: 2 }\n')]:
            source.write_text(text)
            for compiler in [ROOT / 'bin' / x for x in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2')] + self.emitters:
                with self.subTest(case=label, compiler=compiler):
                    output = self.work / 'routing-preserved.output'
                    output.write_bytes(b'previous verified publication')
                    args = [compiler, source]
                    if compiler not in self.emitters:
                        args.append('--emit-nvm')
                    result = subprocess.run([*args, '-o', output], cwd=ROOT, capture_output=True, text=True, timeout=180)
                    if label == 'no-transfer' and compiler.name in ('nanoc_stage1', 'nanoc_stage2'):
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertNotEqual(output.read_bytes(), b'previous verified publication')
                        self.execute_pair(output)
                        failed_source = self.work / 'failed-owned-shadow.nano'
                        self.assertIn('assert (== value 7)', text)
                        failed_source.write_text(text.replace('assert (== value 7)', 'assert false'))
                        output.write_bytes(b'previous verified publication')
                        failed = subprocess.run([compiler, failed_source, '--emit-nvm', '-o', output],
                                                cwd=ROOT, capture_output=True, text=True, timeout=180)
                        self.assertGreater(failed.returncode, 0, failed.stdout + failed.stderr)
                        self.assertIn('shadow', (failed.stdout + failed.stderr).lower())
                        self.assertEqual(output.read_bytes(), b'previous verified publication')
                        continue
                    self.assertGreater(result.returncode, 0, result.stderr)
                    self.assertEqual(output.read_bytes(), b'previous verified publication')
                    self.assertNotRegex(result.stdout + result.stderr, r'(?i)parse (?:error|failed)|unexpected token')
                    self.assertRegex(result.stdout + result.stderr, r'(?i)owner|resource|owned|global|transfer')

    def test_inline_owner_wrappers_refuse_without_ordinary_fallback(self):
        cases = {
            'tuple': 'let wrapped: (int, Leaf) = (2, Leaf { value: 1 })',
            'array': 'let wrapped: array<Leaf> = [Leaf { value: 1 }]',
            'field': 'let wrapped: int = Leaf { value: 1 }.value',
            'call': 'let wrapped: int = (unknown Leaf { value: 1 })',
            'set': 'let mut wrapped: int = 0 set wrapped Leaf { value: 1 }',
        }
        for name, body in cases.items():
            source = self.work / ('inline-owner-' + name + '.nano')
            source.write_text('resource struct Leaf { value: int }\nfn main() -> int { ' + body + ' return 0 } shadow main { assert true }\n')
            for compiler in [ROOT / 'bin' / x for x in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2')] + self.emitters:
                with self.subTest(case=name, compiler=compiler):
                    output = self.work / 'inline-prior.output'
                    output.write_bytes(b'previous verified publication')
                    args = [compiler, source]
                    if compiler not in self.emitters:
                        args.append('--emit-nvm')
                    result = subprocess.run([*args, '-o', output], cwd=ROOT, capture_output=True, text=True, timeout=180)
                    self.assertGreater(result.returncode, 0, result.stderr)
                    self.assertEqual(output.read_bytes(), b'previous verified publication')
                    self.assertNotRegex(result.stdout + result.stderr, r'(?i)parse (?:error|failed)|unexpected token')
                    self.assertRegex(result.stdout + result.stderr, r'(?i)owner|resource|scalar|exact|call|type|borrow|live|field')
                    if compiler in self.emitters:
                        self.assertIn('source borrow profile', result.stdout + result.stderr)

    def test_owned_value_graph_multiple_nested_owners(self):
        baseline, _ = self.graph_positive('nested-values', (FIXTURES / 'source_owned_value_nested.nano').read_text())
        self.assertIn('.parameters 1 struct struct int', baseline)
        self.assertGreaterEqual(baseline.count('OWN_UNPACK_LOCAL'), 3)


    def test_owned_string_unchanged_affine_example(self):
        text = (ROOT / 'examples/language/nl_affine_resource_demo.nano').read_text()
        baseline, shadows = self.graph_positive(
            'affine-demo-unchanged', text,
            b'audit.log\nresource closed\n',
            b'audit.log\nresource closed\nresource closed\naudit.log\nresource closed\n')
        self.assertIn('string', baseline)
        self.assertIn('PRINTLN', shadows)

    def owned_string_fixture(self):
        return r'''resource struct Leaf { value: int }
fn write(text: string, owner: Leaf, tail: string, number: int, yes: bool) -> void {
    (print text)
    (println tail)
    let Leaf { value } = owner
    assert (== value number)
    assert yes
}
shadow write { assert true }
fn forward(text: string, owner: Leaf) -> void {
    (write text owner "" 7 true)
}
shadow forward { assert true }
fn main() -> int {
    let owner: Leaf = Leaf { value: 7 }
    (forward "A\tB\rC\n\"\'\\0\q café" owner)
    (print "")
    return 0
}
shadow main { assert (== (main) 0) }
'''

    def test_owned_string_exact_bytes_and_forwarding(self):
        text = self.owned_string_fixture().replace(' café', ' \x01\x7f café')
        expected = b'A\tB\rC\n"\'\\0\\q \x01\x7f caf\xc3\xa9\n'
        # My raw source contains an escaped backslash before zero; it never
        # is a decoded NUL. Unknown escapes retain their backslash.
        self.graph_positive('string-bytes', text, expected, expected)

    def test_owned_string_refusals_preserve_publication(self):
        base = self.owned_string_fixture()
        cases = {
            'nul': base.replace('A\\tB', 'A\\0B'),
            'field': base.replace('value: int', 'value: string').replace('value: 7', 'value: "seven"'),
            'operation': base.replace('(print text)', '(print (str_concat text "!"))'),
            'argument': base.replace('owner "" 7 true', 'owner 1 7 true'),
            'result': base.replace('-> void {\n    (print text)', '-> string {\n    (print text)').replace('assert yes\n}', 'assert yes return text\n}'),
            'print_binding': base.replace('(print "")', 'let print: int = 2 (print "")'),
            'false_shadow': base.replace('shadow write { assert true }', 'shadow write { assert false }'),
        }
        for name, text in cases.items():
            source = self.work / ('string-refused-' + name + '.nano')
            source.write_text(text)
            compilers = [ROOT / 'bin' / x for x in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2')]
            if name != 'false_shadow':
                compilers += self.emitters
            for compiler in compilers:
                with self.subTest(case=name, compiler=compiler.name):
                    output = self.work / 'string-refused.output'
                    output.write_bytes(b'previous verified publication')
                    args = [compiler, source]
                    if compiler not in self.emitters:
                        args.append('--emit-nvm')
                    result = subprocess.run([*args, '-o', output], cwd=ROOT, capture_output=True, text=True, timeout=180)
                    self.assertGreater(result.returncode, 0, result.stderr)
                    self.assertEqual(output.read_bytes(), b'previous verified publication')
                    self.assertNotRegex(result.stdout + result.stderr, r'(?i)parse (?:error|failed)|unexpected token')
                    self.assertRegex(result.stdout + result.stderr, r'(?i)string|scalar|field|type|argument|shadow|assert|source borrow|call|builtin')

    def test_owned_string_output_precedes_assertion_cleanup(self):
        text = self.owned_string_fixture().replace('assert yes', 'assert false')
        source = self.work / 'string-assert.nano'
        source.write_text(text)
        for emitter in self.emitters:
            assembly, module = self.work / 'string-assert.nasm', self.work / 'string-assert.nvm'
            self.command(emitter, source, '-o', assembly)
            self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
            self.execute_pair(module, expected=1,
                              expected_output=b'A\tB\rC\n"\'\\0\\q caf\xc3\xa9\n')


    def temporary_owner_fixture(self):
        return r'''resource struct Leaf { value: int }
fn scalar(value: int, text: string) -> int { (print text) return value }
shadow scalar { assert true }
fn make(value: int, text: string) -> Leaf { (print text) return Leaf { value: value } }
shadow make { assert true }
fn forward(owner: Leaf) -> Leaf { return owner }
shadow forward { assert true }
fn take(first: Leaf, second: Leaf, last: int) -> int {
    let Leaf { value } = first
    let left: int = value
    let Leaf { value } = second
    assert (== last 3)
    return (+ left value)
}
shadow take { assert true }
fn main() -> int {
    for index in (range 0 3) {
        assert (== (take Leaf { value: (scalar 1 "A") } (forward (make 2 "B")) (scalar 3 "C")) 3)
    }
    return 0
}
shadow main { assert (== (main) 0) }
'''

    def test_float_local_unsafe_original_pattern(self):
        from tests.test_owned_record_patterns import OwnedRecordPatterns, PREFIX
        cases = []
        def capture(source, accepted, stdout=None, diagnostic=None, prefix=PREFIX):
            self.assertTrue(accepted)
            cases.append(prefix + source)
        original = OwnedRecordPatterns()
        original.check_case = capture
        original.test_unsafe_pattern_keeps_outer_shadow()
        self.assertEqual(len(cases), 1)
        self.graph_positive('float-unsafe-original', cases[0])

    def test_float_local_unsafe_control_flow(self):
        from tests.test_owned_record_patterns import PREFIX
        text = PREFIX + """fn worker(owned: Handle, stop: bool) -> int {
    let mut value: float = 1.5
    let copy: float = value
    while false { unsafe { set value 99.0 } }
    let mut index: int = 0
    while (< index 3) {
        unsafe {
            set index (+ index 1)
            if stop { break }
            set value (+ value 1.0)
            continue
        }
    }
    assert (== copy 1.5)
    if stop { assert (== value 1.5) } else { assert (== value 4.5) }
    unsafe {
        let value: float = (- 2.5)
        assert (< value 0.0)
        let mut inf: float = 10000000000.0
        set inf (* inf inf)
        set inf (* inf inf)
        set inf (* inf inf)
        set inf (* inf inf)
        set inf (* inf inf)
        let nan: float = (- inf inf)
        assert (!= nan nan)
        assert (not (<= nan 1.0))
        assert (not (>= nan 1.0))
        return (close owned)
    }
}
shadow worker { assert (== (worker Handle { fd: 7 } true) 7) }
fn main() -> int {
    assert (== (worker Handle { fd: 8 } false) 8)
    assert (== (worker Handle { fd: 9 } true) 9)
    return 0
}
shadow main { assert (== (main) 0) }
"""
        self.graph_positive('float-unsafe-control', text)

    def test_float_local_in_borrowed_control_flow(self):
        text = (FIXTURES / 'source_borrow_control_flow.nano').read_text()
        text = text.replace('set total (+ total 2)',
                            'let value: float = 2.0 assert (== value 2.0) set total (+ total 2)')
        self.graph_positive('float-borrowed-branch', text)

    def test_float_local_unsafe_refusals_preserve_publication(self):
        from tests.test_owned_record_patterns import PREFIX
        base = PREFIX + """fn main() -> int {
    let value: float = 2.5
    unsafe { assert (== (close Handle { fd: 42 }) 42) }
    assert (== value 2.5)
    return 0
}
shadow main { assert (== (main) 0) }
"""
        cases = {
            'false-shadow': base.replace('shadow main { assert (== (main) 0) }', 'shadow main { assert false }'),
            'mixed-tag': base.replace('(== value 2.5)', '(== (+ value 1) 3.5)'),
            'escaped-name': base.replace('unsafe { assert', 'unsafe { let inner: float = 1.5 assert').replace('(== value 2.5)', '(== inner 1.5)'),
        }
        for label, text in cases.items():
            source = self.work / (label + '.nano')
            source.write_text(text)
            for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
                with self.subTest(case=label, compiler=compiler):
                    output = self.work / 'float-refused.nvm'
                    output.write_bytes(b'previous verified publication')
                    result = subprocess.run([ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=180)
                    self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(output.read_bytes(), b'previous verified publication')
                    self.assertNotRegex(result.stdout + result.stderr, r'(?i)parse (?:error|failed)|unexpected token')
                    if label == 'false-shadow':
                        self.assertIn('shadow', (result.stdout + result.stderr).lower())

    def test_temporary_owners_restore_original_pattern_sources(self):
        from tests.test_owned_record_patterns import OwnedRecordPatterns, PREFIX
        cases = []
        def capture(source, accepted, stdout=None, diagnostic=None, prefix=PREFIX):
            self.assertTrue(accepted)
            cases.append((prefix + source, stdout))
        original = OwnedRecordPatterns()
        original.check_case = capture
        original.test_scalar_terminal_operation()
        original.test_empty_resource_pattern()
        original.test_initializer_evaluated_once()
        cases.append(((ROOT / 'tests/test_resource_tracking.nano').read_text(), None))
        for index, (text, output) in enumerate(cases):
            with self.subTest(original_case=index):
                self.graph_positive('original-pattern-' + str(index), text,
                                    None if output is None else output.encode())

    def test_temporary_owner_actuals_preserve_order_and_roots(self):
        baseline, shadows = self.graph_positive(
            'temporary-actuals', self.temporary_owner_fixture(), b'ABCABCABC', b'ABCABCABC')
        self.assertIn('OWN_PACK', baseline)
        self.assertIn('.function forward', shadows)

    def test_temporary_owner_factory_failure_cleans_prepared_arguments(self):
        text = self.temporary_owner_fixture().replace(
            '(print text) return Leaf', '(print text) assert false return Leaf')
        source = self.work / 'temporary-factory-failure.nano'
        source.write_text(text)
        for emitter in self.emitters:
            assembly, module = self.work / 'temporary-failure.nasm', self.work / 'temporary-failure.nvm'
            self.command(emitter, source, '-o', assembly)
            self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
            self.execute_pair(module, expected=1, expected_output=b'AB')

    def test_temporary_owner_actual_refusals_preserve_output(self):
        base = self.temporary_owner_fixture()
        cases = {
            'constructor_nominal': base.replace('fn scalar', 'resource struct Other { value: int }\nfn scalar', 1)
                .replace('take Leaf { value:', 'take Other { value:'),
            'factory_nominal': base.replace('fn scalar', 'resource struct Other { value: int }\nfn scalar', 1)
                .replace('-> Leaf { (print text) return Leaf', '-> Other { (print text) return Other'),
            'missing_field': base.replace('Leaf { value: (scalar 1 "A") }', 'Leaf {}'),
            'duplicate_field': base.replace('Leaf { value: (scalar 1 "A") }', 'Leaf { value: 1, value: 2 }'),
        }
        for name, text in cases.items():
            source = self.work / ('temporary-refused-' + name + '.nano')
            source.write_text(text)
            for compiler in [ROOT / 'bin' / x for x in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2')] + self.emitters:
                with self.subTest(case=name, compiler=compiler.name):
                    output = self.work / 'temporary-refused.output'
                    output.write_bytes(b'previous verified publication')
                    args = [compiler, source]
                    if compiler not in self.emitters:
                        args.append('--emit-nvm')
                    result = subprocess.run([*args, '-o', output], cwd=ROOT, capture_output=True, text=True, timeout=180)
                    self.assertGreater(result.returncode, 0, result.stderr)
                    self.assertEqual(output.read_bytes(), b'previous verified publication')
                    self.assertNotRegex(result.stdout + result.stderr, r'(?i)parse (?:error|failed)|unexpected token')
                    self.assertRegex(result.stdout + result.stderr, r'(?i)owner|resource|nominal|field|type|expected|duplicate|exact owned call result')
                    if name == 'factory_nominal' and compiler in self.emitters:
                        self.assertIn('exact nominal owned call result', result.stdout + result.stderr)
                        self.assertNotIn('exact positional scalar argument', result.stdout + result.stderr)

    def test_transitive_wrappers_restore_unchanged_affine_integration(self):
        text = (ROOT / 'tests/test_affine_integration.nano').read_text()
        baseline, shadows = self.graph_positive('transitive-original', text, b'File closed\n' * 5)
        self.assertIn('Connection', baseline)
        self.assertIn('OWN_PACK', baseline)
        self.assertIn('OWN_UNPACK_LOCAL', shadows)

    def transitive_wrapper_fixture(self):
        return '''resource struct Leaf { value: int }
struct Inner { leaf: Leaf, yes: bool }
struct Outer { inner: Inner, extra: int }
fn scalar(value: int, text: string) -> int { (print text) return value }
shadow scalar { assert true }
fn take(owner: Outer) -> int {
    let Outer { inner, extra } = owner
    let Inner { leaf, yes } = inner
    let Leaf { value } = leaf
    assert yes
    return (+ value extra)
}
shadow take { assert true }
fn main() -> int {
    for index in (range 0 1) {
        let leaf: Leaf = Leaf { value: (scalar 7 "A") }
        let inner: Inner = Inner { leaf: leaf, yes: true }
        let owner: Outer = Outer { inner: inner, extra: (scalar 9 "B") }
        assert (== (take owner) 16)
        let second_leaf: Leaf = Leaf { value: (scalar 7 "A") }
        let second_inner: Inner = Inner { leaf: second_leaf, yes: true }
        assert (== (take Outer { inner: second_inner, extra: (scalar 9 "B") }) 16)
    }
    return 0
}
shadow main { assert (== (main) 0) }
'''

    def test_transitive_wrapper_arguments_preserve_once_order_and_children(self):
        baseline, shadows = self.graph_positive('transitive-arguments', self.transitive_wrapper_fixture(), b'ABAB', b'ABAB')
        self.assertIn('.ownership', baseline)
        self.assertIn('.parameters 2 struct', baseline)
        self.assertGreaterEqual(shadows.count('OWN_UNPACK_LOCAL'), 3)

    def test_transitive_wrapper_refusals_preserve_output(self):
        base = self.transitive_wrapper_fixture()
        cases = {
            'wrong_nominal': base.replace('struct Outer', 'struct Other { leaf: Leaf, yes: bool }\nstruct Outer', 1)
                .replace('let inner: Inner = Inner', 'let inner: Other = Other'),
            'inline_child': base.replace('leaf: leaf, yes:', 'leaf: Leaf { value: 7 }, yes:', 1),
            'missing_field': base.replace(', extra: (scalar 9 "B")', ''),
            'duplicate_field': base.replace('yes: true', 'yes: true, yes: false'),
            'managed_string_order': base.replace('extra: int', 'extra: string').replace('extra: (scalar 9 "B")', 'extra: "B"')
                .replace('return (+ value extra)', 'assert (< extra "Z") return value'),
            'managed_array': base.replace('extra: int', 'extra: array<float>').replace('extra: (scalar 9 "B")', 'extra: [1.5]')
                .replace('return (+ value extra)', 'return value'),
            'forward_child': base.replace('resource struct Leaf { value: int }\nstruct Inner { leaf: Leaf, yes: bool }',
                'struct Inner { leaf: Leaf, yes: bool }\nresource struct Leaf { value: int }'),
        }
        for name, text in cases.items():
            source = self.work / ('transitive-refused-' + name + '.nano')
            source.write_text(text)
            for compiler in [ROOT / 'bin' / x for x in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2')] + self.emitters:
                with self.subTest(case=name, compiler=compiler.name):
                    output = self.work / 'transitive-refused.output'
                    output.write_bytes(b'previous verified publication')
                    args = [compiler, source]
                    if compiler not in self.emitters:
                        args.append('--emit-nvm')
                    result = subprocess.run([*args, '-o', output], cwd=ROOT, capture_output=True, text=True, timeout=180)
                    self.assertGreater(result.returncode, 0, result.stderr)
                    self.assertEqual(output.read_bytes(), b'previous verified publication')
                    self.assertNotRegex(result.stdout + result.stderr, r'(?i)parse (?:error|failed)|unexpected token')
                    diagnostic = r'(?i)owner|resource|nominal|field|type|expected|duplicate|exact|earlier'
                    if name == 'managed_string_order':
                        diagnostic += r'|I require a supported scalar operator in my source borrow profile'
                    self.assertRegex(result.stdout + result.stderr, diagnostic)


    def nested_owner_result_source(self):
        return '''resource struct Leaf { value: int }
struct Inner { leaf: Leaf, yes: bool }
struct Outer { inner: Inner, extra: int }
fn scalar(value: int, text: string) -> int { (print text) return value }
shadow scalar { assert true }
fn make(flag: bool) -> Outer {
    let leaf: Leaf = Leaf { value: (scalar 7 "A") }
    let inner: Inner = Inner { leaf: leaf, yes: true }
    let owner: Outer = Outer { inner: inner, extra: (scalar 9 "B") }
    if flag { return owner } else { return owner }
}
shadow make { assert true }
fn relay(owner: Outer) -> Outer { return owner }
shadow relay { assert true }
fn take(owner: Outer) -> int {
    let Outer { inner, extra } = owner
    let Inner { leaf, yes } = inner
    let Leaf { value } = leaf
    assert yes
    return (+ value extra)
}
shadow take { assert true }
fn main() -> int {
    for index in (range 0 2) {
        let owner: Outer = (relay (make (== index 0)))
        assert (== (take owner) 16)
    }
    return 0
}
shadow main { assert (== (main) 0) }
'''

    def test_nested_owner_results_preserve_exact_transfer_and_order(self):
        baseline, shadows = self.graph_positive('nested-owner-results', self.nested_owner_result_source(), b'ABAB', b'ABAB')
        self.assertIn('.parameters 3 struct', baseline)
        self.assertIn('struct 1', baseline)
        self.assertGreaterEqual(shadows.count('OWN_UNPACK_LOCAL'), 3)

    def test_nested_owner_result_refusals_preserve_output(self):
        base = self.nested_owner_result_source()
        cases = {
            'wrong_result': base.replace('if flag { return owner } else { return owner }',
                'if flag { return 7 } else { return owner }'),
            'same_shape_nominal': base.replace('struct Outer',
                'struct Other { inner: Inner, extra: int }\nstruct Outer', 1)
                .replace('let owner: Outer = Outer', 'let owner: Other = Other', 1),
            'unconsumed_sibling': base.replace('if flag { return owner }',
                'let sibling: Leaf = Leaf { value: 99 }\n    if flag { return owner }', 1),
            'use_after_transfer': base.replace('assert (== (take owner) 16)',
                'assert (== (take owner) 16)\n        assert (== (take owner) 16)'),
        }
        for name, text in cases.items():
            source = self.work / ('nested-result-refused-' + name + '.nano')
            source.write_text(text)
            for compiler in [ROOT / 'bin' / x for x in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2')] + self.emitters:
                with self.subTest(case=name, compiler=compiler.name):
                    output = self.work / 'nested-result-refused.output'
                    output.write_bytes(b'previous verified publication')
                    args = [compiler, source]
                    if compiler not in self.emitters:
                        args.append('--emit-nvm')
                    result = subprocess.run([*args, '-o', output], cwd=ROOT, capture_output=True, text=True, timeout=180)
                    self.assertGreater(result.returncode, 0, result.stderr)
                    self.assertEqual(output.read_bytes(), b'previous verified publication')
                    self.assertNotRegex(result.stdout + result.stderr, r'(?i)parse (?:error|failed)|unexpected token')
                    self.assertRegex(result.stdout + result.stderr, r'(?i)owner|owned|resource|nominal|field|type|expected|exact|consum|live|move')


if __name__ == '__main__':
    unittest.main()
