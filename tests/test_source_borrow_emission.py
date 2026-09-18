"""I admit a closed source-reference profile only with executable ownership."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/nanoisa/fixtures'


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
                cls.command(ROOT / 'bin' / compiler, ROOT / 'src_nano/nanoisa_emit.nano', '-o', emitter)
                cls.emitters.append(emitter)
            shadow_tool = cls.work / (compiler + '-shadows')
            cls.command(ROOT / 'bin' / compiler, shadow_source, '-o', shadow_tool)
            cls.shadow_tools.append(shadow_tool)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def command(*args, expected=0):
        result = subprocess.run([str(arg) for arg in args], cwd=ROOT, capture_output=True,
                                text=True, timeout=180)
        if result.returncode != expected:
            raise AssertionError(f'{args}: {result.returncode}\n{result.stdout}\n{result.stderr}')
        return result

    def execute_pair(self, module, expected=0):
        self.command(ROOT / 'bin/nano_vm', '--verify-only', module)
        vm = self.command(ROOT / 'bin/nano_vm', module, expected=expected)
        source, native = self.work / 'native.c', self.work / 'native'
        self.command(ROOT / 'bin/nvm2c', module, '-o', source)
        self.command(os.environ.get('CC', 'cc'), '-std=c11', '-Wall', '-Wextra', '-Werror',
                     '-fsanitize=address,undefined', '-fno-omit-frame-pointer', source, '-o', native)
        result = subprocess.run([native], cwd=ROOT, capture_output=True, text=True, timeout=30,
                                env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1'})
        # My standalone wrapper maps internal assertion status 2 to exit status 1.
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        self.assertNotIn('Sanitizer', result.stderr)
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

    def names_and_strip(self, module):
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
        self.execute_pair(module)
        self.execute_pair(stripped)
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

    def test_return_paths_preserve_ownership_and_fallthrough(self):
        text = (FIXTURES / 'source_borrow_returns.nano').read_text()
        ending = 'if true { let code: int = 0 return code } else { return 1 }'
        variants = {
            'then': text,
            'else': text.replace(ending, 'if false { return 1 } else { let code: int = 0 return code }'),
            'loop': text.replace(ending, 'while true { let code: int = 0 return code } return 1'),
            'zero': text.replace(ending, 'while false { return 1 } return 0'),
        }
        for fixture, content in variants.items():
            source = self.work / ('returns-' + fixture + '.nano')
            source.write_text(content)
            seed = self.work / 'nested-seed.nvm'
            self.command(ROOT / 'bin/nano_virt', source, '--emit-nvm', '--strip-debug', '-o', seed)
            baseline = self.command(ROOT / 'bin/nanoisa', 'dump', seed).stdout
            self.assertIn('.ownership', baseline)
            self.assertIn('JMP_FALSE', baseline)
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
            'branch_float': text.replace('set total (+ total 2)', 'let value: float = 2.0'),
            'loop_string': text.replace('set j 0', 'let value: string = "unsupported"'),
            'branch_owner': text.replace('set total (+ total 2)', 'let moved: Pair = root'),
            'branch_destructure': text.replace('set total (+ total 2)', 'let Pair { left, right } = root'),
            'loop_destructure': text.replace('set j 0', 'let Pair { left, right } = root'),
            'branch_return': text.replace('set total (+ total 2)', 'return 0'),
            'loop_return': text.replace('set j 0', 'return 0'),
            'break': text.replace('set j 0', 'break'),
            'continue': text.replace('set j 0', 'continue'),
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
            if name not in ('failed_shadow', 'live_tree_exit'):
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
            # My owned profile requires an actual transfer in entry.
            refused = self.command(tool, source, 2, 'raw', expected=1)
            self.assertIn('owned transfer', refused.stdout)

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
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertGreater(result.returncode, 0, 'I require an ordinary reported refusal')
                    self.assertEqual(output.read_bytes(), b'previous verified publication')
                    if label in ('false shadow', 'false helper'):
                        self.assertIn('shadow', (result.stdout + result.stderr).lower())


if __name__ == '__main__':
    unittest.main()
