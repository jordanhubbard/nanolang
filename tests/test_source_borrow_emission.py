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
        self.assertEqual(result.returncode == 0, expected == 0, result.stdout + result.stderr)
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
            'loop': text.replace('return 0', 'while false { assert true } return 0'),
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
