"""I preserve VM truthiness and effects on native conditional branches."""
try:
    from tests.sanitizer_options import asan_options
except ModuleNotFoundError:
    from sanitizer_options import asan_options
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeJumpTrue(unittest.TestCase):
    def run_command(self, args):
        return subprocess.run([str(x) for x in args], capture_output=True, text=True,
                              timeout=90, env={**os.environ, 'ASAN_OPTIONS': asan_options()})

    def checked(self, args):
        result = self.run_command(args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def source(self, body):
        return ('.entry main\n.types 1 0 0\n.string empty ""\n.string text "kept"\n'
                '.string other "other"\n.function main 0 3 0 int 1\n' + body +
                'PUSH_I64 0\nRET\n.end\n')

    def execute_pair(self, work, body):
        assembly, module, source, binary = (work / x for x in
                                           ('input.nasm', 'input.nvm', 'input.c', 'program'))
        assembly.write_text(self.source(body))
        self.checked([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
        vm = self.checked([ROOT / 'bin/nano_vm', module])
        self.checked([ROOT / 'bin/nvm2c', module, '-o', source])
        self.checked(['cc', '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                      '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                      source, '-o', binary])
        native = self.checked([binary])
        self.assertEqual(native.stdout, vm.stdout)
        return source.read_text()

    def test_truthiness_and_exactly_one_branch_effect(self):
        conditions = [('PUSH_BOOL 0', False), ('PUSH_BOOL 1', True),
                      ('PUSH_I64 0', False), ('PUSH_I64 -7', True),
                      ('PUSH_STR empty', True), ('PUSH_STR text', True),
                      ('ARR_NEW 1', True), ('HM_NEW 5 1', True),
                      ('PUSH_I64 4\nAGG_PACK 0 0 0 1', True),
                      ('PUSH_F64 0', False), ('PUSH_F64 -0', False),
                      ('PUSH_F64 1.5', True), ('PUSH_F64 nan', True),
                      ('LOAD_GLOBAL 1', False),
                      ('PUSH_STR empty\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0', True),
                      ('PUSH_F64 -0\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0', False),
                      ('PUSH_F64 2.5\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0', True)]
        body = ''
        for polarity in ('TRUE', 'FALSE'):
            for index, (condition, truth) in enumerate(conditions):
                i = f'{polarity}{index}'
                taken = truth if polarity == 'TRUE' else not truth
                body += (condition + f'\nJMP_{polarity} yes{i}\nPUSH_I64 1\nSTORE_LOCAL 0\nJMP done{i}\n'
                         f'yes{i}:\nPUSH_I64 2\nSTORE_LOCAL 0\ndone{i}:\n'
                         f'LOAD_LOCAL 0\nPUSH_I64 {2 if taken else 1}\nEQ\nASSERT\n')
        with tempfile.TemporaryDirectory(prefix='nano-true-truth-') as tmp:
            self.execute_pair(Path(tmp), body)

    def test_taken_stack_transfer_and_backward_owned_loop(self):
        body = ('PUSH_I64 7\nPUSH_STR text\nPUSH_BOOL 1\nJMP_TRUE join\n'
                'POP\nPOP\nPUSH_I64 9\nPUSH_STR other\njoin:\n'
                'PUSH_STR text\nEQ\nASSERT\nPUSH_I64 7\nEQ\nASSERT\n'
                'PUSH_STR text\nPUSH_STR empty\nSTR_CONCAT\nSTORE_GLOBAL 0\n'
                'PUSH_I64 0\nSTORE_LOCAL 0\nloop:\n'
                'PUSH_STR empty\nPUSH_STR empty\nSTR_CONCAT\nJMP_TRUE owned_true\n'
                'PUSH_BOOL 0\nASSERT\nowned_true:\n'
                'LOAD_GLOBAL 0\nPUSH_STR text\nEQ\nASSERT\n'
                'LOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 0\n'
                'LOAD_LOCAL 0\nPUSH_I64 1000\nI64_LT_S\nJMP_TRUE loop\n'
                'LOAD_LOCAL 0\nPUSH_I64 1000\nEQ\nASSERT\n')
        with tempfile.TemporaryDirectory(prefix='nano-true-loop-') as tmp:
            source = self.execute_pair(Path(tmp), body)
            self.assertIn('nmap_collect_if_needed();', source)

    def test_malformed_branch_preserves_existing_artifact(self):
        with tempfile.TemporaryDirectory(prefix='nano-true-invalid-') as tmp:
            work = Path(tmp)
            output, source = work / 'existing.nvm', work / 'invalid.nasm'
            for body in ('JMP_TRUE absent\n', 'PUSH_BOOL 1\nJMP_TRUE 10000\n',
                         'JMP_TRUE missing_operand\nmissing_operand:\n'):
                with self.subTest(body=body):
                    output.write_text('retained')
                    source.write_text(self.source(body))
                    result = self.run_command([ROOT / 'bin/nanoisa', 'asm', source, '-o', output])
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(output.read_text(), 'retained')

    def test_skipped_record_initialization_preserves_refusal(self):
        with tempfile.TemporaryDirectory(prefix='nano-true-initialization-') as tmp:
            work = Path(tmp)
            source, module, output = (work / x for x in ('input.nasm', 'input.nvm', 'existing.c'))
            source.write_text(self.source(
                'PUSH_BOOL 1\nJMP_TRUE done\nPUSH_I64 4\nAGG_PACK 0 0 0 1\nSTORE_LOCAL 0\n'
                'done:\nLOAD_LOCAL 0\nTYPE_CHECK 0\nASSERT\n'))
            self.checked([ROOT / 'bin/nanoisa', 'asm', source, '-o', module])
            self.checked([ROOT / 'bin/nano_vm', module])
            output.write_text('retained')
            result = self.run_command([ROOT / 'bin/nvm2c', module, '-o', output])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('record to optional', result.stderr)
            self.assertEqual(output.read_text(), 'retained')
