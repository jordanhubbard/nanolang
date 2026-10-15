"""I preserve exact and tagged scalar fields through record-array writes."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from tests import test_native_byte_arrays as arrays
from tests.test_native_closures import CHURN


class RecordFieldTransport(unittest.TestCase):
    command = arrays.NativeByteArrays.command
    paired = arrays.NativeByteArrays.paired

    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='nano-record-field-transport-'))
        cls.sequence = 0
        print('I retain record-field transport evidence at', cls.work, flush=True)

    def test_scalar_write_matrix(self):
        scalars = [('int', 1, 'PUSH_I64 7\n', 'PUSH_I64 23\n'),
                   ('bool', 4, 'PUSH_BOOL 0\n', 'PUSH_BOOL 1\n'),
                   ('float', 3, 'PUSH_F64 1.25\n', 'PUSH_F64 -2.5\n'),
                   ('string', 5, 'PUSH_STR a\n', 'PUSH_STR a\nPUSH_STR b\nSTR_CONCAT\n')]
        for name, tag, old, new in scalars:
            for tagged_destination in (False, True):
                for operation in ('ARR_SET', 'ARR_PUSH'):
                    with self.subTest(kind=name, tagged_destination=tagged_destination, operation=operation):
                        box = f'ARR_LITERAL {tag} 1\nPUSH_I64 0\nARR_GET\n'
                        initial = old + (box if tagged_destination else '')
                        written = new + ('' if tagged_destination else box)
                        index = 0 if operation == 'ARR_SET' else 1
                        body = initial + 'AGG_PACK 0 0 0 1\nARR_LITERAL 8 1\nSTORE_LOCAL 0\n'
                        body += 'LOAD_LOCAL 0\n' + ('PUSH_I64 0\n' if operation == 'ARR_SET' else '')
                        body += written + 'AGG_PACK 0 0 0 1\n' + operation + '\nPOP\n'
                        body += 'CALL churn\n'
                        body += f'LOAD_LOCAL 0\nPUSH_I64 {index}\nARR_GET\nAGG_GET 0\n'
                        body += new + 'EQ\nASSERT\n'
                        self.paired(f'{name}-{tagged_destination}-{operation}', body,
                                    '.string a "a"\n.string b "b"\n' + CHURN, collections=True)

    def test_incompatible_boxed_payload_preserves_output(self):
        for operation in ('ARR_SET', 'ARR_PUSH'):
            with self.subTest(operation=operation):
                source = self.work / (operation + '-wrong-payload.nasm')
                source.write_text('.entry main\n.string text "wrong"\n.function main 0 0 0 int 1\n'
                    'PUSH_I64 1\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\n' +
                    ('PUSH_I64 0\n' if operation == 'ARR_SET' else '') +
                    'PUSH_STR text\nARR_LITERAL 5 1\nPUSH_I64 0\nARR_GET\n'
                    'AGG_PACK 0 0 0 1\n' + operation + '\nPOP\nPUSH_I64 0\nRET\n.end\n')
                module = source.with_suffix('.nvm')
                self.command(arrays.ROOT / 'bin/nanoisa', 'asm', source, '-o', module)
                self.command(arrays.ROOT / 'bin/nano_vm', '--verify-only', module)
                output = source.with_suffix('.c')
                output.write_text('prior output\n')
                result = subprocess.run([str(arrays.ROOT / 'bin/nvm2c'), str(module), '-o', str(output)],
                                        capture_output=True, text=True, timeout=120)
                source.with_suffix('.log').write_text(result.stdout + result.stderr)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('shape', result.stderr)
                self.assertEqual(output.read_text(), 'prior output\n')

    def test_tagged_nested_array_field_retains_children(self):
        body = ('PUSH_STR a\nPUSH_STR b\nSTR_CONCAT\nARR_LITERAL 5 1\nARR_LITERAL 7 1\n'
                'DUP\nSTORE_GLOBAL 0\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\nSTORE_LOCAL 0\n'
                'LOAD_LOCAL 0\nPUSH_I64 0\nLOAD_GLOBAL 0\nAGG_PACK 0 0 0 1\nARR_SET\nPOP\n'
                'PUSH_VOID\nSTORE_GLOBAL 0\nCALL churn\n'
                'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nAGG_GET 0\n'
                'PUSH_I64 0\nARR_GET\nPUSH_I64 0\nARR_GET\nPUSH_STR ab\nSTR_EQ\nASSERT\n')
        self.paired('nested-field', body,
                    '.string a "a"\n.string b "b"\n.string ab "ab"\n' + CHURN,
                    collections=True)

    def test_absent_payload_cannot_become_an_exact_field(self):
        source = self.work / 'absent.nasm'
        source.write_text('.entry main\n.function main 0 0 0 int 1\n'
            'PUSH_I64 1\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\nPUSH_I64 0\n'
            'ARR_NEW 1\nPUSH_I64 0\nARR_GET\nAGG_PACK 0 0 0 1\nARR_SET\n'
            'PUSH_I64 0\nARR_GET\nAGG_GET 0\nPUSH_I64 1\nI64_ADD\nRET\n.end\n')
        module, native_c, binary = source.with_suffix('.nvm'), source.with_suffix('.c'), source.with_suffix('.native')
        self.command(arrays.ROOT / 'bin/nanoisa', 'asm', source, '-o', module)
        self.command(arrays.ROOT / 'bin/nano_vm', '--verify-only', module)
        self.command(arrays.ROOT / 'bin/nvm2c', module, '-o', native_c)
        self.command(*shlex.split(os.environ.get('CC', 'clang')), '-std=c11',
                     '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                     '-fno-sanitize-recover=all', native_c, '-lm', '-o', binary)
        for label, args in [('vm', [str(arrays.ROOT / 'bin/nano_vm'), str(module)]),
                            ('native', [str(binary)])]:
            result = subprocess.run(args, capture_output=True, text=True, timeout=120)
            (self.work / ('absent-' + label + '.log')).write_text(result.stdout + result.stderr)
            self.assertNotEqual(result.returncode, 0, label)
            if label == 'native':
                self.assertIn('native invariant', result.stderr)
