"""I check tagged array fields before writing exact record-array storage."""
import os
import shlex
import subprocess
from pathlib import Path
import re
import tempfile
import unittest
from tests import test_native_closures as closures

ROOT = closures.ROOT


class TaggedRecordArrayFields(unittest.TestCase):
    checked = closures.NativeClosures.checked
    emit = closures.NativeClosures.emit
    sanitized = closures.NativeClosures.sanitized
    run_module = closures.NativeClosures.run_module

    def fixture(self, tag, literal=False):
        values = {
            1: ('PUSH_I64 42', 'PUSH_I64 42\nEQ'),
            3: ('PUSH_F64 1.5', 'PUSH_F64 1.5\nEQ'),
            4: ('PUSH_BOOL 1', 'PUSH_BOOL 1\nEQ'),
            5: ('PUSH_STR value', 'PUSH_STR value\nEQ'),
            11: ('FUNCREF answer', 'CALL_INDIRECT 0 1\nPUSH_I64 42\nEQ'),
        }
        value, check = values[tag]
        constructor = f"ARR_LITERAL {tag} 0" if literal else f"ARR_NEW {tag}"
        return ('.string value "retained"\n.string a "a"\n.string b "b"\n.entry main\n'
                '.function answer 0 0 0 int 1\nPUSH_I64 42\nRET\n.end\n'
                '.function main 0 1 0 int 1\n'
                f'{constructor}\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\nSTORE_LOCAL 0\n'
                f'{value}\nARR_LITERAL {tag} 1\nSTORE_GLOBAL 0\n'
                'LOAD_LOCAL 0\nPUSH_I64 0\nLOAD_GLOBAL 0\nAGG_PACK 0 0 0 1\nARR_SET\nPOP\n'
                'LOAD_LOCAL 0\nLOAD_GLOBAL 0\nAGG_PACK 0 0 0 1\nARR_PUSH\nPOP\n'
                'CALL churn\n'
                f'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nAGG_GET 0\nPUSH_I64 0\nARR_GET\n{check}\nASSERT\n'
                f'LOAD_LOCAL 0\nPUSH_I64 1\nARR_GET\nAGG_GET 0\nPUSH_I64 0\nARR_GET\n{check}\nASSERT\n'
                'PUSH_I64 0\nRET\n.end\n' + closures.CHURN)

    def test_all_word_and_string_arrays_survive_checked_writes(self):
        for tag in (1, 3, 4, 5, 11):
            for literal in (False, True):
                with self.subTest(tag=tag, literal=literal):
                    self.run_module(self.fixture(tag, literal), '', collections=True)

    def test_incompatible_concrete_field_preserves_prior_output(self):
        with tempfile.TemporaryDirectory(prefix='nano-record-array-incompatible-') as tmp:
            work = Path(tmp)
            for op in ('ARR_SET', 'ARR_PUSH'):
                with self.subTest(operation=op):
                    assembly, module, output = [work / name for name in ('input.nasm', 'input.nvm', 'prior.c')]
                    index = 'PUSH_I64 0\n' if op == 'ARR_SET' else ''
                    assembly.write_text('.entry main\n.function main 0 1 0 int 1\n'
                        'ARR_LITERAL 5 0\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\nSTORE_LOCAL 0\n'
                        + 'LOAD_LOCAL 0\n' + index + 'PUSH_F64 1.5\nARR_LITERAL 3 1\nAGG_PACK 0 0 0 1\n'
                        + op + '\nPOP\nPUSH_I64 0\nRET\n.end\n')
                    self.checked([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
                    output.write_text('prior output\n')
                    result = subprocess.run([ROOT / 'bin/nvm2c', module, '-o', output],
                                            capture_output=True, text=True, timeout=30)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('record field representation mismatch', result.stderr)
                    self.assertEqual(output.read_text(), 'prior output\n')

    def test_wrong_runtime_tag_element_kind_and_null_are_refused(self):
        with tempfile.TemporaryDirectory(prefix='nano-record-array-guards-') as tmp:
            work = Path(tmp)
            source = self.emit(work, self.fixture(5))
            original = source.read_text()
            pattern = r'    if \(r\[(\d+)\]\.vk\[0\] != 7 \|\| r\[\1\]\.f\[0\] != 5 \|\| !r\[\1\]\.s\[0\]\) NVM2C_ABORT\(\);'
            match = re.search(pattern, original)
            self.assertIsNotNone(match)
            slot = match.group(1)
            for name, expression in [('tag', f'r[{slot}].vk[0] = 1;'),
                                     ('element', f'r[{slot}].f[0] = 3;'),
                                     ('null', f'r[{slot}].s[0] = NULL;')]:
                with self.subTest(fault=name):
                    source.write_text(original[:match.start()] + '    ' + expression + '\n' + original[match.start():])
                    binary = work / name
                    self.checked([*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', os.environ.get('CC', 'cc'))), '-std=c11', '-O0', '-g', '-Wall', '-Wextra', '-Werror',
                                  '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                                  source, '-o', binary])
                    result = subprocess.run([binary], capture_output=True, text=True, timeout=30,
                                            env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1'})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('native invariant', result.stderr)


if __name__ == '__main__':
    unittest.main()
