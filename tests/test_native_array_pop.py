"""I preserve raw pop absence, exact scalar bits and retained callable environments."""
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
import unittest
from tests import test_native_closures as closures

class NativeArrayPop(unittest.TestCase):
    checked = closures.NativeClosures.checked
    emit = closures.NativeClosures.emit
    def sanitized(self, source, binary):
        self.checked([*shlex.split(os.environ.get('CC', 'cc')), '-std=c11', '-O0', '-g',
                      '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                      '-fno-sanitize-recover=all', source, '-o', binary])
        return self.checked([binary], env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1'})

    run_module = closures.NativeClosures.run_module

    def test_scalar_values_aliases_and_empty_results(self):
        for tag, push, check in (
            (1, 'PUSH_I64 -9223372036854775808\n', 'PUSH_I64 -9223372036854775808\nEQ\nASSERT\n'),
            (4, 'PUSH_BOOL 1\n', 'ASSERT\n'),
            (5, 'PUSH_STR text\n', 'PUSH_STR text\nEQ\nASSERT\n'),
        ):
            with self.subTest(tag=tag):
                text = ('.string text "retained"\n.entry main\n.function main 0 2 0 int 1\n'
                        +push+f'ARR_LITERAL {tag} 1\nSTORE_LOCAL 0\n'
                        'LOAD_LOCAL 0\nSTORE_LOCAL 1\nLOAD_LOCAL 0\nARR_POP\n'+check+
                        'LOAD_LOCAL 1\nARR_LEN\nPUSH_I64 0\nEQ\nASSERT\n'
                        'LOAD_LOCAL 1\nARR_POP\nTYPE_CHECK 0\nASSERT\n'
                        'LOAD_LOCAL 0\nARR_POP\nTYPE_CHECK 0\nASSERT\n'
                        'PUSH_I64 0\nRET\n.end\n')
                self.run_module(text, '')

    def test_record_children_survive_owner_reuse_and_collection(self):
        text = ('.string a "a"\n.string b "b"\n.entry main\n'
                '.function main 0 2 0 int 1\n'
                'PUSH_STR a\nPUSH_STR b\nSTR_CONCAT\nPUSH_I64 77\nARR_LITERAL 1 1\n'
                'AGG_PACK 0 0 0 2\nARR_LITERAL 8 1\nSTORE_LOCAL 0\n'
                'LOAD_LOCAL 0\nARR_POP\nSTORE_LOCAL 1\n'
                'LOAD_LOCAL 0\nPUSH_STR a\nPUSH_I64 99\nARR_LITERAL 1 1\n'
                'AGG_PACK 0 0 0 2\nARR_PUSH\nPOP\nCALL churn\n'
                'LOAD_LOCAL 1\nTYPE_CHECK 8\nASSERT\n'
                'LOAD_LOCAL 1\nAGG_GET 0\nSTR_LEN\nPUSH_I64 2\nEQ\nASSERT\n'
                'LOAD_LOCAL 1\nAGG_GET 1\nPUSH_I64 0\nARR_GET\nPUSH_I64 77\nEQ\nASSERT\n'
                'LOAD_LOCAL 0\nARR_POP\nAGG_GET 1\nPUSH_I64 0\nARR_GET\nPUSH_I64 99\nEQ\nASSERT\n'
                'LOAD_LOCAL 0\nARR_POP\nTYPE_CHECK 0\nASSERT\n'
                'PUSH_I64 0\nRET\n.end\n'+closures.CHURN)
        self.run_module(text, '', collections=True)

    def test_popped_record_crosses_return_and_argument(self):
        self.run_module('.entry main\n'
                        '.function take 1 1 0 struct 1\nLOAD_LOCAL 0\nARR_POP\nRET\n.end\n'
                        '.function read 1 1 0 int 1\nLOAD_LOCAL 0\nAGG_GET 0\nRET\n.end\n'
                        '.function main 0 0 0 int 1\nPUSH_I64 77\nAGG_PACK 0 0 0 1\n'
                        'ARR_LITERAL 8 1\nCALL take\nCALL read\nPUSH_I64 77\nEQ\nASSERT\n'
                        'PUSH_I64 0\nRET\n.end\n', '')

    def test_empty_record_projection_traps(self):
        with tempfile.TemporaryDirectory(prefix='native-pop-absent-') as tmp:
            work = Path(tmp)
            assembly, module, source, native = (work/name for name in ('input.nasm', 'input.nvm', 'input.c', 'program'))
            assembly.write_text('.entry main\n.function main 0 0 0 int 1\n'
                                'ARR_NEW 8\nARR_POP\nAGG_GET 0\nPOP\nPUSH_I64 0\nRET\n.end\n')
            self.checked([closures.ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
            self.checked([closures.ROOT/'bin/nano_vm', '--verify-only', module])
            vm = subprocess.run([str(closures.ROOT/'bin/nano_vm'), str(module)], capture_output=True, text=True, timeout=90)
            self.assertNotEqual(vm.returncode, 0)
            self.assertIn('AGG_GET', vm.stderr)
            self.checked([closures.ROOT/'bin/nvm2c', module, '-o', source])
            self.checked([*shlex.split(os.environ.get('CC', 'cc')), '-std=c11', '-Wall', '-Wextra', '-Werror',
                          '-fsanitize=address,undefined', '-fno-sanitize-recover=all', source, '-o', native])
            result = subprocess.run([str(native)], capture_output=True, text=True, timeout=90,
                                    env={**os.environ, 'ASAN_OPTIONS':'detect_leaks=1'})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('native invariant', result.stderr)
            self.assertNotIn('ERROR: AddressSanitizer', result.stderr)

    def test_empty_record_array_returns_void(self):
        self.run_module('.entry main\n.function main 0 0 0 int 1\n'
                        'ARR_NEW 8\nARR_POP\nTYPE_CHECK 0\nASSERT\n'
                        'PUSH_I64 0\nRET\n.end\n', '')

    def test_binary64_payloads(self):
        for bits in (-9223372036854775808, 9221120237041090561, 9218868437227405313, 1):
            with self.subTest(bits=bits):
                self.run_module('.entry main\n.function main 0 1 0 int 1\n'+
                    f'PUSH_I64 {bits}\nF64_FROM_BITS\nARR_LITERAL 3 1\nSTORE_LOCAL 0\n'
                    'LOAD_LOCAL 0\nARR_POP\nF64_TO_BITS\n'+f'PUSH_I64 {bits}\nEQ\nASSERT\n'
                    'LOAD_LOCAL 0\nARR_POP\nTYPE_CHECK 0\nASSERT\nPUSH_I64 0\nRET\n.end\n', '')

    def test_popped_closure_survives_owner_reuse_and_collection(self):
        text = ('.string a "a"\n.string b "b"\n.entry main\n'
                '.function read 0 0 1 int 1\nLOAD_UPVALUE 0 0\nRET\n.end\n'
                '.function main 0 2 0 int 1\n'
                'PUSH_I64 41\nCLOSURE_NEW read 1\nARR_LITERAL 15 1\nSTORE_LOCAL 0\n'
                'LOAD_LOCAL 0\nARR_POP\nSTORE_LOCAL 1\n'
                'LOAD_LOCAL 0\nPUSH_I64 99\nCLOSURE_NEW read 1\nARR_PUSH\nPOP\n'
                'CALL churn\nLOAD_LOCAL 1\nCALL_INDIRECT 0 1\nPUSH_I64 41\nEQ\nASSERT\n'
                'LOAD_LOCAL 0\nARR_POP\nCALL_INDIRECT 0 1\nPUSH_I64 99\nEQ\nASSERT\n'
                'LOAD_LOCAL 0\nARR_POP\nTYPE_CHECK 0\nASSERT\n'
                'PUSH_I64 0\nRET\n.end\n'+closures.CHURN)
        self.run_module(text, '', collections=True)

if __name__ == '__main__':
    unittest.main()
