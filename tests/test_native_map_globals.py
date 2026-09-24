"""I preserve tagged map globals across calls, mutation and collection."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SANITIZER_CC = shlex.split(os.environ.get("NANOLANG_GUARD_SAN_CC", os.environ.get("CC", "cc")))
HEADER = '.entry main\n.string key "key"\n.string text "value"\n'

class NativeMapGlobals(unittest.TestCase):
    def command(self, args):
        return subprocess.run([str(x) for x in args], capture_output=True, text=True,
                              timeout=60, env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1'})

    def check(self, body, helpers='', bad=False, vm_bad=None):
        with tempfile.TemporaryDirectory(prefix='nano-map-global-') as tmp:
            base = Path(tmp)
            nasm, nvm, source, binary = [base / n for n in ('in.nasm', 'in.nvm', 'in.c', 'run')]
            nasm.write_text(HEADER + '.function main 0 2 0 int 1\n' + body +
                            'PUSH_I64 0\nRET\n.end\n' + helpers)
            for command in ([ROOT/'bin/nanoisa', 'asm', nasm, '-o', nvm],
                            [ROOT/'bin/nvm2c', nvm, '-o', source],
                            [*SANITIZER_CC, '-std=c11', '-Wall', '-Wextra', '-Werror', '-g',
                             '-fsanitize=address,undefined', '-fno-sanitize-recover=all', source, '-o', binary]):
                result = self.command(command)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                if command[0] == ROOT/'bin/nvm2c':
                    generated = source.read_text()
                    self.assertIn('    nmap_release_owned();', generated)
                    source.write_text(generated.replace('    nmap_release_owned();',
                        '    nmap_release_owned();\n'
                        '    if (nmap_owned_live || nmap_live_bytes || nmap_peak_bytes > 69632) abort();\n'))
            for native, command in enumerate(([ROOT/'bin/nano_vm', nvm], [binary])):
                result = self.command(command)
                expected_failure = bad if native or vm_bad is None else vm_bad
                if expected_failure:
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertNotIn('ERROR: AddressSanitizer', result.stderr)
                    self.assertNotIn('runtime error:', result.stderr)
                else:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_initializer_alias_call_return_and_reassignment(self):
        self.check('LOAD_GLOBAL 0\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nCALL put\nPOP\n'
                   'CALL get\nPUSH_STR key\nHM_GET\nPUSH_STR text\nEQ\nASSERT\n'
                   'HM_NEW 5 1\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_STR key\nPUSH_I64 42\nHM_SET\nPOP\n'
                   'LOAD_GLOBAL 0\nPUSH_STR key\nHM_GET\nPUSH_I64 42\nEQ\nASSERT\n'
                   'LOAD_LOCAL 0\nPUSH_STR key\nHM_GET\nPUSH_STR text\nEQ\nASSERT\n'
                   'LOAD_GLOBAL 0\nTYPE_CHECK 13\nASSERT\nLOAD_GLOBAL 0\nASSERT\nLOAD_GLOBAL 0\nLOAD_GLOBAL 0\nEQ\nASSERT\n'
                   'LOAD_GLOBAL 0\nCALL tail_length\nPUSH_I64 1\nEQ\nASSERT\n'
                   'LOAD_GLOBAL 0\nPUSH_STR key\nHM_DELETE\nSTORE_GLOBAL 0\n'
                   'LOAD_GLOBAL 0\nPUSH_STR key\nHM_GET\nTYPE_CHECK 0\nASSERT\n',
                   '.function fresh 0 0 0 hashmap 1\nHM_NEW 5 5\nRET\n.end\n'
                   '.function __init__ 0 0 0 void 0\nCALL fresh\nSTORE_GLOBAL 0\nRET\n.end\n'
                   '.function put 1 1 0 hashmap 1\nLOAD_LOCAL 0\nPUSH_STR key\nPUSH_STR text\nHM_SET\nRET\n.end\n'
                   '.function get 0 0 0 hashmap 1\nLOAD_GLOBAL 0\nRET\n.end\n'
                   '.function tail_length 1 1 0 int 1\nLOAD_LOCAL 0\nTAIL_CALL length\n.end\n'
                   '.function length 1 1 0 int 1\nLOAD_LOCAL 0\nHM_LEN\nRET\n.end\n')

    def test_bad_receiver_and_value_tags(self):
        for value in ('PUSH_I64 9', 'PUSH_STR text', 'PUSH_BOOL 1'):
            with self.subTest(value=value):
                self.check(value + '\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nHM_LEN\nPOP\n', bad=True)
        # I require exact declared value tags in both ordinary backends.
        self.check('HM_NEW 5 1\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_STR key\nPUSH_STR text\nHM_SET\nPOP\n', bad=True, vm_bad=True)
        self.check('LOAD_GLOBAL 0\nHM_LEN\nPOP\n', bad=True)

    def test_global_local_record_and_operand_roots_survive_collection(self):
        from tests.test_native_map_lifetimes import HELPERS
        for keep, read in (
            ('', 'LOAD_GLOBAL 0\n'),
            ('LOAD_GLOBAL 0\nSTORE_LOCAL 0\n', 'LOAD_LOCAL 0\n'),
            ('LOAD_GLOBAL 0\nAGG_PACK 0 0 0 1\nSTORE_LOCAL 0\n', 'LOAD_LOCAL 0\nAGG_GET 0\n'),
            ('LOAD_GLOBAL 0\n', ''),
        ):
            with self.subTest(read=read):
                self.check('HM_NEW 5 5\nPUSH_STR key\nPUSH_STR text\nHM_SET\nSTORE_GLOBAL 0\n' +
                           keep + ('PUSH_I64 0\nSTORE_GLOBAL 0\n' if keep else '') + 'CALL churn\n' + read +
                           'PUSH_STR key\nHM_GET\nPUSH_STR text\nEQ\nASSERT\n', HELPERS)

    def test_distinct_maps_keep_identity_and_vm_ordering(self):
        body = 'HM_NEW 5 1\nSTORE_GLOBAL 0\nHM_NEW 5 1\nSTORE_GLOBAL 1\n'
        # VM val_compare returns zero for same-tag maps; equality is identity.
        for opcode, expected in (('EQ', 0), ('NE', 1), ('LT', 0), ('LE', 1), ('GT', 0), ('GE', 1)):
            body += 'LOAD_GLOBAL 0\nLOAD_GLOBAL 1\n' + opcode + '\n'
            body += ('BOOL_NOT\n' if not expected else '') + 'ASSERT\n'
        self.check(body)

    def test_mixed_direct_and_tagged_callers(self):
        self.check('HM_NEW 5 1\nCALL length\nPUSH_I64 0\nEQ\nASSERT\n'
                   'HM_NEW 5 1\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nCALL length\nPUSH_I64 0\nEQ\nASSERT\n',
                   '.function length 1 1 0 int 1\nLOAD_LOCAL 0\nHM_LEN\nRET\n.end\n')

if __name__ == '__main__':
    unittest.main()
