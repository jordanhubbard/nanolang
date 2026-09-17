"""I preserve reachable map values while reclaiming loop garbage."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HEADER = '.string key "key"\n.string text "retained"\n.entry main\n'
GET = 'HM_NEW 5 5\nPUSH_STR key\nPUSH_STR text\nHM_SET\nPUSH_STR key\nHM_GET\n'
CHECK_TEXT = 'PUSH_STR text\nEQ\nASSERT\n'
HELPERS = (
    '.function text_value 0 0 0 string 1\n' + GET + 'RET\n.end\n'
    '.function churn 0 1 0 void 0\nPUSH_I64 0\nSTORE_LOCAL 0\nagain:\n'
    + GET + 'POP\nLOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 0\n'
    'LOAD_LOCAL 0\nPUSH_I64 20000\nI64_LT_S\nJMP_FALSE done\nJMP again\n'
    'done:\nRET\n.end\n'
)


class NativeMapLifetimes(unittest.TestCase):
    def run_checked(self, command, **kwargs):
        result = subprocess.run([str(x) for x in command], capture_output=True,
                                text=True, timeout=60, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def check_program(self, text):
        with tempfile.TemporaryDirectory(prefix='nano-map-lifetime-') as tmp:
            work = Path(tmp)
            assembly, module, source, binary = (work / name for name in
                                              ('input.nasm', 'input.nvm', 'input.c', 'program'))
            assembly.write_text(text)
            self.run_checked([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
            self.run_checked([ROOT / 'bin/nano_vm', module])
            translator = os.environ.get('NANO_MAP_LIFETIME_TRANSLATOR', ROOT / 'bin/nvm2c')
            self.run_checked([translator, module, '-o', source])
            generated = source.read_text()
            self.assertIn('    nmap_release_owned();', generated)
            self.assertNotIn('nroot_add(&nroots.live, 11,', generated)
            # I bound bytes across batching, not merely an eventual process exit.
            # The 64 KiB floor plus 4 KiB covers retained fixture roots and one allocation.
            generated = generated.replace('    nmap_release_owned();',
                '    nmap_release_owned();\n'
                '    if (nmap_owned_live || nmap_live_bytes || nmap_peak_bytes > 69632) abort();\n')
            source.write_text(generated)
            self.run_checked(['cc', '-std=c11', '-g', '-Wall', '-Wextra', '-Werror',
                              '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                              source, '-o', binary])
            self.run_checked([binary], env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=0'})

    def test_caller_owned_map(self):
        self.check_program((ROOT / 'tests/nanoisa/fixtures/map_caller_lifetime.nasm').read_text())

    def test_reachable_values_and_bounded_churn(self):
        cases = {
            'boxed_local': GET + 'STORE_LOCAL 0\nCALL churn\nLOAD_LOCAL 0\n' + CHECK_TEXT,
            'string_local': 'CALL text_value\nSTORE_LOCAL 0\nCALL churn\nLOAD_LOCAL 0\n' + CHECK_TEXT,
            'string_operand': 'CALL text_value\nCALL churn\n' + CHECK_TEXT,
            'boxed_operand': GET + 'CALL churn\n' + CHECK_TEXT,
            'string_global': 'CALL text_value\nSTORE_GLOBAL 0\nCALL churn\nLOAD_GLOBAL 0\n' + CHECK_TEXT,
            'string_array': 'CALL text_value\nARR_LITERAL 5 1\nSTORE_LOCAL 0\nCALL churn\n'
                            'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\n' + CHECK_TEXT,
            'global_array': 'CALL text_value\nARR_LITERAL 5 1\nSTORE_GLOBAL 0\nCALL churn\n'
                            'LOAD_GLOBAL 0\nPUSH_I64 0\nARR_GET\n' + CHECK_TEXT,
            'nested_record': 'CALL text_value\nAGG_PACK 0 0 0 1\nAGG_PACK 0 0 0 1\n'
                             'STORE_LOCAL 0\nCALL churn\nLOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 0\n' + CHECK_TEXT,
            'record_array': 'CALL text_value\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\n'
                            'STORE_LOCAL 0\nCALL churn\nLOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nAGG_GET 0\n' + CHECK_TEXT,
            'map_record': 'HM_NEW 5 1\nPUSH_STR key\nPUSH_I64 42\nHM_SET\nAGG_PACK 0 0 0 1\n'
                          'STORE_LOCAL 0\nCALL churn\nLOAD_LOCAL 0\nAGG_GET 0\nPUSH_STR key\nHM_GET\n'
                          'PUSH_I64 42\nEQ\nASSERT\n',
            'map_operand': 'HM_NEW 5 1\nPUSH_STR key\nPUSH_I64 42\nHM_SET\nCALL churn\n'
                           'PUSH_STR key\nHM_GET\nPUSH_I64 42\nEQ\nASSERT\n',
            'float_local': 'PUSH_F64 1.25\nSTORE_LOCAL 0\nCALL churn\nLOAD_LOCAL 0\n'
                           'PUSH_F64 1.25\nEQ\nASSERT\n',
            'float_operand': 'PUSH_F64 1.25\nCALL churn\nPUSH_F64 1.25\nEQ\nASSERT\n',
        }
        for name, body in cases.items():
            with self.subTest(name=name):
                self.check_program(HEADER + '.function main 0 1 0 int 1\n' + body
                                   + 'PUSH_I64 0\nRET\n.end\n' + HELPERS)

    def test_self_tail_reclamation(self):
        self.check_program(HEADER +
            '.function main 0 0 0 int 1\nHM_NEW 5 1\nPUSH_STR key\nPUSH_I64 42\nHM_SET\n'
            'PUSH_I64 20000\nCALL repeat\nPUSH_I64 42\nI64_EQ\nASSERT\nPUSH_I64 0\nRET\n.end\n'
            '.function repeat 2 2 0 int 1\nLOAD_LOCAL 1\nPUSH_I64 0\nI64_EQ\nJMP_FALSE again\n'
            'LOAD_LOCAL 0\nPUSH_STR key\nHM_GET\nCAST_INT\nRET\nagain:\n' + GET +
            'POP\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 1\nI64_SUB\nTAIL_CALL repeat\n.end\n')

    def test_mutable_array_roots_and_backward_conditional(self):
        # The caller's array gains its only owned string in the callee. A
        # flattened snapshot of its old elements would lose that new edge.
        self.check_program(HEADER +
            '.function main 0 1 0 int 1\nPUSH_STR text\nARR_LITERAL 5 1\nSTORE_LOCAL 0\n'
            'LOAD_LOCAL 0\nCALL replace\nCALL churn\nLOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\n'
            + CHECK_TEXT + 'PUSH_I64 0\nRET\n.end\n' + HELPERS +
            '.function replace 1 2 0 void 0\nLOAD_LOCAL 0\nPUSH_I64 0\nCALL text_value\nARR_SET\nPOP\n'
            'PUSH_I64 0\nSTORE_LOCAL 1\nagain:\n' + GET +
            'POP\nLOAD_LOCAL 1\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 1\n'
            'LOAD_LOCAL 1\nPUSH_I64 20000\nI64_EQ\nJMP_FALSE again\nRET\n.end\n')

    def test_non_self_tail_unregisters_root_frame(self):
        self.check_program(HEADER +
            '.function main 0 1 0 int 1\nHM_NEW 5 1\nPUSH_STR key\nPUSH_I64 42\nHM_SET\n'
            'STORE_LOCAL 0\nLOAD_LOCAL 0\nCALL relay\nPUSH_I64 42\nI64_EQ\nASSERT\n'
            'CALL churn\nCALL churn\nPUSH_I64 0\nRET\n.end\n' + HELPERS +
            '.function relay 1 1 0 int 1\nLOAD_LOCAL 0\nTAIL_CALL read\n.end\n'
            '.function read 1 1 0 int 1\nLOAD_LOCAL 0\nPUSH_STR key\nHM_GET\nCAST_INT\nRET\n.end\n')


if __name__ == '__main__':
    unittest.main()
