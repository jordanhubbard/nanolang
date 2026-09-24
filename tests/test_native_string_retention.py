"""I reclaim temporary native strings without losing reachable aliases."""
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
HEADER = '.string text "retained"\n.string empty ""\n.string unit "x"\n.entry main\n'
VALUE = 'PUSH_STR text\nPUSH_STR empty\nSTR_CONCAT\n'
CHECK = 'PUSH_STR text\nEQ\nASSERT\n'
CHURN = ('.function churn 0 1 0 void 0\nPUSH_I64 0\nSTORE_LOCAL 0\nagain:\n'
         'PUSH_STR text\nPUSH_STR text\nSTR_CONCAT\nPUSH_I64 0\nPUSH_I64 7\nSTR_SUBSTR\nPOP\n'
         'LOAD_LOCAL 0\nCAST_STRING\nPOP\n'
         'LOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 0\n'
         'LOAD_LOCAL 0\nPUSH_I64 4000\nI64_LT_S\nJMP_TRUE again\nRET\n.end\n')
HELPERS = '.function value 0 0 0 string 1\n' + VALUE + 'RET\n.end\n' + CHURN


class NativeStringRetention(unittest.TestCase):
    def run_checked(self, args, **kwargs):
        result = subprocess.run([str(x) for x in args], capture_output=True,
                                text=True, timeout=90, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def check_program(self, assembly, peak=200000):
        with tempfile.TemporaryDirectory(prefix='nano-string-retention-') as tmp:
            work = Path(tmp)
            asm, module, source, binary = [work / name for name in
                                          ('input.nasm', 'input.nvm', 'input.c', 'program')]
            asm.write_text(assembly)
            self.run_checked([ROOT / 'bin/nanoisa', 'asm', asm, '-o', module])
            self.run_checked([ROOT / 'bin/nano_vm', module])
            translator = os.environ.get('NANO_STRING_TRANSLATOR', ROOT / 'bin/nvm2c')
            self.run_checked([translator, module, '-o', source])
            generated = source.read_text()
            self.assertIn('    nstr_release_owned();', generated)
            generated = generated.replace('    nstr_release_owned();',
                '    nstr_release_owned();\n'
                f'    if (nstr_live_bytes || nstr_owners || nstr_peak_bytes > {peak}) abort();\n'
                '    printf("string_peak_bytes=%zu\\n", nstr_peak_bytes);\n')
            source.write_text('#include <stdio.h>\n' + generated)
            self.run_checked(['cc', '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                              '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                              source, '-o', binary])
            result = self.run_checked([binary], env={**os.environ, 'ASAN_OPTIONS': asan_options()})
            print(result.stdout, end='')

    def test_growing_string_without_maps_has_bounded_retention(self):
        # I repeatedly replace one growing value; retaining old versions costs
        # over 50 MB, while the live value is only 10 KB.
        self.check_program(HEADER + '.function main 0 2 0 int 1\n'
            'PUSH_STR empty\nSTORE_LOCAL 0\nPUSH_I64 0\nSTORE_LOCAL 1\nagain:\n'
            'LOAD_LOCAL 0\nPUSH_STR unit\nSTR_CONCAT\nSTORE_LOCAL 0\n'
            'LOAD_LOCAL 1\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 1\n'
            'LOAD_LOCAL 1\nPUSH_I64 10000\nI64_LT_S\nJMP_TRUE again\n'
            'LOAD_LOCAL 0\nSTR_LEN\nPUSH_I64 10000\nI64_EQ\nASSERT\n'
            'PUSH_I64 0\nRET\n.end\n')

    def test_reachable_aliases_across_calls_and_mutable_aggregates(self):
        cases = {
            'nonself_tail_return': 'CALL relay\nCALL churn\n' + CHECK,
            'caller_operand': 'CALL value\nCALL churn\n' + CHECK,
            'caller_local': 'CALL value\nSTORE_LOCAL 0\nCALL churn\nLOAD_LOCAL 0\n' + CHECK,
            'global': 'CALL value\nSTORE_GLOBAL 0\nCALL churn\nLOAD_GLOBAL 0\n' + CHECK,
            'array': 'CALL value\nARR_LITERAL 5 1\nSTORE_LOCAL 0\nCALL churn\n'
                     'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\n' + CHECK,
            'record_array': 'CALL value\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\n'
                            'STORE_LOCAL 0\nCALL churn\nLOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nAGG_GET 0\n' + CHECK,
            'nested_record': 'CALL value\nAGG_PACK 0 0 0 1\nAGG_PACK 0 0 0 1\n'
                             'STORE_LOCAL 0\nCALL churn\nLOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 0\n' + CHECK,
            'mutated_array': 'PUSH_STR empty\nARR_LITERAL 5 1\nSTORE_LOCAL 0\n'
                             'LOAD_LOCAL 0\nCALL replace\nCALL churn\nLOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\n' + CHECK,
        }
        replace = ('.function replace 1 1 0 void 0\nLOAD_LOCAL 0\nPUSH_I64 0\nCALL value\n'
                   'ARR_SET\nPOP\nCALL churn\nRET\n.end\n')
        for name, body in cases.items():
            with self.subTest(name=name):
                self.check_program(HEADER + '.function main 0 1 0 int 1\n' + body +
                                   'PUSH_I64 0\nRET\n.end\n' + HELPERS + replace +
                                   '.function relay 0 0 0 string 1\nTAIL_CALL value\n.end\n')

    def test_self_tail_simultaneous_string_handoff(self):
        self.check_program(HEADER + '.function main 0 0 0 int 1\n'
            'CALL value\nPUSH_STR empty\nPUSH_I64 10000\nCALL repeat\n' + CHECK +
            'PUSH_I64 0\nRET\n.end\n' + HELPERS +
            '.function repeat 3 3 0 string 1\nLOAD_LOCAL 2\nPUSH_I64 0\nI64_EQ\nJMP_FALSE again\n'
            'LOAD_LOCAL 0\nRET\nagain:\n'
            'LOAD_LOCAL 1\nPUSH_STR empty\nSTR_CONCAT\n'
            'LOAD_LOCAL 0\nPUSH_STR empty\nSTR_CONCAT\n'
            'LOAD_LOCAL 2\nPUSH_I64 1\nI64_SUB\nTAIL_CALL repeat\n.end\n')

    def test_call_safepoint_collects_without_backwards_branch(self):
        # Every allocating call returns before the next safepoint; its result
        # remains live until consumed, and popped results must become garbage.
        calls = ('CALL value\n' + CHECK) * 3500
        self.check_program(HEADER + '.function main 0 0 0 int 1\n' + calls +
                           'PUSH_I64 0\nRET\n.end\n' + HELPERS)


if __name__ == '__main__':
    unittest.main()
