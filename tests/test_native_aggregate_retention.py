"""I bound native aggregate pools while preserving reachable values and handles."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HEADER = '.types 8 0 0\n.string text "retained"\n.entry main\n'
CHURN_BODY = ('PUSH_I64 7\nARR_LITERAL 1 1\nPOP\n'
              'PUSH_BOOL 1\nARR_LITERAL 4 1\nPOP\n'
              'PUSH_STR text\nARR_LITERAL 5 1\nPOP\n'
              'PUSH_I64 9\nAGG_PACK 0 0 0 1\nAGG_PACK 0 1 0 1\n'
              'ARR_LITERAL 8 1\nPOP\n')
CHURN = ('.function churn 0 1 0 void 0\nPUSH_I64 0\nSTORE_LOCAL 0\nloop:\n' + CHURN_BODY +
         'LOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 0\n'
         'LOAD_LOCAL 0\nPUSH_I64 3000\nI64_LT_S\nJMP_TRUE loop\nRET\n.end\n')


class NativeAggregateRetention(unittest.TestCase):
    def run_checked(self, args, **kwargs):
        result = subprocess.run([str(x) for x in args], capture_output=True,
                                text=True, timeout=90, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def emit(self, work, text):
        asm, module, source = [work / name for name in ('input.nasm', 'input.nvm', 'input.c')]
        asm.write_text(text)
        self.run_checked([ROOT / 'bin/nanoisa', 'asm', asm, '-o', module])
        self.run_checked([ROOT / 'bin/nano_vm', module])
        self.run_checked([ROOT / 'bin/nvm2c', module, '-o', source])
        return source

    def compile_run(self, source, binary):
        self.run_checked(['cc', '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                          '-fsanitize=address,undefined', '-fno-sanitize-recover=all', source, '-o', binary])
        return self.run_checked([binary], env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1'})

    def check_program(self, text, peak=200000):
        with tempfile.TemporaryDirectory(prefix='nano-aggregate-retention-') as tmp:
            work = Path(tmp)
            source = self.emit(work, text)
            generated = source.read_text()
            # I measure all owned handles, owner nodes, buffers and snapshots,
            # and demand final cleanup instead of relying on reachable globals.
            self.assertIn('    return result;\n}', generated)
            generated = generated.replace('    return result;\n}',
                f'    if (nagg_live_bytes || nagg_peak_bytes > {peak}) abort();\n'
                '    printf("aggregate_peak_bytes=%zu\\n", nagg_peak_bytes);\n'
                '    return result;\n}')
            source.write_text('#include <stdio.h>\n' + generated)
            print(self.compile_run(source, work / 'program').stdout, end='')

    def test_dead_aggregate_churn_without_maps_or_allocated_strings(self):
        self.check_program(HEADER + '.function main 0 0 0 int 1\nCALL churn\n'
                           'PUSH_I64 0\nRET\n.end\n' + CHURN)

    def test_caller_global_nested_and_returned_aliases(self):
        cases = {
            'int_array': 'PUSH_I64 42\nARR_LITERAL 1 1\nSTORE_LOCAL 0\nCALL churn\n'
                         'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nPUSH_I64 42\nI64_EQ\nASSERT\n',
            'bool_array': 'PUSH_BOOL 1\nARR_LITERAL 4 1\nSTORE_LOCAL 0\nCALL churn\n'
                          'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nASSERT\n',
            'string_array': 'PUSH_STR text\nARR_LITERAL 5 1\nSTORE_LOCAL 0\nCALL churn\n'
                            'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nPUSH_STR text\nEQ\nASSERT\n',
            'record_array': 'PUSH_I64 42\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\nSTORE_LOCAL 0\nCALL churn\n'
                            'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nAGG_GET 0\nPUSH_I64 42\nI64_EQ\nASSERT\n',
            'nested_record': 'PUSH_I64 42\nAGG_PACK 0 0 0 1\nAGG_PACK 0 1 0 1\nSTORE_LOCAL 0\nCALL churn\n'
                             'LOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 0\nPUSH_I64 42\nI64_EQ\nASSERT\n',
            'record_int_array': 'PUSH_I64 42\nARR_LITERAL 1 1\nAGG_PACK 0 2 0 1\nSTORE_LOCAL 0\nCALL churn\n'
                                'LOAD_LOCAL 0\nAGG_GET 0\nPUSH_I64 0\nARR_GET\nPUSH_I64 42\nI64_EQ\nASSERT\n',
            'nested_array': 'PUSH_I64 42\nARR_LITERAL 1 1\nAGG_PACK 0 2 0 1\nAGG_PACK 0 6 0 1\n'
                            'STORE_LOCAL 0\nCALL churn\nLOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 0\n'
                            'PUSH_I64 0\nARR_GET\nPUSH_I64 42\nI64_EQ\nASSERT\n',
            'escaped_inner_copy': 'PUSH_I64 42\nARR_LITERAL 1 1\nAGG_PACK 0 2 0 1\nAGG_PACK 0 6 0 1\n'
                                  'AGG_GET 0\nSTORE_LOCAL 0\nCALL churn\nLOAD_LOCAL 0\nAGG_GET 0\n'
                                  'PUSH_I64 0\nARR_GET\nPUSH_I64 42\nI64_EQ\nASSERT\n',
            'global_array': 'CALL make\nSTORE_GLOBAL 0\nCALL churn\nLOAD_GLOBAL 0\nPUSH_I64 0\nARR_GET\n'
                            'PUSH_I64 42\nEQ\nASSERT\n',
            'returned_operand': 'CALL relay\nCALL churn\nPUSH_I64 0\nARR_GET\nPUSH_I64 42\nI64_EQ\nASSERT\n',
            'aliased_mutation': 'CALL make\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nCALL mutate\nCALL churn\n'
                                'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nPUSH_I64 43\nI64_EQ\nASSERT\n',
        }
        helpers = ('.function make 0 0 0 array 1\nPUSH_I64 42\nARR_LITERAL 1 1\nRET\n.end\n'
                   '.function relay 0 0 0 array 1\nTAIL_CALL make\n.end\n'
                   '.function mutate 1 1 0 void 0\nLOAD_LOCAL 0\nPUSH_I64 0\nPUSH_I64 43\nARR_SET\n'
                   'POP\nCALL churn\nRET\n.end\n')
        for name, body in cases.items():
            with self.subTest(name=name):
                self.check_program(HEADER + '.function main 0 1 0 int 1\n' + body +
                                   'PUSH_I64 0\nRET\n.end\n' + CHURN + helpers)

    def test_self_tail_moves_two_array_arguments_simultaneously(self):
        self.check_program(HEADER + '.function main 0 0 0 int 1\n'
            'PUSH_I64 42\nARR_LITERAL 1 1\nPUSH_I64 7\nARR_LITERAL 1 1\nPUSH_I64 10000\nCALL repeat\n'
            'PUSH_I64 42\nI64_EQ\nASSERT\nPUSH_I64 0\nRET\n.end\n'
            '.function repeat 3 3 0 int 1\nLOAD_LOCAL 2\nPUSH_I64 0\nI64_EQ\nJMP_FALSE again\n'
            'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nRET\nagain:\n' + CHURN_BODY +
            'LOAD_LOCAL 1\nLOAD_LOCAL 0\nLOAD_LOCAL 2\nPUSH_I64 1\nI64_SUB\nTAIL_CALL repeat\n.end\n')

    def test_borrowed_handles_growth_and_escaped_copied_string(self):
        with tempfile.TemporaryDirectory(prefix='nano-borrowed-pool-') as tmp:
            work = Path(tmp)
            source = self.emit(work, HEADER + '.function main 0 0 0 int 1\nCALL churn\n'
                               'PUSH_I64 0\nRET\n.end\n' + CHURN)
            source.write_text('#define main nano_program_main\n' + source.read_text() + r'''
#undef main
int main(void) {
    int64_t ints[] = {42}; narr_s a = {ints, 1, NULL};
    const char *texts[] = {"literal"}; nsarr_s s = {texts, 1, NULL};
    nrec_t records[1] = {{.n=1}}; records[0].f[0] = 77;
    nrarr_s r = {records, 1, NULL};
    narr_reserve(&a, 16); nsarr_reserve(&s, 16); nrarr_reserve(&r, 16);
    if (a.owner->handle || s.owner->handle || r.owner->handle) abort();
    nroot_frame roots = {0}; nroot_head = &roots;
    nroot_add(&roots.live, 3, &a); nroot_add(&roots.live, 5, &s); nroot_add(&roots.live, 6, &r);
    s.data[0] = nsarr_copy_string("escaped");
    const char *escaped = s.data[0];
    nroot_add(&roots.live, 1, escaped);
    nmap_collect();
    if (a.data[0] != 42 || r.data[0].f[0] != 77 || strcmp(s.data[0], "escaped")) abort();
    if (nagg_allocation_debt) abort();
    narr_reserve(&a, 100000);
    if (nagg_allocation_debt < 700000) abort();
    nmap_collect_if_needed();
    if (nagg_allocation_debt || a.data[0] != 42) abort();
    /* I release the borrowed handles' owned buffers, retaining only an escaped
       copied element. I never dereference the discarded handles afterward. */
    nroot_reset(&roots.live); nroot_add(&roots.live, 1, escaped);
    nmap_collect();
    if (narr_owners || nsarr_owners || nrarr_owners || !nsarr_strings) abort();
    if (strcmp(escaped, "escaped") || ints[0] != 42 || records[0].f[0] != 77 || strcmp(texts[0], "literal")) abort();
    nroot_reset(&roots.live); nmap_collect();
    if (nagg_live_bytes || nsarr_strings) abort();
    nroot_head = NULL; nroot_destroy(&roots.live);
    nrec_release_snapshots(); narr_release_owned(); nsarr_release_owned(); nrarr_release_owned();
    nmap_release_owned();
    return 0;
}
''')
            self.compile_run(source, work / 'program')


if __name__ == '__main__':
    unittest.main()
