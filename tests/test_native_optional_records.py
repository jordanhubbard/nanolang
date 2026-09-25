"""I preserve optional record values through nested arrays and collection."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TRANSLATOR = Path(os.environ.get('NANO_OPTIONAL_RECORD_TRANSLATOR', ROOT / 'bin/nvm2c'))
PREFIX = ('.types 1 0 0\n.entry main\n.function main 0 2 0 int 1\n'
          'PUSH_I64 42\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\nARR_LITERAL 7 1\nSTORE_LOCAL 0\n')
READ = 'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nPUSH_I64 {index}\nARR_GET\n'
SUFFIX = 'PUSH_I64 0\nRET\n.end\n'


class OptionalRecords(unittest.TestCase):
    def run_command(self, args):
        return subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True, text=True,
                              timeout=90, env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1',
                                               'UBSAN_OPTIONS': 'halt_on_error=1'})

    def checked(self, args):
        result = self.run_command(args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def emit(self, work, body):
        assembly, module, source = (work / n for n in ('input.nasm', 'input.nvm', 'output.c'))
        assembly.write_text(PREFIX + body + SUFFIX)
        self.checked([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
        self.checked([TRANSLATOR, module, '-o', source])
        return module, source

    def compile(self, source, binary):
        compiler = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC') or os.environ.get('CC') or 'cc')
        self.checked([*compiler, '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror',
                      '-fsanitize=address,undefined', '-fno-sanitize-recover=all', source, '-o', binary])

    def test_present_and_absent(self):
        for index in (0, -1, 1, 4294967296, 9223372036854775807):
            with self.subTest(index=index), tempfile.TemporaryDirectory(prefix='nano-optional-record-') as d:
                work = Path(d)
                body = READ.format(index=index) + 'STORE_LOCAL 1\nLOAD_LOCAL 1\n'
                body += ('DUP\nTYPE_CHECK 8\nASSERT\nAGG_GET 0\nPUSH_I64 42\nEQ\nASSERT\n'
                         if index == 0 else 'TYPE_CHECK 0\nASSERT\n')
                module, source = self.emit(work, body)
                self.checked([ROOT / 'bin/nano_vm', module])
                self.compile(source, work / 'program')
                self.checked([work / 'program'])

    def test_absent_field_traps(self):
        with tempfile.TemporaryDirectory(prefix='nano-optional-record-trap-') as d:
            work = Path(d)
            module, source = self.emit(work, READ.format(index=1) + 'AGG_GET 0\nPOP\n')
            vm = self.run_command([ROOT / 'bin/nano_vm', module])
            self.assertNotEqual(vm.returncode, 0)
            self.assertIn('AGG_GET field 0 is unavailable', vm.stderr)
            self.compile(source, work / 'program')
            native = self.run_command([work / 'program'])
            self.assertNotEqual(native.returncode, 0)
            self.assertIn('native invariant in nvalue_require_record', native.stderr)
            self.assertNotIn('ERROR: AddressSanitizer', native.stderr)

    def test_snapshot_retains_children_after_replacement_and_collection(self):
        with tempfile.TemporaryDirectory(prefix='nano-record-snapshot-') as d:
            work = Path(d)
            _, source = self.emit(work, READ.format(index=0) + 'AGG_GET 0\nPOP\nPUSH_I64 7\nCAST_STRING\nPOP\n')
            source.write_text('#define main nano_program_main\n' + source.read_text() + r'''
#undef main
int main(void) {
    nrec_t record = {.n = 1};
    record.k[0] = 1;
    record.s[0] = nstr_copy("retained child");
    nrarr_t array = nrarr_new();
    nrarr_push(array, record);
    nmap_value value = nvalue_array_get((nmap_value){7, 6, (char *)array}, 0);
    nroot_frame roots = {0}; roots.prev = nroot_head; nroot_head = &roots;
    nroot_value(&roots.live, value);
    array->data[0].s[0] = NULL;
    for (int i = 0; i < 600; ++i) nrarr_push(array, (nrec_t){.n = 1});
    nmap_collect();
    nrec_t saved = nvalue_require_record(value);
    if (strcmp(saved.s[0], "retained child")) abort();
    nroot_destroy(&roots.live); nroot_head = roots.prev;
    nmap_collect();
    if (nrec_owned_head || nrarr_owners || nstr_owners) abort();
    return 0;
}
''')
            self.compile(source, work / 'program')
            self.checked([work / 'program'])

    def test_generated_roots_retain_snapshot_across_mutation(self):
        text = PREFIX.replace('main 0 2', 'main 0 4').replace(
            'PUSH_I64 42\nAGG_PACK', 'PUSH_I64 42\nCAST_STRING\nAGG_PACK').replace(
            'ARR_LITERAL 8 1\n', 'ARR_LITERAL 8 1\nDUP\nSTORE_LOCAL 2\n')
        text += READ.format(index=0) + 'STORE_LOCAL 1\n'
        text += ('LOAD_LOCAL 2\nPUSH_I64 0\nPUSH_I64 9\nCAST_STRING\n'
                 'AGG_PACK 0 0 0 1\nARR_SET\nPOP\nPUSH_I64 0\nSTORE_LOCAL 3\n'
                 'loop:\nLOAD_LOCAL 2\nLOAD_LOCAL 3\nCAST_STRING\nAGG_PACK 0 0 0 1\n'
                 'ARR_PUSH\nPOP\nLOAD_LOCAL 3\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 3\n'
                 'LOAD_LOCAL 3\nPUSH_I64 600\nI64_LT_S\nJMP_TRUE loop\n'
                 'LOAD_LOCAL 1\nAGG_GET 0\nPUSH_I64 42\nCAST_STRING\nEQ\nASSERT\n') + SUFFIX
        with tempfile.TemporaryDirectory(prefix='nano-record-generated-roots-') as d:
            work = Path(d)
            assembly, module, source = (work / n for n in ('input.nasm', 'input.nvm', 'output.c'))
            assembly.write_text(text)
            self.checked([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
            self.checked([ROOT / 'bin/nano_vm', module])
            self.checked([TRANSLATOR, module, '-o', source])
            self.compile(source, work / 'program')
            self.checked([work / 'program'])

    def test_original_nested_generic_source(self):
        from tests.test_union_literal_context import UnionLiteralContext
        with tempfile.TemporaryDirectory(prefix='nano-original-record-') as d:
            work = Path(d)
            source, module, native = (work / n for n in ('original.nano', 'original.nvm', 'original.c'))
            original = UnionLiteralContext()
            original.check = source.write_text
            original.test_nested_generic_record_array()
            self.checked([ROOT / 'bin/nano_virt', source, '--emit-nvm', '-o', module])
            self.checked([ROOT / 'bin/nano_vm', module])
            self.checked([TRANSLATOR, module, '-o', native])
            self.compile(native, work / 'program')
            self.checked([work / 'program'])

    def test_record_comparison_retains_prior_output(self):
        for opcode in ('EQ', 'NE', 'LT', 'LE', 'GT', 'GE'):
            with self.subTest(opcode=opcode), tempfile.TemporaryDirectory(prefix='nano-record-comparison-') as d:
                work = Path(d)
                assembly, module, output = (work / n for n in ('input.nasm', 'input.nvm', 'output.c'))
                assembly.write_text(PREFIX + READ.format(index=0) + 'DUP\n' + opcode + '\nPOP\n' + SUFFIX)
                self.checked([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
                output.write_bytes(b'prior artifact')
                result = self.run_command([TRANSLATOR, module, '-o', output])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('record identity', result.stderr)
                self.assertEqual(output.read_bytes(), b'prior artifact')

    def test_snapshot_allocation_failure_preserves_source(self):
        with tempfile.TemporaryDirectory(prefix='nano-record-allocation-') as d:
            work = Path(d)
            _, source = self.emit(work, READ.format(index=0) + 'AGG_GET 0\nPOP\n')
            source.write_text(r'''#include <stdlib.h>
static int fail_malloc;
static int failed_calls;
static void *checked_malloc(size_t bytes) {
    if (fail_malloc) { ++failed_calls; return NULL; }
    return malloc(bytes);
}
_Noreturn static void checked_abort(void);
#define malloc checked_malloc
#define abort checked_abort
#define main nano_program_main
''' + source.read_text() + r'''
#undef main
#undef abort
#undef malloc
static nrec_t initial;
static nrarr_s array;
_Noreturn static void checked_abort(void) {
    if (failed_calls != 1 || nrec_owned_head || array.data != &initial ||
        array.len != 1 || initial.f[0] != 42) exit(3);
    exit(77);
}
int main(void) {
    initial.n = 1; initial.f[0] = 42;
    array.data = &initial; array.len = 1;
    fail_malloc = 1;
    (void)nvalue_array_get((nmap_value){7, 6, (char *)&array}, 0);
    return 5;
}
''')
            self.compile(source, work / 'program')
            result = self.run_command([work / 'program'])
            self.assertEqual(result.returncode, 77, result.stdout + result.stderr)

    def test_wrong_runtime_tags_trap_before_pointer_use(self):
        with tempfile.TemporaryDirectory(prefix='nano-record-tag-') as d:
            work = Path(d)
            _, source = self.emit(work, READ.format(index=0) + 'AGG_GET 0\nPOP\n')
            source.write_text('#define main nano_program_main\n' + source.read_text() + r'''
#undef main
int main(int argc, char **argv) {
    (void)argv;
    nrec_t record = {.kind = 1, .n = 1};
    nmap_value value = argc == 1 ? (nmap_value){5, 0, (char *)1} :
        argc == 2 ? (nmap_value){8, 0, NULL} : (nmap_value){8, 0, (char *)&record};
    (void)nvalue_require_record(value);
    return 0;
}
''')
            self.compile(source, work / 'program')
            for args in ([], ['null'], ['wrong', 'kind']):
                result = self.run_command([work / 'program', *args])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('native invariant in nvalue_require_record', result.stderr)
                self.assertNotIn('ERROR: AddressSanitizer', result.stderr)


if __name__ == '__main__':
    unittest.main()
