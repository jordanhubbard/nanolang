"""I grow native record arrays without losing aliases, values or owned edges."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests.native_toolchain import native_cc

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/nanoisa/fixtures/record_array_growth_257.nasm'


class NativeRecordGrowth(unittest.TestCase):
    def run_checked(self, args, **kwargs):
        result = subprocess.run([str(x) for x in args], capture_output=True,
                                text=True, timeout=90, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def emit(self, work, text):
        assembly, module, source = (work / name for name in ('input.nasm', 'input.nvm', 'input.c'))
        assembly.write_text(text)
        self.run_checked([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
        self.run_checked([ROOT / 'bin/nano_vm', module])
        self.run_checked([ROOT / 'bin/nvm2c', module, '-o', source])
        return source

    def compile(self, source, binary):
        self.run_checked([*native_cc(), '-std=c11', '-g', '-Wall', '-Wextra', '-Werror',
                          '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                          source, '-o', binary])

    def test_growth_retains_handle_alias_and_values(self):
        text = FIXTURE.read_text().replace('main 0 2', 'main 0 3').replace(
            'ARR_NEW 8\nSTORE_LOCAL 0', 'ARR_NEW 8\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nSTORE_LOCAL 2')
        checks = ''.join(f'LOAD_LOCAL 2\nPUSH_I64 {i}\nARR_GET\nAGG_GET 0\n'
                         f'PUSH_I64 {i}\nEQ\nASSERT\n' for i in (0, 7, 8, 127, 255, 256))
        text = text.replace('done:\n', 'done:\n' + checks)
        with tempfile.TemporaryDirectory(prefix='nano-record-growth-') as tmp:
            work = Path(tmp)
            source = self.emit(work, text)
            self.compile(source, work / 'program')
            self.run_checked([work / 'program'], env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1'})

    def test_growth_preserves_owned_strings_through_collection(self):
        text = ('.entry main\n.types 1 0 0\n.string key "key"\n.string text "retained"\n'
                '.function value 0 0 0 string 1\nHM_NEW 5 5\nPUSH_STR key\nPUSH_STR text\n'
                'HM_SET\nPUSH_STR key\nHM_GET\nRET\n.end\n'
                '.function main 0 2 0 int 1\nARR_NEW 8\nSTORE_LOCAL 0\nPUSH_I64 0\nSTORE_LOCAL 1\n'
                'loop:\nLOAD_LOCAL 0\nCALL value\nLOAD_LOCAL 1\nAGG_PACK 0 0 0 2\nARR_PUSH\nPOP\n'
                'LOAD_LOCAL 1\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nPUSH_I64 600\n'
                'I64_LT_S\nJMP_FALSE done\nJMP loop\ndone:\n')
        for index in (0, 255, 256, 599):
            text += (f'LOAD_LOCAL 0\nPUSH_I64 {index}\nARR_GET\nAGG_GET 0\nPUSH_STR text\nEQ\nASSERT\n'
                     f'LOAD_LOCAL 0\nPUSH_I64 {index}\nARR_GET\nAGG_GET 1\nPUSH_I64 {index}\nEQ\nASSERT\n')
        text += 'PUSH_I64 0\nRET\n.end\n'
        with tempfile.TemporaryDirectory(prefix='nano-record-growth-roots-') as tmp:
            work = Path(tmp)
            source = self.emit(work, text)
            self.compile(source, work / 'program')
            self.run_checked([work / 'program'], env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1'})

    def test_borrowed_storage_growth_preserves_stack_owner(self):
        with tempfile.TemporaryDirectory(prefix='nano-record-borrowed-growth-') as tmp:
            work = Path(tmp)
            source = self.emit(work, FIXTURE.read_text())
            source.write_text('#define main nano_program_main\n' + source.read_text() + '''
#undef main
int main(void) {
    nrec_t initial[2] = {{.n = 1}, {.n = 1}};
    initial[0].f[0] = 11; initial[1].f[0] = 22;
    nrarr_s array = {.data = initial, .len = 2};
    nrarr_t alias = &array;
    nrec_t next = {.n = 1}; next.f[0] = 33;
    nrarr_push(&array, next);
    if (array.data == initial || alias->len != 3 || alias->data[0].f[0] != 11 ||
        alias->data[1].f[0] != 22 || alias->data[2].f[0] != 33 || initial[1].f[0] != 22) abort();
    nrarr_release_owned();
    if (initial[0].f[0] != 11) abort();
    return 0;
}
''')
            self.compile(source, work / 'program')
            self.run_checked([work / 'program'], env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1'})

    def test_growth_failure_preserves_existing_storage(self):
        with tempfile.TemporaryDirectory(prefix='nano-record-growth-failure-') as tmp:
            work = Path(tmp)
            source = self.emit(work, FIXTURE.read_text())
            source.write_text('''#include <stdlib.h>
static int fail_realloc;
static size_t resize_calls;
static void *checked_realloc(void *p, size_t n) {
    if (fail_realloc) { ++resize_calls; return NULL; }
    return realloc(p, n);
}
_Noreturn static void checked_abort(void);
#define realloc checked_realloc
#define abort checked_abort
#define main nano_program_main
''' + source.read_text() + '''
#undef main
#undef abort
#undef realloc
static nrarr_t observed;
static nrec_t *previous;
static int overflow_case;
_Noreturn static void checked_abort(void) {
    if (!observed || observed->data != previous || observed->len != 8 ||
        observed->owner->data != previous || observed->owner->cap != 8 ||
        resize_calls != (overflow_case ? 0u : 1u)) exit(3);
    for (size_t i = 0; i < 8; ++i) if (observed->data[i].f[0] != (int64_t)i) exit(4);
    nrarr_release_owned();
    exit(77);
}
int main(int argc, char **argv) {
    (void)argv;
    observed = nrarr_new();
    for (int64_t i = 0; i < 8; ++i) { nrec_t record = {.n = 1}; record.f[0] = i; nrarr_push(observed, record); }
    previous = observed->data;
    fail_realloc = 1; overflow_case = argc > 1;
    if (overflow_case) nrarr_reserve(observed, SIZE_MAX);
    else { nrec_t record = {.n = 1}; nrarr_push(observed, record); }
    return 5;
}
''')
            self.compile(source, work / 'program')
            for args in ([], ['overflow']):
                result = subprocess.run([str(work / 'program'), *args], capture_output=True,
                                        timeout=30, env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1'})
                self.assertEqual(result.returncode, 77, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
