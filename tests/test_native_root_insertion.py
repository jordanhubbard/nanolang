"""I preserve native root insertion semantics and bound repeated lookup work."""
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PREFIX = '''#include <setjmp.h>
static size_t root_test_lookup_calls;
static size_t root_test_allocations, root_test_fail_at;
static char root_test_allocation_kinds[2048];
static jmp_buf root_test_jump;
static int root_test_abort_armed;
static _Noreturn void root_test_abort(void) {
    if (root_test_abort_armed) longjmp(root_test_jump, 1);
    abort();
}
static void root_test_note_allocation(char kind) {
    if (root_test_allocations >= sizeof root_test_allocation_kinds) abort();
    root_test_allocation_kinds[root_test_allocations++] = kind;
}
static void *root_test_calloc(size_t n, size_t size) {
    root_test_note_allocation('I');
    if (root_test_fail_at && root_test_allocations == root_test_fail_at) return NULL;
    return calloc(n, size);
}
static void *root_test_realloc(void *p, size_t size) {
    root_test_note_allocation('A');
    if (root_test_fail_at && root_test_allocations == root_test_fail_at) return NULL;
    return realloc(p, size);
}
'''


class NativeRootInsertion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='nano-root-insertion-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        assembly, module, source = (cls.work / n for n in ('input.nasm', 'input.nvm', 'runtime.c'))
        assembly.write_text('.entry main\n.function main 0 0 0 int 1\n'
                            'HM_NEW 5 5\nPOP\nPUSH_I64 0\nRET\n.end\n')
        cls.checked([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
        translator = os.environ.get('NANO_ROOT_SCALING_TRANSLATOR', ROOT / 'bin/nvm2c')
        cls.checked([translator, module, '-o', source])
        cls.generated = source.read_text()

    @staticmethod
    def checked(command):
        result = subprocess.run([str(x) for x in command], capture_output=True,
                                text=True, timeout=90)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return result

    def instrument(self, variant):
        runtime = self.generated
        start = runtime.index('static void nroot_add(')
        end = runtime.index('static void nroot_value(', start)
        body = runtime[start:end]
        if variant == 'previous':
            body = (ROOT / 'tests/support/native_root_insertion_previous.inc').read_text()
        elif variant == 'missing-rehash':
            loop = ('        for (size_t i = 0; i < list->count; ++i)\n'
                    '            list->slots[nroot_slot(list, list->items[i].kind, list->items[i].ptr)] = i + 1;\n')
            self.assertEqual(body.count(loop), 1)
            body = body.replace(loop, '')
        self.assertEqual(body.count('calloc('), 1)
        self.assertEqual(body.count('realloc('), 1)
        self.assertEqual(body.count('NVM2C_ABORT()'), 4)
        body = body.replace('calloc(', 'root_test_calloc(').replace('realloc(', 'root_test_realloc(')
        body = body.replace('NVM2C_ABORT()', 'root_test_abort()')
        runtime = runtime[:start] + body + runtime[end:]
        marker = 'typedef struct { uint8_t kind; const void *ptr; } nroot_ref;'
        self.assertEqual(runtime.count(marker), 1)
        runtime = runtime.replace(marker, PREFIX + marker)
        marker = 'static size_t nroot_slot(const nroot_list *list, uint8_t kind, const void *ptr) {\n'
        self.assertEqual(runtime.count(marker), 1)
        runtime = runtime.replace(marker, marker + '    ++root_test_lookup_calls;\n')
        return ('#define main nano_program_main\n' + runtime + '\n#undef main\n' +
                (ROOT / 'tests/native_root_insertion_probe.c').read_text())

    def run_variant(self, variant):
        source = self.work / (variant + '.c')
        source.write_text(self.instrument(variant))
        for mode, flags in [('ordinary', ['-O2']), ('sanitized', ['-O1', '-g',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all'])]:
            with self.subTest(mode=mode):
                binary = self.work / (variant + '-' + mode)
                self.checked(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', *flags,
                              source, '-o', binary])
                result = subprocess.run([str(binary)], capture_output=True, text=True,
                    timeout=90, preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
                if variant == 'missing-rehash':
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('Assertion', result.stderr)
                    continue
                self.assertEqual(result.returncode, 7 if variant == 'previous' else 0,
                                 result.stdout + result.stderr)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertNotIn('ERROR:', result.stderr)
                rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
                self.assertEqual([row['roots'] for row in rows], [16, 32, 1024, 4096])
                self.assertIn('eight allocation-abort boundaries.', result.stdout)
                for row in rows:
                    n = row['roots']
                    self.assertEqual(row['duplicate_lookups'], n)
                    self.assertEqual(row['reused_lookups'], n * (2 if variant == 'previous' else 1))
                    self.assertEqual(row['allocations'], 2 * (n.bit_length() - 4))
                    if variant == 'current':
                        self.assertLessEqual(row['fresh_lookups'], 2 * n)
                    elif n > 16:
                        self.assertLess(2 * n, row['fresh_lookups'])
                print(variant, mode, result.stdout, end='')

    def test_current_preserves_identity_allocation_and_lookup_budget(self):
        self.run_variant('current')

    def test_previous_helper_exceeds_single_lookup_budget(self):
        self.run_variant('previous')

    def test_missing_rehash_loses_required_membership(self):
        self.run_variant('missing-rehash')


if __name__ == '__main__':
    unittest.main()
