"""I bound root lookup work without changing graph reachability or collection."""
try:
    from tests.sanitizer_options import asan_options
except ModuleNotFoundError:
    from sanitizer_options import asan_options
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeRootScaling(unittest.TestCase):
    def run_checked(self, command, **kwargs):
        result = subprocess.run([str(x) for x in command], capture_output=True,
                                text=True, timeout=90, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_growing_aliased_graph_has_bounded_lookup_work(self):
        with tempfile.TemporaryDirectory(prefix='nano-root-scaling-') as tmp:
            work = Path(tmp)
            assembly, module, source, binary = (work / name for name in
                                                ('input.nasm', 'input.nvm', 'input.c', 'input'))
            assembly.write_text('.entry main\n.function main 0 0 0 int 1\n'
                                'HM_NEW 5 5\nPOP\nPUSH_I64 0\nRET\n.end\n')
            self.run_checked([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
            translator = os.environ.get('NANO_ROOT_SCALING_TRANSLATOR', ROOT / 'bin/nvm2c')
            self.run_checked([translator, module, '-o', source])
            generated = source.read_text()
            # I count actual identity comparisons, avoiding host-speed timing assertions.
            generated = generated.replace('typedef struct { uint8_t kind; const void *ptr; } nroot_ref;',
                                          'static size_t root_probes;\n'
                                          'typedef struct { uint8_t kind; const void *ptr; } nroot_ref;')
            generated, count = re.subn(
                r'list->items\[(i|index)\]\.kind == kind && list->items\[\1\]\.ptr == ptr',
                lambda m: '(++root_probes, (' + m[0] + '))', generated)
            self.assertEqual(count, 1)
            for expression in ('v->value.text == root.ptr', 'm->map == root.ptr'):
                generated = generated.replace(expression, '(++root_probes, (' + expression + '))')
            source.write_text('#define main nano_program_main\n' + generated + '''
#undef main
#include <time.h>
#include <stdio.h>
int main(void) {
#ifdef NROOT_INDEX_TEST
    nroot_list identity = {0};
    char marker = 0;
    nroot_add(&identity, 1, &marker);
    nroot_add(&identity, 7, &marker);
    nroot_add(&identity, 1, &marker);
    if (identity.count != 2) abort();
    nroot_reset(&identity);
    if (identity.count || nroot_contains(&identity, 1, &marker)) abort();
    nroot_add(&identity, 7, &marker);
    if (identity.count != 1 || nroot_contains(&identity, 1, &marker)) abort();
    nroot_destroy(&identity);
    nrec_t cyclic = {0};
    cyclic.n = 1; cyclic.k[0] = 4; cyclic.rec[0] = &cyclic;
    nroot_add(&identity, 4, &cyclic);
    nroot_trace(&identity);
    if (identity.count != 1) abort();
    nroot_destroy(&identity);
#endif
    int excessive = 0;
    const size_t sizes[] = {1024, 2048, 4096};
    for (size_t trial = 0; trial < 3; ++trial) {
        size_t n = sizes[trial];
        nmap_t map = nmap_owned_new(5);
        nmap_set(map, "key", (nmap_value){5, 0, "retained"});
        const char **items = calloc(n * 2, sizeof *items);
        if (!items) abort();
        for (size_t i = 0; i < n; ++i) {
            nmap_value value = nmap_owned_get(map, "key");
            items[i] = items[n + i] = value.text;
        }
        nsarr_s array = {items, n * 2, NULL};
        nroot_frame frame = {0}; nroot_head = &frame;
        nroot_add(&frame.live, 5, &array);
        nroot_add(&frame.live, 7, map);
        root_probes = 0;
        clock_t begin = clock();
        nmap_collect();
        double elapsed = (double)(clock() - begin) / CLOCKS_PER_SEC;
        printf("roots=%zu comparisons=%zu seconds=%.6f\\n", n, root_probes, elapsed);
        if (nmap_owned_live != n + 1) abort();
        for (size_t i = 0; i < n * 2; ++i)
            if (strcmp(items[i], "retained")) abort();
        /* I retain duplicate array edges and require roughly linear lookup work. */
        if (root_probes > 64 * n) excessive = 2;
        array.len = 0;
        nmap_collect();
        if (nmap_owned_live != 1) abort();
        nroot_head = NULL;
        nmap_collect();
        if (nmap_owned_live != 0) abort();
#ifdef NROOT_INDEX_TEST
        nroot_destroy(&frame.live);
#else
        free(frame.live.items);
#endif
        free(items);
    }
    return excessive;
}
''')
            flags = ['-DNROOT_INDEX_TEST'] if 'static inline void nroot_reset(' in generated else []
            self.run_checked(['cc', '-std=c11', '-O2', '-g', '-Wall', '-Wextra', '-Werror',
                              '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                              *flags, source, '-o', binary])
            result = self.run_checked([binary], env={**os.environ, 'ASAN_OPTIONS': asan_options()})
            print(result.stdout, end='')


if __name__ == '__main__':
    unittest.main()
