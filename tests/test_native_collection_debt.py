"""I trace fresh roots when owners are allocated and bound allocation-free work."""
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
IDLE = ('.function idle 0 1 0 void 0\nPUSH_I64 0\nSTORE_LOCAL 0\nloop:\n'
        'LOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 0\n'
        'LOAD_LOCAL 0\nPUSH_I64 5000\nI64_LT_S\nJMP_FALSE done\nJMP loop\n'
        'done:\nRET\n.end\n')


class NativeCollectionDebt(unittest.TestCase):
    def run_checked(self, args, **kwargs):
        result = subprocess.run([str(x) for x in args], capture_output=True,
                                text=True, timeout=90, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_allocation_free_loops_and_new_owner_debt(self):
        text = ('.string key "key"\n.string first "first"\n.string second "second"\n.entry main\n'
                '.function main 0 2 0 int 1\n'
                'HM_NEW 5 5\nPUSH_STR key\nPUSH_STR first\nHM_SET\nSTORE_LOCAL 0\n'
                'LOAD_LOCAL 0\nPUSH_STR key\nHM_GET\nSTORE_LOCAL 1\nCALL idle\n'
                'LOAD_LOCAL 1\nPUSH_STR first\nEQ\nASSERT\n'
                # I drop the old copied string, then allocate a replacement owner.
                'LOAD_LOCAL 0\nPUSH_STR key\nPUSH_STR second\nHM_SET\nPOP\n'
                'LOAD_LOCAL 0\nPUSH_STR key\nHM_GET\nSTORE_LOCAL 1\nCALL idle\n'
                'LOAD_LOCAL 1\nPUSH_STR second\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n' + IDLE)
        with tempfile.TemporaryDirectory(prefix='nano-collection-debt-') as tmp:
            work = Path(tmp)
            assembly, module, source, binary = (work / name for name in
                                                ('input.nasm', 'input.nvm', 'input.c', 'input'))
            assembly.write_text(text)
            self.run_checked([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
            self.run_checked([ROOT / 'bin/nano_vm', module])
            translator = os.environ.get('NANO_DEBT_TRANSLATOR', ROOT / 'bin/nvm2c')
            self.run_checked([translator, module, '-o', source])
            generated = source.read_text()
            generated = generated.replace('typedef struct { uint8_t kind; const void *ptr; } nroot_ref;',
                                          'static size_t root_scans;\n'
                                          'typedef struct { uint8_t kind; const void *ptr; } nroot_ref;')
            needle = 'static void nroot_trace(nroot_list *work) {\n'
            self.assertIn(needle, generated)
            generated = generated.replace(needle, needle + '    ++root_scans;\n')
            generated = generated.replace('    nmap_release_owned();',
                '    nmap_release_owned();\n'
                '    printf("scans=%zu peak=%zu\\n", root_scans, nmap_owned_peak);\n'
                '    if (nmap_owned_live || nmap_owned_peak > 3) abort();\n'
                '    if (root_scans != 0 || nmap_live_bytes || nmap_peak_bytes > 65536) return 2;\n')
            source.write_text('#include <stdio.h>\n' + generated)
            self.run_checked(['cc', '-std=c11', '-O2', '-g', '-Wall', '-Wextra', '-Werror',
                              '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                              source, '-o', binary])
            result = self.run_checked([binary], env={**os.environ, 'ASAN_OPTIONS': asan_options()})
            print(result.stdout, end='')


    def test_forced_drops_and_current_mutable_edges(self):
        with tempfile.TemporaryDirectory(prefix='nano-deferred-roots-') as tmp:
            work = Path(tmp)
            assembly, module, source, binary = (work / name for name in
                                                ('input.nasm', 'input.nvm', 'input.c', 'input'))
            assembly.write_text('.entry main\n.function main 0 0 0 int 1\n'
                                'HM_NEW 5 5\nPOP\nPUSH_I64 0\nRET\n.end\n')
            self.run_checked([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
            self.run_checked([ROOT / 'bin/nvm2c', module, '-o', source])
            generated = source.read_text()
            generated = generated.replace('typedef struct { uint8_t kind; const void *ptr; } nroot_ref;',
                                          'static size_t root_scans;\n'
                                          'typedef struct { uint8_t kind; const void *ptr; } nroot_ref;')
            generated = generated.replace('static void nroot_trace(nroot_list *work) {\n',
                                          'static void nroot_trace(nroot_list *work) {\n    ++root_scans;\n')
            source.write_text('#define main nano_program_main\n' + generated + '''
#undef main
int main(void) {
    nmap_t map = nmap_owned_new(5);
    nmap_set(map, "key", (nmap_value){5, 0, "first"});
    const char *items[] = {nmap_owned_get(map, "key").text};
    nsarr_s array = {items, 1, NULL};
    nroot_frame frame = {0}; nroot_head = &frame;
    nroot_add(&frame.live, 5, &array);
    nmap_collect();
    if (nmap_owned_live != 1 || root_scans != 1) abort();
    array.len = 0;
    for (size_t i = 0; i < 10000; ++i) nmap_collect_if_needed();
    if (nmap_owned_live != 1 || root_scans != 1) abort();
    map = nmap_owned_new(5);
    nmap_set(map, "key", (nmap_value){5, 0, "second"});
    items[0] = nmap_owned_get(map, "key").text; array.len = 1;
    nmap_collect();
    if (nmap_owned_live != 1 || root_scans != 2 || strcmp(items[0], "second")) abort();
    array.len = 0;
    nmap_collect_if_needed();
    if (nmap_owned_live != 1 || root_scans != 2) abort();
    nmap_collect();
    if (nmap_owned_live || root_scans != 3) abort();
    (void)nmap_owned_new(5);
    nmap_collect();
    nmap_collect_if_needed();
    if (nmap_owned_live || root_scans != 4 || nmap_owned_peak > 3) abort();
    nroot_head = NULL; nroot_destroy(&frame.live);
    nmap_release_owned();
    return 0;
}
''')
            self.run_checked(['cc', '-std=c11', '-O2', '-g', '-Wall', '-Wextra', '-Werror',
                              '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                              source, '-o', binary])
            self.run_checked([binary], env={**os.environ, 'ASAN_OPTIONS': asan_options()})


if __name__ == '__main__':
    unittest.main()
