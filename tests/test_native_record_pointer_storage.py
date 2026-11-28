"""I retain mixed record pointer fields and independent boxed scalar bits."""
import os
from pathlib import Path
import shlex
import tempfile
import unittest

from tests.native_process import native_link_flags, run_captured

ROOT = Path(__file__).resolve().parents[1]


class NativeRecordPointerStorage(unittest.TestCase):
    def command(self, args):
        result = run_captured(list(map(str, args)), cwd=ROOT, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_mixed_pointer_fields_preserve_boxed_scalar_payload(self):
        with tempfile.TemporaryDirectory(prefix='nano-record-pointer-storage-') as directory:
            work = Path(directory)
            assembly, module, generated = work / 'input.nasm', work / 'input.nvm', work / 'generated.c'
            assembly.write_text('.entry main\n.function main 0 0 0 int 1\n'
                                'HM_NEW 5 1\nPOP\n' + 'PUSH_I64 1\n' * 8 +
                                'AGG_PACK 0 0 0 8\nPOP\nPUSH_I64 0\nRET\n.end\n')
            self.command([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
            self.command([ROOT / 'bin/nano_vm', module])
            self.command([ROOT / 'bin/nvm2c', module, '-o', generated])
            flags = [*shlex.split(os.environ.get('CC', 'cc')), '-std=c11', '-O1', '-g',
                     '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                     '-fno-sanitize-recover=all']
            libraries = [ROOT / 'bin/nano_aot_runtime.o', '-lm', *native_link_flags()]
            native = work / 'native'
            self.command([*flags, generated, *libraries, '-o', native])
            self.command([native])
            probe = work / 'probe.c'
            probe.write_text('''#define main original_main
#include "generated.c"
#undef main
#include <assert.h>
#include <stddef.h>
int main(void) {
    _Alignas(max_align_t) unsigned char markers[7][sizeof(max_align_t)] = {{0}};
    nrec_t record = {0}, child = {0};
    record.n = 8;
    record.k[0] = 1; record.s[0] = (char *)markers[0];
    record.k[1] = 3; record.a[1] = (narr_t)markers[1];
    record.k[2] = 5; record.sa[2] = (nsarr_t)markers[2];
    record.k[3] = 6; record.ra[3] = (nrarr_t)markers[3];
    record.k[4] = 4; record.rec[4] = &child;
    record.k[5] = 7; record.m[5] = (nmap_t)markers[5];
    record.k[6] = 8; record.vk[6] = 7;
    record.f[6] = 17; record.s[6] = (char *)markers[6];
    uint64_t bits = UINT64_C(0xfff8000000000011);
    record.k[7] = 11; memcpy(&record.f[7], &bits, sizeof bits);
    assert(record.s[0] == (char *)markers[0]);
    assert(record.a[1] == (narr_t)markers[1]);
    assert(record.sa[2] == (nsarr_t)markers[2]);
    assert(record.ra[3] == (nrarr_t)markers[3]);
    assert(record.rec[4] == &child);
    assert(record.m[5] == (nmap_t)markers[5]);
    assert(record.f[6] == 17 && record.s[6] == (char *)markers[6]);
    assert(record.k[6] == 8 && record.vk[6] == 7);
    uint64_t actual; memcpy(&actual, &record.f[7], sizeof actual);
    assert(actual == bits);
    record.s[0] = NULL;
    record.a[1] = (narr_t)markers[0];
    assert(!record.s[0] && record.sa[2] == (nsarr_t)markers[2]);
    assert(record.f[6] == 17 && record.s[6] == (char *)markers[6]);
    memcpy(&actual, &record.f[7], sizeof actual); assert(actual == bits);
    return 0;
}
''')
            binary = work / 'probe'
            self.command([*flags, probe, *libraries, '-o', binary])
            self.command([binary])


if __name__ == '__main__':
    unittest.main()
