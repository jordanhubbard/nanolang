"""I batch complete map allocation bytes without losing published aliases."""
import os
import signal
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests.native_toolchain import native_cc

ROOT = Path(__file__).resolve().parents[1]
ASM = ('.types 1 0 0\n.entry main\n.function main 0 0 0 int 1\n'
       'PUSH_I64 42\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\nPOP\n'
       'HM_NEW 5 5\nPOP\nPUSH_I64 0\nRET\n.end\n')


class NativeMapByteDebt(unittest.TestCase):
    def run_checked(self, args, **kwargs):
        result = subprocess.run(list(map(str, args)), capture_output=True, text=True,
                                timeout=90, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def check_harness(self, body, refusals=(), counters=()):
        with tempfile.TemporaryDirectory(prefix='nano-map-byte-debt-') as tmp:
            work = Path(tmp)
            asm, module, source, binary = [work / x for x in ('input.nasm', 'input.nvm', 'input.c', 'program')]
            asm.write_text(ASM)
            self.run_checked([ROOT / 'bin/nanoisa', 'asm', asm, '-o', module])
            self.run_checked([ROOT / 'bin/nano_vm', module])
            self.run_checked([ROOT / 'bin/nvm2c', module, '-o', source])
            generated = source.read_text()
            if counters:
                declaration = 'static size_t ' + ', '.join(counters) + ';\n'
                scan = '\n++scans;' if 'scans' in counters else ''
                hook = 'static void nroot_trace(nroot_list *work) {'
                self.assertEqual(generated.count(hook), 1)
                generated = generated.replace(hook, declaration + hook + scan)
            if 'visits' in counters:
                hook = '        nroot_ref root = work->items[cursor];'
                self.assertEqual(generated.count(hook), 1)
                generated = generated.replace(hook, '        ++visits;\n' + hook)
            source.write_text('#define main original_main\n' + generated + '\n#undef main\n#include <stdio.h>\n' + body)
            self.run_checked([*native_cc(), '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                              '-fsanitize=address,undefined', '-fno-sanitize-recover=all', source, '-o', binary])
            env = {**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1'}
            print(self.run_checked([binary], env=env).stdout, end='')
            for mode in refusals:
                refused = subprocess.run([str(binary), mode], capture_output=True, timeout=30, env=env)
                self.assertEqual(refused.returncode, -signal.SIGABRT, mode)
                self.assertNotIn(b'AddressSanitizer', refused.stderr)
                self.assertNotIn(b'runtime error:', refused.stderr)

    def test_copied_reads_batch_large_live_graph(self):
        self.check_harness(r'''
int main(void) {
    nrarr_t rows=nrarr_new(); nrec_t record={0}; record.n=1; record.k[0]=2; record.f[0]=42;
    for(size_t i=0;i<10000;i++) nrarr_push(rows,record);
    nmap_t map=nmap_owned_new(5); nmap_set(map,"key",(nmap_value){5,0,"value"});
    nroot_frame frame={0}; nroot_head=&frame;
    nroot_add(&frame.live,6,rows); nroot_add(&frame.live,7,map);
    const char *escaped=nmap_owned_get(map,"key").text;
    nroot_add(&frame.live,1,escaped);
    nmap_collect(); scans=visits=0;
    for(size_t i=0;i<5000;i++) {
        nmap_value value=nmap_owned_get(map,"key");
        if(strcmp(value.text,"value")) abort();
        nmap_collect_if_needed();
    }
    if(scans<1 || scans>4 || visits>40032 || strcmp(escaped,"value")) abort();
    if(rows->len!=10000 || rows->data[9999].f[0]!=42) abort();
    if(nmap_peak_bytes>69632) abort();
    printf("scans=%zu visits=%zu map_peak_bytes=%zu\n",scans,visits,nmap_peak_bytes);
    nroot_head=NULL; nroot_destroy(&frame.live); nmap_collect();
    if(nmap_owned_live || nmap_live_bytes || nagg_live_bytes) abort();
    nmap_release_owned(); nrarr_release_owned(); nrec_release_snapshots();
    return 0;
}
''', counters=('scans', 'visits'))

    def test_accounting_rejects_wrap_and_underflow(self):
        self.check_harness(r'''
int main(int argc, char **argv) {
    if(argc>1) {
        if(argv[1][0]=='l') nmap_live_bytes=SIZE_MAX;
        else if(argv[1][0]=='d') nmap_allocation_debt=SIZE_MAX;
        else { nmap_bytes_drop(1); return 0; }
        nmap_bytes_add(1); return 0;
    }
    nmap_bytes_add(42); nmap_bytes_drop(42);
    if(nmap_live_bytes || nmap_allocation_debt!=42 || nmap_peak_bytes!=42) abort();
    return 0;
}
''', refusals=('live', 'debt', 'underflow'))

    def test_map_storage_growth_replacement_and_forced_release(self):
        self.check_harness(r'''
int main(void) {
    nmap_t map=nmap_owned_new(5); nroot_frame frame={0}; nroot_head=&frame;
    nroot_add(&frame.live,7,map);
    char payload[2049]; memset(payload,'x',2048); payload[2048]=0;
    for(size_t i=0;i<100;i++) {
        char key[32]; snprintf(key,sizeof key,"key-%zu",i);
        nmap_set(map,key,(nmap_value){5,0,payload});
        nmap_collect_if_needed();
    }
    if(nmap_live_bytes<204900 || !scans || nmap_len(map)!=100) abort();
    nmap_value escape=nmap_owned_get(map,"key-0");
    nroot_add(&frame.live,1,escape.text);
    size_t large=nmap_live_bytes;
    nmap_set(map,"key-0",(nmap_value){5,0,"short"});
    if(nmap_live_bytes>=large || strlen(escape.text)!=2048) abort();
    for(size_t i=0;i<100;i++) { char key[32]; snprintf(key,sizeof key,"key-%zu",i); nmap_delete(map,key); }
    if(nmap_len(map) || nmap_live_bytes>=large/2) abort();
    nmap_collect();
    if(strlen(escape.text)!=2048 || nmap_owned_live!=2) abort();
    nroot_head=NULL; nroot_destroy(&frame.live); nmap_collect();
    if(nmap_owned_live || nmap_live_bytes || nmap_allocation_debt) abort();
    nmap_release_owned();
    return 0;
}
''', counters=('scans',))


if __name__ == '__main__':
    unittest.main()
