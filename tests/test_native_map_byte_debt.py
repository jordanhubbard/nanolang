"""I batch complete map allocation bytes without losing published aliases."""
import os
import signal
from pathlib import Path
import subprocess
import tempfile
import unittest

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

    def check_harness(self, body, refusals=(), *, trace_scans=False, trace_visits=False, assembly=ASM):
        with tempfile.TemporaryDirectory(prefix='nano-map-byte-debt-') as tmp:
            work = Path(tmp)
            asm, module, source, binary = [work / x for x in ('input.nasm', 'input.nvm', 'input.c', 'program')]
            asm.write_text(assembly)
            self.run_checked([ROOT / 'bin/nanoisa', 'asm', asm, '-o', module])
            self.run_checked([ROOT / 'bin/nano_vm', module])
            self.run_checked([ROOT / 'bin/nvm2c', module, '-o', source])
            generated = source.read_text()
            if trace_scans:
                generated = generated.replace('static void nroot_trace(nroot_list *work) {',
                    'static size_t scans;\nstatic void nroot_trace(nroot_list *work) {\n++scans;')
            if trace_visits:
                generated = generated.replace('static void nroot_trace(nroot_list *work) {',
                    'static size_t visits;\nstatic void nroot_trace(nroot_list *work) {')
                generated = generated.replace('        nroot_ref root = work->items[cursor];',
                    '        ++visits;\n        nroot_ref root = work->items[cursor];')
            source.write_text('#define main original_main\n' + generated + '\n#undef main\n#include <stdio.h>\n' + body)
            self.run_checked(['cc', '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
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
    if(scans>4 || visits>40032 || strcmp(escaped,"value")) abort();
    if(rows->len!=10000 || rows->data[9999].f[0]!=42) abort();
    if(nmap_peak_bytes>nagg_live_bytes+69632) abort();
    printf("scans=%zu visits=%zu map_peak_bytes=%zu\n",scans,visits,nmap_peak_bytes);
    nroot_head=NULL; nroot_destroy(&frame.live); nmap_collect();
    if(nmap_owned_live || nmap_live_bytes || nagg_live_bytes) abort();
    nmap_release_owned(); nrarr_release_owned(); nrec_release_snapshots();
    return 0;
}
''', trace_scans=True, trace_visits=True)

    def test_small_map_debt_amortizes_over_retained_aggregates(self):
        self.check_harness(r'''
int main(void) {
    nrarr_t rows=nrarr_new(); nrec_t record={0};
    record.n=1; record.k[0]=2; record.f[0]=42;
    for(size_t i=0;i<10000;i++) nrarr_push(rows,record);
    nmap_t map=nmap_owned_new(5); nmap_set(map,"key",(nmap_value){5,0,"value"});
    nroot_frame frame={0}; nroot_head=&frame;
    nroot_add(&frame.live,6,rows); nroot_add(&frame.live,7,map);
    const char *escaped=nmap_owned_get(map,"key").text;
    nroot_add(&frame.live,1,escaped);
    nmap_collect(); scans=visits=0;
    size_t retained=nmap_live_bytes+nagg_live_bytes;
    size_t allocated=0;
    while(allocated<2*retained) {
        size_t before=nmap_allocation_debt;
        nmap_value value=nmap_owned_get(map,"key");
        allocated+=nmap_allocation_debt-before;
        if(strcmp(value.text,"value")) abort();
        nmap_collect_if_needed();
    }
    printf("mixed scans=%zu visits=%zu retained=%zu allocated=%zu peak=%zu\n",
           scans,visits,retained,allocated,nmap_peak_bytes);
    fflush(stdout);
    if(scans<1 || scans>3 || visits>30024) abort();
    if(nmap_peak_bytes>2*retained+4096) abort();
    if(strcmp(escaped,"value") || rows->data[9999].f[0]!=42) abort();
    nroot_head=NULL; nroot_destroy(&frame.live); nmap_collect();
    if(nmap_owned_live || nmap_live_bytes || nagg_live_bytes) abort();
    nmap_release_owned(); nrarr_release_owned(); nrec_release_snapshots();
    return 0;
}
''', trace_scans=True, trace_visits=True)

    def test_combined_string_map_and_record_debt_preserves_aliases(self):
        assembly=ASM.replace('HM_NEW 5 5', 'PUSH_I64 42\nCAST_STRING\nPOP\nHM_NEW 5 5')
        self.check_harness(r'''
int main(void) {
    nrarr_t rows=nrarr_new(); nrec_t record={0};
    record.n=1; record.k[0]=2; record.f[0]=42;
    for(size_t i=0;i<10000;i++) nrarr_push(rows,record);
    nmap_t map=nmap_owned_new(5); nmap_set(map,"key",(nmap_value){5,0,"value"});
    const char *escaped=nstr_from_i64(123456789);
    nroot_frame frame={0}; nroot_head=&frame;
    nroot_add(&frame.live,6,rows); nroot_add(&frame.live,7,map);
    nroot_add(&frame.live,1,escaped);
    nmap_collect(); scans=visits=0;
    size_t retained=nmap_live_bytes+nagg_live_bytes+nstr_live_bytes;
    size_t allocated=0;
    while(allocated<2*retained) {
        size_t before=nmap_allocation_debt+nagg_allocation_debt+nstr_allocation_debt;
        char *text=nstr_allocate(1024); text[0]=0;
        (void)nrec_snapshot(record);
        nmap_value value=nmap_owned_get(map,"key");
        allocated+=nmap_allocation_debt+nagg_allocation_debt+nstr_allocation_debt-before;
        if(strcmp(value.text,"value")) abort();
        nmap_collect_if_needed();
        if(nmap_live_bytes+nagg_live_bytes+nstr_live_bytes>2*retained+4096) abort();
    }
    printf("all families scans=%zu visits=%zu retained=%zu allocated=%zu\n",
           scans,visits,retained,allocated);
    if(scans<1 || scans>3 || visits>30024) abort();
    if(strcmp(escaped,"123456789") || rows->data[9999].f[0]!=42) abort();
    nroot_head=NULL; nroot_destroy(&frame.live); nmap_collect();
    if(nmap_owned_live || nmap_live_bytes || nagg_live_bytes || nstr_live_bytes) abort();
    nmap_release_owned(); nrarr_release_owned(); nrec_release_snapshots(); nstr_release_owned();
    return 0;
}
''', trace_scans=True, trace_visits=True, assembly=assembly)

    def test_accounting_rejects_wrap_and_underflow(self):
        self.check_harness(r'''
int main(int argc, char **argv) {
    if(argc>1) {
        if(argv[1][0]=='l') nmap_live_bytes=SIZE_MAX;
        else if(argv[1][0]=='d') nmap_allocation_debt=SIZE_MAX;
        else if(argv[1][0]=='s') { (void)nheap_bytes_sum(SIZE_MAX,1); return 0; }
        else { nmap_bytes_drop(1); return 0; }
        nmap_bytes_add(1); return 0;
    }
    nmap_bytes_add(42); nmap_bytes_drop(42);
    if(nmap_live_bytes || nmap_allocation_debt!=42 || nmap_peak_bytes!=42) abort();
    return 0;
}
''', refusals=('live', 'debt', 'underflow', 'sum'))

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
''', trace_scans=True)


if __name__ == '__main__':
    unittest.main()
