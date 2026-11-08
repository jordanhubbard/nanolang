"""I execute actual translated binary-string programs at both native optimization levels."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest

from scripts.diagnose_bstring_contracts import probes
from tests import test_bstring_vm_contracts as binary
from tests.native_process import native_link_flags

ROOT = binary.ROOT


class BstringNativeContracts(unittest.TestCase):
    command = binary.BstringVmContracts.command

    def exercise(self, source, expected='', *, source_path=None):
        output = ROOT/'.test_output/bstring-native-contracts'
        output.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=self._testMethodName+'-', dir=output))
        path = work/'main.nano' if source_path is None else source_path
        if source_path is None:
            path.write_text(source)
        providers = [path, ROOT/'bin/nano_virt', ROOT/'bin/nano_vm', ROOT/'bin/nvm2c',
                     ROOT/'bin/nano_aot_runtime.o']
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in providers}
        module, generated = work/'program.nvm', work/'program.c'
        products = []
        try:
            self.command(work, 'emit', [providers[1], path, '--emit-nvm', '-o', module], build=True)
            self.command(work, 'verify', [providers[2], '--verify-only', module])
            vm = self.command(work, 'vm', [providers[2], module])
            self.assertEqual(vm.stdout, expected)
            self.command(work, 'translate', [providers[3], module, '-o', generated], build=True)
            for optimization in ['-O0', '-O2']:
                product = work/('native'+optimization)
                self.command(work, 'build'+optimization, [*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc')),
                    '-std=c11', optimization, '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                    generated, providers[4], '-lm', *native_link_flags(),
                    *(['-Wl,--export-dynamic', '-ldl'] if sys.platform.startswith('linux') else []),
                    '-o', product], build=True)
                pin = hashlib.sha256(product.read_bytes()).hexdigest()
                result = self.command(work, 'run'+optimization, [product])
                self.assertEqual(result.stdout, expected)
                after = hashlib.sha256(product.read_bytes()).hexdigest()
                self.assertEqual(pin, after)
                products.append({'path': str(product), 'before': pin, 'after': after})
        finally:
            after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in providers}
            (work/'inputs.json').write_text(json.dumps({'scope':
                'I execute C-created NanoISA and actual translated native products; source-emitter and container qualification remain separate.',
                'before': before, 'after': after, 'native_products': products,
                'module_sha256': hashlib.sha256(module.read_bytes()).hexdigest() if module.exists() else None,
                'generated_sha256': hashlib.sha256(generated.read_bytes()).hexdigest() if generated.exists() else None}, indent=2)+'\n')
            self.assertEqual(before, after)

    def test_all_retained_reference_observations(self):
        expected = {'binary-nul': '3\n65\n0\n66\n1\n',
                    'unicode-and-ownership': '6\n-1\n-1\n-1\n195\nAé茶!\nAé茶!kept\n',
                    'invalid-utf8': 'false\n-1\n-1\n',
                    'substring-bounds': '0\n0\nlo\n', 'byte-negative': '0\n', 'byte-beyond': '0\n'}
        for label, source in probes().items():
            with self.subTest(case=label):
                self.exercise(source, expected[label])

    def test_unchanged_original_fixture(self):
        path = ROOT/'tests/test_bstring.nano'
        self.exercise(path.read_text(), 'Testing bstring type...\n\nCreated bstring\nLength: 11\n'
            'First byte (H): 72\nContent: Hello World\nConcatenated: Hello World - UTF-8 test\n'
            'UTF-8 string: Hello 世界\nByte length: 12\nCharacter length: 8\n'
            'Substring[0:5]: Hello\nEquality test: PASS\n\nAll bstring tests passed!\n', source_path=path)

    def test_alias_identity_snapshot_and_extreme_bounds(self):
        self.exercise('''fn identity(value:bstring)->bstring{return value}
shadow identity{
 let value:bstring=(bstr_new "held")
 assert (== (identity value) value)
 (bstr_free value)
}
fn main()->int{
 let first:bstring=(bstr_new "Aé茶")
 let alias:bstring=(identity first)
 let equal:bstring=(bstr_new "Aé茶")
 assert (== first alias)
 assert (not (== first equal))
 assert (bstr_equals first equal)
 assert (== (bstr_utf8_length alias) -1)
 assert (bstr_validate_utf8 first)
 assert (== (bstr_utf8_length alias) 3)
 assert (== (bstr_utf8_char_at alias 2) 33590)
 let snapshot:string=(bstr_to_cstr alias)
 let tail:bstring=(bstr_substring first 1 9223372036854775807)
 assert (== (bstr_length tail) 5)
 assert (== (bstr_byte_at tail 9223372036854775807) 0)
 (bstr_free alias)
 assert (== snapshot "Aé茶")
 (bstr_free tail) (bstr_free equal)
 return 0
}
shadow main{assert (== (main) 0)}
''', '')

    def test_typed_owner_array_aliases(self):
        binary.BstringVmContracts.test_typed_functions_arrays_aliases_and_useful_shadows(self)

    def test_owner_array_mutation_and_higher_order_operations(self):
        self.exercise('''fn identity(value:bstring)->bstring{return value}
shadow identity{let v:bstring=(bstr_new "x") assert (== (identity v) v) (bstr_free v)}
fn present(value:bstring)->bool{return (> (bstr_length value) 0)}
shadow present{let v:bstring=(bstr_new "x") assert (present v) (bstr_free v)}
fn count(total:int,value:bstring)->int{return (+ total (bstr_length value))}
shadow count{let v:bstring=(bstr_new "x") assert (== (count 2 v) 3) (bstr_free v)}
fn main()->int{
 let first:bstring=(bstr_new "left")
 let second:bstring=(bstr_new "right")
 let mut values:array<bstring> = []
 (array_push values first)
 let alias:array<bstring> = values
 (array_push values second)
 assert (== (array_length alias) 2)
 (array_set values 0 second)
 assert (== (at alias 0) second)
 let slice:array<bstring> = (array_slice values 0 1)
 assert (== (at slice 0) second)
 let mapped:array<bstring> = (map values identity)
 assert (== (at mapped 1) second)
 let filtered:array<bstring> = (filter mapped present)
 assert (== (array_length filtered) 2)
 assert (== (reduce filtered 0 count) 10)
 let filled:array<bstring> = (array_new 2 first)
 assert (== (at filled 1) first)
 assert (== (array_pop values) second)
 assert (== (array_length alias) 1)
 (bstr_free first) (bstr_free second)
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_generated_owner_collection_and_refusals(self):
        output = ROOT/'.test_output/bstring-native-contracts'
        output.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='generated-collection-', dir=output))
        source, module, generated = work/'main.nano', work/'program.nvm', work/'program.c'
        source.write_text('fn main()->int{let value:bstring=(bstr_new "seed") (bstr_free value) return 0}\n'
                          'shadow main{assert (== (main) 0)}\n')
        providers = [ROOT/'bin/nano_virt', ROOT/'bin/nano_vm', ROOT/'bin/nvm2c', ROOT/'bin/nano_aot_runtime.o']
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in providers}
        products = []
        try:
            self.command(work, 'emit', [providers[0], source, '--emit-nvm', '-o', module], build=True)
            self.command(work, 'verify', [providers[1], '--verify-only', module])
            self.command(work, 'translate', [providers[2], module, '-o', generated], build=True)
            wrapper = work/'collection.c'
            wrapper.write_text(r'''#define main generated_main
#include "program.c"
#undef main
#include <assert.h>
#include <sys/resource.h>
int main(int argc, char **argv) {
    struct rlimit core_limit = {0, 0}; assert(setrlimit(RLIMIT_CORE, &core_limit) == 0);
    assert(generated_main(0, NULL) == 0);
    nroot_frame frame = {0}; frame.prev = nroot_head; nroot_head = &frame;
    nmap_value held = nhost_vm_bstring_new("kept");
    nroot_add(&frame.live, 8, &held);
    if (argc == 2) {
        if (!strcmp(argv[1], "freed")) {
            nhost_vm_bstring_free(held);
            (void)nhost_vm_bstring_length(held);
        } else if (!strcmp(argv[1], "double-free")) {
            nhost_vm_bstring_free(held); nhost_vm_bstring_free(held);
        } else if (!strcmp(argv[1], "wrong-tag")) {
            (void)nhost_vm_bstring_length((nmap_value){5, 0, "text"});
        } else if (!strcmp(argv[1], "unowned")) {
            int unrelated = 0;
            (void)nhost_vm_bstring_length((nmap_value){6, 0, (char *)&unrelated});
        }
        return 91;
    }
    for (size_t i = 0; i < 1000; ++i) {
        (void)nhost_vm_bstring_new("temporary");
        if (!(i % 31)) nmap_collect();
        assert(nhost_vm_bstring_length(held) == 4);
    }
    nmap_collect(); assert(nmap_owned_live == 1);
    nmap_value alias = held; nroot_add(&frame.live, 8, &alias);
    const char *snapshot = nhost_vm_bstring_to_cstr(alias);
    nroot_add(&frame.live, 1, snapshot);
    size_t payload_bytes = nmap_live_bytes;
    nhost_vm_bstring_free(held);
    assert(nmap_live_bytes < payload_bytes);
    nmap_collect(); assert(nmap_owned_live == 1);
    assert(!strcmp(snapshot, "kept"));
    nroot_reset(&frame.live); nmap_collect();
    assert(nmap_owned_live == 0 && nmap_live_bytes == 0 && nstr_live_bytes == 0);
    nroot_head = frame.prev; nroot_destroy(&frame.live);
    nmap_release_owned(); nstr_release_owned();
    assert(nmap_live_bytes == 0 && nstr_live_bytes == 0);
    puts("I passed native binary-string collection and refusal controls.");
    return 0;
}
''')
            for optimization in ['-O0', '-O2']:
                product = work/('native'+optimization)
                self.command(work, 'build'+optimization, [*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc')),
                    '-std=c11', optimization, '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                    wrapper, providers[3], '-lm', *native_link_flags(),
                    *(['-Wl,--export-dynamic', '-ldl'] if sys.platform.startswith('linux') else []),
                    '-o', product], build=True)
                pin = hashlib.sha256(product.read_bytes()).hexdigest()
                run = self.command(work, 'collection'+optimization, [product])
                self.assertEqual(run.stdout, 'I passed native binary-string collection and refusal controls.\n')
                for mode in ['freed', 'double-free', 'wrong-tag', 'unowned']:
                    refused = self.command(work, mode+optimization, [product, mode], accepted=False)
                    self.assertNotEqual(refused.returncode, 0)
                    self.assertNotEqual(refused.returncode, 91)
                    self.assertIn('I stopped at a native invariant', refused.stderr)
                    self.assertNotIn('ERROR: AddressSanitizer', refused.stderr)
                    self.assertNotIn('runtime error:', refused.stderr)
                self.assertEqual(pin, hashlib.sha256(product.read_bytes()).hexdigest())
                products.append({'path': str(product), 'before': pin, 'after': pin})
        finally:
            after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in providers}
            (work/'inputs.json').write_text(json.dumps({'scope':
                'I exercise the actual generated private owner/collector boundary, not generic foreign or NativeSDK admission.',
                'before': before, 'after': after, 'native_products': products}, indent=2)+'\n')
            self.assertEqual(before, after)
