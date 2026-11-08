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

    def exercise(self, source, expected, *, source_path=None):
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
