"""I exercise length-aware VM execution without claiming native-emitter parity."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.diagnose_bstring_contracts import probes
from tests import test_numeric_builtin_selection as numeric

ROOT = numeric.ROOT


class BstringVmContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    compiler = ROOT/'bin/nano_virt'
    artifact_directory = 'bstring-vm-contracts'
    scope = 'I test C-seed NanoISA emission and VM execution only.'

    def exercise(self, source, expected='', *, source_path=None, refused=False):
        output = ROOT/'.test_output'/self.artifact_directory
        output.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=self._testMethodName+'-', dir=output))
        path = work/'main.nano' if source_path is None else source_path
        if source_path is None:
            path.write_text(source)
        compiler, vm = self.compiler, ROOT/'bin/nano_vm'
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [path, compiler, vm]}
        module = work/'program.nvm'
        prior = b'I retain my previous binary-string program.\n'
        if refused:
            module.write_bytes(prior)
        product_before = None
        try:
            built = self.command(work, 'build', [compiler, path, '--emit-nvm', '-o', module],
                                 build=True, accepted=not refused)
            if refused:
                self.assertNotEqual(built.returncode, 0)
                self.assertRegex(built.stdout+built.stderr, r'(?i)(type|argument|operand|arity)')
                self.assertEqual(module.read_bytes(), prior)
                return
            product_before = hashlib.sha256(module.read_bytes()).hexdigest()
            self.command(work, 'verify', [vm, '--verify-only', module])
            result = self.command(work, 'run', [vm, module])
            self.assertEqual(result.stdout, expected)
        finally:
            after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [path, compiler, vm]}
            product_after = hashlib.sha256(module.read_bytes()).hexdigest() if module.exists() else None
            (work/'inputs.json').write_text(json.dumps({'scope': self.scope,
                'before': before, 'after': after, 'product_before_run': product_before,
                'product_after_run': product_after}, indent=2)+'\n')
            self.assertEqual(before, after)
            if product_before:
                self.assertEqual(product_before, product_after)

    def test_reference_boundary_observations(self):
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

    def test_typed_functions_arrays_aliases_and_useful_shadows(self):
        self.exercise('''fn size(value:bstring)->int{return (bstr_length value)}
shadow size {
 let value:bstring=(bstr_new "shadow")
 assert (== (size value) 6)
 (bstr_free value)
}
fn main()->int{
 let first:bstring=(bstr_new "Aé茶")
 let second:bstring=(bstr_new "other")
 let values:array<bstring> = [first,second]
 let alias:bstring=(at values 0)
 assert (== (size alias) 6)
 assert (bstr_equals first alias)
 assert (== first alias)
 let independent:bstring=(bstr_new "Aé茶")
 assert (bstr_equals first independent)
 assert (not (== first independent))
 (bstr_free independent)
 assert (bstr_validate_utf8 alias)
 assert (== (bstr_utf8_length first) 3)
 assert (== (bstr_utf8_char_at first 2) 33590)
 let snapshot:string=(bstr_to_cstr first)
 (bstr_free alias)
 assert (== snapshot "Aé茶")
 (bstr_free second)
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_extreme_substring_clamps_without_signed_addition(self):
        self.exercise('''fn main()->int{
 let value:bstring=(bstr_new "abc")
 let tail:bstring=(bstr_substring value 1 9223372036854775807)
 let beyond:bstring=(bstr_substring value 9223372036854775807 9223372036854775807)
 assert (== (bstr_length tail) 2)
 assert (== (bstr_to_cstr tail) "bc")
 assert (== (bstr_length beyond) 0)
 (bstr_free tail) (bstr_free beyond) (bstr_free value)
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_wrong_builtin_types_preserve_output(self):
        for body in ['let value:bstring=(bstr_new 42)', '(bstr_length "text")',
                     '(bstr_substring (bstr_new "text") "bad" 1)', '(bstr_free "text")']:
            with self.subTest(body=body):
                self.exercise('fn main()->int{'+body+' return 0}\nshadow main{assert true}\n', refused=True)
