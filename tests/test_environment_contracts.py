"""I retain existing environment mutations and copied getter storage."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import test_numeric_builtin_selection as numeric
from tests import test_string_stdlib_contracts as strings
from tests.test_math_host_contracts import refuse_host_imports

ROOT = numeric.ROOT
PRODUCTION = '''extern fn nl_os_setenv(name:string,value:string,overwrite:int)->int
extern fn nl_os_unsetenv(name:string)->int
fn main()->int{
 unsafe {
  assert (== (nl_os_unsetenv "NANOLANG_ENV_CONTRACT") 0)
  assert (== (getenv "NANOLANG_ENV_CONTRACT") "")
  assert (== (nl_os_setenv "NANOLANG_ENV_CONTRACT" "first" 0) 0)
  let retained:string=(getenv "NANOLANG_ENV_CONTRACT")
  assert (== retained "first")
  assert (== (nl_os_setenv "NANOLANG_ENV_CONTRACT" "ignored" 0) 0)
  assert (== (getenv "NANOLANG_ENV_CONTRACT") "first")
  assert (== (nl_os_setenv "NANOLANG_ENV_CONTRACT" "changed" -1) 0)
  assert (== (getenv "NANOLANG_ENV_CONTRACT") "changed")
  assert (== retained "first")
  assert (== (nl_os_unsetenv "NANOLANG_ENV_CONTRACT") 0)
  assert (== retained "first")
  assert (== (getenv "NANOLANG_ENV_CONTRACT") "")
  assert (== (nl_os_setenv "" "bad" 1) -1)
  assert (== (nl_os_setenv "BAD=NAME" "bad" 1) -1)
  assert (== (nl_os_unsetenv "") -1)
  assert (== (nl_os_unsetenv "BAD=NAME") -1)
 }
 return 0
}
shadow main{assert (== (main) 0)}
'''


class EnvironmentContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise
    exercise_raw = strings.StringStdlibContracts.exercise_raw

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def test_original_complete_environment_fixtures(self):
        expected = (r'\A=== Testing std::env ===\n\nTesting getenv:\n  PATH length: [1-9][0-9]*\n\n'
            r'Testing setenv/unsetenv:\n  setenv returned: 0\n  Value: test_value\n'
            r'  unsetenv returned: 0\n\nTesting get_argc/get_argv:\n  argc: 1\n'
            r'  argv\[0\]: [^\n]+\n\nAll tests passed!\n\Z')
        for name in ['test_env.nano', 'test_setenv_shadow.nano']:
            with self.subTest(fixture=name):
                path = ROOT/'tests'/name
                before = hashlib.sha256(path.read_bytes()).hexdigest()
                try:
                    self.exercise(path.read_text(), source_path=path,
                                  expected_pattern=expected if name == 'test_env.nano' else None)
                finally:
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_overwrite_unset_errors_and_retained_getter(self):
        self.exercise(PRODUCTION, native_optimizations=('-O0', '-O2'))

    def test_c_seed_native_getter_retains_its_snapshot(self):
        output = ROOT/'.test_output/source-checker-component'
        output.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='language-environment-', dir=output))
        source, product = work/'main.nano', work/'native'
        source.write_text(PRODUCTION)
        compiler = ROOT/'bin/nanoc_c'
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, compiler]}
        product_before = None
        cflags = os.environ.get('NANO_CFLAGS', '') + ' -fsanitize=address,undefined -fno-sanitize-recover=all'
        try:
            with patch.dict(os.environ, {'NANO_CFLAGS': cflags}):
                self.command(work, 'source-build', [compiler, source, '--keep-c', '-o', product], build=True)
            self.assertIn(product.read_bytes()[:4], [b'\x7fELF', bytes.fromhex('cffaedfe'), bytes.fromhex('feedfacf')])
            product_before = hashlib.sha256(product.read_bytes()).hexdigest()
            self.command(work, 'native-run', [product])
        finally:
            after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, compiler]}
            product_after = hashlib.sha256(product.read_bytes()).hexdigest() if product.exists() else None
            (work/'inputs.json').write_text(json.dumps({'scope': 'C interpreter and native environment snapshot',
                'native_build_cflags': cflags,
                'before': before, 'after': after, 'product_before_run': product_before,
                'product_after_run': product_after}, indent=2)+'\n')
            self.assertEqual(before, after)
            if product_before is not None:
                self.assertEqual(product_before, product_after)

    def test_raw_two_string_setter_and_all_getter_identities(self):
        self.exercise_raw(''' .entry main
.string key "NANOLANG_ENV_RAW_CONTRACT"
.string first "first"
.string second "second"
.import "" "vm_setenv" int string string
.import "" "vm_getenv" string string
.import "" "nl_os_getenv" string string
.import "" "getenv" string string
.import "" "nl_os_unsetenv" int string
.function main 0 1 0 int 1
PUSH_STR key
PUSH_STR first
CALL_EXTERN 0
PUSH_I64 0
I64_EQ
ASSERT
PUSH_STR key
CALL_EXTERN 1
STORE_LOCAL 0
PUSH_STR key
PUSH_STR second
CALL_EXTERN 0
PUSH_I64 0
I64_EQ
ASSERT
PUSH_STR key
CALL_EXTERN 2
PUSH_STR second
STR_EQ
ASSERT
PUSH_STR key
CALL_EXTERN 3
PUSH_STR second
STR_EQ
ASSERT
PUSH_STR key
CALL_EXTERN 4
PUSH_I64 0
I64_EQ
ASSERT
LOAD_LOCAL 0
PUSH_STR first
STR_EQ
ASSERT
PUSH_STR key
CALL_EXTERN 1
STR_LEN
PUSH_I64 0
I64_EQ
ASSERT
PUSH_I64 0
RET
.end
''', work_prefix='environment-raw-')

    def test_argument_order_lexical_identity_and_overwrite_narrowing(self):
        self.exercise('''extern fn nl_os_setenv(name:string,value:string,overwrite:int)->int
extern fn nl_os_unsetenv(name:string)->int
let mut trace:int=0
fn key_arg()->string{set trace (+ (* trace 10) 1) return "NANOLANG_ENV_ORDER"}
shadow key_arg{let old:int=trace set trace 0 assert (== (key_arg) "NANOLANG_ENV_ORDER") assert (== trace 1) set trace old}
fn value_arg()->string{set trace (+ (* trace 10) 2) return "first"}
shadow value_arg{let old:int=trace set trace 0 assert (== (value_arg) "first") assert (== trace 2) set trace old}
fn overwrite_arg()->int{set trace (+ (* trace 10) 3) return 1}
shadow overwrite_arg{let old:int=trace set trace 0 assert (== (overwrite_arg) 1) assert (== trace 3) set trace old}
fn held(name:string,value:string,overwrite:int)->int{return (+ (str_length name) (+ (str_length value) overwrite))}
shadow held{assert (== (held "aa" "bbb" 7) 12)}
fn kept(name:string)->string{return (str_concat name "!")}
shadow kept{assert (== (kept "kept") "kept!")}
fn main()->int{
 unsafe {
  set trace 0
  assert (== (nl_os_setenv (key_arg) (value_arg) (overwrite_arg)) 0)
  assert (== trace 123)
  assert (== (getenv "NANOLANG_ENV_ORDER") "first")
  assert (== (nl_os_setenv "NANOLANG_ENV_ORDER" "ignored" 4294967296) 0)
  assert (== (getenv "NANOLANG_ENV_ORDER") "first")
  assert (== (nl_os_setenv "NANOLANG_ENV_ORDER" "changed" 4294967297) 0)
  assert (== (getenv "NANOLANG_ENV_ORDER") "changed")
  assert (== (nl_os_unsetenv "NANOLANG_ENV_ORDER") 0)
 }
 let nl_os_setenv:fn(string,string,int)->int=held
 assert (== (nl_os_setenv "aa" "bbb" 7) 12)
 let getenv:fn(string)->string=kept
 assert (== (getenv "lexical") "lexical!")
 return 0
}
shadow main{let old:int=trace assert (== (main) 0) set trace old}
''', native_optimizations=('-O0', '-O2'))

    def test_wrong_source_arguments_preserve_prior_output(self):
        declarations = ('extern fn nl_os_setenv(name:string,value:string,overwrite:int)->int\n'
                        'extern fn nl_os_unsetenv(name:string)->int\n')
        for expression in ['(nl_os_setenv "k" "v")', '(nl_os_setenv "k" "v" 1 1)',
                           '(nl_os_setenv "k" "v" true)', '(nl_os_setenv 7 "v" 1)',
                           '(nl_os_setenv "k" 7 1)', '(nl_os_unsetenv)', '(nl_os_unsetenv 7)']:
            with self.subTest(expression=expression):
                self.exercise(declarations+'fn main()->int{'+expression+' return 0}\n', refused=True)

    def test_exact_namespace_arity_and_parameter_refusals(self):
        cases = [
            ('foreign.so', 'nl_os_setenv', 'int', 'string string int', 'PUSH_STR 0\nDUP\nPUSH_I64 1'),
            ('', 'nl_os_setenv', 'bool', 'string string int', 'PUSH_STR 0\nDUP\nPUSH_I64 1'),
            ('', 'nl_os_setenv', 'int', 'string string string', 'PUSH_STR 0\nDUP\nDUP'),
            ('', 'nl_os_setenv', 'int', 'int string int', 'PUSH_I64 0\nPUSH_STR 0\nPUSH_I64 1'),
            ('', 'nl_os_setenv', 'int', 'string int int', 'PUSH_STR 0\nPUSH_I64 0\nPUSH_I64 1'),
            ('', 'nl_os_setenv', 'int', 'string string', 'PUSH_STR 0\nDUP'),
            ('', 'nl_os_unsetenv', 'int', 'int', 'PUSH_I64 0'),
            ('foreign.so', 'vm_setenv', 'int', 'string string', 'PUSH_STR 0\nDUP'),
            ('', 'vm_setenv', 'int', 'string int', 'PUSH_STR 0\nPUSH_I64 1'),
            ('foreign.so', 'getenv', 'string', 'string', 'PUSH_STR 0'),
            ('', 'getenv', 'string', 'int', 'PUSH_I64 0'),
            ('', 'nl_os_setenv_other', 'int', 'string string int', 'PUSH_STR 0\nDUP\nPUSH_I64 1'),
        ]
        refuse_host_imports(self, cases, 'environment-import-refusal-')
