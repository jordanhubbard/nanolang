"""I retain whole string stdlib behavior and actual production-main operations."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import sys
import tempfile
import unittest

from tests import test_numeric_builtin_selection as numeric
from tests.native_process import native_link_flags
from tests.test_math_host_contracts import refuse_host_imports

ROOT = numeric.ROOT


class StringStdlibContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def exercise_raw(self, text, *, invalid_utf8=False, utf8_invariant='nhost_join',
                     work_prefix='string-stdlib-raw-'):
        output = ROOT/'.test_output/numeric-builtin-selection'
        output.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=work_prefix, dir=output))
        inputs = [ROOT/'bin/nanoisa', ROOT/'bin/nano_vm', ROOT/'bin/nvm2c', ROOT/'bin/nano_aot_runtime.o']
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
        assembly, module, generated = work/'main.nasm', work/'module.nvm', work/'native.c'
        assembly.write_text(text)
        try:
            self.command(work, 'assemble', [ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
            self.command(work, 'verify', [ROOT/'bin/nano_vm', '--verify-only', module])
            vm = self.command(work, 'vm', [ROOT/'bin/nano_vm', module], accepted=not invalid_utf8)
            if invalid_utf8:
                self.assertGreater(vm.returncode, 0, vm.stdout+vm.stderr)
                self.assertRegex(vm.stdout+vm.stderr, r'NUL-free UTF-8|foreign array')
            self.command(work, 'aot', [ROOT/'bin/nvm2c', module, '-o', generated], build=True)
            for optimization in ('-O0', '-O2'):
                native = work/('native'+optimization[1:])
                self.command(work, 'native-build'+optimization, [
                    *shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc')),
                    '-std=c11', optimization, '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                    generated, ROOT/'bin/nano_aot_runtime.o', '-lm', *native_link_flags(),
                    *(['-Wl,--export-dynamic', '-ldl'] if sys.platform.startswith('linux') else []),
                    '-o', native], build=True)
                result = self.command(work, 'native'+optimization, [native], accepted=not invalid_utf8)
                if invalid_utf8:
                    self.assertEqual(result.returncode, -signal.SIGABRT, result.stdout+result.stderr)
                    self.assertIn(utf8_invariant, result.stderr)
        finally:
            after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
            (work/'inputs.json').write_text(json.dumps({'before': before, 'after': after}, indent=2)+'\n')
            self.assertEqual(before, after)

    def test_binary_byte_case_opcodes(self):
        payload = bytes(range(1, 256))
        def constant(name, value):
            return '.string '+name+' "'+''.join('\\x'+format(byte, '02x') for byte in value)+'"\n'
        text = (constant('input', payload)+constant('lower', payload.lower())+constant('upper', payload.upper())+
                '.entry main\n.function main 0 0 0 int 1\n'
                'PUSH_STR input\nSTR_TO_LOWER\nPUSH_STR lower\nSTR_EQ\nASSERT\n'
                'PUSH_STR input\nSTR_TO_UPPER\nPUSH_STR upper\nSTR_EQ\nASSERT\n'
                'PUSH_I64 0\nRET\n.end\n')
        self.exercise_raw(text)

    def test_join_keeps_scalar_separator_bytes(self):
        self.exercise_raw('.string a "a"\n.string b "b"\n.string separator "\\xff"\n'
            '.import "" "vm_str_join" string array string\n.entry main\n'
            '.function main 0 1 0 int 1\nPUSH_STR a\nPUSH_STR b\nARR_LITERAL 5 2\n'
            'PUSH_STR separator\nCALL_EXTERN 0\nSTORE_LOCAL 0\n'
            'LOAD_LOCAL 0\nSTR_LEN\nPUSH_I64 3\nI64_EQ\nASSERT\n'
            'LOAD_LOCAL 0\nPUSH_I64 1\nSTR_CHAR_AT\nPUSH_I64 255\nI64_EQ\nASSERT\n'
            'PUSH_I64 0\nRET\n.end\n')

    def test_native_join_keeps_utf8_element_refusals(self):
        for value in (b'\xc3', b'\x80', b'\xc0\x80', b'\xed\xa0\x80', b'\xf4\x90\x80\x80', b'\xe2\x82', b'\xff'):
            with self.subTest(value=value.hex()):
                encoded = ''.join('\\x'+format(byte, '02x') for byte in value)
                self.exercise_raw('.string bad "'+encoded+'"\n.string separator ""\n'
                    '.import "" "vm_str_join" string array string\n.entry main\n'
                    '.function main 0 0 0 int 1\nPUSH_STR bad\nARR_LITERAL 5 1\n'
                    'PUSH_STR separator\nCALL_EXTERN 0\nPOP\nPUSH_I64 0\nRET\n.end\n', invalid_utf8=True)

    def test_wrong_native_signatures_and_array_storage_preserve_output(self):
        cases = []
        for name in ('vm_str_trim_left', 'vm_str_trim_right'):
            cases += [('', name, 'string', 'int', 'PUSH_I64 7'),
                      ('', name, 'int', 'string', 'PUSH_I64 7\nCAST_STRING'),
                      ('', name, 'string', 'string string', 'PUSH_I64 7\nCAST_STRING\nDUP'),
                      ('other.so', name, 'string', 'string', 'PUSH_I64 7\nCAST_STRING')]
        cases += [('', 'vm_str_join', 'string', 'string string', 'PUSH_I64 7\nCAST_STRING\nDUP'),
                  ('', 'vm_str_join', 'int', 'array string', 'ARR_LITERAL 5 0\nPUSH_I64 7\nCAST_STRING'),
                  ('other.so', 'vm_str_join', 'string', 'array string', 'ARR_LITERAL 5 0\nPUSH_I64 7\nCAST_STRING')]
        for tag, value in ((1, 'PUSH_I64 7'), (2, 'PUSH_U8 7'), (3, 'PUSH_F64 1.5'), (4, 'PUSH_BOOL 1')):
            cases.append(('', 'vm_str_join', 'string', 'array string',
                value+'\nARR_LITERAL '+str(tag)+' 1\nPUSH_I64 7\nCAST_STRING'))
        refuse_host_imports(self, cases, 'string-stdlib-import-refusal-')

    def test_original_complete_string_stdlib_fixture(self):
        path = ROOT/'tests/test_stdlib_strings.nano'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            self.exercise(path.read_text(),
                'OK: test_str_starts_with passed\nOK: test_str_ends_with passed\n'
                'OK: test_str_index_of passed\nhello\nhello  \n  hello\n'
                'hello world\nHELLO WORLD\nbaz bar baz\n3\nx+y\n', source_path=path)
        finally:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_trim_sides_keep_other_bytes(self):
        self.exercise(r'''fn main()->int{
 assert (== (str_trim_left " \t\r\n café \r") "café \r")
 assert (== (str_trim_right "\t café \r\n") "\t café")
 assert (== (str_trim_left "") "") assert (== (str_trim_right " \t\r\n") "")
 assert (== (str_trim_left " café ") " café ")
 assert (== (str_trim_right " café ") " café ")
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_case_and_replacement_keep_original_bytes(self):
        self.exercise(r'''fn main()->int{
 assert (== (str_to_lower "AZ-É-茶") "az-É-茶")
 assert (== (str_to_upper "az-é-茶") "AZ-é-茶")
 assert (== (str_to_lower "") "") assert (== (str_to_upper "") "")
 assert (== (str_replace "aaaaa" "aa" "b") "bba")
 assert (== (str_replace "café café" "café" "茶") "茶 茶")
 assert (== (str_replace "abc" "" "ignored") "abc")
 assert (== (str_replace "abc" "missing" "replacement") "abc")
 assert (== (str_replace "aaa" "a" "") "")
 return 0
}
shadow main{assert (== (main) 0)}
''', expected_opcodes=('STR_TO_LOWER', 'STR_TO_UPPER', 'STR_REPLACE'))

    def test_split_join_empty_elements_and_byte_fragments(self):
        self.exercise(r'''fn main()->int{
 let parts:array<string> = (str_split "--a----b--" "--")
 assert (== (array_length parts) 5)
 assert (== (at parts 0) "") assert (== (at parts 2) "") assert (== (at parts 4) "")
 assert (== (str_join parts "--") "--a----b--")
 let empty:array<string> = (str_split "" ",")
 assert (== (array_length empty) 1) assert (== (at empty 0) "")
 let none:array<string> = (str_split "" "")
 assert (== (array_length none) 0) assert (== (str_join none ",") "")
 let bytes:array<string> = (str_split "café茶" "")
 assert (== (array_length bytes) 8)
 assert (== (char_at (at bytes 3) 0) 195) assert (== (char_at (at bytes 4) 0) 169)
 assert (== (str_join ["", ""] "--") "--")
 return 0
}
shadow main{assert (== (main) 0)}
''', expected_opcodes=('STR_SPLIT',))


    def test_once_only_lexical_identity_and_detached_join_result(self):
        self.exercise(r'''let mut trace:int=0
let mut saved:array<string> = ["initial"]
fn note(value:string,digit:int)->string{set trace (+ (* trace 10) digit) return value}
shadow note{let old:int=trace set trace 0 assert (== (note "x" 7) "x") assert (== trace 7) set trace old}
fn build()->array<string>{return (str_split (str_concat "a," "b,") ",")}
shadow build{let values:array<string> = (build) assert (== (array_length values) 3) assert (== (at values 1) "b")}
fn keep(value:int)->int{return (+ value 100)}
shadow keep{assert (== (keep 7) 107)}
fn main()->int{
 set trace 0
 assert (== (str_replace (note "aba" 1) (note "a" 2) (note "x" 3)) "xbx")
 assert (== trace 123)
 set trace 0
 let values:array<string> = (str_split (note "a,b" 1) (note "," 2))
 assert (== trace 12) assert (== (array_length values) 2)
 set saved (build)
 let joined:string=(str_join saved "--")
 (array_set saved 0 "changed")
 assert (== joined "a--b--")
 let str_trim_left:fn(int)->int=keep
 let str_join:fn(int)->int=keep
 let str_split:fn(int)->int=keep
 assert (== (str_trim_left 7) 107) assert (== (str_join 8) 108) assert (== (str_split 9) 109)
 for index in (range 0 5000) {let trash:string=(str_replace "temporary temporary" "temporary" "discarded")}
 assert (== joined "a--b--") assert (== (at saved 1) "b")
 return 0
}
shadow main{let old_trace:int=trace let old_saved:array<string> = saved assert (== (main) 0) set trace old_trace set saved old_saved}
''')

    def test_wrong_operands_and_arities_preserve_prior_output(self):
        for name in ('str_trim_left', 'str_trim_right', 'str_to_lower', 'str_to_upper',
                     'str_replace', 'str_split', 'str_join'):
            count = {'str_replace': 3, 'str_split': 2, 'str_join': 2}.get(name, 1)
            valid = ['"x"']*count
            if name == 'str_join':
                valid[0] = '["x"]'
            cases = [[], valid+['"extra"']]
            for position in range(count):
                invalid = valid.copy()
                invalid[position] = 'true'
                cases.append(invalid)
            for arguments in cases:
                with self.subTest(name=name, arguments=arguments):
                    self.exercise('fn main()->int{('+name+' '+' '.join(arguments)+r''') return 0}
shadow main{assert (== (main) 0)}
''', refused=True)
        self.exercise(r'''fn main()->int{let values:array<int> = [] let text:string=(str_join values ",") return 0}
shadow main{assert (== (main) 0)}
''', refused=True)

    def test_foreign_join_keeps_its_utf8_array_boundary(self):
        self.exercise(r'''fn main()->int{
 let fragments:array<string> = (str_split "é" "")
 return (str_length (str_join fragments ""))
}
shadow main{assert (== (main) 0)}
''', refused=True, refusal_pattern=r'(?i)(NUL-free UTF-8|foreign array)')
