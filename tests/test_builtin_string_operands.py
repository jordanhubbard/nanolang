"""I retain cached STRING operands and real function identity at source boundaries."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from tests.native_process import native_link_flags

ROOT = Path(__file__).resolve().parents[1]
PRODUCERS = tuple(os.environ.get('NANO_STRING_OPERAND_COMPILERS',
    'nanoc_c,nano_virt,nanoc_stage1,nanoc_stage2').split(','))


class BuiltinStringOperands(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PRODUCERS or any(name not in ('nanoc_c', 'nano_virt', 'nanoc_stage1', 'nanoc_stage2')
                                for name in PRODUCERS) or len(set(PRODUCERS)) != len(PRODUCERS):
            raise AssertionError('I require an explicit nonempty unique source compiler selection.')
        missing = [name for name in PRODUCERS if not (ROOT / 'bin' / name).is_file()]
        if missing:
            raise AssertionError('I require actual source compiler providers: ' + ', '.join(missing))
        print('I exercise source compiler providers:', ', '.join(PRODUCERS), flush=True)

    def command(self, args):
        result = subprocess.run(list(map(str, args)), cwd=ROOT,
                                capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def run_program(self, source, *, helper=None):
        with tempfile.TemporaryDirectory(prefix='nano-string-operands-') as tmp:
            directory = Path(tmp)
            path = directory / 'main.nano'
            path.write_text(source)
            if helper is not None:
                (directory / 'helper.nano').write_text(helper)
            for producer in PRODUCERS:
                with self.subTest(producer=producer):
                    native = directory / (producer + '-native')
                    if producer == 'nanoc_c':
                        self.command([ROOT / 'bin' / producer, path, '-o', native])
                    else:
                        module = directory / (producer + '.nvm')
                        c_source = directory / (producer + '.c')
                        self.command([ROOT / 'bin' / producer, path, '--emit-nvm', '-o', module])
                        self.command([ROOT / 'bin/nano_vm', '--verify-only', module])
                        self.command([ROOT / 'bin/nano_vm', module])
                        self.command([ROOT / 'bin/nvm2c', module, '-o', c_source])
                        self.command([*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc')),
                            '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                            c_source, ROOT / 'bin/nano_aot_runtime.o', '-lm', *native_link_flags(),
                            *(['-Wl,--export-dynamic', '-ldl'] if sys.platform.startswith('linux') else []),
                            '-o', native])
                    self.command([native])

    def test_valid_empty_whitespace_utf8_and_nested_strings(self):
        self.run_program('''fn cleaned(value: string) -> string { return (str_trim value) }
shadow cleaned {
 assert (str_equals (cleaned "") "")
 assert (str_equals (cleaned " \\t\\r\\n ") "")
 assert (str_equals (cleaned " é ") "é")
}
fn main() -> int {
 let text: string = (str_concat (cleaned " é ") "!")
 assert (str_equals text "é!")
 assert (str_contains text "é")
 assert (str_starts_with text "é")
 assert (str_ends_with text "!")
 assert (== (str_length (str_trim " ab ")) 2)
 assert (str_equals (str_substring "abc" 1 1) "b")
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_lexical_function_binding_keeps_its_signature(self):
        self.run_program('''fn increment(value: int) -> int { return (+ value 1) }
shadow increment { assert (== (increment 7) 8) }
fn main() -> int {
 let str_trim: fn(int)->int = increment
 assert (== (str_trim 7) 8)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_real_declaration_keeps_its_parameter_and_result(self):
        self.run_program('''let mut calls: int = 0
fn next() -> int { set calls (+ calls 1) return 7 }
shadow next { set calls 0 assert (== (next) 7) assert (== calls 1) }
fn str_trim(value: int) -> bool { return (== value 7) }
shadow str_trim { assert (str_trim 7) assert (not (str_trim 8)) }
fn main() -> int {
 set calls 0
 assert (str_trim (next))
 assert (== calls 1)
 let chosen: fn(int)->bool = str_trim
 assert (chosen 7)
 assert (not (str_trim 8))
 return 0
}
shadow main { assert (== (main) 0) }
''')
        self.run_program('''fn bool_to_string(value: bool) -> string { return "declared" }
shadow bool_to_string { assert (str_equals (bool_to_string true) "declared") }
fn relay(value: bool) -> string { return (bool_to_string value) }
shadow relay { assert (str_equals (relay false) "declared") }
fn main() -> int {
 assert (str_equals (relay true) "declared")
 assert (str_equals (relay false) "declared")
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_qualified_module_function_keeps_its_signature(self):
        self.run_program('''module "helper.nano" as helper
fn main() -> int {
 assert (str_equals (helper.cleaned " é ") "é")
 return 0
}
shadow main { assert (== (main) 0) }
''', helper='''pub fn cleaned(value: string) -> string { return (str_trim value) }
shadow cleaned { assert (str_equals (cleaned " x ") "x") }
''')

    def refuse(self, expressions, *, diagnostic=None):
        with tempfile.TemporaryDirectory(prefix='nano-string-operand-refusals-') as tmp:
            directory = Path(tmp)
            source = directory / 'bad.nano'
            for expression in expressions:
                source.write_text('fn main()->int{' + expression + ' return 0}\nshadow main{assert true}\n')
                for producer in PRODUCERS:
                    with self.subTest(expression=expression, producer=producer):
                        output = directory / (producer + '-prior')
                        output.write_bytes(b'I retain prior output.\n')
                        command = [ROOT / 'bin' / producer, source, '-o', output]
                        if producer != 'nanoc_c':
                            command.append('--emit-nvm')
                        result = subprocess.run(list(map(str, command)), cwd=ROOT,
                                                capture_output=True, text=True, timeout=180)
                        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertEqual(output.read_bytes(), b'I retain prior output.\n')
                        if diagnostic is not None:
                            self.assertIn(diagnostic, result.stdout + result.stderr)

    def test_known_wrong_string_operands_preserve_prior_output(self):
        self.refuse(['(str_trim 1)', '(str_trim true)', '(str_length 1)',
                     '(str_concat 1 "x")', '(str_concat "x" false)',
                     '(str_starts_with "x" 1)', '(str_ends_with true "x")',
                     '(str_contains "x" false)', '(str_equals 1 "x")',
                     '(str_substring 7 0 1)'], diagnostic='string')

    def test_wrong_arities_preserve_prior_output(self):
        self.refuse(['(str_trim)', '(str_trim "a" "b")', '(str_concat "a")',
                     '(str_starts_with "a")', '(str_substring "abc" 1)'])
