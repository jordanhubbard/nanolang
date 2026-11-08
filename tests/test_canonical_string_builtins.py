"""I retain exact string results and native trim boundaries."""
import os
from pathlib import Path
import struct
import shlex
import subprocess
import sys
import tempfile
import unittest
from tests.native_process import native_link_flags

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANOLANG_SELFHOST_COMPILER', ROOT / 'bin/nanoc_stage2')).resolve()


class CanonicalStringBuiltins(unittest.TestCase):
    def command(self, args, expected=0):
        result = subprocess.run([str(x) for x in args], cwd=ROOT, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, expected, (result.stdout + result.stderr)[-5000:])
        return result

    def both(self, source, directory):
        module = directory / 'main.nvm'
        self.command([COMPILER, source, '--emit-nvm', '-o', module])
        self.execute_module(module, directory)

    def execute_module(self, module, directory):
        c_source, native = (directory / name for name in ('main.c', 'native'))
        self.command([ROOT / 'bin/nano_vm', '--verify-only', module])
        vm = self.command([ROOT / 'bin/nano_vm', module])
        self.command([ROOT / 'bin/nvm2c', module, '-o', c_source])
        self.command([*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc')), '-std=c11', '-Wall', '-Wextra', '-Werror', c_source,
                      ROOT / 'bin/nano_aot_runtime.o', '-lm', *native_link_flags(),
                      *(['-Wl,--export-dynamic', '-ldl'] if sys.platform.startswith('linux') else []),
                      '-o', native])
        self.assertEqual(self.command([native]).stdout, vm.stdout)

    def test_existing_string_edges(self):
        with tempfile.TemporaryDirectory(prefix='canonical-string-edges-') as tmp:
            self.both(ROOT / 'tests/nl_functions_string_edges.nano', Path(tmp))

    def test_nested_string_and_boolean_results(self):
        with tempfile.TemporaryDirectory(prefix='canonical-string-results-') as tmp:
            directory = Path(tmp)
            source = directory / 'main.nano'
            source.write_text('''fn trim(value: string) -> string { return (str_trim value) }
shadow trim { assert (== (trim " a ") "a") }
fn starts(value: string) -> bool { return (str_starts_with value "a") }
shadow starts { assert (starts "abc") }
fn ends(value: string) -> bool { return (str_ends_with value "c") }
shadow ends { assert (ends "abc") }
fn join(value: string) -> string { return (str_concat (str_trim value) "bc") }
shadow join { assert (== (join " a ") "abc") }
fn main() -> int {
 let value: string = (str_trim (str_concat " a" "bc "))
 assert (== value "abc")
 assert (== (trim " a ") "a")
 assert (starts (join " a "))
 assert (ends value)
 assert (== (str_trim "\\v") "\\v")
 return 0
}
shadow main { assert (== (main) 0) }
''')
            self.both(source, directory)

    def test_wrong_trim_operands_preserve_output(self):
        with tempfile.TemporaryDirectory(prefix='canonical-trim-refusal-') as tmp:
            directory = Path(tmp)
            source, output = directory / 'main.nano', directory / 'prior.nvm'
            for expression in ('(str_trim 1)', '(str_trim "a" "b")'):
                with self.subTest(expression=expression):
                    source.write_text('fn main() -> int { ' + expression + ' return 0 }\nshadow main { assert true }\n')
                    output.write_bytes(b'prior')
                    self.command([COMPILER, source, '--emit-nvm', '-o', output], 1)
                    self.assertEqual(output.read_bytes(), b'prior')

    def test_direct_cast_returns_and_single_interpolation_execute_once(self):
        with tempfile.TemporaryDirectory(prefix='canonical-string-tail-') as tmp:
            directory = Path(tmp)
            source = directory / 'main.nano'
            source.write_text('''let mut calls: int = 0
fn next() -> int { set calls (+ calls 1) return 7 }
shadow next { set calls 0 assert (== (next) 7) assert (== calls 1) }
fn integer(value: int) -> string { return (to_string value) }
shadow integer { assert (str_equals (integer 7) "7") }
fn text(value: string) -> string { return (cast_string value) }
shadow text { assert (str_equals (text "kept") "kept") }
fn flag(value: bool) -> string { return (bool_to_string value) }
shadow flag { assert (str_equals (flag true) "true") assert (str_equals (flag false) "false") }
fn fraction(value: float) -> string { return (to_string value) }
shadow fraction { assert (str_equals (fraction 3.5) "3.5") }
fn interpolation(value: int) -> string { return f"{value}" }
shadow interpolation { assert (str_equals (interpolation -7) "-7") }
fn once() -> string { return (to_string (next)) }
shadow once { set calls 0 assert (str_equals (once) "7") assert (== calls 1) }
fn main() -> int {
 assert (str_equals (integer 7) "7")
 assert (str_equals (text "kept") "kept")
 assert (str_equals (flag true) "true")
 assert (str_equals (flag false) "false")
 assert (str_equals (fraction 3.5) "3.5")
 assert (str_equals (interpolation -7) "-7")
 set calls 0
 assert (str_equals (once) "7")
 assert (== calls 1)
 return 0
}
shadow main { assert (== (main) 0) }
''')
            self.both(source, directory)

    def test_declared_cast_names_keep_their_function_results(self):
        cases = [('bool_to_string', 'bool', 'true')]
        for name, kind, argument in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix='canonical-cast-override-') as tmp:
                directory = Path(tmp)
                source = directory / 'main.nano'
                source.write_text(f'''fn {name}(value: {kind}) -> string {{ return "declared" }}
shadow {name} {{ assert (str_equals ({name} {argument}) "declared") }}
fn relay(value: {kind}) -> string {{ return ({name} value) }}
shadow relay {{ assert (str_equals (relay {argument}) "declared") }}
fn main() -> int {{ assert (str_equals (relay {argument}) "declared") return 0 }}
shadow main {{ assert (== (main) 0) }}
''')
                self.both(source, directory)

    def test_wrong_cast_arities_preserve_output(self):
        with tempfile.TemporaryDirectory(prefix='canonical-cast-refusal-') as tmp:
            directory = Path(tmp)
            source, output = directory / 'main.nano', directory / 'prior.nvm'
            for name in ('to_string', 'cast_string', 'bool_to_string'):
                for arguments in ('', 'true false'):
                    with self.subTest(name=name, arguments=arguments):
                        source.write_text(f'fn main() -> int {{ ({name} {arguments}) return 0 }}\n'
                                          'shadow main { assert true }\n')
                        output.write_bytes(b'prior')
                        self.command([COMPILER, source, '--emit-nvm', '-o', output], 1)
                        self.assertEqual(output.read_bytes(), b'prior')

    def test_original_complete_fstring_fixture(self):
        with tempfile.TemporaryDirectory(prefix='canonical-fstring-original-') as tmp:
            self.both(ROOT / 'tests/unit/test_fstring_comprehensive.nano', Path(tmp))

    @staticmethod
    def float_text_cases():
        values = [(0.0, '0.0', '0'), (-0.0, '-0.0', '-0'),
                  (2.0, '2.0', '2'), (-2.0, '-2.0', '-2'),
                  (3.25, '3.25', '3.25'), (1e20, '1e+20', '1e+20')]
        cases = [(struct.unpack('<Q', struct.pack('<d', value))[0], source, raw)
                 for value, source, raw in values]
        cases += [(0x7ff0000000000000, 'inf', 'inf'), (0xfff0000000000000, '-inf', '-inf'),
                  (0x7ff8000000000001, 'nan', 'nan'), (0xfff8000000000001, '-nan', '-nan')]
        return cases

    def test_source_float_text_and_once_only_arguments(self):
        with tempfile.TemporaryDirectory(prefix='canonical-source-float-text-') as tmp:
            directory = Path(tmp)
            source = directory / 'main.nano'
            program = '''let mut calls: int = 0
fn operand() -> float { set calls (+ calls 1) return 2.0 }
shadow operand { set calls 0 assert (== (operand) 2.0) assert (== calls 1) }
fn render(value: float) -> string { return (to_string value) }
shadow render { assert (str_equals (render 2.0) "2.0") }
fn cast(value: float) -> string { return (cast_string value) }
shadow cast { assert (str_equals (cast 2.0) "2.0") }
fn interpolation(value: float) -> string { return f"{value}" }
shadow interpolation { assert (str_equals (interpolation 2.0) "2.0") }
fn once() -> string { return (to_string (operand)) }
shadow once { set calls 0 assert (str_equals (once) "2.0") assert (== calls 1) }
fn main() -> int {
'''
            for bits, expected, _ in self.float_text_cases():
                signed = bits if bits < 1 << 63 else bits - (1 << 64)
                for name in ('render', 'cast', 'interpolation'):
                    program += f' assert (str_equals ({name} (float_from_bits {signed})) "{expected}")\n'
            program += ''' set calls 0
 assert (str_equals (once) "2.0")
 assert (== calls 1)
 return 0
}
shadow main { assert (== (main) 0) }
'''
            source.write_text(program)
            self.both(source, directory)

    def test_raw_cast_string_retains_its_reference_format(self):
        with tempfile.TemporaryDirectory(prefix='canonical-raw-float-text-') as tmp:
            directory = Path(tmp)
            assembly, module = directory / 'main.nasm', directory / 'main.nvm'
            strings, body = '', ''
            for index, (bits, _, expected) in enumerate(self.float_text_cases()):
                strings += f'.string expected{index} "{expected}"\n'
                body += (f'PUSH_F64 bits:{bits:016x}\nCAST_STRING\n'
                         f'PUSH_STR expected{index}\nEQ\nASSERT\n')
            assembly.write_text(strings + '.entry main\n.function main 0 0 0 int 1\n' +
                                body + 'PUSH_I64 0\nRET\n.end\n')
            self.command([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
            self.execute_module(module, directory)


if __name__ == '__main__':
    unittest.main()
