"""I retain the original character builtin fixture through actual source routes."""
import hashlib
import unittest

from tests import test_numeric_builtin_selection as numeric
from tests.test_math_host_contracts import refuse_host_imports

ROOT = numeric.ROOT


class CharacterBuiltinContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def test_original_complete_standard_library_fixture(self):
        path = ROOT/'tests/unit/test_stdlib_comprehensive.nano'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            self.exercise(path.read_text(), 'All standard library tests passed!\n', source_path=path)
        finally:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_exhaustive_byte_oracles_and_full_integer_boundaries(self):
        members = {
            'is_digit': b'0123456789',
            'is_alpha': b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
            'is_alnum': b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
            'is_whitespace': b' \t\n\r',
            'is_upper': b'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            'is_lower': b'abcdefghijklmnopqrstuvwxyz',
        }
        lines = ['fn main()->int{']
        for name, accepted in members.items():
            expected = ','.join('true' if code in accepted else 'false' for code in range(256))
            lines.append('let expected_'+name+':array<bool> = ['+expected+']')
        for name, values in {
                'digit_value': [code-48 if code in b'0123456789' else -1 for code in range(256)],
                'char_to_lower': list(bytes(range(256)).lower()),
                'char_to_upper': list(bytes(range(256)).upper())}.items():
            lines.append('let expected_'+name+':array<int> = ['+','.join(map(str, values))+']')
        lines.append('for code in (range 0 256){')
        for name in (*members, 'digit_value', 'char_to_lower', 'char_to_upper'):
            lines.append('assert (== ('+name+' code) (at expected_'+name+' code))')
        lines.append('}')
        boundaries = [-1, 256, 288, 65536, -(2**63), 2**63-1, 2**32+65, 2**32+48]
        for code in boundaries:
            spelling = '(- -9223372036854775807 1)' if code == -(2**63) else str(code)
            narrowed = (code+2**31) % 2**32-2**31
            for name, accepted in members.items():
                actual_code = code if name == 'is_whitespace' else narrowed
                expected = 'true' if 0 <= actual_code <= 255 and actual_code in accepted else 'false'
                lines.append('assert (== ('+name+' '+spelling+') '+expected+')')
            for name in ('digit_value', 'char_to_lower', 'char_to_upper'):
                # These three existing VM contracts preserve the complete INT input.
                lines.append('assert (== ('+name+' '+spelling+') '+('-1' if name == 'digit_value' else spelling)+')')
        lines += ['return 0}', 'shadow main{assert (== (main) 0)}']
        self.exercise('\n'.join(lines)+'\n', native_optimizations=('-O0', '-O2'),
                      expected_opcodes=('CALL_EXTERN',))

    def test_once_only_operands_and_bound_function_identity(self):
        self.exercise('''let mut trace:int=0
fn note(value:int,digit:int)->int{set trace (+ (* trace 10) digit) return value}
shadow note{let old:int=trace set trace 0 assert (== (note 48 7) 48) assert (== trace 7) set trace old}
fn kept(value:string)->string{return value}
shadow kept{assert (== (kept "kept") "kept")}
fn main()->int{
 set trace 0
 assert (is_digit (note 48 1))
 assert (== (char_to_lower (note 65 2)) 97)
 assert (== trace 12)
 let is_digit:fn(string)->string=kept
 let char_to_lower:fn(string)->string=kept
 assert (== (is_digit "first") "first")
 assert (== (char_to_lower "second") "second")
 assert (== trace 12)
 return 0
}
shadow main{let old:int=trace assert (== (main) 0) set trace old}
''')

    def test_wrong_operands_and_arities_preserve_prior_output(self):
        names = ('is_digit', 'is_alpha', 'is_alnum', 'is_whitespace', 'is_upper',
                 'is_lower', 'digit_value', 'char_to_lower', 'char_to_upper')
        for name in names:
            for arguments in ('', '48 49', 'true', '"wrong"'):
                with self.subTest(name=name, arguments=arguments):
                    self.exercise('fn main()->int{('+name+' '+arguments+''') return 0}
shadow main{assert (== (main) 0)}
''', refused=True)

    def test_wrong_native_host_signatures_preserve_prior_output(self):
        cases = []
        for name in ('vm_char_to_lower', 'vm_char_to_upper'):
            cases += [('', name, 'int', 'bool', 'PUSH_BOOL 1'),
                      ('', name, 'bool', 'int', 'PUSH_I64 65'),
                      ('', name, 'int', 'int int', 'PUSH_I64 65\nPUSH_I64 66'),
                      ('libcharacters.so', name, 'int', 'int', 'PUSH_I64 65')]
        refuse_host_imports(self, cases, 'character-import-refusal-')
