"""I retain my whole original arithmetic fixture before changing its lowering."""
import hashlib
import tempfile
import unittest

from tests import test_numeric_builtin_selection as numeric
from tests import test_aggregate_binary64_policy as policy

ROOT = numeric.ROOT


class ArrayArithmeticContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    def exercise(self, source, expected='', **kwargs):
        return numeric.NumericBuiltinSelection.exercise(self, source, expected,
            native_optimizations=('-O0', '-O2'),
            native_flags=('-ffp-contract=off', '-fno-fast-math'), **kwargs)

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def test_original_complete_array_operators(self):
        path = ROOT/'tests/test_array_operators.nano'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            self.exercise(path.read_text(), expected='PASSED\n', source_path=path,
                          expected_opcodes=('ARRAY_ADD', 'ARRAY_SUB', 'ARRAY_MUL', 'ARRAY_DIV'))
        finally:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_original_fixed_bit_oracle(self):
        self.exercise(policy.arithmetic_source(),
                      expected_opcodes=('ARRAY_ADD', 'ARRAY_SUB', 'ARRAY_MUL', 'ARRAY_DIV'))

    def test_original_order_alias_growth_and_empty_neighbors(self):
        self.exercise(policy.ORDER, expected_opcodes=('ARRAY_SUB', 'ARRAY_DIV'))

    def test_integer_wrap_total_division_and_shorter_lengths(self):
        self.exercise('''fn main()->int{
 let high:array<int> = [9223372036854775807]
 let low:array<int> = [-9223372036854775808]
 let added:array<int> = (+ high 1)
 let subtracted:array<int> = (- low 1)
 let multiplied:array<int> = (* high 2)
 let divided:array<int> = (/ low -1)
 let zero:array<int> = (/ low 0)
 assert (== (at added 0) -9223372036854775808)
 assert (== (at subtracted 0) 9223372036854775807)
 assert (== (at multiplied 0) -2)
 assert (== (at divided 0) -9223372036854775808)
 assert (== (at zero 0) 0)
 assert (== (at high 0) 9223372036854775807)
 assert (== (at low 0) -9223372036854775808)
 assert (!= high added)
 let shorter:array<int> = (+ [1,2,3] [10])
 assert (== (array_length shorter) 1)
 assert (== (at shorter 0) 11)
 let left:array<int> = (- 10 [2,3])
 assert (== (at left 0) 8)
 assert (== (at left 1) 7)
 return 0
}
shadow main{assert (== (main) 0)}
''', expected_opcodes=('ARRAY_ADD', 'ARRAY_SUB', 'ARRAY_MUL', 'ARRAY_DIV'))

    def test_typed_empty_results_keep_independent_storage(self):
        self.exercise('''fn main()->int{
 let integers:array<int> = []
 let int_result:array<int> = (+ integers integers)
 assert (!= integers int_result)
 let grown_ints:array<int> = (array_push int_result 7)
 assert (== (at grown_ints 0) 7)
 assert (== (array_length integers) 0)
 let floats:array<float> = []
 let float_result:array<float> = (/ floats floats)
 assert (!= floats float_result)
 let grown_floats:array<float> = (array_push float_result 1.5)
 assert (== (float_to_bits (at grown_floats 0)) 4609434218613702656)
 assert (== (array_length floats) 0)
 let words:array<string> = []
 let word_result:array<string> = (+ words words)
 assert (!= words word_result)
 let grown_words:array<string> = (array_push word_result "kept")
 assert (== (at grown_words 0) "kept")
 assert (== (array_length words) 0)
 return 0
}
shadow main{assert (== (main) 0)}
''', expected_opcodes=('ARRAY_ADD', 'ARRAY_DIV'))

    def test_string_broadcast_shorter_lengths_and_repeated_lifetimes(self):
        self.exercise('''let mut trace:int=0
fn prefix()->string{set trace (+ (* trace 10) 1) return "("}
shadow prefix{let old:int=trace set trace 0 assert (== (prefix) "(") assert (== trace 1) set trace old}
fn grown(values:array<string>)->array<string>{
 set trace (+ (* trace 10) 2)
 return (array_push values "extra")
}
shadow grown{let old:int=trace set trace 0 let values:array<string> = ["a"] let result:array<string> = (grown values) assert (== (array_length result) 2) assert (== trace 2) set trace old}
fn together(values:array<string>)->array<string>{return (+ values "!")}
shadow together{let result:array<string> = (together ["a"]) assert (== (at result 0) "a!")}
fn main()->int{
 set trace 0
 let values:array<string> = ["é","λ"]
 let result:array<string> = (+ (prefix) (grown values))
 assert (== trace 12)
 assert (== (array_length result) 3)
 assert (== (at result 0) "(é")
 assert (== (at result 2) "(extra")
 assert (== (at values 0) "é")
 assert (!= values result)
 let shortest:array<string> = (+ result ["suffix"])
 assert (== (array_length shortest) 1)
 assert (== (at shortest 0) "(ésuffix")
 for index in (range 0 2000){
  let copy:array<string> = (together values)
  assert (== (at copy 0) "é!")
  assert (== (at copy 2) "extra!")
 }
 assert (== (at result 0) "(é")
 return 0
}
shadow main{let old:int=trace assert (== (main) 0) set trace old}
''', expected_opcodes=('ARRAY_ADD',))

    def test_known_mismatched_source_operands_preserve_output(self):
        for expression in ['(+ [1] 1.0)', '(+ [1.0] 1)',
                           '(+ [1] [1.0])', '(+ ["a"] 1)',
                           '(- ["a"] "b")']:
            with self.subTest(expression=expression):
                self.exercise('fn main()->int{let result='+expression+' return 0}', refused=True)

    def test_excluded_raw_profiles_preserve_native_output(self):
        cases = {
            'scalars': 'PUSH_I64 1\nPUSH_I64 2\nARRAY_ADD\n',
            'mixed': 'PUSH_I64 1\nARR_LITERAL 1 1\nPUSH_F64 2.0\nARRAY_ADD\n',
            'bool': 'PUSH_BOOL 1\nARR_LITERAL 4 1\nDUP\nARRAY_ADD\n',
            'byte': 'PUSH_U8 7\nARR_LITERAL 2 1\nDUP\nARRAY_ADD\n',
            'nested': 'PUSH_I64 1\nARR_LITERAL 1 1\nARR_LITERAL 7 1\nDUP\nARRAY_ADD\n',
            'string-sub': 'PUSH_STR text\nARR_LITERAL 5 1\nDUP\nARRAY_SUB\n',
            'unknown': 'PUSH_VOID\nPUSH_I64 1\nARR_LITERAL 1 1\nARRAY_ADD\n',
        }
        output = ROOT/'.test_output/numeric-builtin-selection'
        output.mkdir(parents=True, exist_ok=True)
        for label, body in cases.items():
            with self.subTest(case=label):
                work = Path(tempfile.mkdtemp(
                    prefix='raw-array-arithmetic-'+label+'-', dir=output))
                assembly, module, prior = [work/name for name in ['input.nasm', 'module.nvm', 'prior.c']]
                assembly.write_text('.entry main\n.string text "kept"\n.function main 0 0 0 int 1\n'+
                                    body+'POP\nPUSH_I64 0\nRET\n.end\n')
                self.command(work, 'assemble', [ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
                self.command(work, 'verify', [ROOT/'bin/nano_vm', '--verify-only', module])
                if label == 'mixed':
                    self.command(work, 'vm-mixed', [ROOT/'bin/nano_vm', module])
                sentinel = b'I retain my prior array arithmetic C output.\n'
                prior.write_bytes(sentinel)
                result = self.command(work, 'native-refusal',
                    [ROOT/'bin/nvm2c', module, '-o', prior], build=True, accepted=False)
                self.assertGreater(result.returncode, 0, result.stdout+result.stderr)
                self.assertEqual(prior.read_bytes(), sentinel)
