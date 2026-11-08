"""I retain my whole original casting fixture and existing conversion semantics."""
import hashlib
import unittest

from tests import test_numeric_builtin_selection as numeric

ROOT = numeric.ROOT


class CastFloatContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def test_original_complete_casting_fixture(self):
        path = ROOT/'tests/test_casting.nano'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = ('=== Type Casting Tests ===\n\ncast_int(3.7) = 3\n'
            'cast_float(42) = 42.0\ncast_bool(0) = false\ncast_bool(42) = true\n'
            'cast_string(42) = 42\ncast_string(true) = true\n\n✅ All casting tests passed!\n')
        try:
            self.exercise(path.read_text(), expected, source_path=path,
                          expected_opcodes=('CAST_FLOAT',))
        finally:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_existing_scalar_bits_fallback_and_once_only_operand(self):
        self.exercise('''let mut trace:int=0
fn next()->int{set trace (+ trace 1) return 42}
shadow next{let old:int=trace set trace 0 assert (== (next) 42) assert (== trace 1) set trace old}
fn main()->int{
 set trace 0
 let converted:float=(cast_float (next))
 assert (== trace 1)
 assert (== (float_to_bits converted) 4631107791820423168)
 assert (== (float_to_bits (cast_float true)) 4607182418800017408)
 assert (== (float_to_bits (cast_float false)) 0)
 assert (== (float_to_bits (cast_float "2.5")) 4612811918334230528)
 let bytes:array<u8> = [7]
 assert (== (float_to_bits (cast_float (at bytes 0))) 4619567317775286272)
 let signed_zero:float=(float_from_bits -9223372036854775808)
 let quiet:float=(float_from_bits 9221120237041090626)
 let signaling:float=(float_from_bits 9218868437227405313)
 assert (== (float_to_bits (cast_float signed_zero)) -9223372036854775808)
 assert (== (float_to_bits (cast_float quiet)) 9221120237041090626)
 assert (== (float_to_bits (cast_float signaling)) 9218868437227405313)
 let values:array<int> = [9]
 assert (== (float_to_bits (cast_float values)) 0)
 assert (== (array_length values) 1)
 assert (== (at values 0) 9)
 return 0
}
shadow main{let old:int=trace assert (== (main) 0) set trace old}
''', expected_opcodes=('CAST_FLOAT',))

    def test_lexical_binding_keeps_its_signature(self):
        self.exercise('''fn word(value:int)->string{return "kept"}
shadow word{assert (== (word 7) "kept")}
fn main()->int{
 let cast_float:fn(int)->string=word
 assert (== (cast_float 7) "kept")
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_wrong_arity_preserves_prior_output(self):
        for expression in ['(cast_float)', '(cast_float 1 2)']:
            with self.subTest(expression=expression):
                self.exercise('fn main()->int{let result:float='+expression+' return 0}', refused=True)
