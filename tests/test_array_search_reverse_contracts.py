"""I preserve typed reverse/search behavior and the whole original fixture."""
import hashlib
import unittest

from tests import test_numeric_builtin_selection as numeric

ROOT = numeric.ROOT


class ArraySearchReverseContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def test_original_complete_search_and_reverse(self):
        path = ROOT/'tests/test_array_search_reverse.nano'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        self.exercise(path.read_text(), source_path=path)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_float_bits_first_match_and_independent_storage(self):
        self.exercise('''fn main()->int{
 let negative:float=(float_from_bits -9223372036854775808)
 let payload:float=(float_from_bits 9221120237041090626)
 let source:array<float> = [negative,payload,2.0]
 let reversed:array<float> = (array_reverse source)
 assert (!= source reversed)
 assert (== (float_to_bits (at reversed 1)) 9221120237041090626)
 assert (== (float_to_bits (at reversed 2)) -9223372036854775808)
 assert (== (array_index_of source 0.0) 0)
 assert (== (array_index_of source payload) -1)
 assert (not (array_contains source payload))
 (array_set reversed 2 7.0)
 assert (== (float_to_bits (at source 0)) -9223372036854775808)
 assert (== (array_index_of ["same","later","same"] "same") 0)
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_byte_elements_and_empty_storage(self):
        self.exercise('''fn main()->int{
 let bytes:array<u8> = [255,0]
 let reversed:array<u8> = (array_reverse bytes)
 assert (!= bytes reversed)
 assert (== (cast_int (at reversed 0)) 0)
 assert (== (cast_int (at reversed 1)) 255)
 assert (== (array_index_of bytes (at bytes 0)) 0)
 assert (array_contains bytes (at bytes 1))
 let empty:array<u8> = []
 let copied:array<u8> = (array_reverse empty)
 assert (== (array_length copied) 0)
 assert (!= empty copied)
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_lexical_bindings_keep_their_signatures(self):
        self.exercise('''fn kept(value:int)->int{return (+ value 1)}
shadow kept{assert (== (kept 2) 3)}
fn main()->int{
 let array_reverse:fn(int)->int=kept
 let array_contains:fn(int)->int=kept
 let array_index_of:fn(int)->int=kept
 assert (== (array_reverse 7) 8)
 assert (== (array_contains 8) 9)
 assert (== (array_index_of 9) 10)
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_wrong_operands_preserve_prior_output(self):
        for expression in ['(array_reverse)', '(array_reverse 1)',
                           '(array_reverse [1] [2])', '(array_index_of [1])',
                           '(array_index_of [1] true)', '(array_contains ["a"] 1)',
                           '(array_contains [1.0] 1)', '(array_contains true false)']:
            with self.subTest(expression=expression):
                self.exercise('fn main()->int{let result='+expression+' return 0}', refused=True)
