"""I preserve the existing scalar sorting contract and the whole original fixture."""
import hashlib
import unittest

from tests import test_numeric_builtin_selection as numeric

ROOT = numeric.ROOT


class ArraySortContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def test_original_complete_array_stdlib_fixture(self):
        path = ROOT/'tests/test_stdlib_arrays.nano'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            self.exercise(path.read_text(), 'OK: test_array_sort passed\n'
                'OK: test_array_reverse passed\nOK: test_array_contains passed\n'
                'OK: test_array_index_of passed\n', source_path=path)
        finally:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_five_scalar_kinds_typed_empty_and_detached_storage(self):
        self.exercise('''fn main()->int{
 let integers:array<int> = [9223372036854775807,-9223372036854775808,0,7,-1]
 let sorted:array<int> = (array_sort integers)
 assert (== (at sorted 0) -9223372036854775808)
 assert (== (at sorted 4) 9223372036854775807)
 (array_set integers 0 3)
 assert (== (at sorted 4) 9223372036854775807)
 let high:u8=255 let low:u8=0 let middle:u8=128
 let bytes:array<u8> = [high,low,middle]
 let byte_order:array<u8> = (array_sort bytes)
 assert (== (at byte_order 0) low) assert (== (at byte_order 2) high)
 let floats:array<float> = [2.5,-1.5,0.0]
 let float_order:array<float> = (array_sort floats)
 assert (== (at float_order 0) -1.5) assert (== (at floats 0) 2.5)
 let flags:array<bool> = (array_sort [true,false,true])
 assert (not (at flags 0)) assert (at flags 2)
 let words:array<string> = ["z","café","a"]
 let word_order:array<string> = (array_sort words)
 assert (== (at word_order 0) "a") assert (== (at word_order 2) "z")
 (array_set words 0 "changed") assert (== (at word_order 2) "z")
 let empty_int:array<int> = [] let empty_byte:array<u8> = []
 let empty_float:array<float> = [] let empty_bool:array<bool> = [] let empty_string:array<string> = []
 assert (== (array_length (array_sort empty_int)) 0)
 assert (== (array_length (array_sort empty_byte)) 0)
 assert (== (array_length (array_sort empty_float)) 0)
 assert (== (array_length (array_sort empty_bool)) 0)
 assert (== (array_length (array_sort empty_string)) 0)
 return 0
}
shadow main{assert (== (main) 0)}
''')
