"""I retain directed scalar argument conversions without widening array storage."""
import unittest

from tests import test_numeric_builtin_selection as numeric


class U8ArgumentContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def test_direct_indirect_and_callee_first_evaluation(self):
        self.exercise('''let mut trace:int=0
fn note(value:u8,digit:int)->u8{set trace (+ (* trace 10) digit) return value}
shadow note{let old:int=trace set trace 0 assert (== (note 255 7) 255) assert (== trace 7) set trace old}
fn take(value:int)->int{return (+ value 1000)}
shadow take{assert (== (take 7) 1007)}
fn choose()->fn(int)->int{set trace (+ (* trace 10) 1) return take}
shadow choose{let old:int=trace set trace 0 assert (== ((choose) 7) 1007) assert (== trace 1) set trace old}
fn main()->int{
 set trace 0 assert (== (take (note 255 2)) 1255) assert (== trace 2)
 let call:fn(int)->int=take
 set trace 0 assert (== (call (note 255 3)) 1255) assert (== trace 3)
 set trace 0 assert (== ((choose) (note 255 2)) 1255) assert (== trace 12)
 return 0
}
shadow main{let old:int=trace assert (== (main) 0) set trace old}
''', expected_opcodes=('CAST_INT', 'CALL_INDIRECT'))

    def test_existing_computed_byte_destinations(self):
        self.exercise('''fn keep(value:u8)->u8{return value}
shadow keep{assert (== (keep 255) 255)}
fn main()->int{
 assert (== (keep (+ 255 2)) 1)
 let call:fn(u8)->u8=keep
 assert (== (call (- 0 1)) 255)
 assert (== (call (+ 255 2)) 1)
 return 0
}
shadow main{assert (== (main) 0)}
''', expected_opcodes=('CAST_U8', 'CALL_INDIRECT'))

    def test_typed_byte_element_caller(self):
        self.exercise('''fn push_u8(buffer:array<u8>,value:int)->array<u8>{return (array_push buffer value)}
shadow push_u8{let values:array<u8> = [1] let result:array<u8> = (push_u8 values 255) assert (== (at result 1) 255)}
fn copy(values:array<u8>)->array<u8>{
 let mut result:array<u8> = []
 for value in values {set result (push_u8 result value)}
 return result
}
shadow copy{let source:array<u8> = [7,255] let result:array<u8> = (copy source) assert (== (array_length result) 2) assert (== (at result 1) 255)}
fn main()->int{
 let source:array<u8> = [0,7,255]
 let result:array<u8> = (copy source)
 assert (== (array_length result) 3)
 assert (== (at result 0) 0) assert (== (at result 1) 7) assert (== (at result 2) 255)
 return 0
}
shadow main{assert (== (main) 0)}
''', expected_opcodes=('CAST_INT',))

    def test_wrong_scalars_and_array_aliases_preserve_output(self):
        for argument in ('true', '"wrong"', '[7]'):
            with self.subTest(argument=argument):
                self.exercise('''fn take(value:int)->int{return value}
shadow take{assert (== (take 7) 7)}
fn main()->int{return (take '''+argument+''')}
shadow main{assert (== (main) 0)}
''', refused=True)
        self.exercise('''fn take(values:array<int>)->int{return (array_length values)}
shadow take{assert (== (take [7]) 1)}
fn main()->int{let values:array<u8> = [7] return (take values)}
shadow main{assert (== (main) 0)}
''', refused=True)
