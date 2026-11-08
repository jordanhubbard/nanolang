"""I preserve the existing byte spelling as an exact u8 type alias."""
import hashlib
import unittest

from tests import test_numeric_builtin_selection as numeric

ROOT = numeric.ROOT


class ByteAliasContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def test_original_complete_u8_fixture(self):
        path = ROOT/'tests/test_u8_basic.nano'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            self.exercise(path.read_text(), source_path=path)
        finally:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_alias_arguments_returns_arrays_and_computed_conversion(self):
        self.exercise('''fn keep(value:byte)->byte{return value}
shadow keep{assert (== (keep 255) 255)}
fn main()->int{
 let direct:byte=255 let canonical:u8=direct
 let callback:fn(byte)->byte=keep
 let result:byte=(callback canonical)
 assert (== result 255)
 let values:array<byte> = [0,7,255]
 assert (== (at values 2) 255)
 let computed:byte=(+ 255 2)
 assert (== computed 1)
 return 0
}
shadow main{assert (== (main) 0)}
''', expected_opcodes=('PUSH_U8', 'ARR_LITERAL'))

    def test_scalar_integer_arithmetic_and_computed_narrowing(self):
        self.exercise('''let mut trace:int=0
fn note(value:u8,digit:int)->u8{set trace (+ (* trace 10) digit) return value}
shadow note{let old:int=trace set trace 0 assert (== (note 255 7) 255) assert (== trace 7) set trace old}
fn main()->int{
 let left:u8=250 let right:u8=7
 let sum:int=(+ left right) assert (== sum 257)
 assert (== (+ left 2) 252) assert (== (+ 2 left) 252)
 assert (== (- left right) 243) assert (== (- 255 right) 248)
 assert (== (* left right) 1750) assert (== (/ left right) 35)
 assert (== (% left right) 5)
 let narrowed:u8=(+ left right) assert (== narrowed 1)
 let negative:u8=(- right left) assert (== negative 13)
 set trace 0
 let once:int=(+ (note 250 1) (note 7 2))
 assert (== once 257) assert (== trace 12)
 return 0
}
shadow main{let old:int=trace assert (== (main) 0) set trace old}
''', expected_opcodes=('I64_ADD', 'CAST_U8'))

    def test_arithmetic_does_not_widen_other_operand_types(self):
        for other in ('true', '"wrong"', '1.5', '[7]'):
            with self.subTest(other=other):
                self.exercise('fn main()->int{let value:u8=7 let result:int=(+ value '+other+''') return result}
shadow main{assert (== (main) 0)}
''', refused=True)
        self.exercise('''fn main()->int{let value:u8=256 return 0}
shadow main{assert (== (main) 0)}
''', refused=True, refusal_pattern=r'(?i)(byte.*literal|literal.*(range|255)|type)')

    def test_wrong_types_refuse_before_replacing_output(self):
        for value in ('true', '"wrong"'):
            with self.subTest(value=value):
                self.exercise('fn main()->int{let value:byte='+value+''' return 0}
shadow main{assert (== (main) 0)}
''', refused=True)
