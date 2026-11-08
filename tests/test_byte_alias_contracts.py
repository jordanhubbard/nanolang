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

    def test_wrong_types_refuse_before_replacing_output(self):
        for value in ('true', '"wrong"'):
            with self.subTest(value=value):
                self.exercise('fn main()->int{let value:byte='+value+''' return 0}
shadow main{assert (== (main) 0)}
''', refused=True)
