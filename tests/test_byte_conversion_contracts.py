"""I preserve my original byte conversion fixture and actual production behavior."""
import hashlib
import unittest

from tests import test_numeric_builtin_selection as numeric

ROOT = numeric.ROOT


class ByteConversionContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def test_original_complete_byte_conversion_fixture(self):
        path = ROOT/'tests/test_bytes_conversion.nano'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            self.exercise(path.read_text(), source_path=path)
        finally:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_empty_unicode_detached_storage_and_typed_calls(self):
        self.exercise('''fn encode(value:string)->array<u8>{return (bytes_from_string value)}
shadow encode{assert (== (array_length (encode "Hi")) 2)}
fn decode(values:array<u8>)->string{return (string_from_bytes values)}
shadow decode{let bytes:array<u8> =(bytes_from_string "Hi") assert (== (decode bytes) "Hi")}
fn main()->int{
 let empty:array<u8> =(encode "")
 assert (== (array_length empty) 0) assert (== (decode empty) "")
 let original:string="Aé茶"
 let bytes:array<u8> =(encode original)
 assert (== (array_length bytes) 6)
 assert (== (at bytes 0) 65) assert (== (at bytes 1) 195)
 assert (== (at bytes 2) 169) assert (== (at bytes 3) 232)
 assert (== (at bytes 4) 140) assert (== (at bytes 5) 182)
 let text:string=(decode bytes)
 let changed:u8=66 (array_set bytes 0 changed)
 assert (== original "Aé茶") assert (== text "Aé茶")
 let other:array<u8> =(encode text) assert (== (at other 0) 65)
 assert (== (decode bytes) "Bé茶")
 return 0
}
shadow main{assert (== (main) 0)}
''', native_optimizations=('-O0', '-O2'))

    def test_once_only_and_bound_function_identity(self):
        self.exercise('''let mut trace:int=0
fn text_value()->string{set trace (+ (* trace 10) 1) return "Hi"}
shadow text_value{let old:int=trace set trace 0 assert (== (text_value) "Hi") assert (== trace 1) set trace old}
fn byte_values()->array<u8>{set trace (+ (* trace 10) 2) return (bytes_from_string "Hi")}
shadow byte_values{let old:int=trace set trace 0 assert (== (array_length (byte_values)) 2) assert (== trace 2) set trace old}
fn keep_text(value:string)->string{return (+ value "!")}
shadow keep_text{assert (== (keep_text "Hi") "Hi!")}
fn main()->int{
 set trace 0
 let bytes:array<u8> =(bytes_from_string (text_value)) assert (== trace 1)
 let text:string=(string_from_bytes (byte_values)) assert (== trace 12)
 assert (== text "Hi") assert (== (at bytes 1) 105)
 let bytes_from_string:fn(string)->string=keep_text
 assert (== (bytes_from_string "Hi") "Hi!")
 let string_from_bytes:fn(string)->string=keep_text
 assert (== (string_from_bytes "Hi") "Hi!")
 return 0
}
shadow main{let old:int=trace assert (== (main) 0) set trace old}
''', native_optimizations=('-O0', '-O2'))

    def test_existing_nul_and_non_utf8_byte_boundaries(self):
        self.exercise('''fn main()->int{
 let letter:u8=65 let zero:u8=0 let tail:u8=66
 let nul_bytes:array<u8> = [letter,zero,tail]
 let visible:string=(string_from_bytes nul_bytes)
 assert (== visible "A") assert (== (str_length visible) 1)
 assert (== (array_length nul_bytes) 3)
 let encoded:array<u8> = (bytes_from_string visible)
 assert (== (array_length encoded) 1) assert (== (at encoded 0) 65)
 let invalid:u8=255 let first:u8=192 let second:u8=128
 let raw:array<u8> = [invalid]
 let raw_text:string=(string_from_bytes raw)
 assert (== (str_length raw_text) 1) assert (== (char_at raw_text 0) 255)
 let roundtrip:array<u8> = (bytes_from_string raw_text)
 assert (== (array_length roundtrip) 1) assert (== (at roundtrip 0) 255)
 let overlong:array<u8> = [first,second]
 let overlong_text:string=(string_from_bytes overlong)
 assert (== (str_length overlong_text) 2)
 assert (== (char_at overlong_text 0) 192) assert (== (char_at overlong_text 1) 128)
 return 0
}
shadow main{assert (== (main) 0)}
''', native_optimizations=('-O0', '-O2'))
