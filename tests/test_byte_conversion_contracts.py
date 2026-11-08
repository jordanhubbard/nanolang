"""I preserve my original byte conversion fixture and actual production behavior."""
import hashlib
import unittest

from tests import test_numeric_builtin_selection as numeric
from tests import test_string_stdlib_contracts as string_stdlib
from tests.test_math_host_contracts import refuse_host_imports

ROOT = numeric.ROOT


class ByteConversionContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise
    exercise_raw = string_stdlib.StringStdlibContracts.exercise_raw

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

    def test_wrong_source_operands_and_arities_preserve_output(self):
        for expression in ['(bytes_from_string)', '(bytes_from_string true)',
                           '(bytes_from_string "a" "b")', '(string_from_bytes)',
                           '(string_from_bytes true)', '(string_from_bytes [1])',
                           '(string_from_bytes [true])', '(string_from_bytes [[1]])',
                           '(string_from_bytes [] [])']:
            with self.subTest(expression=expression):
                self.exercise('fn main()->int{'+expression+' return 0}\n'
                              'shadow main{assert (== (main) 0)}\n', refused=True)

    def test_exact_native_signatures_and_byte_storage_refusals(self):
        refuse_host_imports(self, [
            ('other.so', 'vm_mktemp', 'string', 'string', 'PUSH_STR 0'),
            ('', 'vm_mktemp', 'int', 'string', 'PUSH_STR 0'),
            ('', 'vm_mktemp', 'string', 'int', 'PUSH_I64 1'),
            ('', 'vm_mktemp', 'string', 'string string', 'PUSH_STR 0\nDUP'),
            ('other.so', 'vm_bytes_from_string', 'array', 'string', 'PUSH_STR 0'),
            ('', 'vm_bytes_from_string', 'string', 'string', 'PUSH_STR 0'),
            ('', 'vm_bytes_from_string', 'array', 'int', 'PUSH_I64 7'),
            ('', 'vm_bytes_from_string', 'array', 'string string', 'PUSH_STR 0\nDUP'),
            ('other.so', 'vm_string_from_bytes', 'string', 'array', 'PUSH_U8 65\nARR_LITERAL 2 1'),
            ('', 'vm_string_from_bytes', 'array', 'array', 'PUSH_U8 65\nARR_LITERAL 2 1'),
            ('', 'vm_string_from_bytes', 'string', 'int', 'PUSH_I64 7'),
            ('', 'vm_string_from_bytes', 'string', 'array array', 'PUSH_U8 65\nARR_LITERAL 2 1\nDUP'),
            ('', 'vm_string_from_bytes', 'string', 'array', 'PUSH_I64 65\nARR_LITERAL 1 1'),
            ('', 'vm_string_from_bytes', 'string', 'array', 'PUSH_BOOL 1\nARR_LITERAL 4 1'),
            ('', 'vm_string_from_bytes', 'string', 'array', 'PUSH_F64 1.5\nARR_LITERAL 3 1'),
            ('', 'vm_string_from_bytes', 'string', 'array', 'PUSH_STR 0\nARR_LITERAL 5 1'),
            ('', 'vm_string_from_bytes', 'string', 'array', 'PUSH_U8 65\nARR_LITERAL 2 1\nARR_LITERAL 6 1'),
            ('', 'vm_string_from_bytes', 'string', 'array', 'AGG_PACK 0 0 0 0\nARR_LITERAL 8 1'),
            ('', 'vm_string_from_bytes', 'string', 'array', 'FUNCREF 0\nARR_LITERAL 11 1'),
        ], 'byte-conversion-import-refusal-')

    def test_raw_every_nonzero_byte_and_detached_conversion(self):
        encoded = ''.join('\\x'+format(byte, '02x') for byte in range(1, 256))
        self.exercise_raw('.string input "'+encoded+'"\n'
            '.import "" "vm_bytes_from_string" array string\n'
            '.import "" "vm_string_from_bytes" string array\n.entry main\n'
            '.function main 0 2 0 int 1\nPUSH_STR input\nCALL_EXTERN 0\nSTORE_LOCAL 0\n'
            'LOAD_LOCAL 0\nARR_LEN\nPUSH_I64 255\nI64_EQ\nASSERT\n'
            'LOAD_LOCAL 0\nCALL_EXTERN 1\nSTORE_LOCAL 1\n'
            'LOAD_LOCAL 0\nPUSH_I64 0\nPUSH_U8 42\nARR_SET\nPOP\n'
            'LOAD_LOCAL 1\nPUSH_STR input\nSTR_EQ\nASSERT\n'
            'ARR_NEW 2\nCALL_EXTERN 1\nSTR_LEN\nPUSH_I64 0\nI64_EQ\nASSERT\n'
            'PUSH_I64 0\nRET\n.end\n', work_prefix='byte-conversion-raw-')

    def test_retained_conversion_owners_survive_allocation_churn(self):
        self.exercise('''let mut saved_bytes:array<u8> = []
let mut saved_text:string=""
fn publish()->void{
 let input:string=(str_concat "retained " "café")
 set saved_bytes (bytes_from_string input)
 set saved_text (string_from_bytes saved_bytes)
}
shadow publish{let old_bytes:array<u8> = saved_bytes let old_text:string=saved_text (publish) assert (== saved_text "retained café") set saved_bytes old_bytes set saved_text old_text}
fn main()->int{
 (publish)
 let changed:u8=88 (array_set saved_bytes 0 changed)
 for index in (range 0 5000) {
  let trash:array<u8> = (bytes_from_string "temporary byte conversion storage discarded at the next collection")
  let text:string=(string_from_bytes trash)
  assert (> (str_length text) 0)
 }
 assert (== saved_text "retained café")
 assert (== (at saved_bytes 0) 88)
 assert (== (array_length saved_bytes) 14)
 return 0
}
shadow main{let old_bytes:array<u8> = saved_bytes let old_text:string=saved_text assert (== (main) 0) set saved_bytes old_bytes set saved_text old_text}
''', native_optimizations=('-O0', '-O2'))

    def test_original_file_operations_execute_in_production_main(self):
        self.exercise('''from "modules/std/fs.nano" import write, delete
fn main()->int{
 let first:string=(mktemp "nano_byte_contract_")
 let second:string=(mktemp "")
 assert (!= first "") assert (!= second "") assert (!= first second)
 assert (file_exists first) assert (file_exists second)
 assert (str_starts_with first (+ (tmp_dir) "/nano_byte_contract_"))
 assert (str_starts_with second (+ (tmp_dir) "/nanolang_"))
 assert (== (array_length (file_read_bytes first)) 0)
 assert (== (write first "Aé茶") 0)
 let bytes:array<u8> = (file_read_bytes first)
 assert (== (array_length bytes) 6) assert (== (at bytes 1) 195)
 assert (== (string_from_bytes bytes) "Aé茶")
 assert (== (delete first) 0) assert (== (delete second) 0)
 assert (== (array_length (file_read_bytes first)) 0)
 return 0
}
shadow main{assert (== (main) 0)}
''', native_optimizations=('-O0', '-O2'))
