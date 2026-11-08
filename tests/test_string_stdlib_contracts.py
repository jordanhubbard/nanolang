"""I retain whole string stdlib behavior and actual production-main operations."""
import hashlib
import unittest

from tests import test_numeric_builtin_selection as numeric

ROOT = numeric.ROOT


class StringStdlibContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def test_original_complete_string_stdlib_fixture(self):
        path = ROOT/'tests/test_stdlib_strings.nano'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            self.exercise(path.read_text(),
                'OK: test_str_starts_with passed\nOK: test_str_ends_with passed\n'
                'OK: test_str_index_of passed\nhello\nhello  \n  hello\n'
                'hello world\nHELLO WORLD\nbaz bar baz\n3\nx+y\n', source_path=path)
        finally:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_trim_sides_keep_other_bytes(self):
        self.exercise(r'''fn main()->int{
 assert (== (str_trim_left " \t\r\n café \r") "café \r")
 assert (== (str_trim_right "\t café \r\n") "\t café")
 assert (== (str_trim_left "") "") assert (== (str_trim_right " \t\r\n") "")
 assert (== (str_trim_left " café ") " café ")
 assert (== (str_trim_right " café ") " café ")
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_case_and_replacement_keep_original_bytes(self):
        self.exercise(r'''fn main()->int{
 assert (== (str_to_lower "AZ-É-茶") "az-É-茶")
 assert (== (str_to_upper "az-é-茶") "AZ-é-茶")
 assert (== (str_to_lower "") "") assert (== (str_to_upper "") "")
 assert (== (str_replace "aaaaa" "aa" "b") "bba")
 assert (== (str_replace "café café" "café" "茶") "茶 茶")
 assert (== (str_replace "abc" "" "ignored") "abc")
 assert (== (str_replace "abc" "missing" "replacement") "abc")
 assert (== (str_replace "aaa" "a" "") "")
 return 0
}
shadow main{assert (== (main) 0)}
''', expected_opcodes=('STR_TO_LOWER', 'STR_TO_UPPER', 'STR_REPLACE'))

    def test_split_join_empty_elements_and_byte_fragments(self):
        self.exercise(r'''fn main()->int{
 let parts:array<string> = (str_split "--a----b--" "--")
 assert (== (array_length parts) 5)
 assert (== (at parts 0) "") assert (== (at parts 2) "") assert (== (at parts 4) "")
 assert (== (str_join parts "--") "--a----b--")
 let empty:array<string> = (str_split "" ",")
 assert (== (array_length empty) 1) assert (== (at empty 0) "")
 let none:array<string> = (str_split "" "")
 assert (== (array_length none) 0) assert (== (str_join none ",") "")
 let bytes:array<string> = (str_split "café茶" "")
 assert (== (array_length bytes) 8)
 assert (== (char_at (at bytes 3) 0) 195) assert (== (char_at (at bytes 4) 0) 169)
 assert (== (str_join ["", ""] "--") "--")
 return 0
}
shadow main{assert (== (main) 0)}
''', expected_opcodes=('STR_SPLIT',))

    def test_foreign_join_keeps_its_utf8_array_boundary(self):
        self.exercise(r'''fn main()->int{
 let fragments:array<string> = (str_split "é" "")
 return (str_length (str_join fragments ""))
}
shadow main{assert (== (main) 0)}
''', refused=True, refusal_pattern=r'(?i)(NUL-free UTF-8|foreign array)')
