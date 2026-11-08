"""I retain the whole original runtime-template formatting fixture."""
import hashlib
import unittest

from tests import test_numeric_builtin_selection as numeric

ROOT = numeric.ROOT


class FormatSourceContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def test_original_complete_format_fixture(self):
        path = ROOT/'tests/test_format.nano'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = ('PASS: test_format_string\nPASS: test_format_int\n'
            'PASS: test_format_float\nPASS: test_format_multiple\n'
            'PASS: test_format_no_args\nResults: 5 passed, 0 failed\n')
        try:
            self.exercise(path.read_text(), expected, source_path=path)
        finally:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
