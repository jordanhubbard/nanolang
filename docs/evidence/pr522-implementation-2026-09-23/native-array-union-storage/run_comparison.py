"""I compare retained array/union fixtures without weakening their assertions."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tests import test_canonical_aggregate_formatting as formatting

FIXTURES = Path(__file__).resolve().parent

class ArrayUnionComparison(formatting.AggregateFormatting):
    def test_direct_array(self):
        self.execute((FIXTURES / "array-choice.nano").read_text())

    def test_record_array(self):
        self.execute((FIXTURES / "array-envelope.nano").read_text())

if __name__ == "__main__":
    suite = unittest.TestSuite(ArrayUnionComparison(name) for name in
        ("test_direct_array", "test_record_array"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
