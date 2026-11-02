"""I qualify exact U8/FLOAT/BOOL nested fields using actual public/private routes."""
import unittest
from tests import test_record_array_nested as support


class RecordArrayNestedPrimitive(unittest.TestCase):
    qualify = support.RecordArrayNested.qualify

    def test_public_primitive_consumers(self):
        for leaf in ('TAG_U8', 'TAG_FLOAT', 'TAG_BOOL'):
            with self.subTest(leaf=leaf):
                self.qualify(False, leaf)

    def test_private_primitive_consumers(self):
        for leaf in ('TAG_U8', 'TAG_FLOAT', 'TAG_BOOL'):
            with self.subTest(leaf=leaf):
                self.qualify(True, leaf)
