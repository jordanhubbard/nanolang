"""I qualify exact depth-three primitive array fields before source publication."""
import unittest
from tests import test_record_array_nested as retained

class RecordArrayDepthThree(unittest.TestCase):
    def test_private_primitive_consumers(self):
        for leaf in ('TAG_INT', 'TAG_U8', 'TAG_FLOAT', 'TAG_BOOL', 'TAG_STRING'):
            with self.subTest(leaf=leaf):
                retained.RecordArrayNested().qualify(True, leaf, 3)

    def test_public_primitive_consumers(self):
        for leaf in ('TAG_INT', 'TAG_U8', 'TAG_FLOAT', 'TAG_BOOL', 'TAG_STRING'):
            with self.subTest(leaf=leaf):
                retained.RecordArrayNested().qualify(False, leaf, 3)
