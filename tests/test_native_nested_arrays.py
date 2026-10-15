"""I trace shared nested array children through slicing and collection."""
from pathlib import Path
import tempfile
import unittest
from tests import test_native_byte_arrays as byte_arrays
from tests.test_native_closures import CHURN


class NativeNestedArrays(unittest.TestCase):
    command = byte_arrays.NativeByteArrays.command
    paired = byte_arrays.NativeByteArrays.paired

    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix="nano-native-nested-arrays-"))
        cls.sequence = 0
        print("I retain nested array evidence at", cls.work, flush=True)

    def test_slice_retains_children_after_outer_release(self):
        body = ('PUSH_STR a\nPUSH_STR b\nSTR_CONCAT\nARR_LITERAL 5 1\n'
                'ARR_LITERAL 7 1\nARR_LITERAL 7 1\nSTORE_LOCAL 0\n'
                'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\n'
                'PUSH_I64 0\nPUSH_I64 1\nARR_SLICE\nSTORE_LOCAL 1\n'
                'PUSH_VOID\nSTORE_LOCAL 0\nCALL churn\n'
                'LOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nPUSH_I64 0\nARR_GET\n'
                'PUSH_STR ab\nSTR_EQ\nASSERT\n')
        self.paired('retained', body,
                    '.string a "a"\n.string b "b"\n.string ab "ab"\n' + CHURN,
                    collections=True)

    def test_self_cycle_survives_collection_and_pop(self):
        body = ('ARR_NEW 7\nDUP\nARR_PUSH\nSTORE_LOCAL 0\nCALL churn\n'
                'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nLOAD_LOCAL 0\nEQ\nASSERT\n'
                'LOAD_LOCAL 0\nARR_POP\nLOAD_LOCAL 0\nEQ\nASSERT\n'
                'LOAD_LOCAL 0\nARR_LEN\nPUSH_I64 0\nI64_EQ\nASSERT\n')
        self.paired('cycle', body, '.string a "a"\n.string b "b"\n' + CHURN,
                    collections=True)

    def test_nested_record_child(self):
        body = ('PUSH_STR a\nPUSH_STR b\nSTR_CONCAT\nAGG_PACK 0 0 0 1\n'
                'ARR_LITERAL 8 1\nARR_LITERAL 7 1\nSTORE_LOCAL 0\nCALL churn\n'
                'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nPUSH_I64 0\nARR_GET\n'
                'AGG_GET 0\nPUSH_STR ab\nSTR_EQ\nASSERT\n')
        self.paired('record-child', body,
                    '.string a "a"\n.string b "b"\n.string ab "ab"\n' + CHURN,
                    collections=True)
