"""I compare exact array handles without erasing identity into element equality."""
from pathlib import Path
import tempfile
import unittest
from tests import test_native_byte_arrays as arrays


class NativeArrayIdentity(unittest.TestCase):
    command = arrays.NativeByteArrays.command
    paired = arrays.NativeByteArrays.paired

    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='native-array-identity-'))
        cls.sequence = 0
        print('I retain array identity evidence at', cls.work, flush=True)

    def test_alias_and_independent_slice_all_storage_kinds(self):
        cases = {
            'int': 'PUSH_I64 7\nARR_LITERAL 1 1\n',
            'byte': 'PUSH_U8 7\nARR_LITERAL 2 1\n',
            'bool': 'PUSH_BOOL 1\nARR_LITERAL 4 1\n',
            'float': 'PUSH_F64 1.5\nARR_LITERAL 3 1\n',
            'string': 'PUSH_STR text\nARR_LITERAL 5 1\n',
            'record': 'PUSH_I64 7\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\n',
            'function': 'CLOSURE_NEW leaf 0\nARR_LITERAL 15 1\n',
            'nested': 'PUSH_I64 7\nARR_LITERAL 1 1\nARR_LITERAL 7 1\n',
        }
        for name, constructor in cases.items():
            with self.subTest(kind=name):
                body = constructor + ('STORE_LOCAL 0\nLOAD_LOCAL 0\nSTORE_LOCAL 1\n'
                    'LOAD_LOCAL 0\nLOAD_LOCAL 1\nEQ\nASSERT\n'
                    'LOAD_LOCAL 0\nLOAD_LOCAL 1\nNE\nBOOL_NOT\nASSERT\n'
                    'LOAD_LOCAL 0\nPUSH_I64 0\nPUSH_I64 1\nARR_SLICE\nSTORE_LOCAL 2\n'
                    'LOAD_LOCAL 0\nLOAD_LOCAL 2\nEQ\nBOOL_NOT\nASSERT\n'
                    'LOAD_LOCAL 0\nLOAD_LOCAL 2\nNE\nASSERT\n'
                    'LOAD_LOCAL 0\nPUSH_I64 7\nEQ\nBOOL_NOT\nASSERT\n')
                helpers = '.string text "same"\n'
                if name == 'function':
                    helpers += '.function leaf 0 0 0 int 1\nPUSH_I64 7\nRET\n.end\n'
                self.paired(name, body, helpers)

    def test_record_projection_retains_child_handle(self):
        body = ('PUSH_I64 7\nARR_LITERAL 1 1\nSTORE_LOCAL 0\n'
                'LOAD_LOCAL 0\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\n'
                'PUSH_I64 0\nPUSH_I64 1\nARR_SLICE\nPUSH_I64 0\nARR_GET\n'
                'AGG_GET 0\nLOAD_LOCAL 0\nEQ\nASSERT\n')
        self.paired('record-child', body)
