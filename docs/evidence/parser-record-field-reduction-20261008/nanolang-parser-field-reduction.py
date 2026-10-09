from pathlib import Path
import tempfile,sys,unittest
sys.path.insert(0,'/private/tmp/nanolang-array-types-20261008')
from tests import test_native_byte_arrays as arrays
class ParserFields(unittest.TestCase):
    command=arrays.NativeByteArrays.command
    paired=arrays.NativeByteArrays.paired
    @classmethod
    def setUpClass(cls):
        cls.work=Path(tempfile.mkdtemp(prefix='nano-parser-field-reduction-'))
        cls.sequence=0
        print('I retain parser field reduction at',cls.work,flush=True)
    def test_exact_into_tagged_record(self):
        self.paired('exact-into-tagged',
            'PUSH_I64 1\nARR_LITERAL 1 1\nPUSH_I64 0\nARR_GET\n'
            'AGG_PACK 0 0 0 1\nARR_LITERAL 8 1\nSTORE_LOCAL 0\n'
            'LOAD_LOCAL 0\nPUSH_I64 0\nPUSH_I64 2\nAGG_PACK 0 0 0 1\nARR_SET\nPOP\n'
            'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nAGG_GET 0\nPUSH_I64 2\nI64_EQ\nASSERT\n')
    def test_tagged_into_exact_record(self):
        self.paired('tagged-into-exact',
            'PUSH_I64 1\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\nSTORE_LOCAL 0\n'
            'LOAD_LOCAL 0\nPUSH_I64 0\nPUSH_I64 2\nARR_LITERAL 1 1\nPUSH_I64 0\nARR_GET\n'
            'AGG_PACK 0 0 0 1\nARR_SET\nPOP\n'
            'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nAGG_GET 0\nPUSH_I64 2\nI64_EQ\nASSERT\n')
unittest.main(verbosity=2)
