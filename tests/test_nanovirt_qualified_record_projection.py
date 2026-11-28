"""I project qualified call results using their checked record field layout."""
from pathlib import Path
import os, tempfile, unittest
from unittest.mock import patch
from tests import nanolang_translator_routes as routes

ROOT = Path(__file__).resolve().parents[1]
PROVIDER = '''module projection_provider
pub struct TagFacts { valid: bool, tags: int }
pub struct Holder { code: int, payload: TagFacts }
pub fn scalar() -> TagFacts { return TagFacts { valid: true, tags: 42 } }
shadow scalar { assert (== (scalar).tags 42) assert (scalar).valid }
pub fn wrapped() -> Holder { return Holder { code: 17, payload: (scalar) } }
shadow wrapped { assert (== (wrapped).payload.tags 42) }
'''

def fixture(fields, chained=False):
    projection = '(provider.wrapped).payload.tags' if chained else '(provider.scalar).tags'
    return ('module "projection_provider.nano" as provider\nstruct Decoy {'+fields+'}\n'
        'fn observe() -> int { return '+projection+' }\n'
        'shadow observe { assert (== (observe) 42) }\n'
        'fn main() -> int { assert (== (observe) 42) return 0 }\n'
        'shadow main { assert (== (main) 0) }\n')

class QualifiedRecordProjection(unittest.TestCase):
    def exercise(self, fields, chained=False):
        with tempfile.TemporaryDirectory(prefix='nano-qualified-record-projection-') as directory:
            work=Path(directory);source=work/'main.nano';source.write_text(fixture(fields,chained))
            (work/'projection_provider.nano').write_text(PROVIDER)
            # I qualify this C producer independently. The complete join matrix
            # separately retains both producers through all five existing routes.
            with patch.dict(os.environ,{},clear=False):
                old_native=os.environ.pop('NANOLANG_TRANSLATOR_SOURCE_COMPILER',None)
                old_module=os.environ.pop('NANOLANG_TRANSLATOR_SOURCE_MODULE',None)
                try:
                    products=routes.build_routes(source,work)
                finally:
                    if old_native is not None:os.environ['NANOLANG_TRANSLATOR_SOURCE_COMPILER']=old_native
                    if old_module is not None:os.environ['NANOLANG_TRANSLATOR_SOURCE_MODULE']=old_module
            self.assertEqual(len(products),3)
            for label,command in products:
                with self.subTest(route=label):self.assertEqual(routes.checked(command,timeout=60),'')

    def test_qualified_result_ignores_unrelated_field_offsets(self):
        for fields in ['tags: int, other: int','left: int, right: int, tags: int']:
            with self.subTest(fields=fields):self.exercise(fields)

    def test_chained_qualified_result_keeps_each_checked_layout(self):
        self.exercise('left: int, right: int, payload: int, tags: int',True)

if __name__=='__main__':unittest.main()
