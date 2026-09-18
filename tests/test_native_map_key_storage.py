"""I extract exact string keys without rewriting optional producer storage."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_native_optional_array_reads as arrays

ROOT=Path(__file__).resolve().parents[1]
HEADER='.entry main\n.string key "retained"\n.string name "lookup"\n.types 1 0 0\n'

class NativeMapKeyStorage(unittest.TestCase):
    def checked(self,args):
        result=arrays.OptionalArrayReads.checked(self,args)
        if str(args[0]) == str(ROOT/'bin/nvm2c'):
            self.assertIn('nvalue_require_string(v[',Path(args[-1]).read_text())
        return result
    paired=arrays.OptionalArrayReads.paired

    def test_array_map_and_projected_keys_preserve_aliases(self):
        keys=(
            'PUSH_STR key\nARR_LITERAL 5 1\nPUSH_I64 0\nARR_GET\n',
            'HM_NEW 5 5\nPUSH_STR name\nPUSH_STR key\nHM_SET\nPUSH_STR name\nHM_GET\n',
            'PUSH_STR key\nARR_LITERAL 5 1\nPUSH_I64 0\nARR_GET\nAGG_PACK 0 0 0 1\nAGG_GET 0\n',
        )
        for key in keys:
            with self.subTest(key=key):
                body='HM_NEW 5 1\nSTORE_LOCAL 0\n'+key+'STORE_LOCAL 1\nLOAD_LOCAL 1\nSTORE_LOCAL 2\n'
                body+='LOAD_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 42\nHM_SET\nPOP\n'
                body+='LOAD_LOCAL 0\nLOAD_LOCAL 2\nHM_HAS\nASSERT\n'
                body+='LOAD_LOCAL 0\nLOAD_LOCAL 1\nHM_GET\nPUSH_I64 42\nEQ\nASSERT\n'
                body+='LOAD_LOCAL 0\nLOAD_LOCAL 2\nHM_DELETE\nPOP\n'
                body+='LOAD_LOCAL 0\nLOAD_LOCAL 1\nHM_HAS\nBOOL_NOT\nASSERT\n'
                body+='LOAD_LOCAL 2\nTYPE_CHECK 5\nASSERT\nLOAD_LOCAL 2\nPUSH_STR key\nEQ\nASSERT\n'
                self.paired(HEADER+'.function main 0 3 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n')

    def test_branch_and_call_keys(self):
        for flag in (0,1):
            body=f'HM_NEW 5 1\nSTORE_LOCAL 0\nPUSH_BOOL {flag}\nJMP_FALSE right\n'
            body+='PUSH_STR key\nARR_LITERAL 5 1\nPUSH_I64 0\nARR_GET\nJMP joined\nright:\n'
            body+='PUSH_STR key\nARR_LITERAL 5 1\nPUSH_I64 0\nARR_GET\njoined:\nSTORE_LOCAL 1\n'
            body+='LOAD_LOCAL 0\nLOAD_LOCAL 1\nCALL insert\nPOP\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nHM_HAS\nASSERT\n'
            helpers='.function insert 2 2 0 hashmap 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 7\nHM_SET\nRET\n.end\n'
            self.paired(HEADER+'.function main 0 2 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'+helpers)

    def test_known_nonstring_keys_preserve_previous_output(self):
        values=('PUSH_I64 7\n','PUSH_BOOL 1\n','PUSH_F64 1.5\n',
                'PUSH_I64 7\nARR_LITERAL 1 1\nPUSH_I64 0\nARR_GET\n')
        with tempfile.TemporaryDirectory() as tmp:
            source,module,output=(Path(tmp)/n for n in ('input.nasm','input.nvm','previous.c'))
            for value in values:
                with self.subTest(value=value):
                    source.write_text(HEADER+'.function main 0 0 0 int 1\nHM_NEW 5 1\n'+value+
                                      'HM_HAS\nPOP\nPUSH_I64 0\nRET\n.end\n')
                    self.checked([ROOT/'bin/nanoisa','asm',source,'-o',module])
                    output.write_text('retained output')
                    result=subprocess.run([ROOT/'bin/nvm2c',module,'-o',output],capture_output=True,text=True,timeout=30)
                    self.assertGreater(result.returncode,0)
                    self.assertIn('I ',result.stderr)
                    self.assertEqual(output.read_text(),'retained output')

if __name__=='__main__':unittest.main()
