"""I retain missing scalar array elements as void until a consumer checks them."""
try:
    from tests.sanitizer_options import asan_options
    from tests.native_toolchain import native_cc
except ModuleNotFoundError:
    from sanitizer_options import asan_options
    from native_toolchain import native_cc
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]

class OptionalArrayReads(unittest.TestCase):
    def checked(self,args):
        r=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True,timeout=60,
                         env={**os.environ,'ASAN_OPTIONS':asan_options("halt_on_error=1"),'UBSAN_OPTIONS':'halt_on_error=1'})
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        return r
    def paired(self,text):
        with tempfile.TemporaryDirectory(prefix='nano-optional-array-') as d:
            p=Path(d);assembly=p/'input.nasm';module=p/'input.nvm';source=p/'output.c';binary=p/'output'
            assembly.write_text(text)
            self.checked([ROOT/'bin/nanoisa','asm',assembly,'-o',module])
            self.checked([ROOT/'bin/nano_vm','--verify-only',module])
            self.checked([ROOT/'bin/nano_vm',module])
            self.checked([ROOT/'bin/nvm2c',module,'-o',source])
            self.checked([*native_cc(),'-std=c11','-O1','-Wall','-Wextra','-Werror',
                          '-fsanitize=address,undefined','-fno-sanitize-recover=all',source,'-o',binary])
            self.checked([binary])
    def test_tags_bounds_locals_and_calls(self):
        for tag,value in [(1,'PUSH_I64 73'),(4,'PUSH_BOOL 1'),(5,'PUSH_STR text')]:
            with self.subTest(tag=tag):
                body=f'{value}\nARR_LITERAL {tag} 1\nSTORE_LOCAL 0\n'
                body+=f'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nDUP\nTYPE_CHECK {tag}\nASSERT\n{value}\nEQ\nASSERT\n'
                for index in (-1,1,4294967296,9223372036854775807):
                    read=f'LOAD_LOCAL 0\nPUSH_I64 {index}\nARR_GET\n'
                    body+=read+'POP\n'+read+'STORE_LOCAL 1\nLOAD_LOCAL 1\nTYPE_CHECK 0\nASSERT\n'
                    body+=read+'CALL missing\nASSERT\n'
                body+=f'ARR_NEW {tag}\nPUSH_I64 0\nARR_GET\nTYPE_CHECK 0\nASSERT\n'
                suffix='.function missing 1 1 0 bool 1\nLOAD_LOCAL 0\nTYPE_CHECK 0\nRET\n.end\n'
                self.paired('.string text "kept"\n.entry main\n.function main 0 2 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'+suffix)
    def test_present_values_feed_typed_writes_and_records(self):
        for tag,value in [(1,'PUSH_I64 73'),(4,'PUSH_BOOL 1'),(5,'PUSH_STR text')]:
            with self.subTest(tag=tag):
                body=f'{value}\nARR_LITERAL {tag} 1\nSTORE_LOCAL 0\n'
                read='LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\n'
                body+='LOAD_LOCAL 0\n'+read+'ARR_PUSH\nPOP\n'
                body+='LOAD_LOCAL 0\nPUSH_I64 1\n'+read+'ARR_SET\nPOP\n'
                body+='LOAD_LOCAL 0\nARR_LEN\nPUSH_I64 2\nI64_EQ\nASSERT\n'
                body+=read+'AGG_PACK 0 0 0 1\nAGG_GET 0\n'+value+'\nEQ\nASSERT\n'
                self.paired('.types 1 0 0\n.string text "kept"\n.entry main\n.function main 0 1 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n')
    def test_present_and_missing_branch_join(self):
        for tag,value in [(1,'PUSH_I64 73'),(4,'PUSH_BOOL 1'),(5,'PUSH_STR text')]:
            for present in (0,1):
                with self.subTest(tag=tag,present=present):
                    body=f'{value}\nARR_LITERAL {tag} 1\nSTORE_LOCAL 0\nPUSH_BOOL {present}\nJMP_FALSE missing\n'
                    body+='LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nJMP joined\nmissing:\nLOAD_LOCAL 0\nPUSH_I64 1\nARR_GET\njoined:\n'
                    body+=f'STORE_LOCAL 1\nLOAD_LOCAL 1\nTYPE_CHECK {tag if present else 0}\nASSERT\n'
                    self.paired('.string text "kept"\n.entry main\n.function main 0 2 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n')
    def test_mixed_record_argument_storage(self):
        for tag,value in [(1,'PUSH_I64 73'),(4,'PUSH_BOOL 1'),(5,'PUSH_STR text')]:
            for missing in (0,1):
                with self.subTest(tag=tag,missing=missing):
                    body=f'{value}\nAGG_PACK 0 0 0 1\nCALL relay\nAGG_GET 0\nTYPE_CHECK {tag}\nASSERT\n'
                    body+=f'{value}\nARR_LITERAL {tag} 1\nPUSH_I64 {missing}\nARR_GET\nAGG_PACK 0 0 0 1\nCALL relay\nAGG_GET 0\nTYPE_CHECK {0 if missing else tag}\nASSERT\n'
                    self.paired('.types 1 0 0\n.string text "kept"\n.entry main\n.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n.function relay 1 1 0 struct 1\nLOAD_LOCAL 0\nRET\n.end\n')
    def test_projected_scalar_locals_calls_and_returns(self):
        for tag,name,value in [(1,'int','PUSH_I64 73'),(4,'bool','PUSH_BOOL 1'),(5,'string','PUSH_STR text')]:
            with self.subTest(tag=tag):
                body=f'{value}\nAGG_PACK 0 0 0 1\nCALL unwrap\n{value}\nEQ\nASSERT\n'
                body+=f'{value}\nARR_LITERAL {tag} 1\nPUSH_I64 0\nARR_GET\nAGG_PACK 0 0 0 1\nCALL unwrap\n{value}\nEQ\nASSERT\n'
                helpers=f'.function unwrap 1 3 0 {name} 1\n{value}\nSTORE_LOCAL 1\nLOAD_LOCAL 0\nAGG_GET 0\nSTORE_LOCAL 1\nARR_NEW {tag}\nSTORE_LOCAL 2\nLOAD_LOCAL 2\nLOAD_LOCAL 1\nARR_PUSH\nPOP\nLOAD_LOCAL 2\nPUSH_I64 0\nLOAD_LOCAL 1\nARR_SET\nPOP\nLOAD_LOCAL 2\nPUSH_I64 0\nARR_GET\nCALL identity\nRET\n.end\n.function identity 1 1 0 {name} 1\nLOAD_LOCAL 0\nRET\n.end\n'
                self.paired('.types 1 0 0\n.string text "kept"\n.entry main\n.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'+helpers)
    def test_nested_record_present_scalar_projection(self):
        for tag,name,value in [(1,'int','PUSH_I64 73'),(4,'bool','PUSH_BOOL 1'),(5,'string','PUSH_STR text')]:
            with self.subTest(tag=tag):
                body=f'{value}\nARR_LITERAL {tag} 1\nPUSH_I64 0\nARR_GET\nAGG_PACK 0 0 0 1\nAGG_PACK 0 1 0 1\nCALL relay_nested\nCALL project_nested\n{value}\nEQ\nASSERT\n'
                body+=f'{value}\nAGG_PACK 0 0 0 1\nAGG_PACK 0 1 0 1\nCALL relay_nested\nCALL project_nested\n{value}\nEQ\nASSERT\n'
                helpers='.function relay_nested 1 1 0 struct 1\nLOAD_LOCAL 0\nRET\n.end\n'
                helpers+=f'.function project_nested 1 1 0 {name} 1\nLOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 0\nRET\n.end\n'
                self.paired('.types 2 0 0\n.string text "kept"\n.entry main\n.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'+helpers)
if __name__=='__main__': unittest.main()
