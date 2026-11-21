"""I preserve tags at record joins without erasing incompatible field shapes."""
from pathlib import Path
import os,subprocess,tempfile,unittest
from tests import test_native_optional_array_reads as optional
ROOT=Path(__file__).resolve().parents[1]

class RecordJoinValues(unittest.TestCase):
    checked=optional.OptionalArrayReads.checked
    def pair(self,text,refuse=False):
        with tempfile.TemporaryDirectory(prefix='record-join-values-') as directory:
            p=Path(directory);source=p/'input.nasm';module=p/'input.nvm';generated=p/'output.c';binary=p/'output'
            source.write_text(text);self.checked([ROOT/'bin/nanoisa','asm',source,'-o',module]);self.checked([ROOT/'bin/nano_vm','--verify-only',module]);self.checked([ROOT/'bin/nano_vm',module])
            translator=Path(os.environ.get('NANOLANG_NVM2C',ROOT/'bin/nvm2c'))
            generated.write_bytes(b'prior source\0')
            if refuse:
                result=subprocess.run([translator,module,'-o',generated],capture_output=True,text=True,cwd=ROOT,timeout=60)
                self.assertNotEqual(result.returncode,0);self.assertEqual(generated.read_bytes(),b'prior source\0');return
            self.checked([translator,module,'-o',generated]);self.checked([os.environ.get('CC','cc'),'-std=c11','-O1','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',generated,'-o',binary]);self.checked([binary])
    def test_scalar_payloads_both_orders_and_missing_tags(self):
        for tag,value in [(1,'PUSH_I64 73'),(4,'PUSH_BOOL 1'),(3,'PUSH_F64 1.5'),(5,'PUSH_STR text')]:
            for missing in (0,1):
                for reverse in (False,True):
                    with self.subTest(tag=tag,missing=missing,reverse=reverse):
                        optional=f'{value}\nARR_LITERAL {tag} 1\nPUSH_I64 {missing}\nARR_GET\nAGG_PACK 0 0 0 1\n'
                        exact=f'{value}\nAGG_PACK 0 0 0 1\n'
                        first,second=(exact,optional) if reverse else (optional,exact)
                        text='.types 1 0 0\n.string text "kept"\n.entry main\n.function choose 1 1 0 struct 1\nLOAD_LOCAL 0\nJMP_FALSE other\n'+first+'JMP joined\nother:\n'+second+'joined:\nRET\n.end\n.function main 0 0 0 int 1\n'
                        for branch in (0,1):
                            selected_optional=bool(branch)!=reverse
                            expected=0 if selected_optional and missing else tag
                            payload='PUSH_VOID' if expected==0 else value
                            text+=f'PUSH_BOOL {branch}\nCALL choose\nAGG_GET 0\nDUP\nTYPE_CHECK {expected}\nASSERT\n{payload}\nEQ\nASSERT\n'
                        self.pair(text+'PUSH_I64 0\nRET\n.end\n')
    def test_incompatible_exact_and_aggregate_fields_preserve_output(self):
        for value in ['PUSH_STR text','PUSH_I64 7\nAGG_PACK 0 1 0 1','PUSH_I64 7\nARR_LITERAL 1 1']:
            with self.subTest(value=value):
                text='.types 2 0 0\n.string text "kept"\n.entry main\n.function choose 1 1 0 struct 1\nLOAD_LOCAL 0\nJMP_FALSE other\nPUSH_I64 73\nAGG_PACK 0 0 0 1\nJMP joined\nother:\n'+value+'\nAGG_PACK 0 0 0 1\njoined:\nRET\n.end\n.function main 0 0 0 int 1\nPUSH_BOOL 1\nCALL choose\nAGG_GET 0\nPUSH_I64 73\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n'
                self.pair(text,True)

if __name__=='__main__':unittest.main()
