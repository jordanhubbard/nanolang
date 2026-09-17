"""I resume ordinary callers after verified implicit scalar/void returns."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
VM = Path(os.environ.get('NANOLANG_IMPLICIT_VM', ROOT/'bin/nano_vm'))


class ImplicitReturns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='nano-implicit-')
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def run_cmd(self, args, status=0):
        p = subprocess.run(list(map(str,args)), capture_output=True, text=True, timeout=30)
        self.assertEqual(p.returncode,status,p.stdout+p.stderr)
        return p

    def module(self,text):
        source=self.work/'module.nasm'; module=self.work/'module.nvm'
        source.write_text(text)
        self.run_cmd([ROOT/'bin/nanoisa','asm',source,'-o',module])
        self.run_cmd([VM,'--verify-only',module])
        return module

    def compare(self,text,status):
        module=self.module(text)
        self.run_cmd([VM,module],status)
        source=self.work/'module.c'; exe=self.work/'program'
        self.run_cmd([ROOT/'bin/nvm2c',module,'-o',source])
        self.run_cmd(['cc','-std=c11','-Wall','-Wextra','-Werror','-O2',source,'-o',exe])
        self.run_cmd([exe],status)

    def test_nested_empty_void_resumes_caller(self):
        self.compare('.entry main\n.function main 0 0 0 int 1\nCALL empty\nPUSH_I64 17\nRET\n.end\n'
                     '.function empty 0 0 0 void 0\n.end\n',17)

    def test_nested_scalar_chain_and_entry_fallthrough(self):
        self.compare('.entry main\n.function main 0 0 0 int 1\nCALL outer\nPUSH_I64 2\nI64_ADD\n.end\n'
                     '.function outer 0 0 0 int 1\nCALL inner\nPUSH_I64 3\nI64_ADD\n.end\n'
                     '.function inner 0 0 0 int 1\nPUSH_I64 9\n.end\n',14)

    def test_float_and_bool_result_tags(self):
        self.compare('.entry main\n.function main 0 0 0 int 1\nCALL number\nPUSH_F64 -2.5\nF64_EQ\nASSERT\n'
                     'CALL yes\nDUP\nTYPE_CHECK 4\nASSERT\nASSERT\nPUSH_I64 7\nRET\n.end\n'
                     '.function number 0 0 0 float 1\nPUSH_F64 -2.5\n.end\n'
                     '.function yes 0 0 0 bool 1\nPUSH_BOOL 1\n.end\n',7)

    def test_conditional_edges_to_code_end(self):
        self.compare('.entry main\n.function main 0 0 0 int 1\nPUSH_BOOL 0\nCALL choose\n'
                     'PUSH_BOOL 1\nCALL choose\nI64_ADD\nRET\n.end\n'
                     '.function choose 1 1 0 int 1\n.parameters choose bool\nLOAD_LOCAL 0\nJMP_TRUE high\n'
                     'PUSH_I64 9\nJMP end\nhigh:\nPUSH_I64 17\nend:\n.end\n',26)

    def test_void_loop_then_caller_continuation(self):
        self.compare('.entry main\n.function main 0 0 0 int 1\nCALL loop\nPUSH_I64 11\nRET\n.end\n'
                     '.function loop 0 1 0 void 0\nPUSH_I64 0\nSTORE_LOCAL 0\nagain:\nLOAD_LOCAL 0\n'
                     'PUSH_I64 3\nI64_LT_S\nJMP_FALSE end\nLOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\n'
                     'STORE_LOCAL 0\nJMP again\nend:\n.end\n',11)

    def test_wrong_result_shape_is_not_synthesized(self):
        for signature,body in [('int 1','NOP\n'),('void 0','PUSH_I64 1\n')]:
            with self.subTest(signature=signature,body=body):
                source=self.work/'bad.nasm'; target=self.work/'prior.nvm'
                source.write_text('.entry main\n.function main 0 0 0 '+signature+'\n'+body+'.end\n')
                target.write_bytes(b'previous')
                result=subprocess.run([ROOT/'bin/nanoisa','asm',source,'-o',target],capture_output=True,text=True,timeout=30)
                self.assertNotEqual(result.returncode,0,result.stdout)
                self.assertEqual(target.read_bytes(),b'previous')

    def test_wrong_result_tag_is_rejected_at_runtime(self):
        module=self.module('.entry main\n.function main 0 0 0 int 1\nPUSH_BOOL 1\n.end\n')
        result=subprocess.run([VM,module],capture_output=True,text=True,timeout=30)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('returned bool, expected int',result.stderr)
        target=self.work/'prior.c'; target.write_text('previous')
        result=subprocess.run([ROOT/'bin/nvm2c',module,'-o',target],capture_output=True,text=True,timeout=30)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(target.read_text(),'previous')

    def test_heap_result_remains_explicit_native_refusal(self):
        module=self.module('.string text "retained"\n.entry main\n.function main 0 0 0 int 1\n'
                           'CALL text\nPOP\nPUSH_I64 0\nRET\n.end\n'
                           '.function text 0 0 0 string 1\nPUSH_STR text\n.end\n')
        self.run_cmd([VM,module])
        target=self.work/'prior.c'; target.write_text('previous')
        result=subprocess.run([ROOT/'bin/nvm2c',module,'-o',target],capture_output=True,text=True,timeout=30)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('implicit returns only',result.stderr)
        self.assertEqual(target.read_text(),'previous')


if __name__ == '__main__':
    unittest.main()
