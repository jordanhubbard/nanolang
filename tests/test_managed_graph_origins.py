"""I check graph provenance without changing leaf admission or runtime lifetime."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from tests.managed_probe_flags import compiler_command, compile_flags, link_flags
ROOT=Path(__file__).resolve().parents[1]
class GraphOrigins(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='nano-graph-origins-');cls.work=Path(cls.temp.name);cls.probes=[]
        for name,cc,flags in [('ordinary','cc',[]),('sanitized','clang',shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS',''))+['-fsanitize=address,undefined','-fno-sanitize-recover=all'])]:
            exe=cls.work/name
            p=subprocess.run([*compiler_command(cc),*compile_flags(),*flags,'-std=c11','-O1','-Wall','-Wextra','-Werror','-DNMA_TESTING',ROOT/'src/nanoisa/managed_array_shapes.c',ROOT/'tests/nanoisa/test_managed_graph_origins.c',*shlex.split(os.environ['NMA_LINK_OBJECTS']),'-lm','-lcrypto',*link_flags(),'-o',exe],capture_output=True,text=True)
            if p.returncode:raise RuntimeError(p.stdout+p.stderr)
            cls.probes.append(exe)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def command(self,args):
        p=subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=30)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr);return p.stdout
    def program(self,body,extra='',locals=3):
        return '.string text "leaf"\n.string empty ""\n.entry main\n.function main 0 '+str(locals)+' 0 int 1\n'+body+'\nPUSH_I64 0\nRET\n.end\n'+extra
    def analyze(self,text,status=0,leaf=1,vm=False,budget=None):
        source=self.work/'input.nasm';source.write_text(text);outputs=[]
        for probe in self.probes:
            lines=self.command([probe,source]+([] if budget is None else [budget])).splitlines()
            outputs.append([list(map(int,line.split())) for line in lines])
        self.assertEqual(outputs[0],outputs[1]);fields,*origins=outputs[0]
        self.assertEqual(fields[0],status,fields);self.assertEqual(fields[1],leaf,fields)
        if leaf==1 and status in (0,1):self.assertEqual(fields[2],int(status==0),fields)
        if vm:
            module=self.work/'input.nvm';self.command([ROOT/'bin/nanoisa','asm',source,'-o',module]);self.command([ROOT/'bin/nano_vm',module])
        return fields,origins
    def test_leaf_decisions_and_reports_unchanged(self):
        for body in ('ARR_NEW 1\nPUSH_I64 1\nARR_PUSH\nPOP','PUSH_STR text\nPUSH_STR empty\nSTR_SPLIT\nPOP','PUSH_STR text\nARR_LITERAL 5 1\nPUSH_I64 0\nPUSH_I64 1\nARR_SLICE\nPOP'):
            fields,origins=self.analyze(self.program(body),leaf=0,vm=True)
            self.assertEqual(fields[2],1);self.assertTrue(all(o[5:]==[0,0] for o in origins))
    def test_nested_get_pop_and_alias_mutation(self):
        body=('ARR_NEW 5\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nLOAD_LOCAL 0\nARR_LITERAL 7 2\nSTORE_LOCAL 1\n'
              'LOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nPUSH_STR text\nARR_PUSH\nPOP\n'
              'LOAD_LOCAL 1\nARR_POP\nPUSH_I64 7\nARR_PUSH\nPOP\n'
              'LOAD_LOCAL 0\nARR_LEN\nPUSH_I64 2\nEQ\nASSERT')
        fields,o=self.analyze(self.program(body),vm=True)
        self.assertEqual(fields[3],2);self.assertEqual(o[0][4:],[(1<<5)|(1<<1),0,0]);self.assertEqual(o[1][4:],[1<<7,1,0])
    def test_shallow_slice_keeps_child_identity_and_fresh_outer(self):
        body=('ARR_NEW 5\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nARR_LITERAL 7 1\nSTORE_LOCAL 1\n'
              'LOAD_LOCAL 1\nPUSH_I64 0\nPUSH_I64 1\nARR_SLICE\nSTORE_LOCAL 2\n'
              'LOAD_LOCAL 2\nPUSH_I64 0\nARR_GET\nPUSH_STR text\nARR_PUSH\nPOP\n'
              'LOAD_LOCAL 2\nPUSH_BOOL 1\nARR_PUSH\nPOP\n'
              'LOAD_LOCAL 1\nARR_LEN\nPUSH_I64 1\nEQ\nASSERT')
        fields,o=self.analyze(self.program(body),vm=True)
        self.assertEqual(fields[3],3);self.assertEqual(o[0][4],1<<5)
        self.assertEqual(o[1][4:],[1<<7,1,0]);self.assertEqual(o[2][4:],[(1<<7)|(1<<4),1,0])
    def test_self_mutual_cycles_and_heterogeneous_children(self):
        body=('ARR_NEW 7\nSTORE_LOCAL 0\nARR_NEW 5\nSTORE_LOCAL 1\n'
              'LOAD_LOCAL 0\nLOAD_LOCAL 0\nARR_PUSH\nLOAD_LOCAL 1\nARR_PUSH\nPUSH_I64 3\nARR_PUSH\nPOP\n'
              'LOAD_LOCAL 1\nLOAD_LOCAL 0\nARR_PUSH\nPUSH_STR text\nARR_PUSH\nPOP')
        _,o=self.analyze(self.program(body),vm=True)
        self.assertEqual(o[0][4:],[(1<<7)|(1<<1),3,0]);self.assertEqual(o[1][4:],[(1<<7)|(1<<5),1,0])
    def test_mixed_receiver_storage_and_nested_write_cross_product(self):
        body=('ARR_NEW 5\nSTORE_LOCAL 0\nPUSH_BOOL 1\nJMP_FALSE other\nARR_NEW 7\nJMP joined\nother:\nARR_NEW 1\njoined:\nLOAD_LOCAL 0\nARR_PUSH\nPOP')
        self.analyze(self.program(body),status=1)
        self.analyze(self.program(body.replace('ARR_NEW 1\njoined:','ARR_NEW 5\njoined:')),vm=True)
    def test_repeated_sites_recursive_calls_and_globals(self):
        extra=('.function make 1 1 0 array 1\nLOAD_LOCAL 0\nARR_LITERAL 7 1\nRET\n.end\n'
               '.function recur 2 2 0 array 1\nLOAD_LOCAL 1\nPUSH_I64 0\nEQ\nJMP_TRUE base\n'
               'LOAD_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 1\nI64_SUB\nCALL recur\nRET\nbase:\nLOAD_LOCAL 0\nRET\n.end\n')
        body=('ARR_NEW 5\nCALL make\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nCALL make\nPUSH_I64 2\nCALL recur\n'
              'PUSH_I64 0\nARR_GET\nPUSH_STR text\nARR_PUSH\nPOP')
        _,o=self.analyze(self.program(body,extra),vm=True)
        self.assertEqual(len(o),2);self.assertEqual(o[1][5],3);self.assertEqual(o[1][4],(1<<7)|(1<<5)|(1<<0)) # Globals include initial VOID.
    def test_reentry_global_child_effects_and_loop_copies(self):
        init='.function __init__ 0 0 0 void 0\nARR_NEW 5\nARR_LITERAL 7 1\nSTORE_GLOBAL 0\nRET\n.end\n'
        body=('LOAD_GLOBAL 0\nPUSH_I64 0\nARR_GET\nSTORE_LOCAL 0\n'
              'PUSH_I64 2\nSTORE_LOCAL 1\nloop:\nLOAD_GLOBAL 0\nPUSH_I64 0\nPUSH_I64 1\nARR_SLICE\n'
              'PUSH_I64 0\nARR_GET\nPUSH_STR text\nARR_PUSH\nPOP\n'
              'LOAD_LOCAL 1\nPUSH_I64 1\nI64_SUB\nDUP\nSTORE_LOCAL 1\nJMP_TRUE loop\n'
              'LOAD_LOCAL 0\nPUSH_I64 8\nARR_PUSH\nPOP')
        fields,o=self.analyze(self.program(body,init),vm=True)
        self.assertEqual(fields[3],3)
        self.assertEqual(o[0][4:],[(1<<5)|(1<<1),0,0])
        self.assertEqual(o[1][5],1);self.assertEqual(o[2][5],1)

    def test_unknown_children_and_unused_escape(self):
        extra='.function unused 1 1 0 array 1\nARR_NEW 7\nLOAD_LOCAL 0\nARR_PUSH\nRET\n.end\n'
        self.analyze(self.program('',extra),status=1)
        body='ARR_NEW 1\nARR_NEW 7\nARR_PUSH\nPOP'
        self.analyze(self.program(body),status=1)
    def test_caps_allocation_atomicity_and_backend_refusal(self):
        body='ARR_NEW 7\nDUP\nARR_PUSH\nPOP';text=self.program(body)
        for budget in range(8):self.analyze(text,status=4,budget=budget)
        self.analyze(text,budget=8)
        self.analyze(self.program('ARR_NEW 7\nPOP\n'*65),status=3)
        source=self.work/'refuse.nasm';source.write_text(self.program('ARR_NEW 1\nARR_NEW 7\nARR_PUSH\nPOP'));module=self.work/'refuse.nvm'
        self.command([ROOT/'bin/nanoisa','asm',source,'-o',module])
        for tool in ('nvm2llvm','nvm2wasm'):
            output=self.work/(tool+'.old');output.write_bytes(b'prior output')
            p=subprocess.run([ROOT/'bin'/tool,module,'-o',output],capture_output=True,timeout=30)
            self.assertNotEqual(p.returncode,0);self.assertEqual(output.read_bytes(),b'prior output')
if __name__=='__main__':unittest.main()
