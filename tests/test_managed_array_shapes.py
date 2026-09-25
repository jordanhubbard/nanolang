"""I qualify private array-shape eligibility with stable read-only analysis results."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from tests.managed_probe_flags import compiler_command, compile_flags, link_flags
ROOT=Path(__file__).resolve().parents[1]

class ArrayShapes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='nano-array-shapes-')
        cls.work=Path(cls.temp.name)
        objects=shlex.split(os.environ['NMA_LINK_OBJECTS'])
        cls.probes=[]
        for name,compiler,flags in [('ordinary','cc',[]),('sanitized','clang',
            shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS',''))+['-fsanitize=address,undefined','-fno-sanitize-recover=all'])]:
            exe=cls.work/name
            subprocess.run([*compiler_command(compiler),*compile_flags(),*flags,'-std=c11','-O1','-Wall','-Wextra','-Werror','-DNMA_TESTING',
                ROOT/'src/nanoisa/managed_array_shapes.c',ROOT/'tests/nanoisa/test_managed_array_shapes.c',
                *objects,'-lm','-lcrypto',*link_flags(),'-o',exe],cwd=ROOT,check=True,capture_output=True,text=True)
            cls.probes.append(exe)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def command(self,args):
        r=subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=30,cwd=ROOT)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr);return r
    def program(self,body,extra='',locals=2):
        return '.string text "leaf"\n.string empty ""\n.entry main\n.function main 0 '+str(locals)+' 0 int 1\n'+body+'\nPUSH_I64 0\nRET\n.end\n'+extra
    def analyze(self,text,status=0,vm=False,budget=None):
        source=self.work/'input.nasm';source.write_text(text)
        outputs=[]
        for probe in self.probes:
            args=[probe,source]+([] if budget is None else [str(budget)])
            out=self.command(args).stdout.splitlines();fields=list(map(int,out[0].split()))
            self.assertEqual(fields[0],status,out);outputs.append(out)
        self.assertEqual(outputs[0],outputs[1])
        if vm:
            module=self.work/'input.nvm'
            self.command([ROOT/'bin/nanoisa','asm',source,'-o',module])
            self.command([ROOT/'bin/nano_vm',module])
        return list(map(int,outputs[0][0].split())),[list(map(int,row.split())) for row in outputs[0][2:]]
    def test_packed_matrix_and_declared_identity(self):
        pairs=[(1,'PUSH_I64 -1'),(1,'PUSH_U8 255'),(2,'PUSH_U8 255'),(2,'PUSH_I64 -1'),
               (3,'PUSH_F64 -0.0'),(3,'PUSH_I64 9007199254740993'),(4,'PUSH_BOOL 1')]
        for kind,value in pairs:
            with self.subTest(kind=kind,value=value):
                fields,origins=self.analyze(self.program(f'ARR_NEW {kind}\n{value}\nARR_PUSH\nDUP\nPUSH_I64 0\nARR_GET\nTYPE_CHECK {kind}\nASSERT\nARR_POP\nTYPE_CHECK {kind}\nASSERT'),vm=True)
                self.assertEqual(fields[4],1);self.assertEqual(origins[0][2:],[kind,1,0])
    def test_boxed_heterogeneous_alias_and_pop_optional(self):
        # A get through an alias includes VOID and the complete weak content set.
        body='ARR_NEW 5\nDUP\nSTORE_LOCAL 0\nSTORE_LOCAL 1\nLOAD_LOCAL 0\nPUSH_I64 17\nARR_PUSH\nPOP\nLOAD_LOCAL 1\nPUSH_STR text\nARR_PUSH\nPOP\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nARR_PUSH\nPOP\nLOAD_LOCAL 0\nARR_POP\nPOP'
        fields,origins=self.analyze(self.program(body),vm=True)
        self.assertEqual(fields[4],3);self.assertEqual(origins[0][2:],[5,0,(1<<0)|(1<<1)|(1<<5)])
    def test_split_promotion_and_global_aliases(self):
        body='PUSH_STR text\nPUSH_STR empty\nSTR_SPLIT\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_I64 3\nARR_PUSH\nPOP\nLOAD_GLOBAL 0\nPUSH_I64 0\nPUSH_BOOL 1\nARR_SET\nPOP\nLOAD_GLOBAL 0\nARR_LEN\nPOP'
        fields,origins=self.analyze(self.program(body),vm=True)
        self.assertEqual(origins[0][2:],[5,0,(1<<1)|(1<<4)|(1<<5)]);self.assertGreater(fields[5],0)
    def test_branch_origins_and_portable_cross_product(self):
        body='PUSH_BOOL 1\nJMP_FALSE other\nARR_NEW 1\nJMP joined\nother:\nARR_NEW 2\njoined:\nPUSH_I64 257\nARR_PUSH\nPOP'
        fields,origins=self.analyze(self.program(body),vm=True);self.assertEqual(fields[3],2)
        self.analyze(self.program(body.replace('ARR_NEW 2','ARR_NEW 3').replace('PUSH_I64 257','PUSH_U8 7')),1)
    def test_calls_repeated_sites_and_reordered_arguments(self):
        extra='.function create 0 0 0 array 1\nARR_NEW 5\nRET\n.end\n.function write 2 2 0 array 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nARR_PUSH\nRET\n.end\n'
        for first,second in [('PUSH_STR text','PUSH_I64 7'),('PUSH_I64 7','PUSH_STR text')]:
            body=f'CALL create\n{first}\nCALL write\nPOP\nCALL create\n{second}\nCALL write\nPOP'
            fields,origins=self.analyze(self.program(body,extra),vm=True)
            self.assertEqual(fields[3],1);self.assertEqual(origins[0][4],(1<<1)|(1<<5))
    def test_recursive_scc_effects_and_nonzero_function_branch(self):
        extra='.function recurse 2 2 0 array 1\nLOAD_LOCAL 1\nPUSH_I64 0\nI64_EQ\nJMP_TRUE base\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 1\nI64_SUB\nCALL recurse\nRET\nbase:\nLOAD_LOCAL 0\nPUSH_STR text\nARR_PUSH\nRET\n.end\n'
        fields,origins=self.analyze(self.program('ARR_NEW 5\nPUSH_I64 3\nCALL recurse\nPOP',extra),vm=True)
        self.assertEqual(fields[4],1);self.assertEqual(origins[0][4],1<<5)
    def test_loop_effects_and_initializer_reentry(self):
        body='ARR_NEW 5\nSTORE_LOCAL 0\nPUSH_I64 2\nSTORE_LOCAL 1\nloop:\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nARR_PUSH\nPOP\nLOAD_LOCAL 1\nPUSH_I64 1\nI64_SUB\nDUP\nSTORE_LOCAL 1\nJMP_TRUE loop\nLOAD_GLOBAL 0\nPUSH_STR text\nARR_PUSH\nPOP'
        init='.function __init__ 0 0 0 void 0\nARR_NEW 5\nSTORE_GLOBAL 0\nRET\n.end\n'
        fields,origins=self.analyze(self.program(body,init),vm=True)
        self.assertEqual({o[4] for o in origins},{1<<1,1<<5})
        # Every reentry includes all global writes, even writes after an earlier read.
        bad='LOAD_GLOBAL 0\nPUSH_I64 0\nARR_GET\nSTORE_LOCAL 0\nARR_NEW 1\nLOAD_LOCAL 0\nARR_PUSH\nPOP\nLOAD_GLOBAL 0\nPUSH_STR text\nARR_PUSH\nPOP'
        self.analyze(self.program(bad,init),1)
    def test_missing_pop_and_unproved_nested_or_escape(self):
        for body in ['ARR_NEW 1\nARR_NEW 5\nPUSH_I64 0\nARR_GET\nARR_PUSH\nPOP',
                     'ARR_NEW 5\nARR_NEW 1\nARR_PUSH\nPOP',
                     'ARR_NEW 1\nPUSH_STR text\nARR_PUSH\nPOP',
                     'ARR_NEW 7\nPOP','PUSH_I64 1\nI64_INVERT\nPOP']:
            self.analyze(self.program(body),1)
        extra='.function unused 1 1 0 array 1\nLOAD_LOCAL 0\nPUSH_I64 1\nARR_PUSH\nRET\n.end\n'
        self.analyze(self.program('',extra),1)
    def test_limits_and_invalid_status(self):
        self.analyze(self.program('ARR_NEW 1\nPOP\n'*65),3)
        self.analyze(self.program('LOAD_GLOBAL 256\nPOP'),3)
        self.analyze(self.program('',locals=257),3)
        self.analyze(self.program('PUSH_VOID\n'*257+'POP\n'*257),3)
        self.analyze(self.program('NOP\n'*65536),3)
        self.analyze(self.program('NOP\n'*4100,locals=256),3)
        extras=''.join(f'.function f{i} 0 0 0 void 0\nRET\n.end\n' for i in range(256))
        self.analyze(self.program('',extras),3)
    def test_mutually_recursive_effects_and_unknown_escape(self):
        extra=('.function left 2 2 0 array 1\nLOAD_LOCAL 1\nPUSH_I64 0\nI64_EQ\nJMP_TRUE base\n'
               'LOAD_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 1\nI64_SUB\nCALL right\nRET\nbase:\n'
               'LOAD_LOCAL 0\nPUSH_STR text\nARR_PUSH\nRET\n.end\n'
               '.function right 2 2 0 array 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nCALL left\nRET\n.end\n')
        _,origins=self.analyze(self.program('ARR_NEW 5\nPUSH_I64 2\nCALL left\nPOP',extra),vm=True)
        self.assertEqual(origins[0][4],1<<5)
        for body in ('LOAD_LOCAL 0\nRET','LOAD_LOCAL 0\nSTORE_GLOBAL 0\nPUSH_I64 0\nRET'):
            extra='.function unused 1 1 0 int 1\n'+body+'\n.end\n'
            self.analyze(self.program('',extra),1)

    def test_literal_count_matrix_and_empty_origins(self):
        pairs=[(1,'PUSH_I64 -1'),(1,'PUSH_U8 255'),(2,'PUSH_U8 255'),
               (2,'PUSH_I64 -1'),(3,'PUSH_F64 -0.0'),(3,'PUSH_I64 17'),(4,'PUSH_BOOL 1')]
        for kind,value in pairs:
            for count in (0,1,8,9,17):
                fields,origins=self.analyze(self.program((value+'\n')*count+f'ARR_LITERAL {kind} {count}\nARR_LEN\nPUSH_I64 {count}\nI64_EQ\nASSERT'),vm=True)
                self.assertEqual(fields[4],count);self.assertEqual(origins[0][2:],[kind,1,0])
        values=['PUSH_VOID','PUSH_I64 7','PUSH_U8 255','PUSH_F64 -0.0','PUSH_BOOL 1','PUSH_STR text','ENUM_VAL 2 3']
        for kind in (0,5,9):
            fields,origins=self.analyze(self.program('\n'.join(values)+f'\nARR_LITERAL {kind} 7\nPOP'),vm=True)
            self.assertEqual(fields[4],7);self.assertEqual(origins[0][4],sum(1<<t for t in (0,1,2,3,4,5,9)))
        for body in ('PUSH_STR text\nARR_LITERAL 1 1\nPOP','ARR_NEW 1\nARR_LITERAL 5 1\nPOP','ARR_LITERAL 7 0\nPOP'):
            self.analyze(self.program(body),1)

    def test_slice_fresh_origin_and_independent_writes(self):
        body=('PUSH_STR text\nARR_LITERAL 5 1\nSTORE_LOCAL 0\n'
              'LOAD_LOCAL 0\nPUSH_I64 0\nPUSH_I64 1\nARR_SLICE\nSTORE_LOCAL 1\n'
              'LOAD_LOCAL 1\nPUSH_BOOL 1\nARR_PUSH\nPOP\n'
              'LOAD_LOCAL 0\nARR_LEN\nPUSH_I64 1\nI64_EQ\nASSERT')
        fields,origins=self.analyze(self.program(body),vm=True)
        self.assertEqual(fields[3],2)
        self.assertEqual(origins[0][4],1<<5);self.assertEqual(origins[1][4],(1<<5)|(1<<4))
        # Later source writes conservatively reach the copy summary, never vice versa.
        _,origins=self.analyze(self.program(body+'\nLOAD_LOCAL 0\nPUSH_I64 7\nARR_PUSH\nPOP'),vm=True)
        self.assertEqual(origins[0][4],(1<<5)|(1<<1));self.assertEqual(origins[1][4],(1<<5)|(1<<1)|(1<<4))

    def test_slice_join_keeps_each_declared_storage(self):
        prefix='PUSH_BOOL 1\nJMP_FALSE other\nARR_NEW 1\nJMP joined\nother:\nARR_NEW 2\njoined:\nPUSH_VOID\nPUSH_STR text\nARR_SLICE\n'
        fields,origins=self.analyze(self.program(prefix+'PUSH_I64 257\nARR_PUSH\nPOP'),vm=True)
        self.assertEqual(fields[3],4)
        self.assertEqual(sorted(o[2] for o in origins),[1,1,2,2])
        self.assertEqual(len({o[1] for o in origins}),3)
        self.analyze(self.program(prefix.replace('ARR_NEW 2','ARR_NEW 3')+'PUSH_U8 7\nARR_PUSH\nPOP'),1)

    def test_repeated_slice_site_and_recursive_source_summary(self):
        extra=('.function copy 1 1 0 array 1\nLOAD_LOCAL 0\nPUSH_I64 0\nPUSH_I64 9\nARR_SLICE\nRET\n.end\n')
        body=('PUSH_STR text\nARR_LITERAL 5 1\nCALL copy\nSTORE_LOCAL 0\n'
              'LOAD_LOCAL 0\nPUSH_BOOL 1\nARR_PUSH\nCALL copy\nPOP\n'
              'LOAD_LOCAL 0\nPUSH_I64 7\nARR_PUSH\nPOP')
        fields,origins=self.analyze(self.program(body,extra),vm=True)
        self.assertEqual(fields[3],2);self.assertEqual(origins[1][4],(1<<5)|(1<<4)|(1<<1))
        recursive=('.function copy 2 2 0 array 1\nLOAD_LOCAL 1\nPUSH_I64 0\nI64_EQ\nJMP_TRUE base\n'
                   'LOAD_LOCAL 0\nPUSH_I64 0\nPUSH_I64 9\nARR_SLICE\n'
                   'LOAD_LOCAL 1\nPUSH_I64 1\nI64_SUB\nCALL copy\nRET\nbase:\n'
                   'LOAD_LOCAL 0\nPUSH_STR text\nARR_PUSH\nRET\n.end\n')
        fields,origins=self.analyze(self.program('ARR_NEW 5\nPUSH_I64 2\nCALL copy\nPOP',recursive),vm=True)
        self.assertEqual(fields[3],2);self.assertTrue(all(o[4]==1<<5 for o in origins))

    def test_slice_globals_reentry_unknown_origins_and_limits(self):
        init='.function __init__ 0 0 0 void 0\nARR_NEW 5\nSTORE_GLOBAL 0\nRET\n.end\n'
        body=('LOAD_GLOBAL 0\nPUSH_I64 0\nPUSH_I64 9\nARR_SLICE\nPUSH_I64 2\nARR_PUSH\nPOP\n'
              'LOAD_GLOBAL 0\nPUSH_STR text\nARR_PUSH\nPOP')
        fields,origins=self.analyze(self.program(body,init),vm=True)
        self.assertEqual(fields[3],2);self.assertEqual(origins[1][4],(1<<1)|(1<<5))
        unused='.function unused 1 1 0 array 1\nLOAD_LOCAL 0\nPUSH_I64 0\nPUSH_I64 1\nARR_SLICE\nRET\n.end\n'
        self.analyze(self.program('',unused),1)
        self.analyze(self.program('ARR_NEW 5\n'+'PUSH_I64 0\nPUSH_I64 1\nARR_SLICE\n'*64+'POP'),3)
        self.analyze(self.program('PUSH_VOID\n'*257+'ARR_LITERAL 5 257\nPOP'),3)

    def test_copy_analysis_allocation_failure_and_shared_admission(self):
        text=self.program('PUSH_STR text\nARR_LITERAL 5 1\nPUSH_I64 0\nPUSH_I64 1\nARR_SLICE\nPOP')
        for budget in range(8):self.analyze(text,4,budget=budget)
        self.analyze(text,0,budget=8)
        source=self.work/'copy.nasm';source.write_text(text)
        module=self.work/'copy.nvm';self.command([ROOT/'bin/nanoisa','asm',source,'-o',module])
        for tool in ('nvm2llvm','nvm2wasm'):
            output=self.work/(tool+'.copy')
            self.command([ROOT/'bin'/tool,module,'-o',output])
            self.assertGreater(output.stat().st_size,0)

    def test_every_private_allocation_failure_is_atomic(self):
        text=self.program('ARR_NEW 5\nPUSH_STR text\nARR_PUSH\nPOP')
        # One analysis context, six function allocations, then report publication.
        for budget in range(8):self.analyze(text,4,budget=budget)
        self.analyze(text,0,budget=8)
    def test_unsupported_mutation_keeps_prior_outputs(self):
        source=self.work/'refuse.nasm';source.write_text(self.program('ARR_NEW 1\nPUSH_STR text\nARR_PUSH\nPOP'))
        module=self.work/'refuse.nvm';self.command([ROOT/'bin/nanoisa','asm',source,'-o',module])
        for tool in ('nvm2llvm','nvm2wasm'):
            output=self.work/(tool+'.old');output.write_bytes(b'prior output')
            p=subprocess.run([ROOT/'bin'/tool,module,'-o',output],capture_output=True,timeout=30)
            self.assertNotEqual(p.returncode,0);self.assertEqual(output.read_bytes(),b'prior output')
if __name__=='__main__':unittest.main()
