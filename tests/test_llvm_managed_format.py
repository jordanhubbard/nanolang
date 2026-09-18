"""I preserve exact formatting and floating instruction/signature publication."""
import unittest
from tests import test_llvm_managed_strings as managed
ROOT = managed.ROOT


class ManagedFormat(unittest.TestCase):
    setUp = managed.ManagedStrings.setUp
    run_cmd = managed.ManagedStrings.run_cmd
    program = managed.ManagedStrings.program
    compile = managed.ManagedStrings.compile
    native_harness = managed.ManagedStrings.native_harness
    node = managed.ManagedStrings.node

    def test_exact_scalar_bytes_and_string_owner_transfer(self):
        cases=[('PUSH_I64 0','0'),('PUSH_I64 -1','-1'),('PUSH_I64 42','42'),
               ('PUSH_I64 9223372036854775807','9223372036854775807'),
               ('PUSH_I64 -9223372036854775808','-9223372036854775808'),
               ('PUSH_U8 0','0'),('PUSH_U8 255','255'),('PUSH_BOOL 0','false'),
               ('PUSH_BOOL 1','true'),('PUSH_VOID',''),('ENUM_VAL 0 17','')]
        strings,body='',''
        for i,(value,expected) in enumerate(cases):
            strings += f'.string expected{i} "{expected}"\n'
            body += f'{value}\nCAST_STRING\nPUSH_STR expected{i}\nEQ\nASSERT\n'
        body += ('PUSH_STR a\nCAST_STRING\nPUSH_STR a\nEQ\nASSERT\n'
                 'PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nDUP\nSTORE_GLOBAL 0\n'
                 'CALL identity\nDUP\nSTORE_LOCAL 0\nPUSH_STR a\nEQ\nASSERT\n'
                 'LOAD_GLOBAL 0\nLOAD_LOCAL 0\nEQ\nASSERT\n')
        suffix='.function identity 1 1 0 string 1\n.parameters identity string\nLOAD_LOCAL 0\nCAST_STRING\nRET\n.end\n'
        _,ir,wasm=self.compile(strings+self.program(body,suffix))
        self.native_harness(ir,'for(int i=0;i<20;i++)if(nano_try_entry()||nms_module_live_objects()!=1||nms_module_live_bytes()!=3)return 1;return nano_dispose();')
        self.node(wasm,'for(let i=0;i<20;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===1n);check(e.nms_module_live_bytes()===3n);}check(e.nano_dispose()===0);')

    def test_allocation_failure_and_allocation_free_identity(self):
        # No literal strings are needed to select the managed formatting profile.
        _,ir,wasm=self.compile('.entry main\n.function main 0 0 0 int 1\nPUSH_I64 -9223372036854775808\nCAST_STRING\nPOP\nPUSH_I64 0\nRET\n.end\n')
        extra='static long budget=-1;extern void *__real_malloc(size_t);void *__wrap_malloc(size_t n){if(!budget)return 0;if(budget>0)--budget;return __real_malloc(n);}'
        for fail in (0,1):
            self.native_harness(ir,f'budget={fail};if(nano_try_entry()!=((uint64_t)3<<32)||nms_module_live_objects())return 1;budget=-1;if(nano_try_entry()||nms_module_live_objects())return 2;return nano_dispose();',extra,['-Wl,--wrap=malloc'])
        self.node(wasm,'for(let i=0;i<20;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===0n);}check(e.nano_dispose()===0);')
        body=('LOAD_GLOBAL 0\nPUSH_VOID\nEQ\nJMP_FALSE ready\nPUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nSTORE_GLOBAL 0\nready:\n'
              'LOAD_GLOBAL 0\nCAST_STRING\nPUSH_STR a\nEQ\nASSERT\n')
        _,ir,_=self.compile(self.program(body))
        self.native_harness(ir,'if(nano_try_entry())return 1;budget=0;for(int i=0;i<20;i++)if(nano_try_entry()||nms_module_live_objects()!=1)return 2;return nano_dispose();',extra,['-Wl,--wrap=malloc'])

    def test_float_instruction_and_signature_publication(self):
        base=self.program('PUSH_I64 1\nCAST_STRING\nPOP\n')
        bodies=['PUSH_F64 1.0\nPOP\n','PUSH_I64 1\nCAST_FLOAT\nPOP\n']
        for op in ('F64_ADD','F64_SUB','F64_MUL','F64_DIV','F64_NEG',
                   'F64_EQ','F64_NE','F64_LT','F64_LE','F64_GT','F64_GE'):
            bodies.append('LOAD_GLOBAL 0\n'+('LOAD_GLOBAL 0\n' if op!='F64_NEG' else '')+op+'\nPOP\n')
        suffixes=['.function unused 0 0 0 void 0\n'+body+'RET\n.end\n' for body in bodies]
        suffixes += ['.function unused 1 1 0 void 0\n.parameters unused float\nRET\n.end\n',
                     '.function unused 0 0 0 float 1\nLOAD_GLOBAL 0\nRET\n.end\n']
        for suffix in suffixes:
            with self.subTest(suffix=suffix):
                asm,mod=self.work/'refuse.nasm',self.work/'refuse.nvm'
                asm.write_text(base+suffix)
                self.run_cmd([ROOT/'bin/nanoisa','asm',asm,'-o',mod])
                self.run_cmd([ROOT/'bin/nano_vm',mod])
                for tool in ('nvm2llvm','nvm2wasm'):
                    out=self.work/'previous';out.write_bytes(b'previous')
                    self.run_cmd([ROOT/'bin'/tool,mod,'-o',out])
                    self.assertNotEqual(out.read_bytes(),b'previous')


if __name__=='__main__':
    unittest.main()
