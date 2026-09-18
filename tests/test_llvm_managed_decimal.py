"""I compare closed C-locale managed decimal conversions across real targets."""
import unittest
from tests import test_llvm_managed_strings as managed


class ManagedDecimal(unittest.TestCase):
    setUp = managed.ManagedStrings.setUp
    run_cmd = managed.ManagedStrings.run_cmd
    program = managed.ManagedStrings.program
    compile = managed.ManagedStrings.compile
    native_harness = managed.ManagedStrings.native_harness
    node = managed.ManagedStrings.node

    def test_decimal_bytes_limits_and_dynamic_aliases(self):
        cases = [(b'',0),(b' ',0),(b'+',0),(b'-',0),(b'++1',0),(b'word',0),
                 (b'00017tail',17),(b'  -42rest',-42),(b'+17',17),(b'-0',0),
                 (b'0x20',0),(b'12\x0099',12),(b'\x0012',0),(b'1.5',1),
                 (b'9223372036854775807',2**63-1),(b'9223372036854775808',2**63-1),
                 (b'-9223372036854775808',-2**63),(b'-9223372036854775809',-2**63),
                 (b'999999999999999999999999999999',2**63-1),
                 (b'-999999999999999999999999999999',-2**63)]
        cases += [(bytes([c])+b'23',23) for c in b' \t\r\n\v\f']
        strings, body = '', ''
        for i,(text,expected) in enumerate(cases):
            escaped=''.join('\\x%02x'%c for c in text)
            strings += f'.string s{i} "{escaped}"\n'
            body += (f'PUSH_STR s{i}\nCAST_INT\nPUSH_I64 {expected}\nEQ\nASSERT\n'
                     f'PUSH_STR s{i}\nPUSH_STR empty\nSTR_CONCAT\nDUP\nSTORE_LOCAL 0\n'
                     f'CALL parse\nPUSH_I64 {expected}\nEQ\nASSERT\n'
                     f'LOAD_LOCAL 0\nCAST_INT\nPUSH_I64 {expected}\nEQ\nASSERT\n')
        suffix=('.function parse 1 1 0 int 1\n.parameters parse string\n'
                'LOAD_LOCAL 0\nCAST_INT\nRET\n.end\n')
        _,ir,wasm=self.compile(strings+self.program(body,suffix))
        self.native_harness(ir,'for(int i=0;i<20;i++)if(nano_try_entry()||nms_module_live_objects())return 1;return nano_dispose();')
        self.node(wasm,'for(let i=0;i<20;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===0n);}check(e.nano_dispose()===0);')

    def test_globals_nonstrings_and_error_cleanup(self):
        body=('PUSH_STR number\nPUSH_STR empty\nSTR_CONCAT\nDUP\nSTORE_GLOBAL 0\n'
              'CAST_INT\nPUSH_I64 -42\nEQ\nASSERT\nLOAD_GLOBAL 0\nCAST_INT\nPUSH_I64 -42\nEQ\nASSERT\n')
        for value,expected in [('PUSH_BOOL 1',1),('PUSH_U8 255',255),('PUSH_VOID',0),
                               ('PUSH_I64 -17',-17),('PUSH_F64 -1.75',-1),('ENUM_VAL 0 9',9)]:
            body += f'{value}\nCAST_INT\nPUSH_I64 {expected}\nEQ\nASSERT\n'
        _,ir,wasm=self.compile('.string number "-42suffix"\n'+self.program(body))
        self.native_harness(ir,'for(int i=0;i<20;i++)if(nano_try_entry()||nms_module_live_objects()!=1)return 1;return nano_dispose();')
        self.node(wasm,'for(let i=0;i<20;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===1n);}check(e.nano_dispose()===0);')
        _,ir,wasm=self.compile('.string number "-42suffix"\n'+self.program(body+'PUSH_F64 inf\nCAST_INT\nPOP\n'),vm_ok=False)
        self.native_harness(ir,'for(int i=0;i<5;i++)if(nano_try_entry()!=((uint64_t)1<<32)||nms_module_live_objects()!=1)return 1;return nano_dispose();')
        self.node(wasm,'for(let i=0;i<5;i++){check(e.nano_try_entry()===(1n<<32n));check(e.nms_module_live_objects()===1n);}check(e.nano_dispose()===0);')


if __name__=='__main__':
    unittest.main()
