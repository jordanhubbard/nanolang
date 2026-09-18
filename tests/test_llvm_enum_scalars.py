"""I compare enum scalar semantics on the same VM/C/LLVM/Wasm modules."""
import signal
import unittest
from tests import test_nvm2wasm as wasm
from tests import test_nvm2llvm as llvm
from tests import test_native_enum_scalars as native_enum
from tests import test_native_typed_enum as native_typed
ROOT=wasm.ROOT


class EnumScalars(unittest.TestCase):
    setUp=wasm.ScalarWasm.setUp
    run_cmd=wasm.ScalarWasm.run_cmd
    module=wasm.ScalarWasm.module

    def program(self, body, helpers=''):
        return '.types 0 1 0\n.entry main\n'+helpers+'.function main 0 2 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'

    def compare(self, text, trap=False):
        ir=wasm.ScalarWasm.compare(self,text,trap)
        native=self.work/'sanitized-llvm'
        self.run_cmd(['clang','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all',ir,'-o',native])
        self.run_cmd([native],success=not trap)
        c=self.work/'program.c';binary=self.work/'sanitized-c'
        self.run_cmd(['cc','-std=c11','-O1','-Wall','-Wextra','-Werror',
                      '-fsanitize=address,undefined','-fno-sanitize-recover=all',c,'-o',binary])
        result=self.run_cmd([binary],success=not trap)
        if trap:
            self.assertEqual(result.returncode,-signal.SIGABRT,result.stderr)
            self.assertIn('I stopped at a native invariant',result.stderr)
            self.assertNotIn('Sanitizer',result.stderr)
            self.assertNotIn('runtime error:',result.stderr)
            self.run_cmd(['node','-e',
                'const fs=require("fs");const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));'
                'if(WebAssembly.Module.imports(m).length)process.exit(2);'
                'try{new WebAssembly.Instance(m).exports.nano_entry();process.exit(3);}'
                'catch(e){if(!(e instanceof WebAssembly.RuntimeError))throw e;}',self.work/'program.wasm'])
        return ir

    def paired(self, body, helpers=''):
        return self.compare(self.program(body,helpers))

    test_generic_arithmetic_matrix=native_enum.EnumScalars.test_arithmetic_matrix_and_tags
    test_typed_binary_matrix=native_typed.TypedEnum.test_all_typed_binary_operations_and_orders
    test_original_producer_tags=native_typed.TypedEnum.test_stored_and_duplicated_enum_tags_survive_consumption
    test_typed_wrapping_and_zero=native_typed.TypedEnum.test_wrapping_signed_division_and_zero_totality

    def test_comparison_cast_and_truthiness_compatibility(self):
        body=''
        for ordinal in (0,1,127,255,65535):
            body+=f'ENUM_VAL 0 {ordinal}\nDUP\nTYPE_CHECK 9\nASSERT\nCAST_INT\nPUSH_I64 {ordinal}\nEQ\nASSERT\n'
            body+=f'ENUM_VAL 0 {ordinal}\nCAST_FLOAT\nPUSH_F64 0\nF64_EQ\nASSERT\n'
            body+=f'ENUM_VAL 0 {ordinal}\nCAST_BOOL\nPUSH_BOOL {int(bool(ordinal))}\nEQ\nASSERT\n'
        for left,right in [('ENUM_VAL 0 2','ENUM_VAL 0 3'),('ENUM_VAL 0 3','ENUM_VAL 0 2')]:
            body+=f'{left}\n{right}\nNE\nASSERT\n{left}\n{right}\nLE\nASSERT\n{left}\n{right}\nGE\nASSERT\n'
        for left,right in [('ENUM_VAL 0 2','PUSH_I64 2'),('PUSH_I64 2','ENUM_VAL 0 2')]:
            body+=f'{left}\n{right}\nEQ\nASSERT\n'
        body+='ENUM_VAL 0 2\nPUSH_F64 2\nNE\nASSERT\nENUM_VAL 0 2\nPUSH_F64 99\nGT\nASSERT\n'
        body+='PUSH_F64 99\nENUM_VAL 0 2\nLT\nASSERT\n'
        self.paired(body)

    def test_globals_initializer_calls_returns_and_joins(self):
        helpers=('.function __init__ 0 0 0 void 0\nENUM_VAL 0 65535\nSTORE_GLOBAL 0\nRET\n.end\n'
                 '.function identity 1 1 0 enum 1\n.parameters identity enum\nLOAD_LOCAL 0\nRET\n.end\n')
        body='LOAD_GLOBAL 0\nCALL identity\nDUP\nTYPE_CHECK 9\nASSERT\nCAST_INT\nPUSH_I64 65535\nEQ\nASSERT\n'
        for flag in (0,1):
            body+=f'PUSH_BOOL {flag}\nJMP_FALSE other{flag}\nENUM_VAL 0 2\nJMP joined{flag}\nother{flag}:\nENUM_VAL 0 3\njoined{flag}:\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nCALL identity\nDUP\nTYPE_CHECK 9\nASSERT\nCAST_INT\nPUSH_I64 {2 if flag else 3}\nEQ\nASSERT\n'
        self.paired(body,helpers)

    def test_exact_refusals(self):
        bodies=['ENUM_VAL 0 2\nNEG\n','ENUM_VAL 0 2\nI64_NEG\n',
                'ENUM_VAL 0 2\nPUSH_I64 1\nMOD\n']
        for wrong in ('PUSH_BOOL 1','PUSH_U8 1','PUSH_F64 1','PUSH_VOID'):
            bodies.append(wrong+'\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nENUM_VAL 0 2\nI64_ADD\n')
        for body in bodies:
            with self.subTest(body=body):
                self.compare(self.program(body+'POP\n'),trap=True)

    def test_profile_refusals_preserve_output(self):
        for text in (
            self.program('ENUM_VAL 0 1\nCAST_STRING\nPOP\n'),
            self.program('CALL relay\nPOP\n','.function identity 0 0 0 enum 1\nENUM_VAL 0 1\nRET\n.end\n.function relay 0 0 0 enum 1\nTAIL_CALL identity\n.end\n'),
            '.types 1 1 0\n.entry main\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n',
            self.program('PUSH_STR s\nPOP\nENUM_VAL 0 2\nCAST_FLOAT\nPOP\n','.string s "kept"\n'),
        ):
            with self.subTest(text=text):
                module=self.module(text)
                for tool in (llvm.LLVM,wasm.WASM):
                    output=self.work/'previous';output.write_bytes(b'previous')
                    self.run_cmd([tool,module,'-o',output],success=False)
                    self.assertEqual(output.read_bytes(),b'previous')

    def test_literal_profile_preserves_enum_tag_order(self):
        self.paired('PUSH_STR text\nSTR_LEN\nPUSH_I64 4\nEQ\nASSERT\n'
                    'ENUM_VAL 0 2\nPUSH_STR text\nNE\nASSERT\n'
                    'ENUM_VAL 0 2\nPUSH_STR text\nGT\nASSERT\n'
                    'ENUM_VAL 0 2\nENUM_VAL 0 3\nI64_LT_S\nASSERT\n',
                    '.string text "word"\n')

    def test_declared_enum_result_checks_actual_tag(self):
        helpers=('.function identity 1 1 0 enum 1\n.parameters identity enum\n'
                 'LOAD_LOCAL 0\nRET\n.end\n')
        self.compare(self.program('PUSH_U8 2\nCALL identity\nPOP\n',helpers),trap=True)
