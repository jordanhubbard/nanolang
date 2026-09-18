"""I preserve zero-result calls and scalar implicit exits in shared lowering."""
import unittest
from tests import test_nvm2llvm as llvm
from tests import test_nvm2wasm as wasm


class LLVMImplicitReturns(unittest.TestCase):
    setUp = wasm.ScalarWasm.setUp
    run_cmd = wasm.ScalarWasm.run_cmd
    module = wasm.ScalarWasm.module
    compare = wasm.ScalarWasm.compare

    def test_empty_explicit_void_and_caller_operands(self):
        self.compare('.entry main\n.function main 0 0 0 int 1\nPUSH_I64 42\nCALL empty\n'
            'PUSH_I64 7\nCALL checked_void\nCALL explicit_void\nPUSH_I64 42\nI64_EQ\nASSERT\nPUSH_I64 0\n.end\n'
            '.function empty 0 0 0 void 0\n.end\n'
            '.function checked_void 1 1 0 void 0\n.parameters checked_void int\n'
            'LOAD_LOCAL 0\nPUSH_I64 7\nI64_EQ\nASSERT\n.end\n'
            '.function explicit_void 0 0 0 void 0\nRET\n.end\n')

    def test_nested_scalar_float_bool_implicit_results(self):
        self.compare('.entry main\n.function main 0 0 0 int 1\nCALL outer\nPUSH_I64 12\nI64_EQ\nASSERT\n'
            'CALL real\nPUSH_F64 -2.5\nF64_EQ\nASSERT\nCALL truth\nDUP\nTYPE_CHECK 4\nASSERT\nASSERT\nPUSH_I64 0\n.end\n'
            '.function outer 0 0 0 int 1\nCALL inner\nPUSH_I64 3\nI64_ADD\n.end\n'
            '.function inner 0 0 0 int 1\nPUSH_I64 9\n.end\n'
            '.function real 0 0 0 float 1\nPUSH_F64 -2.5\n.end\n'
            '.function truth 0 0 0 bool 1\nPUSH_BOOL 1\n.end\n')

    def test_recursive_void_and_conditional_end_edges(self):
        self.compare('.entry main\n.function main 0 0 0 int 1\nPUSH_I64 12\nCALL descend\n'
            'PUSH_BOOL 0\nCALL choose\nPUSH_I64 9\nI64_EQ\nASSERT\n'
            'PUSH_BOOL 1\nCALL choose\nPUSH_I64 17\nI64_EQ\nASSERT\nPUSH_I64 0\nRET\n.end\n'
            '.function descend 1 1 0 void 0\n.parameters descend int\nLOAD_LOCAL 0\nPUSH_I64 0\n'
            'I64_GT_S\nJMP_FALSE end\nLOAD_LOCAL 0\nPUSH_I64 1\nI64_SUB\nCALL descend\nend:\n.end\n'
            '.function choose 1 1 0 int 1\n.parameters choose bool\nLOAD_LOCAL 0\nJMP_TRUE high\n'
            'PUSH_I64 9\nJMP done\nhigh:\nPUSH_I64 17\ndone:\n.end\n')

    def test_implicit_result_tag_check_traps(self):
        module=self.module('.entry main\n.function main 0 0 0 int 1\nPUSH_BOOL 1\n.end\n')
        self.run_cmd([llvm.VM,'--verify-only',module])
        self.run_cmd([llvm.VM,module],success=False)
        ir=self.work/'invalid-tag.ll'
        self.run_cmd([llvm.LLVM,module,'-o',ir])
        self.run_cmd(['lli',ir],success=False)
        bitcode=self.work/'invalid-tag.bc'
        self.run_cmd(['opt','-passes=default<O2>',ir,'-o',bitcode])
        self.run_cmd(['lli',bitcode],success=False)
        target=self.work/'invalid-tag.wasm'
        self.run_cmd([wasm.WASM,module,'-o',target])
        self.run_cmd(['wasmtime','run','--invoke','nano_entry',target],success=False)
        self.run_cmd(['node','-e',
            'const fs=require("fs"),m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));'
            'try{new WebAssembly.Instance(m).exports.nano_entry();process.exit(1)}'
            'catch(e){if(!(e instanceof WebAssembly.RuntimeError))process.exit(2)}',target])
        self.run_cmd([llvm.C,module],success=False)

    def test_unsupported_result_profiles_preserve_output(self):
        texts=[
            '.entry main\n.function main 0 0 0 void 0\n.end\n',
            '.entry main\n.function main 0 0 0 float 1\nPUSH_F64 0.0\n.end\n',
            '.entry main\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n'
            '.function unused 0 0 0 int 2\nPUSH_I64 1\nPUSH_I64 2\n.end\n',
            '.string text "text"\n.entry main\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n'
            '.function unused 0 0 0 array 1\nARR_NEW 1\n.end\n',
        ]
        for text in texts:
            with self.subTest(text=text):
                module=self.module(text)
                self.run_cmd([llvm.VM,'--verify-only',module])
                for translator in (llvm.LLVM,wasm.WASM):
                    target=self.work/'prior';target.write_bytes(b'previous')
                    self.run_cmd([translator,module,'-o',target],success=False)
                    self.assertEqual(target.read_bytes(),b'previous')
                    self.assertEqual(list(self.work.glob('.nano-wasm-*')),[])


if __name__ == '__main__':
    unittest.main()
