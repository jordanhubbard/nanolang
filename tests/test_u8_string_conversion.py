"""I convert bytes to owned decimal strings without broadening LLVM strings."""
import tempfile
import unittest
from pathlib import Path
from tests import test_nvm2llvm as llvm
from tests import test_nvm2wasm as wasm


class U8Strings(unittest.TestCase):
    run_cmd = llvm.ScalarLLVM.run_cmd
    module = llvm.ScalarLLVM.module

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='nano-byte-string-')
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def paired(self, text):
        module = self.module(text)
        self.run_cmd([llvm.VM, '--verify-only', module])
        vm = self.run_cmd([llvm.VM, module])
        c, exe = self.work/'out.c', self.work/'out'
        self.run_cmd([llvm.C, module, '-o', c])
        self.run_cmd(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', c, '-o', exe])
        native = self.run_cmd([exe])
        self.assertEqual(native.stdout, vm.stdout)
        return vm.stdout

    def test_all_unsigned_decimal_values_and_tags(self):
        body = ''
        for value in range(256):
            body += f'PUSH_U8 {value}\nCAST_STRING\nDUP\nTYPE_CHECK 5\nASSERT\nDUP\nPRINTLN\n'
            body += f'PUSH_U8 {value}\nCAST_INT\nCAST_STRING\nEQ\nASSERT\n'
        text = '.entry main\n.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'
        self.assertEqual(self.paired(text), ''.join(f'{v}\n' for v in range(256)))

    def test_returned_alias_survives_conversion_churn(self):
        body = 'PUSH_U8 255\nCALL render\nDUP\nSTORE_LOCAL 0\nSTORE_LOCAL 1\n'
        # Each helper call reaches normal native collection safepoints while
        # two caller roots keep the returned string alive.
        for value in range(256):
            body += f'PUSH_U8 {value}\nCALL render\nPOP\n'
        body += ('PUSH_I64 0\nSTORE_LOCAL 2\nchurn:\n'
                 'PUSH_U8 128\nCALL render\nPOP\n'
                 'LOAD_LOCAL 2\nPUSH_I64 1\nI64_ADD\nDUP\nSTORE_LOCAL 2\n'
                 'PUSH_I64 20000\nI64_LT_S\nJMP_TRUE churn\n'
                 'LOAD_LOCAL 0\nPRINTLN\nLOAD_LOCAL 1\nPRINTLN\n')
        text = ('.entry main\n.function main 0 3 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'
                '.function render 1 1 0 string 1\n.parameters render u8\nLOAD_LOCAL 0\nCAST_STRING\nRET\n.end\n')
        self.assertEqual(self.paired(text), '255\n255\n')

    def test_llvm_wasm_string_refusal_preserves_output(self):
        module = self.module('.entry main\n.function main 0 0 0 int 1\nPUSH_U8 128\nCAST_STRING\nPOP\nPUSH_I64 0\nRET\n.end\n')
        for translator in (llvm.LLVM, wasm.WASM):
            output = self.work/'previous'
            output.write_bytes(b'previous')
            result = self.run_cmd([translator, module, '-o', output], success=False)
            self.assertIn('do not support opcode', result.stderr)
            self.assertEqual(output.read_bytes(), b'previous')


if __name__ == '__main__':
    unittest.main()
