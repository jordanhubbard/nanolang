"""I preserve byte identity and exact unsigned conversions across scalar backends."""
import signal
import unittest
from tests import test_nvm2llvm as llvm
from tests import test_nvm2wasm as wasm


class ScalarU8(unittest.TestCase):
    setUp = wasm.ScalarWasm.setUp
    run_cmd = wasm.ScalarWasm.run_cmd
    module = wasm.ScalarWasm.module
    compare = wasm.ScalarWasm.compare

    def program(self, body, suffix=''):
        return '.entry main\n.function main 0 2 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'+suffix

    def test_exact_conversions_tags_and_unsigned_order(self):
        body = ''
        for value in (0, 1, 127, 128, 254, 255):
            body += f'PUSH_U8 {value}\nDUP\nTYPE_CHECK 2\nASSERT\nDUP\nTYPE_CHECK 1\nBOOL_NOT\nASSERT\nSTORE_LOCAL 0\n'
            body += f'LOAD_LOCAL 0\nCAST_INT\nPUSH_I64 {value}\nI64_EQ\nASSERT\n'
            body += f'LOAD_LOCAL 0\nCAST_FLOAT\nDUP\nTYPE_CHECK 3\nASSERT\nPUSH_F64 {value}.0\nF64_EQ\nASSERT\n'
            body += 'LOAD_LOCAL 0\nCAST_BOOL\n'+('BOOL_NOT\n' if value == 0 else '')+'ASSERT\n'
            body += f'LOAD_LOCAL 0\nPUSH_U8 1\nOR\nASSERT\nLOAD_LOCAL 0\nPUSH_U8 0\nAND\nBOOL_NOT\nASSERT\n'
        for a, b in ((0, 255), (127, 128), (254, 255), (255, 255)):
            body += f'PUSH_U8 {a}\nCAST_INT\nPUSH_U8 {b}\nCAST_INT\nI64_LE_S\nASSERT\n'
        self.compare(self.program(body))

    def test_calls_locals_branch_and_implicit_result_transport(self):
        suffix = ('.function identity 1 1 0 u8 1\n.parameters identity u8\nLOAD_LOCAL 0\n.end\n'
                  '.function choose 1 1 0 u8 1\n.parameters choose bool\nLOAD_LOCAL 0\nJMP_FALSE low\n'
                  'PUSH_U8 255\nJMP done\nlow:\nPUSH_U8 128\ndone:\nCALL identity\nRET\n.end\n')
        body = ''
        for flag, expected in ((0, 128), (1, 255)):
            body += f'PUSH_BOOL {flag}\nCALL choose\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nTYPE_CHECK 2\nASSERT\n'
            body += f'LOAD_LOCAL 1\nCAST_INT\nPUSH_I64 {expected}\nI64_EQ\nASSERT\n'
        self.compare(self.program(body, suffix))

    def test_generic_byte_order_across_scalar_backends(self):
        body = ''
        for a, b in ((0, 255), (128, 127), (255, 255)):
            for op, expected in (('EQ', a == b), ('NE', a != b), ('LT', a < b),
                                 ('LE', a <= b), ('GT', a > b), ('GE', a >= b)):
                body += f'PUSH_U8 {a}\nPUSH_U8 {b}\n{op}\nDUP\nTYPE_CHECK 4\nASSERT\n'
                body += ('BOOL_NOT\n' if not expected else '')+'ASSERT\n'
        self.compare(self.program(body))

    def test_native_byte_display_and_unused_signature(self):
        text = self.program('PUSH_U8 255\nPRINTLN\nPUSH_U8 128\nPRINTLN\n',
            '.function unused 1 1 0 u8 1\n.parameters unused u8\nLOAD_LOCAL 0\nRET\n.end\n')
        module = self.module(text)
        vm = self.run_cmd([llvm.VM, module])
        self.assertEqual(vm.stdout, '255\n128\n')
        c, executable = self.work/'display.c', self.work/'display'
        self.run_cmd([llvm.C, module, '-o', c])
        self.run_cmd(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', c, '-o', executable])
        self.assertEqual(self.run_cmd([executable]).stdout, vm.stdout)

    def test_declared_byte_argument_retains_runtime_tag_guard(self):
        suffix = ('.function checked 1 1 0 int 1\n.parameters checked u8\n'
                  'LOAD_LOCAL 0\nTYPE_CHECK 2\nASSERT\nPUSH_I64 0\nRET\n.end\n')
        self.compare(self.program('PUSH_I64 255\nCALL checked\nPOP\n', suffix), trap=True)

    def test_declared_byte_result_checks_actual_return_tag(self):
        suffix = ('.function wrong 1 1 0 u8 1\n.parameters wrong int\n'
                  'LOAD_LOCAL 0\nRET\n.end\n')
        module = self.module(self.program('PUSH_I64 255\nCALL wrong\nPOP\n', suffix))
        self.run_cmd([llvm.VM, module], success=False)
        ir, target = self.work/'wrong.ll', self.work/'wrong.wasm'
        self.run_cmd([llvm.LLVM, module, '-o', ir])
        self.run_cmd(['lli', ir], success=False)
        self.run_cmd([wasm.WASM, module, '-o', target])
        self.run_cmd(['wasmtime', 'run', '--invoke', 'nano_entry', target], success=False)
        c, native = self.work/'wrong.c', self.work/'wrong-native'
        self.run_cmd([llvm.C, module, '-o', c])
        self.run_cmd(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                      '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                      c, '-o', native])
        refusal = self.run_cmd([native], success=False)
        self.assertEqual(refusal.returncode, -signal.SIGABRT, refusal.stderr)
        self.assertIn('I stopped at a native invariant', refusal.stderr)
        self.assertNotIn('Sanitizer', refusal.stderr)
        self.assertNotIn('runtime error:', refusal.stderr)

    def test_typed_integer_instruction_still_checks_byte_tag(self):
        suffix = '.function add_one 1 1 0 int 1\n.parameters add_one int\nLOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nRET\n.end\n'
        self.compare(self.program('PUSH_U8 1\nCALL add_one\nPOP\n', suffix), trap=True)


if __name__ == '__main__':
    unittest.main()
