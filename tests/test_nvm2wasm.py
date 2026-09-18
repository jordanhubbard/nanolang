"""I execute the same scalar modules in VM, C, LLVM and freestanding Wasm."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests import test_nvm2llvm as llvm_tests
from tests import test_nvm2llvm_floats as float_tests

ROOT = Path(__file__).resolve().parents[1]
WASM = ROOT / 'bin/nvm2wasm'


class ScalarWasm(unittest.TestCase):
    run_cmd = llvm_tests.ScalarLLVM.run_cmd
    module = llvm_tests.ScalarLLVM.module

    def setUp(self):
        for tool in ('llvm-as', 'lli', 'opt', 'llc', 'wasm-ld', 'wasmtime', 'node', 'cc'):
            self.assertIsNotNone(shutil.which(tool), tool + ' is required')
        self.tmp = tempfile.TemporaryDirectory(prefix='nano wasm ')
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def compare(self, text, trap=False):
        ir = llvm_tests.ScalarLLVM.compare(self, text, trap)
        module, wasm = self.work / 'input.nvm', self.work / 'program.wasm'
        self.run_cmd([WASM, module, '-o', wasm])
        self.assertEqual(wasm.read_bytes()[:8], b'\0asm\x01\0\0\0')
        result = self.run_cmd(['wasmtime', 'run', '--invoke', 'nano_entry', wasm], success=not trap)
        if not trap:
            self.assertEqual(result.stdout, '0\n')
            # I inspect imports and instantiate without any host-provided imports.
            self.run_cmd(['node', '-e',
                'const fs=require("fs"); const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));'
                'if(WebAssembly.Module.imports(m).length) process.exit(2);'
                'if(new WebAssembly.Instance(m).exports.nano_entry()!==0) process.exit(3);', wasm])
        return ir

    test_integer_boundaries = llvm_tests.ScalarLLVM.test_integer_boundaries
    test_calls_loops_branch_effects_and_bool_transport = llvm_tests.ScalarLLVM.test_calls_loops_branch_effects_and_bool_transport
    test_scalar_tags_boolean_ops_and_argument_order = llvm_tests.ScalarLLVM.test_scalar_tags_boolean_ops_and_argument_order
    test_recursive_call_and_stack_join = llvm_tests.ScalarLLVM.test_recursive_call_and_stack_join

    program = float_tests.LLVMFloats.program
    test_float_arithmetic_comparisons_and_nan = float_tests.LLVMFloats.test_arithmetic_comparisons_and_nan
    test_float_calls_locals_joins_truthiness_and_signed_zero = float_tests.LLVMFloats.test_float_calls_locals_joins_truthiness_and_signed_zero
    test_checked_float_conversions = float_tests.LLVMFloats.test_checked_integer_and_float_conversions
    test_invalid_float_to_int_traps = float_tests.LLVMFloats.test_invalid_float_to_int_traps_before_conversion

    def test_scalar_entry_result_and_custom_name(self):
        module = self.module('.entry main\n.function main 0 0 0 int 1\nPUSH_I64 7\nRET\n.end\n')
        wasm, ir = self.work / 'status.wasm', self.work / 'status.ll'
        self.run_cmd([WASM, module, '-o', wasm])
        result = self.run_cmd(['wasmtime', 'run', '--invoke', 'nano_entry', wasm])
        self.assertEqual(result.stdout, '7\n')
        vm = subprocess.run([llvm_tests.VM, module], capture_output=True, timeout=30)
        self.assertEqual(vm.returncode, 7, vm.stderr)
        self.run_cmd([llvm_tests.LLVM, module, '--entry-name', 'nano_status', '-o', ir])
        llvm = subprocess.run(['lli', '--entry-function=nano_status', ir],
                              capture_output=True, timeout=30)
        self.assertEqual(llvm.returncode, 7, llvm.stderr)
        for name in ('check', 'nano_bad-name'):
            with self.subTest(name=name):
                ir.write_text('previous')
                self.run_cmd([llvm_tests.LLVM, module, '--entry-name', name, '-o', ir], success=False)
                self.assertEqual(ir.read_text(), 'previous')

    def test_output_and_source_preservation(self):
        module = self.module('.entry main\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n')
        output = self.work / 'retained.wasm'
        source = module.read_bytes()
        self.run_cmd([WASM, module, '-o', module], success=False)
        self.assertEqual(module.read_bytes(), source)
        alias = self.work / 'alias.nvm'
        os.link(module, alias)
        self.run_cmd([WASM, module, '-o', alias], success=False)
        self.assertEqual(module.read_bytes(), source)
        for variable in ('NANO_NVM2LLVM', 'NANO_LLC', 'NANO_WASM_LD'):
            with self.subTest(tool=variable):
                output.write_bytes(b'previous')
                result = subprocess.run([WASM, module, '-o', output], capture_output=True,
                    timeout=30, env={**os.environ, variable: str(self.work / 'absent-tool')})
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output.read_bytes(), b'previous')
                self.assertEqual(list(self.work.glob('.nano-wasm-*')), [])
        self.run_cmd([WASM, module, '-o', output])
        stdout = subprocess.run([WASM, module], capture_output=True, timeout=30)
        self.assertEqual(stdout.returncode, 0, stdout.stderr)
        self.assertEqual(stdout.stdout, output.read_bytes())

    def test_shared_profile_refusals_preserve_output(self):
        for extra, body in (
            ('.string outside "outside profile"\n', 'PUSH_STR outside\nDUP\nSTR_CONCAT\nPOP\nPUSH_I64 0\nRET\n'),
            ('.types 1 0 0\n', 'PUSH_I64 0\nRET\n'),
            ('.import "" "get_argc" int\n', 'PUSH_I64 0\nRET\n'),
        ):
            with self.subTest(body=body, extra=extra):
                module = self.module(extra + '.entry main\n.function main 0 0 0 int 1\n' + body + '.end\n')
                output = self.work / 'prior.wasm'
                output.write_bytes(b'previous')
                self.run_cmd([WASM, module, '-o', output], success=False)
                self.assertEqual(output.read_bytes(), b'previous')
                self.assertEqual(list(self.work.glob('.nano-wasm-*')), [])


if __name__ == '__main__':
    unittest.main()
