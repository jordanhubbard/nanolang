"""I execute one verified module on VM, C AOT and my scalar LLVM translator."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
VM = Path(os.environ.get('NANOLANG_LLVM_VM', ROOT / 'bin/nano_vm'))
C = Path(os.environ.get('NANOLANG_LLVM_C', ROOT / 'bin/nvm2c'))
LLVM = Path(os.environ.get('NANOLANG_LLVM_TRANSLATOR', ROOT / 'bin/nvm2llvm'))


class ScalarLLVM(unittest.TestCase):
    def run_cmd(self, args, success=True):
        p = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=30)
        if success:
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        else:
            self.assertNotEqual(p.returncode, 0, str(args))
        return p

    def setUp(self):
        # This is an execution gate: absent LLVM tools are failures, not coverage.
        for tool in ('llvm-as', 'lli', 'opt', 'llc', 'cc'):
            self.assertIsNotNone(shutil.which(tool), tool + ' is required')
        self.tmp = tempfile.TemporaryDirectory(prefix='nano-llvm-')
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def module(self, text):
        asm, module = self.work/'input.nasm', self.work/'input.nvm'
        asm.write_text(text)
        self.run_cmd([ROOT/'bin/nanoisa', 'asm', asm, '-o', module])
        return module

    def compare(self, text, trap=False):
        module = self.module(text)
        self.run_cmd([VM, '--verify-only', module])
        vm = self.run_cmd([VM, module], success=not trap)
        ir = self.work/'program.ll'
        self.run_cmd([LLVM, module, '-o', ir])
        self.assertNotIn('nano_vm', ir.read_text())
        self.assertEqual(self.run_cmd([LLVM, module]).stdout, ir.read_text())
        self.run_cmd(['llvm-as', ir, '-o', self.work/'program.bc'])
        llvm = self.run_cmd(['lli', ir], success=not trap)
        optimized = self.work/'optimized.bc'
        self.run_cmd(['opt', '-passes=default<O2>', ir, '-o', optimized])
        optimized_run = self.run_cmd(['lli', optimized], success=not trap)
        self.assertEqual(llvm.stdout, optimized_run.stdout)
        obj, llvm_native = self.work/'llvm.o', self.work/'llvm-native'
        self.run_cmd(['llc', '-filetype=obj', '-relocation-model=pic', optimized, '-o', obj])
        self.run_cmd(['cc', obj, '-o', llvm_native])
        machine = self.run_cmd([llvm_native], success=not trap)
        self.assertEqual(vm.stdout, machine.stdout)
        source, exe = self.work/'program.c', self.work/'program'
        translated = subprocess.run([C, module, '-o', source], capture_output=True, text=True, timeout=30)
        if trap and translated.returncode:
            self.assertIn('require integer operands', translated.stderr)
            return ir
        self.assertEqual(translated.returncode, 0, translated.stderr)
        self.run_cmd(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', source, '-o', exe])
        native = self.run_cmd([exe], success=not trap)
        self.assertEqual(vm.stdout, native.stdout)
        self.assertEqual(vm.stdout, llvm.stdout)
        return ir

    def test_verified_implicit_entry(self):
        self.compare('.entry main\n.function main 0 0 0 int 1\nPUSH_I64 0\n.end\n')

    def test_integer_boundaries(self):
        checks = []
        cases = [
            ('I64_ADD', 9223372036854775807, 1, -9223372036854775808),
            ('I64_SUB', -9223372036854775808, 1, 9223372036854775807),
            ('I64_MUL', 9223372036854775807, 2, -2),
            ('I64_DIV_S', -9223372036854775808, -1, -9223372036854775808),
            ('I64_REM_S', -9223372036854775808, -1, 0),
            ('I64_DIV_S', 8, 0, 0), ('I64_REM_S', 8, 0, 0),
            ('I64_DIV_S', -17, 5, -3), ('I64_REM_S', -17, 5, -2),
        ]
        for op, a, b, answer in cases:
            checks.append(f'PUSH_I64 {a}\nPUSH_I64 {b}\n{op}\nPUSH_I64 {answer}\nI64_EQ\nASSERT\n')
        checks.append('PUSH_I64 -9223372036854775808\nI64_NEG\nPUSH_I64 -9223372036854775808\nI64_EQ\nASSERT\n')
        self.compare('.entry main\n.function main 0 0 0 int 1\n'+''.join(checks)+'PUSH_I64 0\nRET\n.end\n')

    def test_calls_loops_branch_effects_and_bool_transport(self):
        self.compare('''.entry main
.function main 0 2 0 int 1
PUSH_I64 0
STORE_LOCAL 0
PUSH_I64 0
STORE_LOCAL 1
loop:
LOAD_LOCAL 0
PUSH_I64 6
I64_LT_S
JMP_FALSE done
LOAD_LOCAL 1
LOAD_LOCAL 0
CALL square
I64_ADD
STORE_LOCAL 1
LOAD_LOCAL 0
PUSH_I64 1
I64_ADD
STORE_LOCAL 0
JMP loop
done:
LOAD_LOCAL 1
PUSH_I64 55
I64_EQ
ASSERT
PUSH_BOOL 1
CALL invert
BOOL_NOT
ASSERT
PUSH_BOOL 1
JMP_TRUE taken
PUSH_BOOL 0
ASSERT
taken:
PUSH_I64 0
RET
.end
.function square 1 1 0 int 1
.parameters square int
LOAD_LOCAL 0
DUP
I64_MUL
RET
.end
.function invert 1 1 0 bool 1
.parameters invert bool
LOAD_LOCAL 0
BOOL_NOT
RET
.end
''')

    def test_scalar_tags_boolean_ops_and_argument_order(self):
        self.compare(""".entry main
.function main 0 1 0 int 1
PUSH_BOOL 1
PUSH_BOOL 0
BOOL_OR
PUSH_BOOL 1
BOOL_AND
ASSERT
PUSH_BOOL 0
BOOL_NOT
TYPE_CHECK 4
ASSERT
PUSH_I64 9
PUSH_I64 4
CALL difference
PUSH_I64 5
I64_EQ
ASSERT
PUSH_I64 1
PUSH_I64 2
SWAP
I64_GT_S
ASSERT
PUSH_I64 2
PUSH_I64 2
I64_GE_S
ASSERT
PUSH_I64 1
PUSH_I64 2
I64_NE
ASSERT
PUSH_BOOL 0
JMP_TRUE wrong
PUSH_I64 0
RET
wrong:
PUSH_BOOL 0
ASSERT
PUSH_I64 1
RET
.end
.function difference 2 2 0 int 1
.parameters difference int int
LOAD_LOCAL 0
LOAD_LOCAL 1
I64_SUB
RET
.end
""")

    def test_recursive_call_and_stack_join(self):
        self.compare('''.entry main
.function main 0 0 0 int 1
PUSH_I64 6
CALL factorial
PUSH_I64 720
I64_EQ
ASSERT
PUSH_I64 77
PUSH_BOOL 1
JMP_FALSE alternate
PUSH_I64 3
JMP join
alternate:
PUSH_I64 9
join:
I64_ADD
PUSH_I64 80
I64_EQ
ASSERT
PUSH_I64 0
RET
.end
.function factorial 1 1 0 int 1
.parameters factorial int
LOAD_LOCAL 0
PUSH_I64 1
I64_LE_S
JMP_FALSE recurse
PUSH_I64 1
RET
recurse:
LOAD_LOCAL 0
LOAD_LOCAL 0
PUSH_I64 1
I64_SUB
CALL factorial
I64_MUL
RET
.end
''')

    def test_runtime_tag_preserved_across_call(self):
        self.compare('''.entry main
.function main 0 0 0 int 1
PUSH_BOOL 1
CALL typed
RET
.end
.function typed 1 1 0 int 1
.parameters typed int
LOAD_LOCAL 0
PUSH_I64 1
I64_ADD
RET
.end
''', trap=True)

    def test_runtime_declared_result_tag(self):
        module = self.module('.entry main\n.function main 0 0 0 int 1\nPUSH_BOOL 1\nCALL typed\nPOP\nPUSH_I64 0\nRET\n.end\n.function typed 1 1 0 int 1\n.parameters typed int\nLOAD_LOCAL 0\nRET\n.end\n')
        self.run_cmd([VM, '--verify-only', module])
        self.run_cmd([VM, module], success=False)
        ir = self.work/'result-tag.ll'
        self.run_cmd([LLVM, module, '-o', ir])
        self.run_cmd(['llvm-as', ir, '-o', self.work/'result-tag.bc'])
        self.run_cmd(['lli', ir], success=False)

    def test_refused_profile_preserves_output(self):
        module = self.module('.string outside "heap"\n.entry main\n.function main 0 0 0 int 1\nPUSH_STR outside\nDUP\nSTR_CONCAT\nPOP\nPUSH_I64 0\nRET\n.end\n')
        output = self.work/'kept.ll'; output.write_text('prior output')
        result = self.run_cmd([LLVM, module, '-o', output], success=False)
        self.assertIn('scalar LLVM profile', result.stderr)
        self.assertEqual(output.read_text(), 'prior output')
        self.assertEqual(list(self.work.glob('kept.ll.*')), [])
        original = module.read_bytes()
        self.run_cmd([LLVM, module, '-o', module], success=False)
        self.assertEqual(module.read_bytes(), original)
        module.write_bytes(original[:20])
        self.run_cmd([LLVM, module, '-o', output], success=False)
        self.assertEqual(output.read_text(), 'prior output')
        module.write_bytes(b'not a module')
        self.run_cmd([LLVM, module, '-o', output], success=False)
        self.assertEqual(output.read_text(), 'prior output')

    def test_nominal_metadata_and_nonscalar_signature_refused(self):
        for text in (
            '.types 1 0 0\n.entry main\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n',
            '.entry main\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n.function unused 1 1 0 int 1\n.parameters unused array\nPUSH_I64 0\nRET\n.end\n',
        ):
            with self.subTest(text=text):
                module = self.module(text)
                self.run_cmd([VM, '--verify-only', module])
                result = self.run_cmd([LLVM, module], success=False)
                self.assertEqual(result.stdout, '')
                self.assertIn('parameters' if '.parameters' in text else 'scalar', result.stderr)

    def test_nonzero_arity_initializer_refused(self):
        module = self.module('.entry main\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n'
                             '.function __init__ 1 1 0 int 1\n.parameters __init__ int\nPUSH_I64 0\nRET\n.end\n')
        self.run_cmd([VM, '--verify-only', module])
        result = self.run_cmd([LLVM, module], success=False)
        self.assertEqual(result.stdout, '')
        self.assertIn('initializer', result.stderr)

    def test_unused_import_refused(self):
        module = self.module('.import "" "get_argc" int\n.entry main\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n')
        result = self.run_cmd([LLVM, module], success=False)
        self.assertIn('closed scalar', result.stderr)
        self.assertEqual(result.stdout, '')


if __name__ == '__main__':
    unittest.main()
