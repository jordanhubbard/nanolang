"""I derive closed-call scalar purity from concrete code and actual inputs."""
from pathlib import Path
import tempfile
import unittest
import tests.test_passive_inputs as inputs


def program(body, arity=1, locals_count=1, argument='PUSH_I64 4', extra=''):
    return (f'.entry 2\n.function callee {arity} {locals_count} 0 int 1\n'
            f'{body}\n.end\n{extra}'
            '.function owner 1 2 0 int 1\n'
            'LOAD_LOCAL 0\nTYPE_CHECK 1\nASSERT\n'
            '.par_begin\n.par_node 1 0\nLOAD_LOCAL 0\nCALL callee\nSTORE_LOCAL 1\n.par_end\n'
            'LOAD_LOCAL 1\nRET\n.end\n.parameters 1 int\n'
            f'.function main 0 0 0 int 1\n{argument}\nCALL owner\nPRINTLN\nPUSH_I64 0\nRET\n.end\n')


class PassiveCalls(unittest.TestCase):
    command = inputs.PassiveInputs.command
    assemble = inputs.PassiveInputs.assemble
    paired_roundtrip = inputs.PassiveInputs.paired_roundtrip

    def test_scalar_square_does_not_require_parameter_annotation(self):
        source = program('LOAD_LOCAL 0\nLOAD_LOCAL 0\nI64_MUL\nRET')
        # A declared tag does not establish the summary. Actual producers do.
        source += '.parameters 0 void\n'
        self.paired_roundtrip(source, b'16\n')

    def test_local_loop_and_definite_assignment_join(self):
        body = ('PUSH_I64 0\nSTORE_LOCAL 1\nPUSH_I64 0\nSTORE_LOCAL 2\n'
                'loop:\nLOAD_LOCAL 1\nLOAD_LOCAL 0\nI64_LT_S\nJMP_FALSE done\n'
                'LOAD_LOCAL 2\nLOAD_LOCAL 1\nI64_ADD\nSTORE_LOCAL 2\n'
                'LOAD_LOCAL 1\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 1\nJMP loop\n'
                'done:\nLOAD_LOCAL 2\nRET')
        self.paired_roundtrip(program(body, locals_count=3), b'6\n')
        branch = ('LOAD_LOCAL 0\nPUSH_I64 0\nI64_GT_S\nJMP_FALSE other\n'
                  'PUSH_I64 17\nSTORE_LOCAL 1\nJMP join\n'
                  'other:\nPUSH_I64 29\nSTORE_LOCAL 1\n'
                  'join:\nLOAD_LOCAL 1\nRET')
        self.paired_roundtrip(program(branch, locals_count=2), b'17\n')

    def test_nested_and_tail_calls_use_actual_stack_effects(self):
        source = program('LOAD_LOCAL 0\nCALL square\nLOAD_LOCAL 0\nI64_MUL\nRET')
        source += '.function square 1 1 0 int 1\nLOAD_LOCAL 0\nTAIL_CALL multiply\n.end\n'
        source += '.function multiply 1 1 0 int 1\nLOAD_LOCAL 0\nLOAD_LOCAL 0\nI64_MUL\nRET\n.end\n'
        self.paired_roundtrip(source, b'64\n')

    def test_zero_argument_and_void_callee_stack_effects(self):
        source = program('PUSH_I64 42\nRET', arity=0, locals_count=0)
        source = source.replace('.par_node 1 0\nLOAD_LOCAL 0\nCALL callee',
                                '.par_node 1\nCALL callee')
        self.paired_roundtrip(source, b'42\n')
        source = program('RET', arity=0, locals_count=0)
        source = source.replace('callee 0 0 0 int 1', 'callee 0 0 0 void 0')
        source = source.replace('.par_node 1 0\nLOAD_LOCAL 0\nCALL callee',
                                '.par_node 1\nCALL callee\nPUSH_I64 17')
        self.paired_roundtrip(source, b'17\n')

    def test_arctan_loop_with_two_arguments(self):
        source = """.entry 2
.function arctan 2 8 0 float 1
PUSH_F64 0
STORE_LOCAL 2
PUSH_I64 0
STORE_LOCAL 3
LOAD_LOCAL 0
LOAD_LOCAL 0
F64_MUL
STORE_LOCAL 4
LOAD_LOCAL 0
STORE_LOCAL 5
loop:
LOAD_LOCAL 3
LOAD_LOCAL 1
I64_LT_S
JMP_FALSE done
PUSH_F64 2
LOAD_LOCAL 3
CAST_FLOAT
F64_MUL
PUSH_F64 1
F64_ADD
STORE_LOCAL 6
LOAD_LOCAL 5
LOAD_LOCAL 6
F64_DIV
STORE_LOCAL 7
LOAD_LOCAL 3
PUSH_I64 2
I64_REM_S
PUSH_I64 0
I64_EQ
JMP_FALSE subtract
LOAD_LOCAL 2
LOAD_LOCAL 7
F64_ADD
STORE_LOCAL 2
JMP advance
subtract:
LOAD_LOCAL 2
LOAD_LOCAL 7
F64_SUB
STORE_LOCAL 2
advance:
LOAD_LOCAL 5
LOAD_LOCAL 4
F64_MUL
STORE_LOCAL 5
LOAD_LOCAL 3
PUSH_I64 1
I64_ADD
STORE_LOCAL 3
JMP loop
done:
LOAD_LOCAL 2
RET
.end
.function owner 1 3 0 float 1
LOAD_LOCAL 0
TYPE_CHECK 1
ASSERT
.par_begin
.par_node 1 0
PUSH_F64 0.2
LOAD_LOCAL 0
CALL arctan
STORE_LOCAL 1
.par_node 2 0
PUSH_F64 0.0041841
LOAD_LOCAL 0
CALL arctan
STORE_LOCAL 2
.par_end
PUSH_F64 4
PUSH_F64 4
LOAD_LOCAL 1
F64_MUL
LOAD_LOCAL 2
F64_SUB
F64_MUL
RET
.end
.parameters 1 int
.function main 0 0 0 int 1
PUSH_I64 50
CALL owner
PRINTLN
PUSH_I64 0
RET
.end
"""
        # The verifier slice is executable in VM. Exact native CAST_FLOAT
        # lowering remains task_3d6c3314d8924dd2b0aca679647e7048; the frontend
        # cutover must retain paired arctan acceptance after that companion.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            module = self.assemble(directory, source)
            original = module.read_bytes()
            vm = self.command(inputs.ROOT/'bin/nano_vm', module)
            self.assertEqual(vm.returncode, 0, vm.stderr)
            self.assertEqual(vm.stdout, b'3.14159\n')
            dump = self.command(inputs.ROOT/'bin/nanoisa', 'dump', module)
            self.assertEqual(dump.returncode, 0, dump.stderr)
            self.assertEqual(self.assemble(directory, dump.stdout.decode()).read_bytes(), original)
            native = self.command(inputs.ROOT/'bin/nvm2c', module, '-o', directory/'arctan.c')
            self.assertNotEqual(native.returncode, 0)
            self.assertIn(b'CAST_FLOAT', native.stderr)

    def test_effectful_recursive_and_uninitialized_callees_refuse(self):
        cases = {
            'printing': program('LOAD_LOCAL 0\nPRINTLN\nPUSH_I64 1\nRET'),
            'recursive': program('LOAD_LOCAL 0\nCALL callee\nRET'),
            'tail recursive': program('LOAD_LOCAL 0\nTAIL_CALL callee'),
            'uninitialized local': program('LOAD_LOCAL 1\nRET', locals_count=2),
            'one branch uninitialized': program(
                'LOAD_LOCAL 0\nPUSH_I64 0\nI64_GT_S\nJMP_FALSE join\n'
                'PUSH_I64 17\nSTORE_LOCAL 1\njoin:\nLOAD_LOCAL 1\nRET', locals_count=2),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, source in cases.items():
                with self.subTest(name=name):
                    self.assemble(Path(tmp), source, accepted=False)


if __name__ == '__main__':
    unittest.main()
