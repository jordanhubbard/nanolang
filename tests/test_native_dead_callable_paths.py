"""I retain unresolved-call refusal unless bounded constant flow proves it dead."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from tests import test_native_callables as harness

ROOT = harness.ROOT
TRANSLATOR = Path(os.environ.get("NANO_DEAD_CALLABLE_TRANSLATOR", ROOT/"bin/nvm2c"))
EMPTY = '''ARR_LITERAL 11 0
STORE_LOCAL 0
PUSH_I64 0
LOAD_LOCAL 0
ARR_LEN
LT
JMP_FALSE done
LOAD_LOCAL 0
PUSH_I64 0
ARR_GET
CALL_INDIRECT 0 0
done:
RET
'''


class NativeDeadCallablePaths(unittest.TestCase):
    def execute(self, assembly):
        with tempfile.TemporaryDirectory(prefix='nano-dead-callable-') as tmp:
            work = Path(tmp)
            runner = harness.NativeCallables()
            text, module, source = work/'input.nasm', work/'input.nvm', work/'input.c'
            text.write_text(assembly)
            runner.checked([ROOT/'bin/nanoisa','asm',text,'-o',module])
            runner.checked([ROOT/'bin/nano_vm',module])
            runner.checked([TRANSLATOR,module,'-o',source])
            runner.sanitized(source, work/'native')

    def refuse(self, assembly):
        with tempfile.TemporaryDirectory(prefix='nano-live-callable-') as tmp:
            work = Path(tmp)
            source, module, output = work/'input.nasm', work/'input.nvm', work/'output.c'
            source.write_text(assembly)
            harness.NativeCallables().checked([ROOT/'bin/nanoisa','asm',source,'-o',module])
            output.write_bytes(b'prior accepted output')
            result = subprocess.run([TRANSLATOR,module,'-o',output],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertRegex(result.stderr,r'resolved indirect-call targets|I have no callable targets')
            self.assertEqual(output.read_bytes(),b'prior accepted output')

    def test_empty_literal_loop(self):
        self.execute('.entry main\n.function main 0 1 0 int 1\n'+EMPTY.replace('RET\n','PUSH_I64 0\nRET\n')+'.end\n')

    def test_source_empty_loop_with_result_and_break_continue(self):
        with tempfile.TemporaryDirectory(prefix='nano-source-empty-loop-') as tmp:
            work = Path(tmp)
            module, source = work/'input.nvm', work/'input.c'
            runner = harness.NativeCallables()
            runner.checked([ROOT/'bin/nano_virt',ROOT/'docs/evidence/callable-array-loops-20261010/empty-loop.nano','-o',module])
            runner.checked([ROOT/'bin/nano_vm',module])
            runner.checked([TRANSLATOR,module,'-o',source])
            runner.sanitized(source,work/'native')

    def test_unknown_parameter_remains_reachable(self):
        body = EMPTY.replace('PUSH_I64 0\nLOAD_LOCAL 0\nARR_LEN\nLT','LOAD_LOCAL 1')
        self.refuse('.entry main\n.function main 0 0 0 void 0\nPUSH_BOOL 1\nCALL apply\nRET\n.end\n'
            +'.function apply 1 2 0 void 0\nLOAD_LOCAL 0\nSTORE_LOCAL 1\n'+body+'.end\n')

    def test_join_does_not_keep_one_branch_constant(self):
        body = EMPTY.replace('PUSH_I64 0\nLOAD_LOCAL 0\nARR_LEN\nLT','LOAD_LOCAL 1')
        self.refuse('.entry main\n.function main 0 0 0 void 0\nPUSH_BOOL 1\nCALL apply\nRET\n.end\n'
            +'.function apply 1 2 0 void 0\nLOAD_LOCAL 0\nJMP_FALSE other\nPUSH_BOOL 1\nSTORE_LOCAL 1\nJMP joined\n'
            +'other:\nPUSH_BOOL 0\nSTORE_LOCAL 1\njoined:\n'+body+'.end\n')

    def test_backedge_widens_initial_constant(self):
        self.refuse('.entry main\n.function main 0 2 0 void 0\nARR_LITERAL 11 0\nSTORE_LOCAL 0\n'
            +'PUSH_BOOL 0\nSTORE_LOCAL 1\nagain:\nLOAD_LOCAL 1\nJMP_FALSE update\n'
            +'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nCALL_INDIRECT 0 0\nRET\n'
            +'update:\nPUSH_BOOL 1\nSTORE_LOCAL 1\nJMP again\n.end\n')

    def test_mutation_disables_length_assumptions(self):
        body = EMPTY.replace('PUSH_I64 0\nLOAD_LOCAL 0\nARR_LEN',
            'ARR_LITERAL 1 0\nPUSH_I64 1\nARR_PUSH\nPOP\nPUSH_I64 0\nLOAD_LOCAL 0\nARR_LEN')
        self.refuse('.entry main\n.function main 0 1 0 void 0\n'+body+'.end\n')

    def test_new_length_facts_after_mutation(self):
        self.execute('.entry main\n.function main 0 1 0 int 1\n'
            +'ARR_LITERAL 1 0\nPUSH_I64 1\nARR_PUSH\nPOP\n'
            +EMPTY.replace('RET\n','PUSH_I64 0\nRET\n')+'.end\n')

    def test_call_boundary_discards_array_length(self):
        body = EMPTY.replace('PUSH_I64 0\nLOAD_LOCAL 0\nARR_LEN',
            'CALL boundary\nPUSH_I64 0\nLOAD_LOCAL 0\nARR_LEN')
        self.refuse('.entry main\n.function main 0 1 0 void 0\n'+body+'.end\n'
            +'.function boundary 0 0 0 void 0\nRET\n.end\n')

    def test_uninvoked_reference_does_not_require_body(self):
        self.execute('.entry main\n.function main 0 0 0 int 1\nFUNCREF unresolved\nPOP\nPUSH_I64 0\nRET\n.end\n'
            +'.function unresolved 1 1 0 void 0\nLOAD_LOCAL 0\nCALL_INDIRECT 0 0\nRET\n.end\n')

    def test_invoked_reference_requires_body(self):
        self.refuse('.entry main\n.function main 0 0 0 void 0\n'
            +'ARR_LITERAL 11 0\nPUSH_I64 0\nARR_GET\nFUNCREF unresolved\nCALL_INDIRECT 1 0\nRET\n.end\n'
            +'.function unresolved 1 1 0 void 0\nLOAD_LOCAL 0\nCALL_INDIRECT 0 0\nRET\n.end\n')

    def test_uninvoked_indirect_caller_is_omitted_with_callee(self):
        self.execute('.entry main\n.function main 0 0 0 int 1\nFUNCREF unused\nPOP\nPUSH_I64 0\nRET\n.end\n'
            +'.function unused 1 1 0 void 0\nLOAD_LOCAL 0\nFUNCREF unresolved\nCALL_INDIRECT 1 0\nRET\n.end\n'
            +'.function unresolved 1 1 0 void 0\nLOAD_LOCAL 0\nCALL_INDIRECT 0 0\nRET\n.end\n')
