"""I preserve short-circuit behavior inside independently checked passive nodes."""
from pathlib import Path
import tempfile
import unittest
from tests import test_passive_inputs as inputs


class PassiveInternalCFG(unittest.TestCase):
    command = inputs.PassiveInputs.command
    assemble = inputs.PassiveInputs.assemble
    paired_roundtrip = inputs.PassiveInputs.paired_roundtrip

    @staticmethod
    def source(body):
        return ('.entry 0\n.function main 0 1 0 int 1\n.par_begin\n.par_node 0\n'
                + body + 'STORE_LOCAL 0\n.par_end\nLOAD_LOCAL 0\nPRINTLN\n'
                'PUSH_I64 0\nRET\n.end\n')

    def test_selected_and_skipped_boolean_arms_execute_and_roundtrip(self):
        for opcode, short_value in [('JMP_FALSE', 0), ('JMP_TRUE', 1)]:
            for flag in [0, 1]:
                with self.subTest(opcode=opcode, flag=flag):
                    # A skipped RHS would trap if eager execution were restored.
                    divisor = 0 if flag == short_value else 1
                    body = (f'PUSH_BOOL {flag}\n{opcode} short\n'
                            f'PUSH_I64 1\nPUSH_I64 {divisor}\nI64_DIV_S\nPOP\n'
                            f'PUSH_BOOL {1-short_value}\nJMP store\n'
                            f'short:\nPUSH_BOOL {short_value}\nstore:\n')
                    self.paired_roundtrip(self.source(body), b'true\n' if flag else b'false\n')

    def test_nested_branches_rejoin_with_one_result(self):
        body = ('PUSH_BOOL 1\nJMP_FALSE no\nPUSH_BOOL 0\nJMP_TRUE yes\n'
                'PUSH_BOOL 1\nJMP store\nyes:\nPUSH_BOOL 0\nJMP store\n'
                'no:\nPUSH_BOOL 0\nstore:\n')
        self.paired_roundtrip(self.source(body), b'true\n')

    def test_conditional_input_read_keeps_exact_static_input_claim(self):
        source = ('.entry 1\n.function helper 1 2 0 int 1\n'
                  'LOAD_LOCAL 0\nTYPE_CHECK 4\nASSERT\n'
                  '.par_begin\n.par_node 1 0\nPUSH_BOOL 0\nJMP_FALSE no\n'
                  'LOAD_LOCAL 0\nJMP store\nno:\nPUSH_BOOL 0\nstore:\n'
                  'STORE_LOCAL 1\n.par_end\nLOAD_LOCAL 1\nPRINTLN\nPUSH_I64 0\nRET\n.end\n'
                  '.parameters 0 bool\n.function main 0 0 0 int 1\n'
                  'PUSH_BOOL 1\nCALL helper\nRET\n.end\n')
        self.paired_roundtrip(source, b'false\n')
        with tempfile.TemporaryDirectory(prefix='nano-passive-cfg-refusal-') as tmp:
            self.assemble(Path(tmp), source.replace('.par_node 1 0', '.par_node 1'), accepted=False)

    def test_cross_node_and_external_interior_entries_remain_refused(self):
        body = ('PUSH_BOOL 1\nJMP_FALSE no\nPUSH_BOOL 1\nJMP store\n'
                'no:\nPUSH_BOOL 0\nstore:\n')
        original = self.source(body)
        cases = [
            original.replace('.par_begin\n', 'JMP no\n.par_begin\n'),
            original.replace('.par_begin\n', 'JMP store\n.par_begin\n'),
            original.replace('PUSH_BOOL 1\nJMP store', 'PUSH_BOOL 1\nJMP beyond')
                    .replace('.par_end\n', '.par_end\nbeyond:\n'),
            original.replace('JMP store', 'JMP next_node').replace(
                'STORE_LOCAL 0\n.par_end', 'STORE_LOCAL 0\n.par_node 1\nnext_node:\n'
                'PUSH_BOOL 0\nSTORE_LOCAL 1\n.par_end').replace('main 0 1 0', 'main 0 2 0'),
        ]
        with tempfile.TemporaryDirectory(prefix='nano-passive-cfg-boundary-') as tmp:
            for index, source in enumerate(cases):
                with self.subTest(index=index):
                    self.assemble(Path(tmp), source, accepted=False)


if __name__ == '__main__':
    unittest.main()
