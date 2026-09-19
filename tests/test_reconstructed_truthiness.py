"""I reconstruct exact scalar casts without dropping evaluated operands."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from tests import test_reconstructed_integer_addition as addition

ROOT = Path(__file__).resolve().parents[1]
SUCCESS = 'LOAD_LOCAL 0\nJMP_FALSE bad\nPUSH_I64 0\nRET\nbad:\nPUSH_I64 1\nRET\n'
SHADOW = 'shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }'


class ScalarTruthiness(unittest.TestCase):
    assemble = addition.IntegerReconstruction.assemble
    paired = addition.IntegerReconstruction.paired
    check_eager = False

    def checked(self, args, expected=0):
        result = addition.IntegerReconstruction.checked(self, args, expected)
        if self.check_eager and Path(args[0]).name == 'nvm2hl':
            source = Path(args[-1]).read_text()
            # I inspect executable call assignments, not declarations or fixture shadows.
            calls = list(re.finditer(r'nlr_t\d+[^\n=]*= [^\n]*nlr_f1_right[^\n]*', source))
            self.assertEqual(len(calls), 2)
            logic = list(re.finditer(r'nlr_t\d+[^\n=]*= [^\n]*(?:&&|\|\||\(and |\(or )[^\n]*', source))
            self.assertEqual(len(logic), 2)
            for call, operation in zip(calls, logic):
                self.assertLess(call.start(), operation.start())
        return result

    def test_mixed_truth_tables(self):
        values = (('PUSH_I64 0', False), ('PUSH_I64 -7', True),
                  ('PUSH_BOOL 0', False), ('PUSH_BOOL 1', True))
        for opcode in ('AND', 'OR'):
            with self.subTest(opcode=opcode):
                body = 'PUSH_BOOL 1\nSTORE_LOCAL 0\n'
                for left, a in values:
                    for right, b in values:
                        expected = int(a and b if opcode == 'AND' else a or b)
                        body += f'{left}\n{right}\n{opcode}\nCAST_INT\nPUSH_I64 {expected}\nI64_EQ\nLOAD_LOCAL 0\nBOOL_AND\nSTORE_LOCAL 0\n'
                self.paired('.entry main\n.function main 0 1 0 int 1\n'+body+SUCCESS+'.end\n', 0, SHADOW)

    def test_casts_and_not_endpoints(self):
        values = (-(1 << 63), -1, 0, 1, (1 << 63)-1)
        body = 'PUSH_BOOL 1\nSTORE_LOCAL 0\n'
        for value in values:
            for operation, expected in (('CAST_INT', value), ('CAST_BOOL\nCAST_INT', int(value != 0)),
                                        ('NOT\nCAST_INT', int(value == 0))):
                body += f'PUSH_I64 {value}\n{operation}\nPUSH_I64 {expected}\nI64_EQ\nLOAD_LOCAL 0\nBOOL_AND\nSTORE_LOCAL 0\n'
        for value in (0, 1):
            body += f'PUSH_BOOL {value}\nCAST_BOOL\nCAST_INT\nPUSH_I64 {value}\nI64_EQ\nLOAD_LOCAL 0\nBOOL_AND\nSTORE_LOCAL 0\n'
        self.paired('.entry main\n.function main 0 1 0 int 1\n'+body+SUCCESS+'.end\n', 0, SHADOW)

    def test_eager_calls_and_local_snapshot(self):
        self.check_eager = True
        try:
            self.paired('''.entry main
.function main 0 1 0 int 1
PUSH_I64 -1
STORE_LOCAL 0
LOAD_LOCAL 0
PUSH_I64 0
STORE_LOCAL 0
CAST_BOOL
JMP_FALSE bad
PUSH_BOOL 0
PUSH_BOOL 0
CALL right
AND
JMP_TRUE bad
PUSH_BOOL 1
PUSH_BOOL 1
CALL right
OR
JMP_FALSE bad
PUSH_I64 0
RET
bad:
PUSH_I64 1
RET
.end
.function right 1 1 0 bool 1
LOAD_LOCAL 0
NOT
RET
.end
.parameters right bool
''', 0, SHADOW+'\nshadow nlr_f1_right { assert (nlr_f1_right false) assert (not (nlr_f1_right true)) }')
        finally:
            self.check_eager = False

    def test_total_cast_loop_condition(self):
        self.paired('''.entry main
.function main 0 1 0 int 1
PUSH_I64 3
STORE_LOCAL 0
loop:
LOAD_LOCAL 0
CAST_BOOL
CAST_INT
PUSH_I64 0
I64_GT_S
JMP_FALSE done
LOAD_LOCAL 0
PUSH_I64 1
I64_SUB
STORE_LOCAL 0
JMP loop
done:
LOAD_LOCAL 0
RET
.end
''', 0, SHADOW)

    def test_excluded_tags_and_wrong_arity_preserve_output(self):
        for producer in ('PUSH_F64 0.0', 'PUSH_VOID', 'PUSH_STR text', 'ARR_NEW 1'):
            for opcode in ('CAST_BOOL', 'CAST_INT', 'NOT', 'AND', 'OR'):
                with self.subTest(producer=producer, opcode=opcode), tempfile.TemporaryDirectory() as temp:
                    directory = Path(temp)
                    body = producer+'\n'+('PUSH_BOOL 1\n' if opcode in ('AND', 'OR') else '')+opcode+'\n'
                    if opcode != 'CAST_INT': body += 'CAST_INT\n'
                    module = self.assemble(directory, '.string text "text"\n.entry main\n.function main 0 0 0 int 1\n'+body+'RET\n.end\n')
                    for language in ('c', 'nano'):
                        output = directory/('previous.'+language); output.write_text('previous')
                        result = self.checked([ROOT/'bin/nvm2hl', '--language', language, module, '-o', output], 1)
                        self.assertIn('I ', result.stderr)
                        self.assertEqual(output.read_text(), 'previous')
        for opcode in ('NOT', 'AND', 'OR'):
            with self.subTest(producer='PUSH_U8 1', opcode=opcode), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                body = 'PUSH_U8 1\n'+('PUSH_BOOL 1\n' if opcode in ('AND', 'OR') else '')+opcode+'\nCAST_INT\n'
                module = self.assemble(directory, '.entry main\n.function main 0 0 0 int 1\n'+body+'RET\n.end\n')
                for language in ('c', 'nano'):
                    output = directory/('previous.'+language); output.write_text('previous')
                    result = self.checked([ROOT/'bin/nvm2hl', '--language', language, module, '-o', output], 1)
                    self.assertIn('I ', result.stderr)
                    self.assertEqual(output.read_text(), 'previous')
        for body in ('CAST_BOOL', 'CAST_INT', 'NOT', 'PUSH_BOOL 1\nAND', 'PUSH_I64 1\nOR', 'PUSH_I64 1\nBOOL_NOT'):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp); source = directory/'wrong.nasm'; output = directory/'previous.nvm'
                source.write_text('.entry main\n.function main 0 0 0 int 1\n'+body+'\nPOP\nPUSH_I64 0\nRET\n.end\n')
                output.write_bytes(b'previous')
                result = subprocess.run([ROOT/'bin/nanoisa', 'asm', source, '-o', output], text=True, capture_output=True)
                self.assertEqual(result.returncode, 1, result.stdout+result.stderr)
                self.assertEqual(output.read_bytes(), b'previous')


if __name__ == '__main__': unittest.main()
