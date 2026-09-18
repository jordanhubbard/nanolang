"""I retain indexed scalar stack order and evaluated operand snapshots."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from tests import test_reconstructed_integer_addition as addition

ROOT = Path(__file__).resolve().parents[1]


class IndexedStack(unittest.TestCase):
    check_once = False
    assemble = addition.IntegerReconstruction.assemble
    paired = addition.IntegerReconstruction.paired

    def checked(self, args, expected=0):
        result = addition.IntegerReconstruction.checked(self, args, expected)
        if self.check_once and Path(args[0]).name == 'nvm2hl':
            source = Path(args[-1]).read_text()
            calls = re.findall(r'nlr_f1_identity\(nlr_t\d+\)|\(nlr_f1_identity nlr_t\d+\)', source)
            self.assertEqual(len(calls), 1, 'I retain one evaluated call before indexed reuse')
        return result

    def test_mixed_tags_all_depths_and_stack_order(self):
        values = (('bool', 1), ('int', 7), ('bool', 0), ('int', 19))
        body = 'PUSH_BOOL 1\nSTORE_LOCAL 0\n'
        for opcode in ('PICK', 'ROLL'):
            for depth in range(len(values)):
                expected = list(values)
                item = expected[-1-depth] if opcode == 'PICK' else expected.pop(-1-depth)
                expected.append(item)
                for tag,value in values:
                    body += f'PUSH_{"BOOL" if tag == "bool" else "I64"} {value}\n'
                body += f'{opcode} {depth}\n'
                for tag,value in reversed(expected):
                    if tag == 'int': body += f'PUSH_I64 {value}\nI64_EQ\n'
                    elif value == 0: body += 'BOOL_NOT\n'
                    body += 'LOAD_LOCAL 0\nBOOL_AND\nSTORE_LOCAL 0\n'
        body += 'LOAD_LOCAL 0\nJMP_FALSE bad\nPUSH_I64 0\nRET\nbad:\nPUSH_I64 1\nRET\n'
        self.paired('.entry main\n.function main 0 1 0 int 1\n'+body+'.end\n', 0,
                    'shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }')

    def test_call_snapshot_and_indexed_loop_condition(self):
        self.check_once = True
        text = '''.entry main
.function main 0 1 0 int 1
PUSH_I64 7
STORE_LOCAL 0
LOAD_LOCAL 0
CALL identity
PUSH_I64 99
STORE_LOCAL 0
PICK 0
I64_ADD
PUSH_I64 14
I64_EQ
JMP_FALSE bad
PUSH_I64 0
STORE_LOCAL 0
loop:
LOAD_LOCAL 0
PUSH_I64 3
PICK 1
ROLL 1
I64_LT_S
SWAP
POP
JMP_FALSE done
LOAD_LOCAL 0
PUSH_I64 1
I64_ADD
STORE_LOCAL 0
JMP loop
done:
LOAD_LOCAL 0
PUSH_I64 3
I64_EQ
JMP_FALSE bad
PUSH_I64 0
RET
bad:
PUSH_I64 1
RET
.end
.function identity 1 1 0 int 1
LOAD_LOCAL 0
RET
.end
.parameters identity int
'''
        self.paired(text, 0, 'shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }\n'
                            'shadow nlr_f1_identity { assert (== (nlr_f1_identity 7) 7) }')

    def test_out_of_range_depth_preserves_output(self):
        for opcode in ('PICK', 'ROLL'):
            for values,depth in (('', 0), ('PUSH_I64 1\n', 1), ('PUSH_I64 1\n', 65535)):
                with self.subTest(opcode=opcode, depth=depth), tempfile.TemporaryDirectory(prefix='nano-hl-index-refusal-') as tmp:
                    p=Path(tmp); source=p/'input.nasm'; output=p/'previous.nvm'
                    source.write_text('.entry main\n.function main 0 0 0 int 1\n'+values+f'{opcode} {depth}\nRET\n.end\n')
                    output.write_bytes(b'previous')
                    result=subprocess.run([ROOT/'bin/nanoisa','asm',source,'-o',output], capture_output=True,text=True)
                    self.assertEqual(result.returncode,1,result.stdout+result.stderr)
                    self.assertIn('underflow',result.stderr)
                    self.assertEqual(output.read_bytes(),b'previous')


if __name__ == '__main__': unittest.main()
