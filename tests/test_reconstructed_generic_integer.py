"""I retain exact INT generic arithmetic without admitting dynamic promotion."""
from pathlib import Path
import tempfile
import unittest
from tests import test_reconstructed_integer_addition as addition
from tests import test_reconstructed_wide_multiply as wide

ROOT = Path(__file__).resolve().parents[1]
LOW, HIGH = -(1 << 63), (1 << 63)-1
SHADOW = 'shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }'

class GenericInteger(unittest.TestCase):
    check_once = False
    checked = wide.WideMultiply.checked
    assemble = wide.WideMultiply.assemble
    paired = addition.IntegerReconstruction.paired

    def test_small_total_integer_cases(self):
        cases = (
            ('ADD', HIGH, 1, LOW), ('SUB', LOW, 1, HIGH),
            ('MUL', LOW, -1, LOW), ('NEG', LOW, None, LOW),
            ('DIV', -17, 3, -5), ('MOD', -17, 3, -2),
            ('DIV', LOW, -1, LOW), ('MOD', LOW, -1, 0),
            ('DIV', 13, 0, 0), ('MOD', 13, 0, 0),
            ('MUL', 65537, 65537, 4295098369), ('NEG', 19, None, -19))
        for start in range(0, len(cases), 3):
            with self.subTest(chunk=start):
                body = ''
                for op, a, b, expected in cases[start:start+3]:
                    body += f'PUSH_I64 {a}\n'
                    if b is not None: body += f'PUSH_I64 {b}\n'
                    body += f'{op}\nPUSH_I64 {expected}\nI64_EQ\nJMP_FALSE bad\n'
                self.paired('.entry main\n.function main 0 0 0 int 1\n'+body+
                            'PUSH_I64 0\nRET\nbad:\nPUSH_I64 1\nRET\n.end\n',0,SHADOW)

    def test_call_local_snapshots_and_loop_condition(self):
        self.paired('''.entry main
.function main 0 2 0 int 1
PUSH_I64 9
CALL identity
STORE_LOCAL 0
LOAD_LOCAL 0
PUSH_I64 2
STORE_LOCAL 0
PUSH_I64 3
SUB
PUSH_I64 6
I64_EQ
JMP_FALSE bad
PUSH_I64 0
STORE_LOCAL 1
loop:
LOAD_LOCAL 1
PUSH_I64 1
ADD
PUSH_I64 4
I64_LT_S
JMP_FALSE done
LOAD_LOCAL 1
PUSH_I64 1
ADD
STORE_LOCAL 1
JMP loop
done:
LOAD_LOCAL 1
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
.parameters identity int
LOAD_LOCAL 0
RET
.end
''',0,SHADOW+'\nshadow nlr_f1_identity { assert (== (nlr_f1_identity 9) 9) }')

    def test_other_tags_refused_before_publication(self):
        for op in ('ADD','SUB','MUL','DIV','MOD','NEG'):
            for operand in ('PUSH_BOOL 1','PUSH_U8 7','PUSH_F64 1.5'):
                with self.subTest(op=op,operand=operand), tempfile.TemporaryDirectory() as temp:
                    directory=Path(temp)
                    body=operand+'\n'+('PUSH_I64 2\n' if op!='NEG' else '')+op+'\nPOP\nPUSH_I64 0\nRET\n'
                    # I inspect checked refusal only, never execute an invalid-tag operation.
                    module=addition.IntegerReconstruction.assemble(self,directory,'.entry main\n.function main 0 0 0 int 1\n'+body+'.end\n')
                    for language in ('c','nano'):
                        output=directory/('prior.'+language);output.write_text('previous')
                        result=self.checked([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],1)
                        self.assertIn('I ',result.stderr)
                        self.assertEqual(output.read_text(),'previous')

if __name__ == '__main__': unittest.main()
