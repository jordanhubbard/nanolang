"""I test small independent wide products with successful pinned compilers."""
from pathlib import Path
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from tests import test_reconstructed_integer_addition as addition
from tests.native_toolchain import native_cc

ROOT = Path(__file__).resolve().parents[1]
MASK = (1 << 64)-1
LOW, HIGH = -(1 << 63), (1 << 63)-1
SHADOW = 'shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }'


def signed(bits):
    bits &= MASK
    return bits if bits <= HIGH else bits-(1 << 64)


def words(a, b, unsigned):
    product = (a & MASK)*(b & MASK) if unsigned else a*b
    return signed(product), signed(product >> 64)


class WideMultiply(unittest.TestCase):
    check_once = False
    paired = addition.IntegerReconstruction.paired

    def checked(self, args, expected=0):
        try:
            result = addition.IntegerReconstruction.checked(self, args, expected)
        except (AssertionError, subprocess.TimeoutExpired):
            retained = Path('/tmp/nanolang-wide-first-failure')
            suffix = 1
            while retained.exists():
                retained = Path(f'/tmp/nanolang-wide-failure-{suffix}')
                suffix += 1
            if not retained.exists():
                for value in args:
                    candidate = Path(str(value)).parent
                    if candidate.name.startswith('nano-reconstruct-') and candidate.is_dir():
                        shutil.copytree(candidate, retained)
                        (retained/'command.json').write_text(json.dumps(list(map(str, args)), indent=2))
                        break
            raise
        if self.check_once and Path(args[0]).name == 'nvm2hl':
            source = Path(args[-1]).read_text()
            calls = re.findall(r'nlr_f1_identity\(nlr_t\d+\)|\(nlr_f1_identity nlr_t\d+\)', source)
            self.assertEqual(len(calls), 1)
            self.assertRegex(source, r'nlr_t\d+_low')
            self.assertRegex(source, r'nlr_t\d+_high')
        return result

    def assemble(self, directory, text):
        module = addition.IntegerReconstruction.assemble(self, directory, text)
        source, binary = directory/'native.c', directory/'native'
        self.checked([ROOT/'bin/nvm2c', module, '-o', source])
        self.checked([*native_cc(), '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror',
                      '-fsanitize=address,undefined', '-fno-sanitize-recover=all', source, '-o', binary])
        self.checked([binary])
        return module

    def test_small_signed_and_unsigned_word_cases(self):
        # I use distinct small modules, not the historical carry endpoint corpus.
        cases = ((0, HIGH), (1, LOW), (-1, -1), (LOW, LOW), (HIGH, HIGH),
                 (65535, 65537), (4294967297, 4294967299), (LOW+17, -65537))
        for unsigned in (False, True):
            opcode = 'I64_MUL_WIDE_U' if unsigned else 'I64_MUL_WIDE_S'
            for start in range(0, len(cases), 3):
                with self.subTest(unsigned=unsigned, chunk=start):
                    body = 'PUSH_BOOL 1\nSTORE_LOCAL 0\n'
                    for a, b in cases[start:start+3]:
                        low, high = words(a,b,unsigned)
                        body += f'PUSH_I64 {a}\nPUSH_I64 {b}\n{opcode}\nSTORE_LOCAL 2\nSTORE_LOCAL 1\n'
                        for slot, expected in ((1, low), (2, high)):
                            body += f'LOAD_LOCAL {slot}\nPUSH_I64 {expected}\nI64_EQ\nLOAD_LOCAL 0\nBOOL_AND\nSTORE_LOCAL 0\n'
                    body += 'LOAD_LOCAL 0\nJMP_FALSE bad\nPUSH_I64 0\nRET\nbad:\nPUSH_I64 1\nRET\n'
                    self.paired('.entry main\n.function main 0 3 0 int 1\n'+body+'.end\n',0,SHADOW)

    def test_call_and_local_snapshots(self):
        self.check_once = True
        low, high = words(123456789,4294967297,True)
        try:
            self.paired(f'''.entry main
.function main 0 3 0 int 1
PUSH_I64 -1
CALL identity
DUP
I64_MUL_WIDE_U
STORE_LOCAL 2
STORE_LOCAL 1
LOAD_LOCAL 2
PUSH_I64 -2
I64_EQ
JMP_FALSE bad
LOAD_LOCAL 1
PUSH_I64 1
I64_EQ
JMP_FALSE bad
PUSH_I64 123456789
STORE_LOCAL 0
LOAD_LOCAL 0
PUSH_I64 0
STORE_LOCAL 0
PUSH_I64 4294967297
I64_MUL_WIDE_U
STORE_LOCAL 2
STORE_LOCAL 1
LOAD_LOCAL 2
PUSH_I64 {high}
I64_EQ
JMP_FALSE bad
LOAD_LOCAL 1
PUSH_I64 {low}
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
''',0,SHADOW+'\nshadow nlr_f1_identity { assert (== (nlr_f1_identity -1) -1) }')
        finally:
            self.check_once = False

    def test_pure_loop_condition_and_body_words(self):
        self.paired('''.entry main
.function main 0 3 0 int 1
PUSH_I64 0
STORE_LOCAL 0
PUSH_I64 0
STORE_LOCAL 1
PUSH_I64 0
STORE_LOCAL 2
loop:
LOAD_LOCAL 0
PUSH_I64 1
I64_MUL_WIDE_U
POP
PUSH_I64 3
I64_LT_S
JMP_FALSE done
LOAD_LOCAL 0
PUSH_I64 -1
I64_MUL_WIDE_S
STORE_LOCAL 2
STORE_LOCAL 1
LOAD_LOCAL 0
PUSH_I64 1
I64_ADD
STORE_LOCAL 0
JMP loop
done:
LOAD_LOCAL 1
PUSH_I64 -2
I64_EQ
JMP_FALSE bad
LOAD_LOCAL 2
PUSH_I64 -1
I64_EQ
JMP_FALSE bad
PUSH_I64 0
RET
bad:
PUSH_I64 1
RET
.end
''',0,SHADOW)

    def test_type_arity_and_profile_refusals_preserve_output(self):
        for opcode in ('I64_MUL_WIDE_U', 'I64_MUL_WIDE_S'):
            for operands in ('PUSH_I64 1', 'PUSH_BOOL 1\nPUSH_I64 2', 'PUSH_I64 2\nPUSH_BOOL 1'):
                with self.subTest(opcode=opcode, operands=operands), tempfile.TemporaryDirectory() as temp:
                    directory=Path(temp); source=directory/'wrong.nasm'; output=directory/'prior.nvm'
                    source.write_text('.entry main\n.function main 0 0 0 int 1\n'+operands+'\n'+opcode+'\nPOP\nRET\n.end\n')
                    output.write_bytes(b'previous')
                    result=subprocess.run([ROOT/'bin/nanoisa','asm',source,'-o',output],capture_output=True,text=True)
                    self.assertEqual(result.returncode,1,result.stdout+result.stderr)
                    self.assertEqual(output.read_bytes(),b'previous')
            with self.subTest(opcode=opcode), tempfile.TemporaryDirectory() as temp:
                directory=Path(temp)
                text='.entry main\n.function main 0 0 0 int 1\nPUSH_I64 2\nPUSH_I64 3\n'+opcode+'\nPOP\nCAST_STRING\nPOP\nPUSH_I64 0\nRET\n.end\n'
                # I assemble the valid excluded profile but never execute it.
                module=addition.IntegerReconstruction.assemble(self,directory,text)
                for language in ('c','nano'):
                    output=directory/('prior.'+language);output.write_text('previous')
                    result=self.checked([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],1)
                    self.assertIn('CAST_STRING',result.stderr)
                    self.assertEqual(output.read_text(),'previous')


if __name__ == '__main__': unittest.main()
