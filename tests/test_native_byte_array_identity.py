"""I retain exact byte-array storage through native calls, fields and globals."""
from pathlib import Path
import os
import shlex
import tempfile
import unittest
from tests import test_file_cyclic, test_cast_u8_backends

ROOT = Path(__file__).resolve().parents[1]


class NativeByteArrayIdentity(unittest.TestCase):
    command = test_file_cyclic.FileCyclic.command
    run_trap = test_cast_u8_backends.CastU8Backends.run_trap

    def setUp(self):
        self.artifacts = Path(tempfile.mkdtemp(prefix='nano-native-byte-array-'))
        print(f'I retain byte-array artifacts at {self.artifacts}', flush=True)
        self.serial = 0

    def run_actual(self, args):
        self.serial += 1
        return self.command(f'{self.serial:03d}', list(map(str, args)))

    def native(self, text):
        asm, module, source = (self.artifacts / n for n in ('input.nasm', 'input.nvm', 'program.c'))
        asm.write_text(text)
        self.run_actual([ROOT / 'bin/nanoisa', 'asm', asm, '-o', module])
        self.run_actual([ROOT / 'bin/nano_vm', '--verify-only', module])
        self.run_actual([ROOT / 'bin/nano_vm', module])
        self.run_actual([ROOT / 'bin/nvm2c', module, '-o', source])
        cc = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc'))
        flags = shlex.split(os.environ.get('NANO_CAST_U8_NATIVE_FLAGS', ''))
        for opt in ('O0', 'O2'):
            output = self.artifacts / opt
            self.run_actual([*cc, *flags, '-std=c11', '-' + opt, '-Wall', '-Wextra', '-Werror', source, '-o', output])
            self.run_actual([output])

    def test_constructor_mutation_calls_joins_records_and_globals(self):
        self.native('''.types 1 0 0
.entry main
.function __init__ 0 0 0 void 0
ARR_NEW 2
STORE_GLOBAL 0
RET
.end
.function identity 1 1 0 array 1
LOAD_LOCAL 0
RET
.end
.function churn 0 1 0 void 0
PUSH_I64 0
STORE_LOCAL 0
again:
LOAD_LOCAL 0
PUSH_I64 2048
I64_LT_S
JMP_FALSE done
ARR_NEW 2
PUSH_U8 3
ARR_PUSH
POP
LOAD_LOCAL 0
PUSH_I64 1
I64_ADD
STORE_LOCAL 0
JMP again
done:
RET
.end
.function main 0 2 0 int 1
LOAD_GLOBAL 0
PUSH_U8 255
ARR_PUSH
STORE_GLOBAL 0
LOAD_GLOBAL 0
CALL identity
STORE_LOCAL 0
PUSH_BOOL 1
JMP_FALSE alternate
LOAD_LOCAL 0
JMP joined
alternate:
PUSH_U8 7
ARR_LITERAL 2 1
joined:
AGG_PACK 0 0 0 1
AGG_GET 0
STORE_LOCAL 1
CALL churn
LOAD_LOCAL 1
PUSH_I64 0
PUSH_I64 258
CAST_U8
ARR_SET
POP
LOAD_GLOBAL 0
PUSH_I64 0
ARR_GET
DUP
TYPE_CHECK 2
ASSERT
CAST_INT
PUSH_I64 2
I64_EQ
ASSERT
LOAD_LOCAL 0
PUSH_U8 19
ARR_PUSH
POP
LOAD_GLOBAL 0
ARR_LEN
PUSH_I64 2
I64_EQ
ASSERT
LOAD_GLOBAL 0
PUSH_I64 9223372036854775807
ARR_GET
TYPE_CHECK 0
ASSERT
PUSH_I64 0
RET
.end
''')

    def test_byte_integer_join_keeps_explicit_conversion(self):
        self.native('''.entry main
.function main 0 1 0 int 1
PUSH_BOOL 0
JMP_FALSE integer
PUSH_U8 7
JMP joined
integer:
PUSH_I64 258
joined:
CAST_U8
STORE_LOCAL 0
ARR_NEW 2
LOAD_LOCAL 0
ARR_PUSH
PUSH_I64 0
ARR_GET
DUP
TYPE_CHECK 2
ASSERT
CAST_INT
PUSH_I64 2
I64_EQ
ASSERT
PUSH_I64 0
RET
.end
''')

    def test_wrong_tag_and_mixed_array_refusals_preserve_output(self):
        cases = [
            'PUSH_I64 7\nARR_LITERAL 2 1\nPOP\n',
            'ARR_NEW 2\nPUSH_I64 7\nARR_PUSH\nPOP\n',
            'PUSH_BOOL 1\nJMP_FALSE integer\nARR_NEW 2\nJMP joined\n'
            'integer:\nARR_NEW 1\njoined:\nPOP\n',
        ]
        for index, body in enumerate(cases):
            with self.subTest(index=index):
                asm = self.artifacts / f'refusal-{index}.nasm'
                module = asm.with_suffix('.nvm')
                output = asm.with_suffix('.c')
                asm.write_text('.entry main\n.function main 0 0 0 int 1\n' + body +
                               'PUSH_I64 0\nRET\n.end\n')
                self.run_actual([ROOT / 'bin/nanoisa', 'asm', asm, '-o', module])
                output.write_bytes(b'I retain the prior product.\n')
                result = self.run_trap([ROOT / 'bin/nvm2c', module, '-o', output])
                self.assertIn('I ', result['stderr'])
                self.assertEqual(output.read_bytes(), b'I retain the prior product.\n')

    def test_byte_literal_policy_before_explicit_member_casts(self):
        for index, value in enumerate(('256', '-1', 'true', '"wrong"')):
            source = self.artifacts / f'literal-refusal-{index}.nano'
            output = source.with_suffix('.nvm')
            source.write_text('fn main() -> int { let a: array<u8> = [' + value +
                              '] return (array_length a) }\n')
            output.write_bytes(b'I retain the prior product.\n')
            self.run_trap([ROOT / 'bin/nano_virt', source, '--emit-nvm', '-o', output])
            self.assertEqual(output.read_bytes(), b'I retain the prior product.\n')

    def test_repeated_source_lowering_retains_byte_constructor_tag(self):
        source = self.artifacts / 'source.nano'
        source.write_text('''fn values() -> array<u8> { let a: array<u8> = [1, (+ 255 2)] return a }
shadow values { let a: array<u8> = (values) assert (== (cast_int (at a 1)) 1) }
fn direct() -> array<u8> { return [2, (+ 255 3)] }
shadow direct { let a: array<u8> = (direct) assert (== (cast_int (at a 1)) 2) }
fn main() -> int { let a: array<u8> = (values) let b: array<u8> = (direct)
 assert (== (cast_int (at a 1)) 1) assert (== (cast_int (at b 1)) 2) return 0 }
''')
        module = self.artifacts / 'source.nvm'
        self.run_actual([ROOT / 'bin/nano_virt', source, '--emit-nvm', '-o', module])
        assembly = self.run_actual([ROOT / 'bin/nanoisa', 'dump', module]).decode()
        self.assertGreaterEqual(assembly.count('ARR_LITERAL 2 2'), 2)
        self.assertNotIn('ARR_LITERAL 1 2', assembly)
        self.native(assembly)


if __name__ == '__main__':
    unittest.main()
