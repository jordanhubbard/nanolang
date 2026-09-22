"""I preserve scalar tail frames across actual consumers and bounded host stacks."""
import sys
import unittest
from pathlib import Path

from tests import test_cast_u8_backends as backend

ROOT = Path(__file__).resolve().parents[1]


class ScalarTailFrames(unittest.TestCase):
    setUp = backend.CastU8Backends.setUp
    command = backend.CastU8Backends.command
    run_actual = backend.CastU8Backends.run_actual
    run_trap = backend.CastU8Backends.run_trap
    native_object = backend.CastU8Backends.native_object
    assert_target = backend.CastU8Backends.assert_target
    compare = backend.CastU8Backends.compare
    program = staticmethod(backend.CastU8Backends.program)

    def bounded_native_stack(self):
        for optimization in ('O0', 'O2'):
            executable = self.artifacts / ('llvm-native-' + optimization)
            wrapper = ('import os,resource,sys\n'
                       '_,hard=resource.getrlimit(resource.RLIMIT_STACK)\n'
                       'soft=524288 if hard==resource.RLIM_INFINITY else min(524288,hard)\n'
                       'resource.setrlimit(resource.RLIMIT_STACK,(soft,hard))\n'
                       'os.execv(sys.argv[1],[sys.argv[1]])\n')
            self.run_actual([sys.executable, '-c', wrapper, executable])

    def test_deep_self_tail_and_ordinary_caller(self):
        self.compare(self.program('PUSH_I64 100000\nPUSH_I64 0\nCALL caller\n'
            'PUSH_I64 100001\nI64_EQ\nASSERT\n', '''
.function caller 2 2 0 int 1
LOAD_LOCAL 0
LOAD_LOCAL 1
CALL down
PUSH_I64 1
I64_ADD
RET
.end
.function down 2 2 0 int 1
LOAD_LOCAL 0
PUSH_I64 0
I64_EQ
JMP_TRUE done
LOAD_LOCAL 0
PUSH_I64 1
I64_SUB
LOAD_LOCAL 1
PUSH_I64 1
I64_ADD
TAIL_CALL down
done:
LOAD_LOCAL 1
RET
.end
'''))
        self.bounded_native_stack()

    def test_mutual_different_arity_stages_all_arguments(self):
        self.compare(self.program('PUSH_I64 100000\nPUSH_I64 11\nPUSH_I64 17\nCALL left\n'
            'PUSH_I64 1117\nI64_EQ\nASSERT\n', '''
.function left 3 3 0 int 1
LOAD_LOCAL 0
PUSH_I64 0
I64_EQ
JMP_TRUE done
LOAD_LOCAL 0
PUSH_I64 1
I64_SUB
LOAD_LOCAL 2
LOAD_LOCAL 1
PUSH_BOOL 1
TAIL_CALL right
done:
LOAD_LOCAL 1
PUSH_I64 100
I64_MUL
LOAD_LOCAL 2
I64_ADD
RET
.end
.function right 4 4 0 int 1
LOAD_LOCAL 3
ASSERT
LOAD_LOCAL 0
LOAD_LOCAL 1
LOAD_LOCAL 2
TAIL_CALL left
.end
'''))
        self.bounded_native_stack()

    def test_void_initializer_tail_preserves_globals(self):
        self.compare(self.program('LOAD_GLOBAL 0\nPUSH_I64 42\nI64_EQ\nASSERT\n', '''
.function __init__ 0 0 0 void 0
PUSH_I64 42
TAIL_CALL initialize
.end
.function initialize 1 1 0 void 0
LOAD_LOCAL 0
STORE_GLOBAL 0
RET
.end
'''))

    def test_exact_scalar_and_literal_string_carriers(self):
        cases = (
            ('u8', 'PUSH_U8 255\n', 'CAST_INT\nPUSH_I64 255\nI64_EQ\nASSERT\n', ''),
            ('bool', 'PUSH_BOOL 1\n', 'ASSERT\n', ''),
            ('float', 'PUSH_I64 9221120237041090561\nF64_FROM_BITS\n',
             'F64_TO_BITS\nPUSH_I64 9221120237041090561\nI64_EQ\nASSERT\n', ''),
            ('enum', 'ENUM_VAL 0 3\n', 'CAST_INT\nPUSH_I64 3\nI64_EQ\nASSERT\n', '.types 0 1 0\n'),
            ('string', 'PUSH_STR text\n', 'STR_LEN\nPUSH_I64 4\nI64_EQ\nASSERT\n', '.string text "kept"\n'),
        )
        original = self.artifacts
        for tag, value, check, prefix in cases:
            with self.subTest(tag=tag):
                self.artifacts = original / tag; self.artifacts.mkdir()
                helpers = (f'.function relay 1 1 0 {tag} 1\n.parameters relay {tag}\n'
                    'LOAD_LOCAL 0\nTAIL_CALL identity\n.end\n' +
                    f'.function identity 1 1 0 {tag} 1\n.parameters identity {tag}\n'
                    'LOAD_LOCAL 0\nRET\n.end\n')
                self.compare(prefix + self.program(value + 'CALL relay\n' + check, helpers))
        self.artifacts = original

    def test_incompatible_tail_result_is_refused_without_publication(self):
        source = self.artifacts / 'incompatible.nasm'
        source.write_text(self.program('CALL relay\nPOP\n', '''
.function relay 0 0 0 int 1
TAIL_CALL identity
.end
.function identity 0 0 0 bool 1
PUSH_BOOL 1
RET
.end
'''))
        module = self.artifacts / 'incompatible.nvm'
        module.write_bytes(b'previous output\n')
        observed = self.run_trap([ROOT / 'bin/nanoisa', 'asm', source, '-o', module])
        self.assertIn('incompatible result signature', observed['stderr'])
        self.assertEqual(module.read_bytes(), b'previous output\n')

    def test_managed_tail_remains_refused_without_publication(self):
        source = self.artifacts / 'managed.nasm'
        source.write_text(self.program('CALL relay\nPOP\n', '''
.string text "kept"
.function relay 0 0 0 string 1
PUSH_STR text
PUSH_STR text
STR_CONCAT
POP
TAIL_CALL identity
.end
.function identity 0 0 0 string 1
PUSH_STR text
RET
.end
'''))
        module = self.artifacts / 'managed.nvm'
        self.run_actual([ROOT / 'bin/nanoisa', 'asm', source, '-o', module])
        self.run_actual([ROOT / 'bin/nano_vm', '--verify-only', module])
        for tool in ('nvm2llvm', 'nvm2wasm'):
            output = self.artifacts / ('previous-' + tool); output.write_bytes(b'previous output\n')
            self.run_trap([ROOT / 'bin' / tool, module, '-o', output])
            self.assertEqual(output.read_bytes(), b'previous output\n')


if __name__ == '__main__':
    unittest.main()
