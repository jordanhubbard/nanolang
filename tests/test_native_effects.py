"""I compare native activation boundaries with the VM, including malformed use."""
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeEffectBoundaries(unittest.TestCase):
    def checked(self, command):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def exercise(self, body, traps=False, expected=0):
        with tempfile.TemporaryDirectory(prefix='native-effect-') as directory:
            work = Path(directory)
            assembly, module, source, binary = (work / name for name in ('input.nasm', 'input.nvm', 'input.c', 'program'))
            assembly.write_text('.string op "Ask.ask"\n.entry main\n' + body)
            self.checked([os.environ.get('NANOLANG_NANOISA', str(ROOT/'bin/nanoisa')), 'asm', assembly, '-o', module])
            self.checked([os.environ.get('NANOLANG_NANO_VM', str(ROOT/'bin/nano_vm')), '--verify-only', module])
            self.checked([os.environ.get('NANOLANG_NVM2C', str(ROOT/'bin/nvm2c')), module, '-o', source])
            self.checked([*shlex.split(os.environ.get('NANOLANG_EFFECT_CC', os.environ.get('CC', 'cc'))), '-std=c11', '-Wall', '-Wextra', '-Werror',
                          *shlex.split(os.environ.get('NANOLANG_EFFECT_CFLAGS', '')), source, '-o', binary, '-lm'])
            for target in ([os.environ.get('NANOLANG_NANO_VM', str(ROOT/'bin/nano_vm')), module], [binary]):
                with self.subTest(target=str(target[0])):
                    ran = subprocess.run(target, capture_output=True, text=True, timeout=30)
                    if traps:
                        self.assertNotEqual(ran.returncode, 0, ran.stdout+ran.stderr)
                        if target[0] == binary:
                            self.assertEqual(ran.returncode, -signal.SIGABRT, ran.stdout+ran.stderr)
                            self.assertIn('native invariant', ran.stderr)
                            self.assertNotIn('AddressSanitizer', ran.stderr)
                            self.assertNotIn('runtime error:', ran.stderr)
                    else:
                        self.assertEqual(ran.returncode, expected, ran.stdout+ran.stderr)

    def test_activation_errors_trap(self):
        cases = {
            'unhandled': '.function main 0 0 0 int 1\nPERFORM op 0\nRET\n.end\n',
            'pop_absent': '.function main 0 0 0 int 1\nHANDLER_POP 1\nPUSH_I64 0\nRET\n.end\n',
            'resume_outside_arm': '.function main 0 0 0 int 1\nPUSH_I64 0\nEFFECT_RESUME\n.end\n',
            'removed_handler': '.function main 0 0 0 int 1\nHANDLER_PUSH op arm 0 0\nHANDLER_POP 1\nPERFORM op 0\nRET\narm:\nPUSH_I64 0\nEFFECT_RESUME\n.end\n',
            'argument_count': '.function main 0 1 0 int 1\nHANDLER_PUSH op arm 0 1\nPERFORM op 0\nRET\narm:\nLOAD_LOCAL 0\nEFFECT_RESUME\n.end\n',
            'pop_other_owner': '.function bad 0 0 0 int 1\nHANDLER_POP 1\nPUSH_I64 0\nRET\n.end\n.function main 0 0 0 int 1\nHANDLER_PUSH op arm 0 0\nCALL bad\nRET\narm:\nPUSH_I64 0\nEFFECT_RESUME\n.end\n',
            'tail_from_arm': '.function callee 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n.function main 0 0 0 int 1\nHANDLER_PUSH op arm 0 0\nPERFORM op 0\nRET\narm:\nTAIL_CALL callee\n.end\n',
        }
        for name, body in cases.items():
            with self.subTest(case=name):
                self.exercise(body, traps=True)

    def test_activation_temporaries_start_void(self):
        self.exercise('''.function main 0 2 0 int 1
HANDLER_PUSH op arm 1 0
PUSH_I64 42
STORE_LOCAL 1
PERFORM op 0
ASSERT
HANDLER_POP 1
LOAD_LOCAL 1
PUSH_I64 42
EQ
ASSERT
PUSH_I64 0
RET
arm:
LOAD_LOCAL 1
TYPE_CHECK 0
EFFECT_RESUME
.end
''')

    def test_lexical_return_prunes_intervening_handlers(self):
        self.exercise('''.string inner "Other.ask"
.function caller 0 0 0 int 1
HANDLER_PUSH inner inner_arm 0 0
PERFORM op 0
RET
inner_arm:
PUSH_I64 99
EFFECT_RESUME
.end
.function owner 0 0 0 int 1
HANDLER_PUSH op arm 0 0
CALL caller
RET
arm:
PUSH_I64 7
RET
.end
.function main 0 0 0 int 1
CALL owner
PUSH_I64 7
EQ
ASSERT
HANDLER_PUSH inner fresh 0 0
PERFORM inner 0
PUSH_I64 8
EQ
ASSERT
HANDLER_POP 1
PUSH_I64 0
RET
fresh:
PUSH_I64 8
EFFECT_RESUME
.end
''')

    def test_callable_arguments_and_results_retain_targets(self):
        self.exercise(""".function plus 1 1 0 int 1
LOAD_LOCAL 0
PUSH_I64 2
I64_ADD
RET
.end
.function main 0 1 0 int 1
HANDLER_PUSH op arm 0 1
PUSH_I64 3
FUNCREF plus
PERFORM op 1
CALL_INDIRECT 1 1
PUSH_I64 5
EQ
ASSERT
HANDLER_POP 1
PUSH_I64 0
RET
arm:
LOAD_LOCAL 0
EFFECT_RESUME
.end
""")

    def test_backward_handler_entry(self):
        self.exercise(""".function main 0 1 0 int 1
JMP body
arm:
LOAD_LOCAL 0
PUSH_I64 1
I64_ADD
EFFECT_RESUME
body:
HANDLER_PUSH op arm 0 1
PUSH_I64 4
PERFORM op 1
PUSH_I64 5
EQ
ASSERT
HANDLER_POP 1
PUSH_I64 0
RET
.end
""")

    def test_implicit_arm_completion_resumes(self):
        self.exercise(""".function main 0 0 0 int 1
HANDLER_PUSH op arm 0 0
PERFORM op 0
PUSH_I64 5
EQ
ASSERT
HANDLER_POP 1
PUSH_I64 0
RET
arm:
PUSH_I64 5
.end
""")

    def test_entry_argument_count(self):
        self.exercise('.function main 0 1 0 int 1\nHANDLER_PUSH op arm 0 1\nHANDLER_POP 1\nHANDLER_PUSH op arm 0 0\nPERFORM op 0\nPUSH_I64 7\nEQ\nASSERT\nHANDLER_POP 1\nPUSH_I64 0\nRET\narm:\nPUSH_I64 7\nEFFECT_RESUME\n.end\n')

    def test_halt_in_arm(self):
        self.exercise('.function owner 0 0 0 int 1\nHANDLER_PUSH op arm 0 0\nPERFORM op 0\nRET\narm:\nPUSH_I64 0\nHALT\n.end\n.function main 0 0 0 int 1\nCALL owner\nPOP\nPUSH_I64 99\nRET\n.end\n')

    def test_halt_preserves_suspended_integer_top(self):
        self.exercise(""".function owner 0 0 0 int 1
HANDLER_PUSH op arm 0 0
PERFORM op 0
RET
arm:
HALT
.end
.function main 0 0 0 int 1
PUSH_I64 7
CALL owner
POP
POP
PUSH_I64 99
RET
.end
""", expected=7)

    def test_halt_releases_allocated_strings(self):
        self.exercise(""".string a "hello"
.string b "world"
.function owner 0 0 0 int 1
HANDLER_PUSH op arm 0 0
PERFORM op 0
RET
arm:
PUSH_STR a
PUSH_STR b
STR_CONCAT
HALT
.end
.function main 0 0 0 int 1
CALL owner
POP
PUSH_I64 99
RET
.end
""")


if __name__ == '__main__':
    unittest.main()
