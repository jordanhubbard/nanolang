"""I retain terminal false-assertion fallthrough without changing HALT policy."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeFalseAssert(unittest.TestCase):
    def command(self, *args):
        result = subprocess.run([str(x) for x in args], cwd=ROOT, text=True,
                                capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        return result

    def exercise(self, body, tag='float', argument=1, refusal=False):
        push = 'PUSH_F64 2.5' if tag == 'float' else 'PUSH_STR 0'
        compare = 'F64_EQ' if tag == 'float' else 'EQ'
        text = f'''.string 0 "kept"
.entry main
.function main 0 0 0 int 1
PUSH_BOOL {argument}
CALL 1
{push}
{compare}
ASSERT
PUSH_I64 0
RET
.end
.function choose 1 1 0 {tag} 1
{body.replace('VALUE', push)}
.end
'''
        with tempfile.TemporaryDirectory(prefix='native-false-assert-') as tmp:
            root = Path(tmp)
            assembly, module = root/'case.nasm', root/'case.nvm'
            assembly.write_text(text)
            self.command(ROOT/'bin/nanoisa', 'asm', assembly, '-o', module)
            self.command(ROOT/'bin/nano_vm', '--verify-only', module)
            source, binary = root/'native.c', root/'native'
            source.write_text('retained')
            result = subprocess.run([ROOT/'bin/nvm2c', module, '-o', source], cwd=ROOT,
                                    text=True, capture_output=True, timeout=120)
            if refusal:
                self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                self.assertIn('HALT with unexpected stack height 0', result.stderr)
                self.assertEqual(source.read_text(), 'retained')
                return
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            vm = self.command(ROOT/'bin/nano_vm', module)
            self.command(os.environ.get('CC', 'cc'), '-std=c11', '-Wall', '-Wextra', '-Werror',
                         '-fsanitize=address,undefined', '-fno-omit-frame-pointer', source, '-lm', '-o', binary)
            native = self.command(binary)
            self.assertEqual(native.stdout, vm.stdout)

    def test_impossible_fallback_is_terminal_for_float_and_string(self):
        for tag in ('float', 'string'):
            with self.subTest(tag=tag):
                self.exercise('LOAD_LOCAL 0\nJMP_FALSE missing\nVALUE\nRET\nmissing:\nPUSH_BOOL 0\nASSERT\nHALT', tag)

    def test_branch_entry_at_assert_keeps_its_incoming_condition(self):
        self.exercise('LOAD_LOCAL 0\nJMP check\nPUSH_BOOL 0\ncheck:\nASSERT\nVALUE\nRET')

    def test_independent_successor_resumes_after_terminal_arm(self):
        self.exercise('LOAD_LOCAL 0\nJMP_FALSE result\nPUSH_BOOL 0\nASSERT\nresult:\nVALUE\nRET', argument=0)

    def test_reachable_halt_and_unproved_conditions_remain_refused(self):
        bodies = (
            'HALT',
            'PUSH_BOOL 1\nASSERT\nHALT',
            'LOAD_LOCAL 0\nASSERT\nHALT',
            'PUSH_BOOL 0\nPOP\nPUSH_BOOL 1\nASSERT\nHALT',
            'LOAD_LOCAL 0\nJMP_FALSE end\nPUSH_BOOL 0\nASSERT\nend:\nHALT',
        )
        for body in bodies:
            for tag in ('float', 'string'):
                with self.subTest(body=body, tag=tag):
                    self.exercise(body, tag, argument=0, refusal=True)


if __name__ == '__main__':
    unittest.main()
