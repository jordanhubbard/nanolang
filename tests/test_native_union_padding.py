"""I leave absent variant fields without fabricated scalar type facts."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeUnionPadding(unittest.TestCase):
    def command(self, *args):
        result = subprocess.run([str(a) for a in args], cwd=ROOT, text=True,
                                capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        return result

    def module(self, directory, text):
        source, module = directory/'input.nasm', directory/'input.nvm'
        source.write_text(text)
        self.command(ROOT/'bin/nanoisa', 'asm', source, '-o', module)
        return module

    def test_empty_and_scalar_data_calls_returns_and_join_orders(self):
        data = 'PUSH_I64 7\nPUSH_STR 0\nPUSH_BOOL 1\nPUSH_F64 1.5\nAGG_PACK 1 0 0 4\n'
        empty = 'AGG_PACK 1 0 1 0\n'
        read = '''.function read 1 1 0 int 1
LOAD_LOCAL 0
AGG_TAG
PUSH_I64 0
EQ
JMP_FALSE empty
LOAD_LOCAL 0
AGG_GET 0
PUSH_I64 7
EQ
ASSERT
LOAD_LOCAL 0
AGG_GET 1
PUSH_STR 0
EQ
ASSERT
LOAD_LOCAL 0
AGG_GET 2
ASSERT
LOAD_LOCAL 0
AGG_GET 3
PUSH_F64 1.5
F64_EQ
ASSERT
empty:
PUSH_I64 0
RET
.end
'''
        for reverse in (False, True):
            for select in (0, 1):
                with self.subTest(reverse=reverse, select=select), tempfile.TemporaryDirectory(prefix='union-padding-') as tmp:
                    directory = Path(tmp)
                    left, right = (empty, data) if reverse else (data, empty)
                    text = '.string 0 "kept"\n.types 0 0 1\n.entry main\n.function main 0 0 0 int 1\n'
                    text += left+'CALL 2\nPOP\n'+right+'CALL 2\nPOP\n'
                    text += f'PUSH_BOOL {select}\nCALL 1\nCALL 2\nRET\n.end\n'
                    text += '.function choose 1 1 0 union 1\nLOAD_LOCAL 0\nJMP_FALSE other\n'+left+'JMP result\nother:\n'+right+'result:\nRET\n.end\n'+read
                    module = self.module(directory, text)
                    vm = self.command(ROOT/'bin/nano_vm', module)
                    source, binary = directory/'native.c', directory/'native'
                    self.command(ROOT/'bin/nvm2c', module, '-o', source)
                    self.command(os.environ.get('CC', 'cc'), '-std=c11', '-Wall', '-Wextra', '-Werror',
                                 '-fsanitize=address,undefined', '-fno-omit-frame-pointer', source, '-lm', '-o', binary)
                    native = self.command(binary)
                    self.assertEqual(vm.stdout, native.stdout)

    def test_conflicting_heap_and_scalar_fields_still_refuse(self):
        text = '''.string 0 "kept"
.types 0 0 1
.entry main
.function main 0 0 0 int 1
PUSH_I64 7
AGG_PACK 1 0 0 1
CALL 1
POP
ARR_NEW 1
AGG_PACK 1 0 1 1
CALL 1
RET
.end
.function read 1 1 0 int 1
LOAD_LOCAL 0
AGG_TAG
RET
.end
'''
        with tempfile.TemporaryDirectory(prefix='union-present-') as tmp:
            directory = Path(tmp)
            module = self.module(directory, text)
            output = directory/'retained.c'
            output.write_text('retained')
            result = subprocess.run([ROOT/'bin/nvm2c', module, '-o', output], cwd=ROOT,
                                    capture_output=True, text=True, timeout=120)
            self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('conflicting kinds', result.stderr)
            self.assertEqual(output.read_text(), 'retained')
