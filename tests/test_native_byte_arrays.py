"""I preserve byte element tags and narrowing through native array storage."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeByteArrays(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix="nano-native-byte-arrays-"))
        cls.sequence = 0
        print("I retain byte-array evidence at", cls.work, flush=True)

    def command(self, *args):
        type(self).sequence += 1
        result = subprocess.run(list(map(str, args)), cwd=ROOT, text=True,
                                capture_output=True, timeout=120)
        (self.work / f"command-{self.sequence}.log").write_text(
            repr(list(map(str, args))) + "\n" + result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def paired(self, name, body):
        assembly = self.work / (name + ".nasm")
        assembly.write_text('.entry main\n.function main 0 3 0 int 1\n' + body +
                            'PUSH_I64 0\nRET\n.end\n')
        module = assembly.with_suffix('.nvm')
        self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
        self.command(ROOT / 'bin/nano_vm', '--verify-only', module)
        vm = self.command(ROOT / 'bin/nano_vm', module)
        native_c, binary = assembly.with_suffix('.c'), assembly.with_suffix('.native')
        self.command(ROOT / 'bin/nvm2c', module, '-o', native_c)
        self.command(*shlex.split(os.environ.get('CC', 'clang')), '-std=c11',
                     '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                     '-fno-sanitize-recover=all', native_c, '-lm', '-o', binary)
        native = self.command(binary)
        self.assertEqual(vm.stdout, native.stdout)

    @staticmethod
    def read(slot, index, expected):
        return (f'LOAD_LOCAL {slot}\nPUSH_I64 {index}\nARR_GET\n'
                'DUP\nTYPE_CHECK 2\nASSERT\nCAST_INT\n'
                f'PUSH_I64 {expected}\nI64_EQ\nASSERT\n')

    def test_literal_tags_and_integer_narrowing(self):
        body = 'PUSH_U8 255\nPUSH_I64 300\nARR_LITERAL 2 2\nSTORE_LOCAL 0\n'
        body += self.read(0, 0, 255) + self.read(0, 1, 44)
        body += 'LOAD_LOCAL 0\nPUSH_I64 -1\nARR_PUSH\nSTORE_LOCAL 0\n'
        body += self.read(0, 2, 255)
        self.paired('literal', body)

    def test_alias_mutation_slice_independence_and_empty(self):
        body = ('ARR_NEW 2\nPUSH_I64 300\nARR_PUSH\nSTORE_LOCAL 0\n'
                'LOAD_LOCAL 0\nSTORE_LOCAL 1\n'
                'LOAD_LOCAL 0\nPUSH_I64 0\nPUSH_I64 1\nARR_SLICE\nSTORE_LOCAL 2\n'
                'LOAD_LOCAL 1\nPUSH_I64 0\nPUSH_I64 511\nARR_SET\nPOP\n')
        body += self.read(0, 0, 255) + self.read(1, 0, 255) + self.read(2, 0, 44)
        body += ('LOAD_LOCAL 0\nPUSH_I64 1\nPUSH_I64 0\nARR_SLICE\n'
                 'ARR_LEN\nPUSH_I64 0\nI64_EQ\nASSERT\n')
        self.paired('slice', body)
