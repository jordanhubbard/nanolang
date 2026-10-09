"""I preserve byte element tags and narrowing through native array storage."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from tests.test_native_closures import CHURN

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

    def paired(self, name, body, helpers='', collections=False):
        assembly = self.work / (name + ".nasm")
        assembly.write_text('.entry main\n' + helpers + '.function main 0 3 0 int 1\n' + body +
                            'PUSH_I64 0\nRET\n.end\n')
        module = assembly.with_suffix('.nvm')
        self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
        self.command(ROOT / 'bin/nano_vm', '--verify-only', module)
        vm = self.command(ROOT / 'bin/nano_vm', module)
        native_c, binary = assembly.with_suffix('.c'), assembly.with_suffix('.native')
        self.command(ROOT / 'bin/nvm2c', module, '-o', native_c)
        if collections:
            generated = native_c.read_text()
            marker = 'static void nmap_collect(void) {\n'
            self.assertIn(marker, generated)
            generated = generated.replace(marker, 'static size_t collections;\n' + marker + '    ++collections;\n')
            native_c.write_text('#define main fixture_main\n' + generated +
                                '\n#undef main\nint main(void) { int result = fixture_main();\n'
                                '    if (collections < 2) abort(); return result; }\n')
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

    def test_calls_globals_records_and_absent_reads(self):
        helpers = ('.function identity 1 1 0 array 1\n.parameters identity array\n'
                   'LOAD_LOCAL 0\nRET\n.end\n')
        body = ('PUSH_U8 9\nARR_LITERAL 2 1\nCALL identity\nSTORE_GLOBAL 0\n'
                'LOAD_GLOBAL 0\nAGG_PACK 0 0 0 1\nSTORE_LOCAL 0\n'
                'LOAD_LOCAL 0\nAGG_GET 0\nSTORE_LOCAL 1\n')
        body += self.read(1, 0, 9)
        body += ('LOAD_LOCAL 1\nPUSH_I64 0\nPUSH_I64 300\nARR_SET\nPOP\n'
                 'LOAD_GLOBAL 0\nSTORE_LOCAL 2\n')
        body += self.read(2, 0, 44)
        body += ('LOAD_LOCAL 2\nPUSH_I64 2\nARR_GET\nTYPE_CHECK 0\nASSERT\n'
                 'LOAD_LOCAL 2\nARR_POP\nDUP\nTYPE_CHECK 2\nASSERT\n'
                 'CAST_INT\nPUSH_I64 44\nI64_EQ\nASSERT\n'
                 'LOAD_LOCAL 2\nARR_POP\nTYPE_CHECK 0\nASSERT\n')
        self.paired('transport', body, helpers)

    def test_tagged_integer_pop_keeps_integer_identity(self):
        body = ('PUSH_I64 300\nARR_LITERAL 1 1\nSTORE_GLOBAL 0\n'
                'LOAD_GLOBAL 0\nARR_POP\nDUP\nTYPE_CHECK 1\nASSERT\n'
                'PUSH_I64 300\nI64_EQ\nASSERT\n'
                'LOAD_GLOBAL 0\nARR_POP\nTYPE_CHECK 0\nASSERT\n')
        self.paired('integer-pop', body)

    def test_tagged_record_pop_preserves_owned_child(self):
        body = ('PUSH_STR retained\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\n'
                'STORE_GLOBAL 0\nLOAD_GLOBAL 0\nARR_POP\nSTORE_LOCAL 0\n'
                'LOAD_GLOBAL 0\nARR_POP\nTYPE_CHECK 0\nASSERT\n'
                'LOAD_LOCAL 0\nAGG_GET 0\nPUSH_STR retained\nSTR_EQ\nASSERT\n')
        self.paired('record-pop', body, '.string retained "retained"\n')

    def test_byte_roots_survive_collection(self):
        body = ('PUSH_U8 44\nARR_LITERAL 2 1\nSTORE_LOCAL 0\n'
                'PUSH_U8 55\nARR_LITERAL 2 1\nAGG_PACK 0 0 0 1\nSTORE_LOCAL 1\n'
                'CALL churn\n')
        body += self.read(0, 0, 44)
        body += 'LOAD_LOCAL 1\nAGG_GET 0\nSTORE_LOCAL 2\n' + self.read(2, 0, 55)
        self.paired('collected-bytes', body, '.string a "a"\n.string b "b"\n' + CHURN,
                    collections=True)
