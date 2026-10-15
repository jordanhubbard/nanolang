"""I check contextual array literals before self-hosted NanoISA publication."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from tests.test_byte_array_literals import BYTE_CONTEXTS

ROOT = Path(__file__).resolve().parents[1]


class CheckedArrayLiterals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='checked-array-literals-'))
        cls.serial = 0
        cls.driver = Path(os.environ.get('NANOLANG_ARRAY_CONTEXT_DRIVER', cls.work / 'driver'))
        print('I retain checked array evidence at', cls.work, flush=True)
        if 'NANOLANG_ARRAY_CONTEXT_DRIVER' not in os.environ:
            source = cls.work / 'driver.nano'
            source.write_text((ROOT / 'tests/nanoisa/fixtures/checked_owner_selection_driver.nano.txt').read_text())
            cls.command(ROOT / 'bin/nanoc_c', source, '-o', cls.driver)

    @classmethod
    def command(cls, *args, expected=0):
        cls.serial += 1
        result = subprocess.run(list(map(str, args)), cwd=ROOT, text=True,
                                capture_output=True, timeout=600)
        (cls.work / f'command-{cls.serial}.log').write_text(repr(list(map(str, args))) + '\n' + result.stdout + result.stderr)
        if result.returncode != expected:
            raise AssertionError(f'{args}: {result.returncode}\n{result.stdout}{result.stderr}')
        return result

    def qualify(self, name, source):
        path = self.work / (name + '.nano')
        path.write_text(source if 'shadow main' in source else source + '\nshadow main { assert true }\n')
        assembly = path.with_suffix('.nasm')
        self.command(self.driver, path, 'program', '0', assembly)
        module = path.with_suffix('.nvm')
        self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
        self.command(ROOT / 'bin/nano_vm', '--verify-only', module)
        self.command(ROOT / 'bin/nano_vm', module)
        generated, binary = path.with_suffix('.c'), path.with_suffix('.native')
        self.command(ROOT / 'bin/nvm2c', module, '-o', generated)
        self.command(*shlex.split(os.environ.get('CC', 'clang')), '-std=c11', '-Wall', '-Wextra', '-Werror',
                     '-fsanitize=address,undefined', '-fno-sanitize-recover=all', generated, '-lm', '-o', binary)
        self.command(binary)

    def test_all_byte_destinations(self):
        for name, source in BYTE_CONTEXTS.items():
            with self.subTest(context=name):
                self.qualify(name, source)

    def test_complete_slice_kind_bit_and_identity_fixture(self):
        self.qualify('all-kinds', (ROOT / 'tests/nanoisa/fixtures/array_slice_all_kinds.nano.txt').read_text())

    def test_string_builtin_literal_element(self):
        self.qualify('trimmed-string', 'fn main()->int { let spelling:string = " Choice<int>" let values:array<string> = [(str_trim (str_substring spelling 0 7))] assert (== (at values 0) "Choice") return 0 }')

    def test_nested_literals_keep_context(self):
        self.qualify('nested', 'fn main()->int { let values:array<array<u8>> = [[300], []] assert (== (at (at values 0) 0) 44) assert (== (array_length (at values 1)) 0) return 0 }')

    def test_invalid_elements_and_existing_storage_preserve_output(self):
        templates = [
            'fn main()->int { let values:array<u8> = VALUE return 0 }',
            'fn main()->int { let mut values:array<u8> = [] set values VALUE return 0 }',
            'let values:array<u8> = VALUE fn main()->int { return 0 }',
            'fn values()->array<u8> { return VALUE } fn main()->int { return 0 }',
            'fn consume(values:array<u8>)->int { return 0 } fn main()->int { return (consume VALUE) }',
            'struct Box { values:array<u8> } fn main()->int { let box:Box = Box { values:VALUE } return 0 }',
        ]
        cases = [template.replace('VALUE', value) for template in templates for value in ('[true]', '[1.5]', '["text"]')]
        cases += ['fn main()->int { let child:array<int> = [7] let values:array<array<u8>> = [child] return 0 }',
                  'fn main()->int { let values:array<array<u8>> = [[true]] return 0 }']
        for i, source in enumerate(cases):
            with self.subTest(case=i):
                path, output = self.work / f'invalid-{i}.nano', self.work / f'invalid-{i}.nasm'
                path.write_text(source if 'shadow main' in source else source + '\nshadow main { assert true }\n')
                output.write_bytes(b'prior output')
                self.command(self.driver, path, 'program', '0', output, expected=1)
                self.assertEqual(output.read_bytes(), b'prior output')
