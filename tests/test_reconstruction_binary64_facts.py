"""I retain operand bits and distinguish admitted arithmetic from refused output effects."""
from pathlib import Path
import json
import struct
import subprocess
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]

class Binary64Facts(unittest.TestCase):
    def checked(self, *args):
        result = subprocess.run([str(a) for a in args], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def module(self, directory, extra=''):
        source, module = directory/'input.nasm', directory/'input.nvm'
        source.write_text('.entry main\n.function main 0 0 0 int 1\n'
                          'PUSH_F64 1.23456789\n'+extra+'POP\nPUSH_BOOL 1\nPOP\nPUSH_I64 37\nRET\n.end\n')
        self.checked(ROOT/'bin/nanoisa', 'asm', source, '-o', module)
        return module

    def test_exact_binary64_encodings(self):
        patterns = (0, 1 << 63, 1, 0x8000000000000001,
                    0x000fffffffffffff, 0x0010000000000000,
                    0x7fefffffffffffff, 0xffefffffffffffff,
                    0x3ff0000000000000, 0xbff0000000000000,
                    0x7ff0000000000000, 0xfff0000000000000,
                    0x7ff8000000000001, 0x7ff8123456789abc,
                    0xfff8000000001234, 0x7ff0000000000001,
                    0xfff0000000000042)
        with tempfile.TemporaryDirectory() as tmp:
            module = self.module(Path(tmp))
            original = module.read_bytes()
            marker = struct.pack('<d', 1.23456789)
            self.assertEqual(original.count(marker), 1)
            # Every replacement is a valid binary64 constant; only its operand
            # bytes change. The facts executable loads and verifies each module.
            for bits in patterns:
                with self.subTest(bits=f'{bits:016x}'):
                    encoded = bytearray(original.replace(marker, struct.pack('<Q', bits)))
                    header_size = struct.unpack_from('<I', encoded, 20)[0]
                    struct.pack_into('<I', encoded, 36, zlib.crc32(encoded[header_size:]))
                    module.write_bytes(encoded)
                    facts = json.loads(self.checked(ROOT/'bin/nanoisa_hl_facts', module))
                    code = facts['functions'][0]['code']
                    self.assertEqual(code[0]['f64_bits'], f'{bits:016x}')
                    self.assertEqual(code[0]['arg'], 0)
                    self.assertEqual(code[2]['arg'], 1)
                    self.assertEqual(code[4]['arg'], 37)
                    self.assertTrue(all('f64_bits' not in ins for ins in code[1:]))

    def test_current_arithmetic_reconstruction_is_admitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            module = self.module(directory, 'DUP\nF64_ADD\n')
            self.checked(ROOT/'bin/nanoisa', 'verify', module)
            for target in ('c', 'nano'):
                output = directory/f'arithmetic.{target}'
                self.checked(ROOT/'bin/nvm2hl', module, '--language', target,
                             '-o', output)
                text = output.read_text()
                self.assertIn('nano_rt_f64_add(' if target == 'c' else '(+ ', text)
                self.assertIn('nlr_f64_from_bits' if target == 'c' else 'float_from_bits', text)

    def test_source_refusal_preserves_previous_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            module = self.module(directory, 'DUP\nF64_ADD\nDUP\nPRINT\n')
            self.checked(ROOT/'bin/nanoisa', 'verify', module)
            for target in ('c', 'nano'):
                output = directory/f'previous.{target}'
                output.write_text('retained output\n')
                result = subprocess.run([ROOT/'bin/nvm2hl', module, '--language', target,
                                         '-o', output], cwd=ROOT, capture_output=True,
                                        text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('PRINT', result.stderr)
                self.assertEqual(output.read_text(), 'retained output\n')

if __name__ == '__main__':
    unittest.main()
