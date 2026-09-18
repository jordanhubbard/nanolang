"""I roundtrip exact F64 operands without executing float programs."""
from pathlib import Path
import json
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = (0, 1 << 63, 1, 0x8000000000000001, 0x000fffffffffffff,
            0x0010000000000000, 0x7fefffffffffffff, 0xffefffffffffffff,
            0x3ff0000000000000, 0xbff0000000000000,
            0x7ff0000000000000, 0xfff0000000000000,
            0x7ff8000000000001, 0x7ff8123456789abc,
            0xfff8000000001234, 0x7ff0000000000001, 0xfff0000000000042)

class CanonicalF64Bits(unittest.TestCase):
    def checked(self, *args):
        result = subprocess.run([str(a) for a in args], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def source(self, operand):
        return ('.entry main\n.function main 0 0 0 int 1\nPUSH_F64 '+operand+
                '\nPOP\nPUSH_I64 0\nRET\n.end\n')

    def roundtrip(self, directory, operand, bits):
        source, module = directory/'input.nasm', directory/'input.nvm'
        dumped, repeated = directory/'dump.nasm', directory/'repeated.nvm'
        source.write_text(self.source(operand))
        self.checked(ROOT/'bin/nanoisa', 'asm', source, '-o', module)
        text = self.checked(ROOT/'bin/nanoisa', 'dump', module)
        self.assertIn(f'PUSH_F64 bits:{bits:016x}', text)
        dumped.write_text(text)
        self.checked(ROOT/'bin/nanoisa', 'asm', dumped, '-o', repeated)
        self.assertEqual(module.read_bytes(), repeated.read_bytes())
        facts = json.loads(self.checked(ROOT/'bin/nanoisa_hl_facts', repeated))
        self.assertEqual(facts['functions'][0]['code'][0]['f64_bits'], f'{bits:016x}')

    def test_exact_patterns_and_uppercase_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bits in PATTERNS:
                with self.subTest(bits=f'{bits:016x}'):
                    self.roundtrip(Path(tmp), f'bits:{bits:016X}', bits)

    def test_legacy_decimal_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            for text, bits in (('0.0', 0), ('-0.0', 1 << 63),
                               ('1.5', 0x3ff8000000000000),
                               ('-2.25', 0xc002000000000000),
                               ('inf', 0x7ff0000000000000)):
                with self.subTest(text=text):
                    self.roundtrip(Path(tmp), text, bits)

    def test_existing_comment_and_whitespace_delimiters(self):
        with tempfile.TemporaryDirectory() as tmp:
            for suffix in ('; adjacent comment', '# adjacent comment',
                           ' \t ; spaced comment', ' \t '):
                with self.subTest(suffix=suffix):
                    self.roundtrip(Path(tmp), 'bits:8000000000000000'+suffix, 1 << 63)

    def test_parser_refusals_preserve_output(self):
        invalid = ('bits:', 'bits:0', 'bits:000000000000000',
                   'bits:00000000000000000', 'bits:000000000000000g',
                   'bits:0000000000000000x', 'bits:0000000000000000 extra',
                   'bits: 0000000000000000', 'Bits:0000000000000000',
                   'raw:0000000000000000', 'bits:-000000000000000')
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'input.nasm', Path(tmp)/'previous.nvm'
            for token in invalid:
                with self.subTest(token=token):
                    source.write_text(self.source(token))
                    output.write_bytes(b'retained output')
                    result = subprocess.run([ROOT/'bin/nanoisa', 'asm', source, '-o', output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=30)
                    self.assertGreater(result.returncode, 0)
                    self.assertNotIn('AddressSanitizer', result.stderr)
                    self.assertNotIn('runtime error:', result.stderr)
                    self.assertEqual(output.read_bytes(), b'retained output')

if __name__ == '__main__':
    unittest.main()
