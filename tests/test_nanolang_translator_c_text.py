"""I compile emitted tokens as real C and compare their bytes and integer values."""
from pathlib import Path
import os
import re
import shlex
import struct
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked
from tests.native_process import run_captured

ROOT = Path(__file__).resolve().parents[1]
HALVES = (-1, 0, 1, 2147483647, 2147483648, 4294967295, 4294967296)
SIGNED = (-(1 << 63), -(1 << 63) + 1, -(1 << 31), -1, 0, 1, (1 << 31) - 1, (1 << 63) - 1)


def byte_cases():
    cases = [b'', bytes(range(256)), b'\x00\xff"\\\n\r\t?/?=/* */123\x80\xc0\xaf',
             'I preserve λ and 日本語 as bytes'.encode()]
    for count in (1, 2, 3, 254, 255, 256, 257, 511, 512, 513, 4095, 4096, 4097, 65536):
        cases.append(bytes((index * 73 + 19) % 256 for index in range(count)))
    return cases


class CText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='nano-c-text-')
        cls.addClassCleanup(cls.workspace.cleanup)
        cls.directory = Path(cls.workspace.name)
        cls.cases = byte_cases()
        cls.paths = []
        for index, data in enumerate(cls.cases):
            path = cls.directory / f'bytes-{index}.bin'
            path.write_bytes(data)
            cls.paths.append(path)
        cls.routes = build_routes(ROOT / 'tests/nvm2c_c_text_driver.nano', cls.directory)

    def reports(self, route):
        rows = [line.split('|', 3 if line.startswith(('U64|', 'N|')) else 2)
                for line in checked([*route, *self.paths]).splitlines()]
        self.assertEqual(len(rows), 7 + 49 + 56 + 8 + 258 + len(self.cases))
        keys = []
        for low in HALVES:
            keys.append(['U32', str(low)])
            keys.extend(['U64', str(low), str(high)] for high in HALVES)
            keys.extend(['N', str(kind), str(low)] for kind in range(-1, 7))
        keys.extend(['I64', str(value)] for value in SIGNED)
        keys.extend(['O', str(value)] for value in range(-1, 257))
        keys.extend(['B', str(len(data))] for data in self.cases)
        self.assertEqual([row[:-1] for row in rows], keys)
        return rows

    def test_actual_c_roundtrips(self):
        for route_name, route in self.routes:
            with self.subTest(route=route_name):
                rows = self.reports(route)
                definitions = ['#include <stdint.h>', '#include <stdio.h>', '#include <limits.h>',
                               '_Static_assert(CHAR_BIT == 8, "I require eight-bit bytes");']
                statements = []
                expected = bytearray()
                counts = dict(U32=0, U64=0, I64=0, N=0, O=0, B=0)
                for row in rows:
                    kind = row[0]
                    index = sum(counts.values())
                    counts[kind] += 1
                    token = row[-1]
                    if kind == 'U32':
                        value = int(row[1])
                        self.assertIn(value, HALVES)
                        if not 0 <= value < 1 << 32:
                            self.assertEqual(token, '')
                            continue
                        statements.append(f'emit({token},4);')
                        expected += struct.pack('<I', value)
                    elif kind == 'U64':
                        low, high = map(int, row[1:3])
                        self.assertIn(low, HALVES)
                        self.assertIn(high, HALVES)
                        if not (0 <= low < 1 << 32 and 0 <= high < 1 << 32):
                            self.assertEqual(token, '')
                            continue
                        statements.append(f'emit({token},8);')
                        expected += struct.pack('<Q', high << 32 | low)
                    elif kind == 'I64':
                        value = int(row[1])
                        self.assertIn(value, SIGNED)
                        statements.append(f'emit((uint64_t)({token}),8);')
                        expected += struct.pack('<q', value)
                    elif kind == 'N':
                        category, ordinal = map(int, row[1:3])
                        if not (0 <= category < 6 and 0 <= ordinal < 1 << 32):
                            self.assertEqual(token, '')
                            continue
                        self.assertRegex(token, r'\A[A-Za-z_][A-Za-z_0-9]*\Z')
                        identity = category << 32 | ordinal
                        definitions.append(f'static const uint64_t {token} = UINT64_C({identity});')
                        statements.append(f'emit({token},8);')
                        expected += struct.pack('<Q', identity)
                    elif kind == 'O':
                        value = int(row[1])
                        if not 0 <= value <= 255:
                            self.assertEqual(token, '')
                            continue
                        self.assertRegex(token, r'\A\\[0-3][0-7][0-7]\Z')
                        definitions.append(f'static const unsigned char octet_{index}[] = "{token}";')
                        statements.append(f'emit(octet_{index}[0],1);')
                        expected.append(value)
                    elif kind == 'B':
                        data = self.cases[counts['B'] - 1]
                        count = int(row[1])
                        self.assertEqual(count, len(data))
                        self.assertTrue(re.fullmatch(r'"(?:\\[0-3][0-7][0-7])*"', token))
                        definitions.append(f'static const unsigned char bytes_{index}[] = {token};')
                        definitions.append(f'_Static_assert(sizeof bytes_{index} - 1 == {count}, "I preserve length");')
                        statements.append(f'emit({count},4); if(fwrite(bytes_{index},1,{count},stdout)!={count}) return 12;')
                        expected += struct.pack('<I', count) + data
                    else:
                        self.fail(f'I do not recognize report {row!r}')
                self.assertEqual(counts, dict(U32=7, U64=49, I64=8, N=56, O=258, B=len(self.cases)))
                definitions.append('static void emit(uint64_t value,unsigned count){'
                                   'for(unsigned i=0;i<count;i++){if(putchar((unsigned char)(value>>(8*i)))==EOF)return;}}')
                source = self.directory / f'roundtrip-{route_name}.c'
                binary = source.with_suffix('')
                source.write_text('\n'.join(definitions) + '\nint main(void){\n' +
                                  '\n'.join(statements) + '\nreturn ferror(stdout)?11:0;}\n')
                checked([*shlex.split(os.environ.get('CC', 'cc')), '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                         '-fsanitize=undefined', '-fno-sanitize-recover=all', source, '-o', binary], build=True)
                # I compare binary stdout; decoding it as UTF-8 would lose arbitrary bytes.
                executed = run_captured([binary], cwd=ROOT, timeout=60)
                self.assertEqual(executed.returncode, 0, executed.stderr)
                self.assertEqual(executed.stdout, expected)


    def test_oversized_bytes_refuse(self):
        path = self.directory / 'oversized.bin'
        with path.open('wb') as stream:
            stream.truncate(16777217)
        for route_name, route in self.routes:
            with self.subTest(route=route_name):
                result = run_captured([*route, path], cwd=ROOT, timeout=60)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(result.stdout.splitlines()[-1], b'B|0|')
                self.assertEqual(result.stderr, b'')


if __name__ == '__main__':
    unittest.main()
