"""I compare complete decoded operands and re-encoded bytes with the actual C ISA."""
from pathlib import Path
import os
import random
import struct
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked

ROOT = Path(__file__).resolve().parents[1]
REFUSED = '0:-1:-1:-1:0:'


def window(code, offset=0, available=None):
    if available is None:
        available = len(code) - offset
    return struct.pack('<QQ', offset, available) + code


def explicit_cases():
    cases = [('nop', window(b'\0'), '1:0:0:0:0:00'),
             ('reserved-prefix', window(b'\xff\0'), REFUSED)]
    for value in (0, 1, -1, -(2**63), 2**63-1, -(2**63)+1, 2**32, -(2**32)):
        bits = value % (2**64)
        code = b'\x01' + struct.pack('<q', value)
        expected = f'9:1:0:1:1:5:{bits & 0xffffffff}:{bits >> 32}:{value}:{code.hex()}'
        cases.append((f'i64-{value}', window(code), expected))
    for value in (0, 1, -1, -(2**31), 2**31-1, -(2**31)+1):
        code = b'\x38' + struct.pack('<i', value)
        cases.append((f'i32-{value}', window(code), f'5:56:0:0:1:4:{value % 2**32}:0:{value}:{code.hex()}'))
    for bits in (0, 1, 2**63, 0x7ff0000000000000, 0xfff0000000000000,
                 0x7ff8000000000001, 0x7ff0000000000001, 0xfff123456789abcd,
                 0xffffffffffffffff, 0x0010000000000000, 0x000fffffffffffff):
        code = b'\x02' + struct.pack('<Q', bits)
        cases.append((f'f64-bits-{bits:x}', window(code), f'9:2:0:1:1:6:{bits & 0xffffffff}:{bits >> 32}:0:{code.hex()}'))
    for value in (0, 1, 127, 128, 255):
        code = bytes([3, value])
        cases.append((f'u8-{value}', window(code), f'2:3:0:1:1:1:{value}:0:{value}:{code.hex()}'))
    for value in (0, 1, 32767, 32768, 65535):
        code = b'\x10' + struct.pack('<H', value)
        cases.append((f'u16-{value}', window(code), f'3:16:0:1:1:2:{value}:0:{value}:{code.hex()}'))
    for left, right in ((0, 0), (2**32-1, 65535), (2**31, 32768), (145, 152)):
        code = b'\x0f' + struct.pack('<IH', left, right)
        cases.append((f'compound-{left}-{right}', window(code), f'7:15:0:1:2:3:{left}:0:{left}:2:{right}:0:{right}:{code.hex()}'))
    for offset, size in ((0, 0), (1, 0), (2, 0), (0, 2), (1, 1),
                         (2**63-1, 1), (2**63, 1), (2**64-1, 1),
                         (0, 2**63-1), (0, 2**63), (1, 2**64-1)):
        cases.append((f'bad-window-{offset}-{size}', window(b'\0', offset, size), REFUSED))
    for length in range(16):
        cases.append((f'short-header-{length}', b'\0'*length, REFUSED))
    return cases


class NanolangTranslatorInstructionsTests(unittest.TestCase):
    def test_primary_decoding(self):
        with tempfile.TemporaryDirectory(prefix='nano-nanolang-instructions-') as name:
            directory = Path(name)
            oracle = ROOT/'obj/nvm2c_instruction_reference'

            def materialize(cases, prefix):
                paths = []
                for index, (_, data, _) in enumerate(cases):
                    path = directory/f'{prefix}-{index}.bin'
                    path.write_bytes(data)
                    paths.append(path)
                return paths

            def execute(command, paths):
                values = []
                for start in range(0, len(paths), 128):
                    values.extend(checked([*command, *paths[start:start+128]]).splitlines())
                self.assertEqual(len(values), len(paths))
                return values

            probes = materialize([(str(op), window(bytes([op])+b'\0'*63), None)
                                  for op in range(256)], 'probe')
            widths = [int(value.split(':', 1)[0]) for value in execute([oracle], probes)]
            cases = explicit_cases()
            explicit_count = len(cases)
            randomizer = random.Random(1142)
            for op, width in enumerate(widths):
                for size in range(width):
                    code = bytes([op])+b'\xff'*max(0, size-1) if size else b''
                    cases.append((f'truncated-{op}-{size}', window(code), REFUSED))
                for pattern in (b'\0'*63, b'\xff'*63, b'\x80'*63,
                                bytes([0x55, 0xaa])*32, randomizer.randbytes(63)):
                    code = bytes([op])+pattern
                    cases.append((f'pattern-{op}-{pattern[:4].hex()}', window(code), None))
                    cases.append((f'offset-{op}-{pattern[:4].hex()}', window(b'\xff'+code+b'\xfe', 1, len(code)), None))
                if width:
                    code = bytes([op])+b'\xff'*(width-1)
                    cases.append((f'exact-{op}', window(code), None))
                    cases.append((f'window-truncation-{op}', window(code+b'\0', 0, width-1), REFUSED))
            paths = materialize(cases, 'case')
            reference = execute([oracle], paths)
            for (case, _, expected), actual in zip(cases, reference):
                if expected is not None:
                    with self.subTest(reference=case): self.assertEqual(actual, expected)
            routes = build_routes(ROOT/'tests/nvm2c_instruction_driver.nano', directory)
            for route, command in routes:
                values = execute(command, paths)
                for (case, _, _), expected, actual in zip(cases, reference, values):
                    with self.subTest(route=route, case=case): self.assertEqual(actual, expected)
            print(f'I matched {len(cases)} instruction decisions across {len(routes)} routes; '
                  f'{sum(c[2] is not None for c in cases)} explicit decisions include '
                  f'{explicit_count} integer/bit/window controls. All 256 primary allocations were probed.')
