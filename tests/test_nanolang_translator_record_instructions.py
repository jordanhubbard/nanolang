"""I copy existing ARRAY/MAP recipes without granting checked-profile admission."""
from pathlib import Path
import hashlib
import json
import os
import random
import struct
import tempfile
import unittest
from tests.nanolang_translator_routes import build_routes, checked

ROOT = Path(__file__).resolve().parents[1]


def carrier(code, signatures=((0, 1),), current=0, maps=False, start=0, length=None):
    return (struct.pack('<5I', start, len(code)-start if length is None else length,
                        len(signatures), current, int(maps))
            + b''.join(struct.pack('<II', *signature) for signature in signatures) + code)


def corpus(widths):
    cases = []
    for maps in (False, True):
        def add(name, code=b'', expected=None, **kwargs):
            cases.append((f'{int(maps)}-{name}', carrier(code, maps=maps, **kwargs), expected))
        add('empty', expected=True)
        add('nop', b'\0', True)
        add('last-function', signatures=((0, 0),)*256, current=255, expected=True)
        add('too-many-functions', signatures=((0, 0),)*257, expected=False)
        add('no-functions', signatures=(), expected=False)
        add('current-outside', current=1, expected=False)
        add('wide-arity', signatures=((65536, 0),), expected=False)
        add('wide-result', signatures=((0, 256),), expected=False)
        add('unused-wide-signature', signatures=((65535, 255),), expected=True)
        add('maximum-instructions', b'\0'*65536, True)
        add('too-many-instructions', b'\0'*65537, False)
        add('unreachable-unknown', b'\x3d\xff', False)
        for opcode, width in enumerate(widths):
            instruction = bytes([opcode]) + bytes(max(0, width-1))
            add(f'opcode-{opcode}', instruction, False if not width else None,
                signatures=((2, 1),))
            if width:
                add(f'offset-{opcode}', b'\xff'+instruction+b'\xff',
                    False if not width else None, start=1, length=width,
                    signatures=((2, 1),))
                for size in range(1, width):
                    add(f'truncated-{opcode}-{size}', instruction[:size], False)
        for opcode in (88, 99, 251):
            for count in (0, 1, 255, 256, 257, 65535):
                literal = (bytes([opcode]) + (struct.pack('<BH', 1, count) if opcode == 88 else
                    struct.pack('<IH', 0, count) if opcode == 99 else struct.pack('<BIHH', 0, 0, 0, count)))
                add(f'literal-{opcode}-{count}', literal, count <= 256)
        for opcode in (51, 59):
            for arity in (0, 1, 255, 256, 257, 65535):
                for results in (0, 1, 255):
                    add(f'call-{opcode}-{arity}-{results}', bytes([opcode])+struct.pack('<I', 1),
                        arity <= 256, signatures=((0, 1), (arity, results)))
            add(f'invalid-callee-{opcode}', bytes([opcode])+struct.pack('<I', 2), False)
        for results in (0, 1, 255):
            add(f'return-{results}', b'\x3d', True, signatures=((0, results),))
        for opcode in (18, 19):
            for index in (0, 1, 254, 255, 256, 2**32-2, 2**32-1):
                # I copy this low-level unsigned increment; full profile analysis
                # separately rejects invalid global operands before execution.
                add(f'global-{opcode}-{index}', bytes([opcode])+struct.pack('<I', index),
                    (index+1) % 2**32 <= 256)
        for opcode in (56, 57, 58):
            for relative in range(-3, 12):
                instruction = bytes([opcode])+struct.pack('<i', relative)
                add(f'edge-{opcode}-{relative}', b'\0'+instruction+b'\0',
                    1+relative in (0, 1, 6, 7))
        for opcode in (1, 2):
            for bits in (0, 1, 2**63, 2**64-1, 0x7ff0000000000001, 0xfff123456789abcd):
                add(f'bits-{opcode}-{bits}', bytes([opcode])+struct.pack('<Q', bits), True)
        for start, length, valid in ((0, 0, True), (1, 0, True), (2, 0, False),
                                      (0, 2, False), (2**32-1, 0, False)):
            add(f'window-{start}-{length}', b'\0', valid, start=start, length=length)
        rng = random.Random(114277 + int(maps))
        supported = [op for op, width in enumerate(widths) if width]
        for i in range(128):
            code = b''.join(bytes([op])+rng.randbytes(widths[op]-1)
                            for op in rng.choices(supported, k=rng.randrange(1, 16)))
            add(f'random-{i}', code)
    for size in range(28):
        cases.append((f'short-header-{size}', bytes(size), False))
    return cases


class RecordInstructions(unittest.TestCase):
    def test_exact_recipes_and_successors(self):
        work = Path(tempfile.mkdtemp(prefix='record-instructions-', dir=ROOT/'.test_output'))
        print(f'I retain instruction-recipe products at {work}', flush=True)
        oracle = Path(os.environ.get('NANOLANG_RECORD_INSTRUCTIONS_REFERENCE', ROOT/'obj/nvm2c_record_instructions_reference'))
        primary = Path(os.environ.get('NANOLANG_PRIMARY_REFERENCE', ROOT/'obj/nvm2c_instruction_reference'))
        probes = []
        for opcode in range(256):
            path = work/f'primary-{opcode}.bin'
            path.write_bytes(struct.pack('<QQ', 0, 64)+bytes([opcode])+bytes(63)); probes.append(path)
        widths = []
        for i in range(0, 256, 64):
            widths += [int(line.split(':', 1)[0]) for line in checked([primary, *probes[i:i+64]]).splitlines()]
        cases = corpus(widths)
        paths = []
        for i, (_, data, _) in enumerate(cases):
            path = work/f'case-{i}.bin'; path.write_bytes(data); paths.append(path)
        record = dict(cases=[dict(name=name, sha256=hashlib.sha256(data).hexdigest(), expected=valid)
                             for name, data, valid in cases], routes=[])

        def execute(command, name):
            values = []
            for i in range(0, len(paths), 32):
                values += checked([*command, *paths[i:i+32]], timeout=600).splitlines()
            self.assertEqual(len(values), len(cases))
            (work/(name+'.reports')).write_text('\n'.join(values)+'\n')
            return values

        expected = execute([oracle], 'actual-c')
        by_name = {name: value for (name, _, _), value in zip(cases, expected)}
        for (name, _, valid), value in zip(cases, expected):
            if valid is not None:
                with self.subTest(expectation=name): self.assertEqual(value != '0', valid)
        self.assertEqual(by_name['0-nop'], '1:0:1:0,1,0,0,64,0,0,4294967295,1,1,0')
        self.assertEqual(by_name['0-maximum-instructions'], '1:0:65536')
        self.assertEqual(sum(by_name[f'0-opcode-{op}'] != '0' for op in range(256)), 94)
        self.assertEqual(sum(by_name[f'1-opcode-{op}'] != '0' for op in range(256)), 100)
        for name, command in build_routes(ROOT/'tests/nvm2c_record_instructions_driver.nano', work):
            actual = execute(command, name)
            for (case, _, _), want, got in zip(cases, expected, actual):
                with self.subTest(route=name, case=case): self.assertEqual(got, want)
            record['routes'].append(dict(name=name, command=list(map(str, command)), reports=len(actual)))
            (work/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
        print(f'I matched {len(cases)} instruction-recipe reports per route, including '
              f'{sum(valid is not None for _, _, valid in cases)} explicit validity expectations.', flush=True)
