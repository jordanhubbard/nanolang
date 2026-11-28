"""I compare closed-recipe height walks with the actual shared stack verifier."""
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


def carrier(code, signatures=((0, 0),), current=0, maps=False, locals_=0, start=0, length=None):
    return (struct.pack('<6I', start, len(code)-start if length is None else length,
                        len(signatures), current, int(maps), locals_)
            + b''.join(struct.pack('<II', *sig) for sig in signatures) + code)


class Assembly:
    def __init__(self):
        self.code = bytearray(); self.labels = {}; self.patches = []

    def label(self, name):
        self.labels[name] = len(self.code)

    def emit(self, code):
        self.code.extend(code)

    def branch(self, opcode, name):
        self.patches.append((len(self.code), name))
        self.code.extend(bytes([opcode])+bytes(4))

    def finish(self):
        for position, name in self.patches:
            struct.pack_into('<i', self.code, position+1, self.labels[name]-position)
        return bytes(self.code)


def corpus():
    cases = []
    for maps in (False, True):
        def add(name, code=b'', maximum=None, valid=True, **kwargs):
            expected = None if maximum is None and valid else '0' if not valid else f'1:{maximum}'
            cases.append((f'{int(maps)}-{name}', carrier(code, maps=maps, **kwargs), expected))
        for result in (0, 1, 255):
            add(f'empty-{result}', signatures=((0, result),), maximum=0)
            for locals_ in (0, 65535):
                add(f'empty-frame-{result}-{locals_}', signatures=((0, result),), locals_=locals_, maximum=0)
        add('wide-frame', locals_=65536, valid=False)
        add('no-functions', signatures=(), valid=False)
        add('bad-current', current=1, valid=False)
        add('invalid-dead-opcode', b'\x3d\xff', valid=False)
        add('valid-dead-underflow', b'\x3d\x08', maximum=0)
        add('valid-dead-call', b'\x3d\x3b\0\0\0\0', maximum=0)
        for pushes in range(9):
            for pops in range(10):
                for result in range(3):
                    valid = pushes >= pops and pushes-pops == result
                    add(f'linear-{pushes}-{pops}-{result}', b'\x05'*pushes+b'\x08'*pops+b'\x3d',
                        maximum=pushes if valid else None, valid=valid, signatures=((0, result),))
        for pushes in range(5):
            for result in range(3):
                valid = pushes == result
                add(f'implicit-{pushes}-{result}', b'\x05'*pushes, maximum=pushes if valid else None,
                    valid=valid or pushes == 0, signatures=((0, result),))
                if pushes == 0:
                    cases[-1] = (*cases[-1][:2], '1:0')
        for opcode in (88, 99, 251):
            for fields in (0, 1, 255, 256, 257):
                literal = (bytes([opcode])+(struct.pack('<BH', 1, fields) if opcode == 88 else
                    struct.pack('<IH', 0, fields) if opcode == 99 else struct.pack('<BIHH', 0, 0, 0, fields)))
                add(f'literal-{opcode}-{fields}', b'\x05'*fields+literal+b'\x3d',
                    maximum=max(1, fields) if fields <= 256 else None,
                    valid=fields <= 256, signatures=((0, 1),))
        for opcode in (51, 59):
            for arity in (0, 1, 3, 256):
                for result in (0, 1, 255):
                    for extra in (-1, 0, 1):
                        if arity+extra < 0: continue
                        count = arity+extra
                        tail = opcode == 51
                        valid = extra >= 0 and (tail or extra+result == 1)
                        code = b'\x05'*count+bytes([opcode])+struct.pack('<I', 1)+(b'' if tail else b'\x3d')
                        add(f'call-{opcode}-{arity}-{result}-{extra}', code,
                            maximum=max(count, extra+result) if valid else None,
                            valid=valid, signatures=((0, 1), (arity, result)))
        for left in range(4):
            for right in range(4):
                for result in range(3):
                    a = Assembly(); a.emit(b'\x03\1'); a.branch(58, 'right')
                    a.emit(b'\x05'*left); a.branch(56, 'join')
                    a.label('right'); a.emit(b'\x05'*right); a.label('join'); a.emit(b'\x3d')
                    valid = left == right == result
                    add(f'diamond-{left}-{right}-{result}', a.finish(),
                        maximum=max(1, left) if valid else None, valid=valid, signatures=((0, result),))
        for pushes in range(5):
            for pops in range(5):
                a = Assembly(); a.label('top'); a.emit(b'\0'+b'\x05'*pushes+b'\x08'*pops); a.branch(56, 'top')
                add(f'backedge-{pushes}-{pops}', a.finish(), maximum=pushes if pushes == pops else None,
                    valid=pushes == pops)
        for pushes in range(4):
            for result in range(3):
                a = Assembly(); a.emit(b'\x05'*pushes); a.branch(56, 'end'); a.emit(b'\x08'*4); a.label('end')
                add(f'terminal-{pushes}-{result}', a.finish(), maximum=pushes if pushes == result else None,
                    valid=pushes == result, signatures=((0, result),))
        for depth in (0, 1, 255, 65535):
            for locals_ in (0, 1, 65535):
                code = b'\x05'*depth+b'\x33'+struct.pack('<I', 1)
                valid = depth+locals_ <= 65535
                add(f'frame-{depth}-{locals_}', code, maximum=depth if valid else None,
                    valid=valid, locals_=locals_, signatures=((0, 0), (0, 0)))
        add('tail-maximum-overflow', b'\x05'*65535+b'\x33'+struct.pack('<I', 1),
            valid=False, signatures=((0, 0), (0, 1)))
        add('bad-late-boundary', b'\x3d\x38\1\0\0\0', valid=False)
        add('nonzero-offset', b'\xff\x05\x3d\xff', maximum=1, start=1, length=2, signatures=((0, 1),))
        if maps:
            # HM_NEW pushes a map; HM_LEN consumes it and pushes one scalar.
            add('map-stack', b'\x78\x05\x01\x7f\x3d', maximum=1, signatures=((0, 1),))
        rng = random.Random(114278+int(maps))
        for number in range(256):
            a = Assembly(); size = rng.randrange(1, 33)
            for index in range(size):
                a.label(str(index))
                op = rng.choice((0, 5, 7, 8, 9, 56, 57, 58, 61))
                if op in (56, 57, 58): a.branch(op, str(rng.randrange(size+1)))
                else: a.emit(bytes([op]))
            a.label(str(size))
            add(f'random-{number}', a.finish(), signatures=((0, rng.randrange(3)),))
    return cases


class RecordStack(unittest.TestCase):
    def test_reachable_heights(self):
        work = Path(tempfile.mkdtemp(prefix='record-stack-', dir=ROOT/'.test_output'))
        print(f'I retain height-walk products at {work}', flush=True)
        oracle = Path(os.environ.get('NANOLANG_RECORD_STACK_REFERENCE', ROOT/'obj/nvm2c_record_stack_reference'))
        cases = corpus(); paths = []
        record = dict(cases=[dict(name=name, expected=want, sha256=hashlib.sha256(data).hexdigest())
                             for name, data, want in cases], routes=[])
        (work/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
        for index, (_, data, _) in enumerate(cases):
            path = work/f'case-{index}.bin'; path.write_bytes(data); paths.append(path)

        def execute(command, name):
            reports = []
            for i in range(0, len(paths), 32):
                reports += checked([*command, *paths[i:i+32]], timeout=600).splitlines()
            self.assertEqual(len(reports), len(cases))
            (work/(name+'.reports')).write_text('\n'.join(reports)+'\n')
            return reports

        expected = execute([oracle], 'actual-c')
        for (name, _, want), got in zip(cases, expected):
            if want is not None:
                with self.subTest(expectation=name): self.assertEqual(got, want)
        for name, command in build_routes(ROOT/'tests/nvm2c_record_stack_driver.nano', work):
            actual = execute(command, name)
            for (case, _, _), want, got in zip(cases, expected, actual):
                with self.subTest(route=name, case=case): self.assertEqual(got, want)
            record['routes'].append(dict(name=name, command=list(map(str, command)), reports=len(actual)))
            (work/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
        print(f'I matched {len(cases)} closed height reports per route, including '
              f'{sum(want is not None for _, _, want in cases)} explicit expectations.', flush=True)
