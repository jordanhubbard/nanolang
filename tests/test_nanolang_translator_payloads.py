"""I compare public payload codecs and their actual decoded byte values."""
from pathlib import Path
import random
import struct
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked

ROOT = Path(__file__).resolve().parents[1]
NO_INDEX = 0xffffffff


def table(*rows):
    return struct.pack('<I', len(rows)) + b''.join(rows)


def constant(tag, payload):
    return struct.pack('<B3xI', tag, len(payload)) + payload + b'\0' * (-len(payload) % 4)


def signature(params, results):
    return (struct.pack('<HH', len(params), len(results)) + bytes(params)
            + b'\0' * (-len(params) % 4) + bytes(results) + b'\0' * (-len(results) % 4))


def layout(kind=0, fields=(), name=NO_INDEX):
    return (struct.pack('<BBHI', kind, 0, len(fields), name)
            + b''.join(struct.pack('<B3xII', tag, nested, field_name)
                       for tag, nested, field_name in fields))


def change(data, offset, encoding, value):
    result = bytearray(data)
    struct.pack_into(encoding, result, offset, value)
    return bytes(result)


def corpus():
    bases = {
        1: table(struct.pack('<II', NO_INDEX, 0x80000000), struct.pack('<II', 0, 1)),
        2: table(*(constant(tag, b'\0\xff\x80abc'[:tag % 7]) for tag in range(16))),
        3: table(signature(range(16), range(15, -1, -1)), signature([1, 5, 7], [8]), signature([], [])),
        4: table(layout(), layout(3, [(15, 0, NO_INDEX)], name=0x80000000)),
        5: table(struct.pack('<IIQQHHHH', NO_INDEX, NO_INDEX, 2**64-1, 2**63, 65535, 65535, 65535, 0)),
        7: table(*(struct.pack('<IBBHI', NO_INDEX, tag, 1, 0, NO_INDEX) for tag in range(16))),
        8: table(*(struct.pack('<IIIB3x', NO_INDEX, 0x80000000, NO_INDEX, kind) for kind in range(4))),
        9: table(struct.pack('<IIII', NO_INDEX, 0x80000000, NO_INDEX, 1)),
        10: table(struct.pack('<QII', 2**64-1, NO_INDEX, 0x80000000)),
        11: table(struct.pack('<IHBBII', 1, 0, 1, 0, NO_INDEX, 0x80000000),
                  struct.pack('<IHBBII', 1, 65535, 1, 1, NO_INDEX, NO_INDEX),
                  struct.pack('<IHBBII', NO_INDEX, 0, 1, 0, 0, 0)),
    }
    result = {}
    for kind, base in bases.items():
        cases = [('lossless-fields', base, 0, base), ('zero-count', table(), 0, table()),
                 ('impossible-count', struct.pack('<I', NO_INDEX), 1, None)]
        cases += [(f'truncated-{i}', base[:i], 1, None) for i in range(len(base))]
        # All ordinary codecs tolerate trailing bytes. Callbacks require exact size.
        cases.append(('trailing-byte', base + b'\xff', 9 if kind == 11 else 0,
                      None if kind == 11 else base))
        cases.append(('zero-count-trailing', table() + b'\xff', 9 if kind == 11 else 0,
                      None if kind == 11 else table()))
        rng = random.Random(1142 + kind)
        for i in range(80):
            mutated = bytearray(base)
            mutated[rng.randrange(len(mutated))] ^= 1 << rng.randrange(8)
            cases.append((f'field-mutation-{i}', bytes(mutated), None, None))
        result[kind] = cases

    def case(kind, name, data, status, canonical=None):
        result[kind].append((name, data, status, canonical))

    value = table(constant(5, b'\0\xff\0'))
    case(2, 'embedded-zero-not-terminated', value, 0, value)
    case(2, 'nonzero-padding', change(value, len(value)-1, '<B', 1), 9)
    case(2, 'bad-tag', change(value, 4, '<B', 16), 8)
    case(2, 'reserved-before-tag', change(change(value, 4, '<B', 16), 5, '<B', 1), 7)
    case(2, 'huge-byte-length', change(value, 8, '<I', NO_INDEX), 1)
    value = table(signature([1, 5, 7], [8]))
    case(3, 'bad-param-tag', change(value, 8, '<B', 16), 8)
    case(3, 'param-padding', change(value, 11, '<B', 1), 9)
    case(3, 'result-padding', change(value, 13, '<B', 1), 9)
    case(3, 'huge-param-count', change(value, 4, '<H', 65535), 1)
    case(5, 'reserved-function-flags', change(bases[5], 34, '<H', 1), 7)
    case(7, 'bad-tag-before-flags', change(change(bases[7], 8, '<B', 16), 9, '<B', 255), 8)
    case(7, 'unknown-global-flag', change(bases[7], 9, '<B', 2), 7)
    case(7, 'global-padding', change(bases[7], 10, '<H', 1), 7)
    case(8, 'import-kind-before-reserved', change(change(bases[8], 16, '<B', 4), 17, '<B', 1), 8)
    case(8, 'import-reserved', change(bases[8], 17, '<B', 1), 7)
    case(9, 'link-unknown-bit', change(bases[9], 16, '<I', 2), 7)
    case(11, 'callback-abi', change(bases[11], 10, '<B', 0), 7)
    case(11, 'callback-execution', change(bases[11], 11, '<B', 2), 7)
    case(11, 'callback-duplicate-parameter', change(bases[11], 24, '<H', 0), 13)
    case(11, 'callback-descending-import', change(bases[11], 20, '<I', 0), 13)
    scalar = layout(fields=[(1, NO_INDEX, NO_INDEX)])
    forward = layout(fields=[(8, 1, NO_INDEX)])
    dag = table(forward, scalar)
    case(4, 'forward-record-dag', dag, 0, dag)
    dag = table(layout(fields=[(8, 1, NO_INDEX), (8, 1, NO_INDEX)]), scalar)
    case(4, 'repeated-complete-child', dag, 0, dag)
    dag = table(layout(fields=[(8, 1, NO_INDEX)]), layout(fields=[(8, 2, NO_INDEX)]), scalar)
    case(4, 'depth-three-dag', dag, 0, dag)
    cycle = table(forward, layout(fields=[(8, 0, NO_INDEX)]))
    case(4, 'cycle', cycle, 13)
    case(4, 'self-edge', table(layout(fields=[(8, 0, NO_INDEX)])), 13)
    case(4, 'outside-edge', table(layout(fields=[(8, 1, NO_INDEX)])), 13)
    case(4, 'forward-disallows-tuple', table(forward, layout(1)), 13)
    case(4, 'forward-disallows-array-leaf', table(forward, layout(fields=[(7, NO_INDEX, NO_INDEX)])), 13)
    case(4, 'forward-disallows-scalar-edge', table(layout(fields=[(1, 1, NO_INDEX)]), scalar), 13)
    case(4, 'forward-requires-record-edge', table(layout(fields=[(8, NO_INDEX, NO_INDEX)]), forward, scalar), 13)
    # These are deliberately transport-only cases: module authority must decide
    # whether their tags/nested indices form an executable layout.
    prior = table(scalar, layout(1, [(7, 0, NO_INDEX)]))
    case(4, 'prior-only-preserves-existing-transport', prior, 0, prior)
    case(4, 'bad-layout-kind', change(table(scalar), 4, '<B', 4), 8)
    case(4, 'reserved-layout', change(table(scalar), 5, '<B', 1), 7)
    case(4, 'field-padding', change(table(scalar), 13, '<B', 1), 7)
    case(4, 'field-tag', change(table(scalar), 12, '<B', 16), 8)
    return result


class NanoLangTranslatorPayloads(unittest.TestCase):
    def test_decoded_tables_and_refusals_across_producers(self):
        with tempfile.TemporaryDirectory(prefix='nano-nanolang-payloads-') as directory:
            directory = Path(directory)
            routes = build_routes(ROOT / 'tests/nvm2c_payload_driver.nano', directory)
            total = 0
            for kind, cases in corpus().items():
                paths = []
                for i, (_, data, _, _) in enumerate(cases):
                    path = directory / f'{kind}-{i}.payload'
                    path.write_bytes(data)
                    paths.append(path)
                reference = checked([ROOT / 'obj/nvm2c_payload_reference', str(kind), *paths]).splitlines()
                self.assertEqual(len(reference), len(cases))
                for (name, _, expected, canonical), actual in zip(cases, reference):
                    if expected is not None:
                        with self.subTest(reference_contract=(kind, name)):
                            status, count, encoded = actual.split(':')
                            self.assertEqual(int(status), expected)
                            if expected:
                                self.assertEqual((count, encoded), ('0', ''))
                            else:
                                self.assertEqual(int(count), struct.unpack_from('<I', canonical)[0])
                                self.assertEqual(encoded, canonical.hex())
                for route, command in routes:
                    output = checked([*command, str(kind), *paths]).splitlines()
                    self.assertEqual(len(output), len(cases), (route, kind))
                    for (case, _, _, _), expected, actual in zip(cases, reference, output):
                        with self.subTest(route=route, kind=kind, case=case):
                            self.assertEqual(actual, expected)
                total += len(cases)
            print(f'I matched {total} decoded payloads across ten codecs and {len(routes)} execution routes.')
