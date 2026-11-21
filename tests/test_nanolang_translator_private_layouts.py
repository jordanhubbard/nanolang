"""I preserve private layout preflight and graph profiles beside public decoding."""
from pathlib import Path
import random
import struct
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked
from tests.test_nanolang_translator_payloads import NO_INDEX, change, layout, table

ROOT = Path(__file__).resolve().parents[1]


def corpus():
    cases = []

    def add(name, data, status=None):
        expected = {i: status for i in range(4)} if isinstance(status, int) else status
        cases.append((name, data, expected))

    add('absent-storage', b'', {0: 1, 1: 9, 2: 9, 3: 9})
    for n in range(1, 4): add(f'short-count-{n}', b'\0' * n, 1)
    add('zero', table(), 0)
    add('zero-tail', table() + b'\xff', {0: 0, 1: 9, 2: 9, 3: 9})
    add('count-before-truncation', struct.pack('<I', 257), {0: 1, 1: 13, 2: 13, 3: 13})
    add('huge-count', struct.pack('<I', NO_INDEX), {0: 1, 1: 13, 2: 13, 3: 13})
    for n in (255, 256, 257):
        add(f'layout-budget-{n}', table(*(layout() for _ in range(n))),
            0 if n <= 256 else {0: 0, 1: 13, 2: 13, 3: 13})
    add('minimum-headers-before-bad-kind', struct.pack('<I', 2) + layout(255), 1)
    add('minimum-headers-before-reserved', struct.pack('<I', 2) + change(layout(), 1, '<B', 1), 1)
    add('entry-truncation-precedes-bad-kind', struct.pack('<I', 2) + layout(255, fields=[(1, NO_INDEX, NO_INDEX)]),
        {0: 8, 1: 1, 2: 1, 3: 1})
    add('entry-truncation-precedes-reserved', struct.pack('<I', 2) + change(layout(fields=[(1, NO_INDEX, NO_INDEX)]), 1, '<B', 1),
        {0: 7, 1: 1, 2: 1, 3: 1})
    add('trailing-precedes-bad-kind', table(layout(255)) + b'\0',
        {0: 8, 1: 9, 2: 9, 3: 9})
    add('self-index', table(layout(fields=[(8, 0, NO_INDEX)])), 13)
    add('index-equals-count', table(layout(fields=[(8, 1, NO_INDEX)])), 13)
    add('wide-index', table(layout(fields=[(8, 2**31, NO_INDEX)])), 13)
    for name in (0, 2**31, NO_INDEX):
        add(f'opaque-name-{name}', table(layout(name=name, fields=[(5, NO_INDEX, name)])), 0)
    for kind in range(256):
        add(f'prior-kind-{kind}', table(layout(kind)), 0 if kind <= 3 else 8)
        add(f'forward-kind-{kind}', table(layout(fields=[(8, 1, NO_INDEX)]), layout(kind)),
            8 if kind > 3 else (0 if kind == 0 else 13))
    for tag in range(256):
        add(f'prior-tag-{tag}', table(layout(fields=[(tag, NO_INDEX, NO_INDEX)])), 0 if tag < 16 else 8)
        expected = {p: (8 if tag >= 16 else (0 if 1 <= tag <= 5 or (tag == 7 and p >= 1)
                     or (tag == 13 and p == 3) else 13)) for p in range(4)}
        add(f'forward-leaf-tag-{tag}', table(layout(fields=[(8, 1, NO_INDEX)]),
                                           layout(fields=[(tag, NO_INDEX, NO_INDEX)])), expected)
    # A prior-only table deliberately does not enter the forward grammar.
    for kind in range(4):
        for tag in range(16):
            add(f'prior-nominal-kind-{kind}-tag-{tag}', table(layout(kind), layout(fields=[(tag, 0, NO_INDEX)])), 0)
    for kind in range(4):
        for tag in (1, 5, 7, 8, 10, 13):
            data = table(layout(kind, fields=[(tag, NO_INDEX, NO_INDEX)]),
                         layout(fields=[(8, 2, NO_INDEX)]), layout())
            expected = {p: (0 if (kind == 0 or (kind == 2 and p >= 2)) and
                                (1 <= tag <= 5 or (tag == 7 and p >= 1 and kind == 0)
                                 or (tag == 13 and p == 3 and kind == 0)) else 13) for p in range(4)}
            add(f'disconnected-kind-{kind}-tag-{tag}', data, expected)
    add('two-cycle', table(layout(fields=[(8, 1, NO_INDEX)]), layout(fields=[(8, 0, NO_INDEX)])), 13)
    add('disconnected-cycle', table(layout(), layout(fields=[(8, 2, NO_INDEX)]), layout(fields=[(8, 1, NO_INDEX)])), 13)
    add('shared-diamond', table(layout(fields=[(8, 1, NO_INDEX), (8, 2, NO_INDEX)]),
                                layout(fields=[(8, 3, NO_INDEX)]), layout(fields=[(8, 3, NO_INDEX)]), layout()), 0)
    add('forward-depth-256', table(*(layout(fields=[(8, i + 1, NO_INDEX)]) if i < 255 else layout()
                                     for i in range(256))), 0)
    add('prior-depth-256', table(*(layout(fields=[(8, i - 1, NO_INDEX)]) if i else layout()
                                   for i in range(256))), 0)
    for offset in (5, 13, 14, 15):
        for byte in (1, 128, 255):
            add(f'reserved-{offset}-{byte}', change(table(layout(fields=[(1, NO_INDEX, NO_INDEX)])), offset, '<B', byte), 7)
    rich = table(layout(fields=[(8, 1, NO_INDEX), (7, NO_INDEX, NO_INDEX)]),
                 layout(fields=[(1, NO_INDEX, NO_INDEX)]), layout(2, fields=[(2, NO_INDEX, NO_INDEX)]))
    add('mixed-forward-array-union', rich, {0: 13, 1: 13, 2: 0, 3: 0})
    add('trailing-before-graph', rich + b'\xff', {0: 13, 1: 9, 2: 9, 3: 9})
    for n in range(len(rich)):
        add(f'rich-truncation-{n}', rich[:n])
    rng = random.Random(114201)
    for n in range(512):
        data = bytearray(rich)
        data[rng.randrange(len(data))] ^= 1 << rng.randrange(8)
        add(f'rich-mutation-{n}', bytes(data))
    leaf = struct.pack('<B3xII', 1, NO_INDEX, NO_INDEX)
    maximum = struct.pack('<BBHI', 0, 0, 65535, NO_INDEX) + leaf * 65535
    add('field-budget-65535', table(maximum), 0)
    add('field-budget-65536', table(maximum, layout(fields=[(1, NO_INDEX, NO_INDEX)])), 0)
    add('field-budget-65537', table(maximum, layout(fields=[(1, NO_INDEX, NO_INDEX)] * 2)),
        {0: 0, 1: 13, 2: 13, 3: 13})
    add('field-budget-precedes-truncation', struct.pack('<I', 2) + maximum + struct.pack('<BBHI', 0, 0, 2, NO_INDEX),
        {0: 1, 1: 13, 2: 13, 3: 13})
    add('field-budget-precedes-reserved', table(maximum, change(layout(fields=[(1, NO_INDEX, NO_INDEX)] * 2), 1, '<B', 1)),
        {0: 7, 1: 13, 2: 13, 3: 13})
    add('byte-budget-16777216', bytes(16777216), {0: 0, 1: 9, 2: 9, 3: 9})
    add('byte-budget-16777217', bytes(16777217), {0: 0, 1: 13, 2: 13, 3: 13})
    return cases


class PrivateLayouts(unittest.TestCase):
    def test_exact_preflight_and_forward_profiles(self):
        cases = corpus()
        with tempfile.TemporaryDirectory(prefix='nano-private-layouts-') as directory:
            work = Path(directory)
            paths = []
            for index, (_, data, _) in enumerate(cases):
                path = work / f'case-{index}.bin'
                path.write_bytes(data)
                paths.append(path)
            routes = build_routes(ROOT / 'tests/nvm2c_private_layout_driver.nano', work)

            def execute(command, profile):
                result = []
                for start in range(0, len(paths), 64):
                    result.extend(checked([*command, str(profile), *paths[start:start+64]], timeout=600).splitlines())
                self.assertEqual(len(result), len(cases))
                return result

            for profile in range(4):
                reference = execute([ROOT / 'obj/nvm2c_private_layout_reference'], profile)
                for (case, data, expected), value in zip(cases, reference):
                    status, count, length, encoded = value.split(':')
                    if expected is not None:
                        with self.subTest(reference=case, profile=profile):
                            self.assertEqual(int(status), expected[profile])
                    if int(status) == 0:
                        self.assertEqual(int(count), struct.unpack_from('<I', data)[0])
                        self.assertEqual(len(encoded), int(length) * 2)
                        self.assertEqual(encoded, data[:int(length)].hex())
                        if profile: self.assertEqual(int(length), len(data))
                    else:
                        self.assertEqual(value, f'{status}:0:0:')
                for route, command in routes:
                    actual = execute(command, profile)
                    for (case, _, _), expected, value in zip(cases, reference, actual):
                        with self.subTest(profile=profile, route=route, case=case):
                            self.assertEqual(value, expected)
            print(f'I matched {len(cases)} exact layout reports per profile on four public/private profiles '
                  f'across {len(routes)} routes; {sum(e is not None for _, _, e in cases)} explicit cases retain '
                  '256-layout/65536-field/16777216-byte boundaries, graph and failure precedence.')


if __name__ == '__main__': unittest.main()
