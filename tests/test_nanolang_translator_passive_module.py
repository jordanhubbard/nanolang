"""I compare complete passive-bearing module decisions and retained claim bytes."""
from pathlib import Path
import os
import struct
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked
from tests.test_nanolang_translator_module import packet
from tests.test_nanolang_translator_payloads import constant, layout, signature, table
from tests.test_nanolang_translator_passive import cases as claims, block, node, payload

ROOT = Path(__file__).resolve().parents[1]
NONE = 0xffffffff


def wrapped(envelope):
    size, count = struct.unpack_from('<II', envelope)
    cursor, functions, signatures = 8, [], []
    valid_tags = True
    for index in range(count):
        start, length, arity, locals_, upvalues, results, tags = struct.unpack_from('<7I', envelope, cursor)
        cursor += 28
        params = envelope[cursor:cursor + tags] if tags else bytes(arity)
        cursor += tags
        valid_tags &= all(tag < 16 for tag in params)
        signatures.append(signature(params, [1] * results))
        functions.append(struct.pack('<IIQQHHHH', index, index, start, length, locals_, upvalues, 0, 0))
    claim = envelope[cursor:cursor + size]
    code = envelope[cursor + size:]
    constants = table(*(constant(5, f'function-{index}'.encode()) for index in range(count)))
    return packet([(2, constants), (3, table(*signatures)), (5, table(*functions)), (6, code), (12, claim)], features=64), size, valid_tags


def corpus(widths):
    result = []
    for name, envelope, expected in claims(widths):
        data, size, valid_tags = wrapped(envelope)
        decision = None
        if not size:
            decision = 9
        elif not valid_tags:
            decision = 8
        elif expected is not None:
            decision = 0 if expected else 13
        result.append((f'claim-{name}', data, decision))

    claim = payload([block([node(0, 5, 0)], 0, 5)])
    base = {2: table(constant(5, b'main')),
            3: table(signature([], [])),
            5: table(struct.pack('<IIQQHHHH', 0, 0, 0, 6, 1, 0, 0, 0)),
            6: b'\x03\x01\x11\0\0\x3d', 12: claim}

    def add(name, changes=None, expected=None, features=64, passive=True):
        sections = base | (changes or {})
        if not passive:
            del sections[12]
        result.append((name, packet(list(sections.items()), features=features), expected))

    for tag, value in [(1, b'\0' * 8), (2, b'\xff'), (3, b'\0' * 8), (4, b'\1'), (6, b'bytes\0')]:
        constants = table(constant(5, b'main'), constant(tag, value))
        add(f'convert-refuses-nonstring-{tag}', {2: constants}, 8)
        add(f'core-keeps-nonstring-{tag}', {2: constants}, 0, features=0, passive=False)
    for name, a, b, expected in [('empty', b'', b'', 13), ('embedded-duplicate', b'a\0same', b'a\0same', 13),
                                ('embedded-distinct', b'a\0left', b'a\0right', 0),
                                ('binary-distinct', b'\xff', b'\xfe', 0), ('prefix-distinct', b'a', b'ab', 0)]:
        constants = table(constant(5, b'main'), constant(5, a), constant(5, b))
        add('constant-identity-' + name, {2: constants}, expected)
        add('core-constant-identity-' + name, {2: constants}, 0, features=0, passive=False)
    add('duplicate-precedes-later-nonstring', {2: table(constant(5, b'main'), constant(5, b'main'), constant(1, b'\0' * 8))}, 13)
    add('nonstring-precedes-later-duplicate', {2: table(constant(5, b'main'), constant(1, b'\0' * 8), constant(5, b'main'))}, 8)
    for count in (0, 1, 255, 256, 65535):
        add(f'function-results-{count}', {3: table(signature([], [1] * count))}, 0 if count <= 255 else 13)
        add(f'core-function-results-{count}', {3: table(signature([], [1] * count))}, 0, features=0, passive=False)
    for offset in (0, 2**32-1, 2**32, 2**63-1, 2**63, 2**64-1):
        debug = table(struct.pack('<QII', offset, 0, 0))
        add(f'debug-narrow-{offset}', {10: debug}, 0 if offset <= NONE else 13, features=64+8)
        add(f'core-debug-wide-{offset}', {10: debug}, 0, features=8, passive=False)
    add('feature-before-constant-conversion', {2: table(constant(1, b'\0' * 8))}, 14, features=0)
    add('reference-before-constant-conversion', {2: table(constant(1, b'\0' * 8)),
                                               5: table(struct.pack('<IIQQHHHH', 1, 0, 0, 6, 1, 0, 0, 0))}, 13)
    add('constant-before-debug-narrowing', {2: table(constant(1, b'\0' * 8)), 10: table(struct.pack('<QII', 2**32, 0, 0))}, 8, features=72)
    add('service-error-before-conversion', {2: table(constant(1, b'\0' * 8))}, 8, features=64+512)
    add('import-multiple-results-transport', {8: table(struct.pack('<IIIB3x', 0, 0, 1, 0)),
                                              3: table(signature([], []), signature([1], [1] * 256))}, 0, features=64+2)
    for names in [(NONE, NONE), (0, 0)]:
        layouts = table(layout(fields=[(1, NONE, names[1])], name=names[0]))
        add('retained-layout-' + str(names), {4: layouts}, 0, features=64+128)
        add('retained-layout-permitted-tail-' + str(names), {4: layouts + b'ignored table tail'}, 0, features=64+128)
    add('empty-retained-layout-reconstruction', {4: table() + b'tail'}, 0, features=64+128)
    constants = table(constant(5, b'main'), constant(5, b'nano.source_file'), constant(5, b'file-a'), constant(5, b'file-b'))
    add('metadata-keeps-repeated-source-key', {2: constants, 1: table(struct.pack('<II', 1, 2), struct.pack('<II', 1, 3))}, 0)
    for name, sections, expected in [('passive-first', [(12, b''), (2, b'')], 9),
                                      ('bad-table-first', [(2, b''), (12, b'')], 1)]:
        result.append((name, packet(sections, features=64), expected))
    return result


class PassiveModule(unittest.TestCase):
    def test_complete_module_decisions_and_retained_claims(self):
        oracle = ROOT / 'obj/nvm2c_module_reference'
        widths = list(map(int, checked([oracle, '--widths']).splitlines()))
        cases = corpus(widths)
        with tempfile.TemporaryDirectory(prefix='nano-passive-module-') as name:
            work = Path(name)
            paths = []
            for index, (_, data, _) in enumerate(cases):
                path = work / f'case-{index}.nvm'
                path.write_bytes(data)
                paths.append(path)

            def execute(command, inputs):
                values = []
                for start in range(0, len(inputs), 64):
                    values += checked([*command, '--passive', *inputs[start:start + 64]], timeout=600).splitlines()
                self.assertEqual(len(values), len(inputs))
                return values

            reference = execute([oracle], paths)
            for (case, _, expected), actual in zip(cases, reference):
                if expected is not None:
                    with self.subTest(reference=case):
                        self.assertEqual(int(actual.split(':')[0]), expected)
            producers = [(ROOT / 'bin/nano_virt', [])]
            if os.environ.get('NANOLANG_TRANSLATOR_SOURCE_COMPILER'):
                producers.append((os.environ['NANOLANG_TRANSLATOR_SOURCE_COMPILER'], ['--emit-nvm']))
            products = []
            for index, (producer, flags) in enumerate(producers):
                for kind in ('par', 'flow'):
                    output = work / f'{index}-{kind}.nvm'
                    checked([producer, ROOT / f'tests/nanoisa/fixtures/passive_{kind}.nano', *flags, '-o', output], build=True)
                    checked([ROOT / 'bin/nano_vm', '--verify-only', output])
                    self.assertEqual(checked([ROOT / 'bin/nano_vm', output]), '13\n' if kind == 'par' else '15\n')
                    products.append(output)
            original_reference = execute([oracle], products)
            self.assertTrue(all(value.startswith('0:') and value != '0:0:' for value in original_reference))
            for route, command in build_routes(ROOT / 'tests/nvm2c_module_driver.nano', work):
                actual = execute(command, paths)
                for (case, _, _), expected, value in zip(cases, reference, actual):
                    with self.subTest(route=route, case=case):
                        self.assertEqual(value, expected)
                self.assertEqual(execute(command, products), original_reference, route)
            print(f'I matched {len(cases)} complete passive-bearing module decisions and exact claim bytes across five routes when both producers are selected; '
                  f'{sum(expected is not None for _, _, expected in cases)} explicit decisions and {len(products)} unchanged producer products.')


if __name__ == '__main__':
    unittest.main()
