"""I compare cross-section decisions without calling structural success execution authority."""
from pathlib import Path
import os
import random
import struct
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked
from tests.test_nanolang_translator_container import container, patched
from tests.test_nanolang_translator_payloads import constant, signature, layout, table

ROOT = Path(__file__).resolve().parents[1]
NONE = 0xffffffff


def packet(sections=(), *, features=0, entry=NONE, isa=2):
    return patched(container(sections, features=features, isa=isa), 28, '<I', entry)


def corpus():
    sections = {
        1: table(struct.pack('<II', 0, 1)),
        2: table(*(constant(5, s) for s in [b'main', b'ffi', b'symbol', b'adapter', b'/absolute', b'adapter'])),
        3: table(signature([], [1]), signature([11, 15], [1]), signature([1, 14], [1]), signature([1], []), signature([], [])),
        4: table(layout(fields=[(1, NONE, NONE)])),
        5: table(struct.pack('<IIQQHHHH', 0, 0, 0, 1, 0, 0, 1, 0)),
        6: b'\x00',
        7: table(struct.pack('<IBBHI', 0, 1, 0, 0, NONE)),
        8: table(struct.pack('<IIIB3x', 1, 2, 1, 0)),
        9: table(struct.pack('<IIII', 1, NONE, NONE, 1)),
        10: table(struct.pack('<QII', 2**64-1, 1, 1)),
        11: table(struct.pack('<IHBBII', 0, 0, 1, 0, 2, 3), struct.pack('<IHBBII', 0, 1, 1, 0, 3, 3)),
    }
    flags = 1 + 2 + 8 + 32 + 128

    def build(changes=None, *, features=flags, entry=0):
        return packet(list((sections | (changes or {})).items()), features=features, entry=entry)

    cases = [('all-core-tables', build(), 0),
             ('default-empty-sections', packet(), 0),
             ('unknown-isa-is-not-this-layer', packet(isa=65535), 0),
             ('empty-ffi-declaration', packet(features=2), 0),
             ('empty-linked-declaration', packet(features=1), 0),
             ('empty-coprocess-declaration', packet(features=4), 0),
             ('empty-closures-declaration', packet(features=16), 0),
             ('empty-retained-layout-declaration', packet(features=128), 0),
             ('callbacks-feature-without-rows', packet(features=32), 14),
             ('passive-feature-without-bytes', packet(features=64), 14),
             ('ownership-feature-without-bytes', packet(features=256), 14),
             ('service-feature-without-bytes', packet(features=512), 8),
             ('entry-without-functions', packet(entry=0), 13),
             ('reverse-directory-order', packet(list(reversed(list(sections.items()))), features=flags, entry=0), 0)]
    for bit in (1, 2, 8, 32, 128):
        cases.append((f'missing-feature-{bit}', build(features=flags-bit), 14))
    cases.append(('service-error-before-index', build({5: patched(sections[5], 4, '<I', NONE)}, features=flags+512), 8))
    mutations = [
        (5, 4, '<I', 6, 'function-name'), (5, 8, '<I', 5, 'function-signature'),
        (5, 12, '<Q', 2, 'function-offset'), (5, 12, '<Q', 2**63, 'function-high-offset'),
        (5, 20, '<Q', 2**64-1, 'function-wrap-length'),
        (7, 4, '<I', NONE, 'global-name'), (7, 12, '<I', 6, 'global-init'),
        (8, 4, '<I', 6, 'import-module'), (8, 8, '<I', 6, 'import-symbol'),
        (8, 12, '<I', 5, 'import-signature'), (9, 4, '<I', NONE, 'link-module'),
        (9, 8, '<I', 6, 'link-symbol'), (9, 12, '<I', 5, 'link-signature'),
        (4, 8, '<I', 6, 'layout-name'), (4, 20, '<I', 6, 'field-name'),
        (1, 4, '<I', NONE, 'metadata-key'), (1, 8, '<I', 6, 'metadata-value'),
        (11, 4, '<I', 1, 'callback-import'), (11, 8, '<H', 2, 'callback-parameter'),
        (11, 12, '<I', 5, 'callback-signature'), (11, 16, '<I', 6, 'callback-adapter-index'),
    ]
    cases.extend((name, build({kind: patched(sections[kind], off, encoding, value)}), 13)
                 for kind, off, encoding, value, name in mutations)
    cases += [('entry-sentinel', build(entry=NONE), 0), ('entry-outside', build(entry=1), 13),
              ('zero-code-span', build({5: patched(patched(sections[5], 12, '<Q', 1), 20, '<Q', 0)}), 0),
              ('bad-opcode-is-not-module-verification', build({6: b'\xfe'}), 0),
              ('file-opcode-without-contract', build({6: b'\x91'}), 8),
              ('file-priority-before-function-signature', build({6: b'\x91', 5: patched(sections[5], 8, '<I', NONE)}), 8),
              ('file-byte-inside-operand', build({6: b'\x01\x91' + b'\0'*7,
                                                5: patched(sections[5], 20, '<Q', 9)}), 0),
              ('unknown-before-file-stops-scan', build({6: b'\xff\x91', 5: patched(sections[5], 20, '<Q', 2)}), 0),
              ('service-import-without-contract', build({8: patched(sections[8], 16, '<B', 3)}), 8),
              ('coprocess-needs-feature', build({8: patched(sections[8], 16, '<B', 1)}), 14),
              ('coprocess-with-feature', build({8: patched(sections[8], 16, '<B', 1)}, features=flags+4), 0),
              ('relative-artifact', build({8: patched(sections[8], 16, '<B', 2)}), 8),
              ('absolute-artifact', build({8: patched(patched(sections[8], 16, '<B', 2), 4, '<I', 4)}), 0),
              ('callback-adapter-identity', build({11: patched(sections[11], 32, '<I', 5)}), 8),
              ('callback-execution-identity', build({11: patched(sections[11], 27, '<B', 1)}), 8),
              ('callback-incomplete-coverage', build({11: table(sections[11][4:20])}), 13)]
    for tag in range(16):
        signatures = table(signature([], [1]), signature([11, 15], [1]), signature([tag, 14], [1]), signature([1], []))
        cases.append((f'callback-parameter-tag-{tag}', build({3: signatures}), 0 if tag in (1,2,3,4,14) else 8))
        signatures = table(signature([], [1]), signature([11, 15], [1]), signature([1, 14], [tag]), signature([1], []))
        cases.append((f'callback-result-tag-{tag}', build({3: signatures}), 0 if tag in (1,2,3,4,14) else 8))
    for label, adapter, expected in [('empty', b'', 8), ('nul', b'a\0b', 8), ('binary-nonzero', b'\xff', 0)]:
        constants = table(*(constant(5, s) for s in [b'main', b'ffi', b'symbol', adapter, b'/absolute', b'adapter']))
        cases.append((f'callback-name-{label}', build({2: constants}), expected))
    signatures = table(signature([], [1]), signature([1]*17, [1]), signature([1,14], [1]), signature([1], []))
    cases.append(('callback-import-over-budget', build({3: signatures}), 13))
    signatures = table(signature([], [1]), signature([11,15], [1]), signature([1]*17, [1]), signature([1], []))
    cases.append(('callback-target-over-budget', build({3: signatures}), 8))
    sentinel = table(struct.pack('<IHBBII', 0, 65535, 1, 0, NONE, 3))
    cases.append(('no-callable-parameter-sentinel', build({11: sentinel, 8: patched(sections[8], 12, '<I', 0)}), 0))
    cases.append(('sentinel-cannot-omit-callables', build({11: sentinel}), 13))
    # A well-formed envelope is deliberately insufficient for execution.
    for i in range(40):
        cases.append((f'truncated-header-{i}', build()[:i], 1))
    rng = random.Random(114204)
    base = build()
    for i in range(128):
        mutated = bytearray(base)
        mutated[rng.randrange(len(mutated))] ^= 1 << rng.randrange(8)
        cases.append((f'wire-mutation-{i}', bytes(mutated), None))
    pending = [(f'pending-section-{kind}', packet([(kind, b'\0')]), -1) for kind in (12, 13, 14)]
    pending += [('pending-array-layout', packet([(4, table(layout(fields=[(7,NONE,NONE)])))]), -1),
                ('pending-map-layout', packet([(4, table(layout(fields=[(13,NONE,NONE)])))]), -1)]
    return cases, pending


class NanoLangTranslatorModule(unittest.TestCase):
    def test_declared_owner_return_and_nominal_refusal(self):
        with tempfile.TemporaryDirectory(prefix='nano-declared-owner-') as name:
            directory = Path(name)
            left, right, bridge = (directory / n for n in ('left.nano', 'right.nano', 'bridge.nano'))
            left.write_text('module left_owner\npub struct Record { value: int }\n')
            right.write_text('module right_owner\npub struct Record { value: int }\n')
            bridge.write_text('module declared_bridge\n'
                              f'module "{left}" as p\n'
                              'pub fn forward(value: p.Record) -> p.Record { return value }\n'
                              'shadow forward { assert (== (forward p.Record { value: 7 }).value 7) }\n')
            fixture = directory / 'client.nano'
            source = (f'module "{left}" as l\nmodule "{right}" as r\nmodule "{bridge}" as b\n'
                      'fn main() -> int { let value: l.Record = l.Record { value: 7 } '
                      'let other: r.Record = r.Record { value: 9 } '
                      'assert (== (b.forward value).value 7) return 0 }\n')
            fixture.write_text(source)
            for route, command in build_routes(fixture, directory):
                self.assertEqual(checked(command), '', route)
            fixture.write_text(source.replace('(b.forward value)', '(b.forward other)'))
            producers = [(ROOT/'bin/nanoc_c', []), (ROOT/'bin/nano_virt', [])]
            if os.environ.get('NANOLANG_TRANSLATOR_SOURCE_COMPILER'):
                producers.append((os.environ['NANOLANG_TRANSLATOR_SOURCE_COMPILER'], ['--emit-nvm']))
            from tests.native_process import run_captured
            for index, (producer, flags) in enumerate(producers):
                prior = directory / f'prior-{index}.nvm'
                prior.write_bytes(b'I retain this prior product.')
                result = run_captured([producer, fixture, *flags, '-o', prior], build=True, timeout=60, cwd=ROOT)
                self.assertNotEqual(result.returncode, 0, str(producer))
                self.assertEqual(prior.read_bytes(), b'I retain this prior product.', str(producer))

    def test_cross_section_contracts_and_existing_programs(self):
        with tempfile.TemporaryDirectory(prefix='nano-nanolang-module-') as name:
            directory = Path(name)
            routes = build_routes(ROOT/'tests/nvm2c_module_driver.nano', directory)
            oracle = ROOT/'obj/nvm2c_module_reference'
            cases, pending = corpus()
            paths = []
            for i, (_, data, _) in enumerate(cases + pending):
                path = directory/f'{i}.nvm'
                path.write_bytes(data)
                paths.append(path)
            reference = list(map(int, checked([oracle, *paths[:len(cases)]]).splitlines()))
            self.assertEqual(len(reference), len(cases))
            for (case, _, expected), actual in zip(cases, reference):
                if expected is not None:
                    with self.subTest(reference=case): self.assertEqual(actual, expected)
            produced = []
            producers = [(ROOT/'bin/nano_virt', [])]
            if os.environ.get('NANOLANG_TRANSLATOR_SOURCE_COMPILER'):
                producers.append((os.environ['NANOLANG_TRANSLATOR_SOURCE_COMPILER'], ['--emit-nvm']))
            for producer_index, (producer, flags) in enumerate(producers):
                for original in ('examples/language/nl_functions_basic.nano',
                                 'examples/language/nl_control_if_while.nano',
                                 'examples/cross_backend/hello_cross_backend.nano'):
                    module = directory/f'original-{producer_index}-{Path(original).stem}.nvm'
                    checked([producer, ROOT/original, *flags, '-o', module], build=True)
                    self.assertEqual(checked([oracle, module]).strip(), '0')
                    produced.append(module)
            for route, command in routes:
                values = list(map(int, checked([*command, *paths, *produced]).splitlines()))
                self.assertEqual(len(values), len(paths)+len(produced))
                for (case, _, _), expected, actual in zip(cases, reference, values):
                    with self.subTest(route=route, case=case): self.assertEqual(actual, expected)
                self.assertEqual(values[len(cases):len(paths)], [-1]*len(pending), route)
                self.assertEqual(values[len(paths):], [0]*len(produced), route)
                self.assertEqual(checked([*command, '--widths']), checked([oracle, '--widths']), route)
            scans = []
            for opcode in range(256):
                path = directory/f'scan-{opcode}.bin'
                path.write_bytes(bytes([opcode,145])+b'\0'*15)
                scans.append(path)
            expected = checked([oracle, '--scan', *scans])
            for route, command in routes:
                self.assertEqual(checked([*command, '--scan', *scans]), expected, route)
            print(f'I matched {len(cases)} module decisions, {len(produced)} original products, 256 widths and 256 File scans across {len(routes)} routes; {len(pending)} profiles remain pending.')
