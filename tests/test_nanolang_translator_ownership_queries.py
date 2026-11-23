"""I compare complete metadata query publications with their actual C APIs."""
from pathlib import Path
import hashlib
import json
import os
import struct
import tempfile
import unittest

from tests.nanolang_translator_routes import ROOT, build_routes, checked
from tests.test_nanolang_translator_ownership import (
    desc, effect_data, extension, function, globals_data, image, record, union_data,
    corpus as declaration_corpus,
)

NO = 0xffffffff


def carrier(model, op, a=0, b=0, flags=None):
    if flags is None:
        layout_size, own_size = struct.unpack_from('<II', model)
        flags = 1 | 8 | 16 | 32 | (2 if own_size else 0) | (4 if layout_size else 0)
    return struct.pack('<5I', op, a, b, flags, len(model))+model


def report(*values):
    return ''.join(f'{value},' for value in values)


def nominal(own=None, *, operations=None, constructors=((0, 0, 0),), revision=3, code=b'\x50\x01\x00\x3d'):
    rows = [(0, 2, [(1, NO, 4)]), (2, 3, [(8, 0, 4)])]
    types = [desc(1), desc(5), desc(7, 0), desc(8, 0), desc(10, 1)]
    if operations is None:
        operations = [(5, desc(8, 0), [desc(7, 2), desc(5)] if revision >= 2 else [desc(5)])]
    if own is None:
        own = record(3, flags=[1, 1], functions=[(0, 0, desc(1), [])], paths=[(1, 65535)],
            extensions=[extension(1, union_data(1, [(0, 0, 1), (1, 1, 0)])),
                        extension(3, globals_data([desc(8, 0, mode=1), desc(10, 1)])),
                        extension(4, effect_data(revision, types=types if revision >= 2 else (),
                            constructors=constructors, operations=operations), revision)])
    return image(own, rows=rows,
        strings=[b'Some', b'Empty', b'Record', b'Choice', b'field', b'op', b'op', b'', b'op\0', b'\xff'],
        functions=[function(results=1, result_tag=1, length=len(code))], code=code)


def corpus():
    cases = []
    def add(name, model, op, a=0, b=0, flags=None, expected=None, expected_status=None):
        cases.append(dict(name=name, data=carrier(model, op, a, b, flags), expected=expected, expected_status=expected_status))
    empty = image(b'')
    v1 = image(record())
    v2 = image(record(2, paths=[(0,), (65535,), tuple(range(32)), (1, 2)]))
    for op in range(6):
        add(f'absent-{op}', empty, op, expected=report(0, 0) if op == 1 else report(13))
        add(f'null-module-{op}', v1, op, flags=62, expected=report(13))
        add(f'present-empty-ownership-{op}', empty, op, flags=63,
            expected=report(9) if op in (0, 1, 3, 5) else report(13))
        add(f'version1-{op}', v1, op, expected=report(3) if op in (0, 5) else report(0, 0) if op == 1 else report(13))
    for index, path in enumerate(((0,), (65535,), tuple(range(32)), (1, 2))):
        for capacity in (0, len(path)-1, len(path), 65535):
            add(f'path-{index}-capacity-{capacity}', v2, 0, index, capacity,
                expected=report(0, len(path), *path) if capacity >= len(path) else report(13))
    for index in (4, NO):
        add(f'path-absent-{index}', v2, 0, index, 32, expected=report(13))
    good = nominal()
    add('nominal-globals', good, 1, 2, expected=report(0, 2, 8, 1, 0, 10, 0, 1))
    add('nominal-globals-count-only', good, 1, 0, flags=55, expected=report(0, 2))
    add('nominal-globals-too-small', good, 1, 1, expected=report(13))
    add('nominal-element-table', good, 3, expected=report(0, 5, 1, NO, 5, NO, 7, 0, 8, 0, 10, 1))
    for key in (5, 6):
        add(f'effect-exact-key-alias-{key}', good, 2, key, expected=report(0, 5, 8, 0, 2, 7, 2, 5, NO))
    for key in (0, 7, 8, 9, 10, NO):
        add(f'effect-missing-binary-key-{key}', good, 2, key, expected=report(13))
    for offset in (0, 1, 2, 3, 4, NO):
        add(f'constructor-byte-offset-{offset}', good, 4, 0, offset,
            expected=report(13) if offset >= 4 else report(0, 0 if offset == 0 else NO))
    add('constructor-function-range', good, 4, 1, 0, expected=report(13))
    add('union-first', good, 5, 0, 0, expected=report(0, 1, 0, 0, 1))
    add('union-second', good, 5, 0, 1, expected=report(0, 1, 1, 1, 0))
    for ordinal, arm in ((1, 0), (NO, 0), (0, 2), (0, 65535)):
        add(f'union-range-{ordinal}-{arm}', good, 5, ordinal, arm, expected=report(13))
    for flags in range(64):
        module, owner, layouts, output, count, functions = (bool(flags & (1 << i)) for i in range(6))
        for op, a, b in ((0, 0, 32), (1, 2, 0), (2, 5, 0), (3, 0, 0), (4, 0, 0), (5, 0, 1)):
            status = 0
            if not module:
                status = 13
            elif op == 0 and not owner:
                status = 13
            elif op == 1 and (not count or not output):
                status = 13
            elif op in (2, 3, 4, 5) and not output:
                status = 13
            elif op == 4 and not functions:
                status = 13
            elif op == 5 and (not owner or not layouts):
                status = 13
            elif not owner or not layouts:
                status = 8
            elif op == 0 and (not output or not count):
                status = 13
            add(f'pointer-precedence-{flags}-{op}', good, op, a, b, flags=flags, expected_status=status)
    # I require every declaration to validate before publishing unrelated facts.
    duplicate = nominal(operations=[(5, desc(), []), (6, desc(), [])])
    invalid_ctor = nominal(constructors=[(0, 1, 0)])
    for label, model in [('duplicate-effect-bytes', duplicate), ('interior-constructor', invalid_ctor)]:
        for op, a, b in ((0, 0, 32), (1, 2, 0), (2, 5, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0)):
            add(f'whole-validation-{label}-{op}', model, op, a, b, expected=report(8))
        add(f'early-missing-key-{label}', model, 2, NO, expected=report(13))
        add(f'early-missing-constructor-site-{label}', model, 4, NO, NO, expected=report(13))
        add(f'early-no-path-index-{label}', model, 0, NO, 0, expected=report(13))
        add(f'early-validation-only-union-sentinel-{label}', model, 5, NO, 0, expected=report(13))
    empty_union = image(record(3, extensions=[extension(1, struct.pack('<I', 0))]))
    add('present-empty-union-table', empty_union, 5, 0, 0, expected=report(8))
    for revision in (1, 2):
        old = nominal(revision=revision)
        add(f'no-constructor-table-revision-{revision}', old, 4, 0, 0, expected=report(0, NO))
        add(f'element-table-revision-{revision}', old, 3, expected=report(0, 0) if revision == 1 else report(0, 5, 1, NO, 5, NO, 7, 0, 8, 0, 10, 1))
    # These are metadata queries, not function-span/stack admission.
    odd_span = image(record(3, functions=[(0, 0, desc(1), [])],
        extensions=[extension(4, effect_data(1))]), strings=[b'op'],
        functions=[function(results=1, result_tag=1, start=NO, length=2)])
    add('unbound-constructor-does-not-admit-code-span', odd_span, 4, 0, 1, expected=report(0, NO))
    huge_paths = image(record(2, paths=[tuple(range(32))]*256))
    add('path-full-domain', huge_paths, 0, 255, 32, expected=report(0, 32, *range(32)))
    add('path-domain-overrun', huge_paths, 0, 256, 32, expected=report(13))
    add('path-depth-capacity', huge_paths, 0, 255, 31, expected=report(13))
    globals_ = image(record(3, extensions=[extension(3, globals_data([desc(1)]*256))]))
    add('global-full-domain', globals_, 1, 256, expected=report(0, 256, *([1, 0, NO]*256)))
    add('global-count-only-full-domain', globals_, 1, 0, flags=55, expected=report(0, 256))
    add('global-capacity-before-copy', globals_, 1, 255, expected=report(13))
    operations = [(i, desc(), []) for i in range(256)]
    names = [f'op{i:03d}'.encode() for i in range(256)]
    effects = image(record(3, extensions=[extension(4, effect_data(operations=operations))]), strings=names)
    add('effect-full-domain', effects, 2, 255, expected=report(0, 255, 0, NO, 0))
    parameters = image(record(3, extensions=[extension(4, effect_data(2, types=[desc(1)]*256,
        operations=[(0, desc(), [desc(1)]*256)]), 2)]), strings=[b'op'])
    add('element-full-domain', parameters, 3, expected=report(0, 256, *([1, NO]*256)))
    add('parameter-full-domain', parameters, 2, 0, expected=report(0, 0, 0, NO, 256, *([1, NO]*256)))
    variants = image(record(3, flags=[0], extensions=[extension(1, union_data(0, [(i, 0, 0) for i in range(256)]))]),
        rows=[(2, 256, [])], strings=[f'Case{i}'.encode() for i in range(256)]+[b'Choice'])
    add('variant-full-domain', variants, 5, 0, 255, expected=report(0, 0, 255, 0, 0))
    add('variant-domain-overrun', variants, 5, 0, 256, expected=report(13))
    # I retain every original declaration carrier through a full-validating query.
    for index, (name, model, expected) in enumerate(declaration_corpus()):
        add('original-'+name, model, 1, 256, expected_status=int(expected.split(':')[0]) if expected is not None else None)
    # Every truncation of the complete nominal ownership payload is retained.
    layout_size, own_size = struct.unpack_from('<II', good)
    # The model helper reconstructs identical tables while varying only ownership.
    own = record(3, flags=[1, 1], functions=[(0, 0, desc(1), [])], paths=[(1, 65535)],
        extensions=[extension(1, union_data(1, [(0, 0, 1), (1, 1, 0)])),
                    extension(3, globals_data([desc(8, 0, mode=1), desc(10, 1)])),
                    extension(4, effect_data(3, types=[desc(1), desc(5), desc(7, 0), desc(8, 0), desc(10, 1)],
                        constructors=[(0, 0, 0)], operations=[(5, desc(8, 0), [desc(7, 2), desc(5)])]), 3)])
    assert len(own) == own_size
    for size in range(len(own)):
        add(f'truncated-complete-declarations-{size}', nominal(own[:size]), 1, 256, flags=63,
            expected=report(9) if size == 0 else None)
    raw = image(record(3, functions=[(0, 0, desc(1), [])], extensions=[extension(4,
        effect_data(3, types=[desc(1)], constructors=[(0, i*2, 0) for i in range(32768)]), 3)]),
        strings=[b'op'], functions=[function(results=1, result_tag=1, length=65536)], code=b'\x50\x01'*32768)
    add('constructor-full-domain', raw, 4, 0, 65534, expected=report(0, 0))
    valid_transport = carrier(good, 1, 2)
    malformed = [('short-query', b'\0'*19), ('bad-op', carrier(good, 6)),
                 ('bad-flags', carrier(good, 1, flags=64)),
                 ('wide-path-capacity', carrier(good, 0, 0, 65536)),
                 ('wide-variant', carrier(good, 5, 0, 65536)),
                 ('wrong-model-length', valid_transport[:16]+struct.pack('<I', 0)+valid_transport[20:]),
                 ('short-model', carrier(b'\0'*27, 0, flags=63))]
    for name, raw in malformed:
        cases.append(dict(name=name, data=raw, expected='D', expected_status=None))
    return cases


class OwnershipQueries(unittest.TestCase):
    def test_complete_queries_against_actual_c(self):
        work = Path(tempfile.mkdtemp(prefix='nano-ownership-queries-'))
        cases = corpus()
        paths = []
        record_ = dict(scope='I check declaration query facts and publication only, not execution admission.',
                       corpus=[], routes=[])
        for index, row in enumerate(cases):
            path = work/f'{index:04d}.bin'; path.write_bytes(row['data']); paths.append(path)
            record_['corpus'].append(dict(name=row['name'], path=str(path), bytes=len(row['data']),
                sha256=hashlib.sha256(row['data']).hexdigest(), expected=row['expected'], expected_status=row['expected_status']))
        def save(): (work/'manifest.json').write_text(json.dumps(record_, indent=2)+'\n')
        save()
        def execute(command, label):
            reports = []
            for start in range(0, len(paths), 16):
                reports += checked([*command, *paths[start:start+16]], timeout=600).splitlines()
            (work/(label+'.reports')).write_text('\n'.join(reports)+'\n')
            self.assertEqual(len(reports), len(cases), label)
            for row, actual in zip(cases, reports):
                with self.subTest(route=label, case=row['name']):
                    if row['expected'] is not None: self.assertEqual(actual, row['expected'])
                    if row['expected_status'] is not None: self.assertTrue(actual.startswith(report(row['expected_status'])), actual)
            record_['routes'].append(dict(label=label, command=list(map(str, command)), reports=len(reports),
                sha256=hashlib.sha256(('\n'.join(reports)+'\n').encode()).hexdigest()))
            save()
            return reports
        reference = Path(os.environ.get('NANOLANG_OWNERSHIP_QUERIES_REFERENCE', ROOT/'obj/nvm2c_ownership_queries_reference'))
        expected = execute([reference], 'actual-c')
        sanitized = Path(os.environ.get('NANOLANG_OWNERSHIP_QUERIES_REFERENCE_ASAN', ROOT/'obj/nvm2c_ownership_queries_reference_asan'))
        self.assertEqual(execute([sanitized], 'actual-c-sanitized'), expected)
        for label, command in build_routes(ROOT/'tests/nvm2c_ownership_queries_driver.nano', work):
            self.assertEqual(execute(command, label), expected, label)
