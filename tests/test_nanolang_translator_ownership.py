"""I compare all shared declaration versions with the production C validator."""
from pathlib import Path
import os
import random
import struct
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked
from tests.test_nanolang_translator_payloads import layout, table

ROOT = Path(__file__).resolve().parents[1]
NONE = 0xffffffff


def u32(value):
    return struct.pack('<I', value)


def desc(tag=0, index=NONE, mode=0, reserved=0):
    return struct.pack('<BBHI', tag, mode, reserved, index)


def path_table(paths=()):
    return u32(len(paths)) + b''.join(struct.pack('<HH', len(path), 0)
                                     + struct.pack('<' + 'H' * len(path), *path)
                                     + b'\0' * (-(len(path) * 2) % 4) for path in paths)


def extension(kind, data, revision=1):
    return struct.pack('<HHI', kind, revision, len(data)) + data + b'\0' * (-len(data) % 4)


def record(version=1, flags=(), functions=(), paths=(), extensions=(), raw_paths=None):
    data = struct.pack('<II', version, len(flags)) + bytes(flags) + b'\0' * (-len(flags) % 4)
    data += u32(len(functions))
    for locals_, params, result, values in functions:
        data += struct.pack('<HH', locals_, params) + result + b''.join(values)
    path_bytes = path_table(paths) if raw_paths is None else raw_paths
    if version == 2:
        data += path_bytes
    if version == 3:
        data += u32(len(path_bytes)) + path_bytes + u32(len(extensions)) + b''.join(extensions)
    return data


def function(locals_=0, arity=0, results=0, result_tag=0, params=(), start=0, length=0):
    return struct.pack('<8I', start, length, arity, locals_, 0, results, result_tag, len(params)) + bytes(params)


def image(owned, rows=(), strings=(), functions=(), code=b'', raw_layouts=None, counts=None):
    layouts = table(*(layout(kind, fields, name) for kind, name, fields in rows)) if raw_layouts is None else raw_layouts
    if counts is None:
        counts = tuple(sum(row[0] == kind for row in rows) for kind in range(3))
    return (struct.pack('<7I', len(layouts), len(owned), len(strings), len(functions), *counts)
            + b''.join(u32(len(name)) + name for name in strings) + b''.join(functions) + layouts + owned + code)


def globals_data(values):
    return u32(len(values)) + b''.join(values)


def union_data(layout_index, variants):
    return u32(1) + struct.pack('<IHH', layout_index, len(variants), 0) + b''.join(struct.pack('<IHH', *row) for row in variants)


def effect_data(revision=1, types=(), constructors=(), operations=None):
    if operations is None:
        operations = [(0, desc(), [])]
    data = b''
    if revision >= 2:
        data += u32(len(types)) + b''.join(types)
    if revision == 3:
        data += u32(len(constructors)) + b''.join(struct.pack('<III', *row) for row in constructors)
    data += u32(len(operations))
    for key, result, params in operations:
        data += struct.pack('<IHH', key, len(params), 0) + result + b''.join(params)
    return data


def corpus():
    cases = []

    def add(name, owned, expected=None, **model):
        cases.append((name, image(owned, **model), expected))

    def expected(status=0, needs=False):
        return f'{status}:{int(needs)}'

    scalar_global = extension(3, globals_data([desc(1)]))
    add('absent-ignores-missing-layouts', b'', expected(), raw_layouts=b'')
    add('present-requires-layouts', record(), expected(8), raw_layouts=b'')
    for version in (1, 2):
        add(f'empty-v{version}', record(version), expected())
        add(f'trailing-v{version}', record(version) + b'\0', expected(9))
    add('v3-minimal-global', record(3, extensions=[scalar_global]), expected(0, True))
    for length in range(12):
        add(f'truncated-base-{length}', record()[:length], expected(0 if length == 0 else 1))
    for version in (0, 4, NONE):
        add(f'unknown-version-{version}', record(version), expected(3))
    row = [(0, NONE, [])]
    for flag in range(256):
        add(f'layout-flag-{flag}', record(flags=[flag]),
            expected(0, flag == 3) if flag in (0, 1, 3) else expected(8 if flag == 2 else 7), rows=row)
    for kind in (0, 1, 2, 3):
        add(f'complete-kind-v1-{kind}', record(flags=[1]), expected() if kind == 0 else expected(8), rows=[(kind, NONE, [])])
    for tag in range(256):
        rows = [(0, NONE, [(tag, NONE, NONE)])]
        add(f'complete-field-tag-{tag}', record(flags=[1]), expected() if tag in range(1, 6) else expected(8), rows=rows)
        add(f'unknown-field-tag-{tag}', record(flags=[0]), None, rows=rows)
    add('retained-tail-is-not-canonical', record(), expected(8), raw_layouts=table() + b'\0')
    add('retained-count-mismatch', record(flags=[0]), expected(8), rows=row, counts=(0, 0, 0))
    add('retained-name-index', record(flags=[0]), expected(8), rows=[(0, 0, [])])
    forward = [(0, NONE, [(8, 1, NONE)]), (0, NONE, [])]
    for flags, status in [([0, 1], 0), ([0, 3], 8), ([5, 3], 8), ([5, 1], 7), ([1, 1], 0)]:
        add(f'forward-resource-order-{flags}', record(flags=flags), expected(status), rows=forward)
    for mode in range(4):
        for tag in (1, 5, 8):
            rows = [(0, NONE, [(tag, NONE if tag != 8 else 0, NONE)])] if tag != 8 else [(0, NONE, []), (0, NONE, [(8, 0, NONE)])]
            root = len(rows) - 1
            flags = [3] * len(rows)
            declarations = [(1, 1, desc(), [desc(8, root, mode)])]
            add(f'borrow-root-{mode}-{tag}', record(flags=flags, functions=declarations),
                expected(0, True) if mode == 0 or (mode in (1, 2) and tag == 1) else expected(8),
                rows=rows, functions=[function(1, 1, params=[8])])
    for tag in range(256):
        add(f'local-tag-{tag}', record(functions=[(1, 0, desc(), [desc(tag)])]),
            expected() if tag < 16 else expected(8), functions=[function(1)])
    for mode in range(256):
        add(f'nonparameter-mode-{mode}', record(functions=[(1, 0, desc(), [desc(1, mode=mode)])]),
            expected() if mode == 0 else expected(8), functions=[function(1)])
    add('return-signature-mismatch', record(functions=[(0, 0, desc(1), [])]), expected(8), functions=[function()])
    add('multiple-results-before-descriptor', record(functions=[(0, 0, b'', [])]), expected(13), functions=[function(results=2)])
    add('parameter-annotation-mismatch', record(functions=[(1, 1, desc(), [desc(1)])]), expected(8), functions=[function(1, 1, params=[4])])
    add('missing-parameter-tags-mean-void', record(functions=[(1, 1, desc(), [desc(0)])]), expected(), functions=[function(1, 1)])
    add('descriptor-reserved-before-tag', record(functions=[(1, 0, desc(), [desc(255, reserved=1)])]), expected(7), functions=[function(1)])
    for paths, status in [([], 0), ([[65535]], 0), ([[1, 1]], 0), ([[]], 13), ([list(range(33))], 13),
                          ([list(range(32))] * 256, 0), ([[0]] * 257, 13)]:
        add(f'path-budget-{len(paths)}-{len(paths[0]) if paths else 0}', record(2, paths=paths), expected(status))
    add('path-padding', record(2, raw_paths=u32(1) + struct.pack('<HHH', 1, 0, 0) + b'\1\0'), expected(9))
    add('path-reserved-before-depth', record(2, raw_paths=u32(1) + struct.pack('<HH', 0, 1)), expected(7))
    for path_bytes in (b'', b'\0' * 3, b'\0' * 6):
        add(f'v3-path-framing-{len(path_bytes)}', record(3, raw_paths=path_bytes, extensions=[scalar_global]), expected(9))
    for exts, status in [([], 13), ([extension(5, b'')], 3), ([extension(0, b'')], 8),
                         ([extension(3, b'', 2)], 3), ([scalar_global, scalar_global], 8),
                         ([scalar_global] * 5, 13), ([extension(2, b'')], 3),
                         ([extension(1, b'')], 8)]:
        add(f'extension-frames-{len(cases)}', record(3, extensions=exts), expected(status))
    for count in (0, 1, 256, 257):
        add(f'global-count-{count}', record(3, extensions=[extension(3, globals_data([desc(1)] * count))]),
            expected(0, True) if 1 <= count <= 256 else expected(13))
    for tag in range(256):
        add(f'global-tag-{tag}', record(3, extensions=[extension(3, globals_data([desc(tag)]))]),
            expected(0, True) if tag in range(1, 6) else expected(13 if tag in (8, 10) else 8))
    add('global-mutable-reserved', record(3, extensions=[extension(3, globals_data([desc(1, mode=2)]))]), expected(7))
    for flag in (0, 1, 3):
        add(f'global-nominal-{flag}', record(3, flags=[flag], extensions=[extension(3, globals_data([desc(8, 0)]))]),
            expected(0, True) if flag else expected(8), rows=row)
    union_rows = [(2, 0, [(1, NONE, 1), (5, NONE, 1)])]
    variants = union_data(0, [(2, 0, 1), (3, 1, 1)])
    names = [b'U', b'field', b'Variant', b'Other']
    for flag in (0, 1, 3):
        add(f'union-authority-{flag}', record(3, flags=[flag], extensions=[extension(1, variants)]), expected(0, True), rows=union_rows, strings=names)
    add('union-required-before-bad-global', record(3, flags=[0], extensions=[extension(3, b'')]), expected(8), rows=union_rows, strings=names)
    for label, entries, status in [('duplicate-name', [(2, 0, 1), (2, 1, 1)], 8),
                                    ('gap', [(2, 0, 1), (3, 2, 0)], 13),
                                    ('incomplete', [(2, 0, 1)], 13), ('no-variants', [], 8)]:
        add('union-' + label, record(3, flags=[0], extensions=[extension(1, union_data(0, entries))]), expected(status), rows=union_rows, strings=names)
    add('union-name-indices-not-text-identity', record(3, flags=[0], extensions=[extension(1, variants)]), expected(0, True), rows=union_rows, strings=[b'U', b'f', b'V', b'V'])
    add('union-field-needs-name', record(3, flags=[0], extensions=[extension(1, union_data(0, [(1, 0, 1)]))]), expected(8), rows=[(2, 0, [(1, NONE, NONE)])], strings=[b'U', b'V'])
    for count in (256, 257):
        variants = union_data(0, [(index + 1, 0, 0) for index in range(count)])
        add(f'variant-budget-{count}', record(3, flags=[0], extensions=[extension(1, variants)]),
            expected(0, True) if count == 256 else expected(8), rows=[(2, 0, [])], strings=[b'U'] + [b'V'] * count)
        unions = u32(count) + b''.join(struct.pack('<IHHIHH', index, 1, 0, 1, 0, 0) for index in range(count))
        add(f'union-budget-{count}', record(3, flags=[0] * count, extensions=[extension(1, unions)]),
            expected(0, True) if count == 256 else expected(13), rows=[(2, 0, [])] * count, strings=[b'U', b'V'])
    for revision in (1, 2, 3):
        types = [desc(1)] if revision >= 2 else []
        add(f'effect-revision-{revision}', record(3, extensions=[extension(4, effect_data(revision, types), revision)]), expected(0, True), strings=[b'op'])
    for key, names_, status in [(0, [b''], 8), (0, [b'a\0b'], 8), (0, [b'\xff'], 0), (1, [b'op'], 13)]:
        add(f'effect-key-{len(cases)}', record(3, extensions=[extension(4, effect_data(operations=[(key, desc(), [])]))]), expected(status, status == 0), strings=names_)
    for names_, keys, status in [([b'a', b'a'], [0, 1], 8), ([b'a', b'b'], [1, 0], 8), ([b'a', b'b'], [0, 0], 8), ([b'a', b'b'], [0, 1], 0)]:
        add(f'effect-key-order-{len(cases)}', record(3, extensions=[extension(4, effect_data(operations=[(key, desc(), []) for key in keys]))]), expected(status, status == 0), strings=names_)
    for count in (0, 256, 257):
        types = [desc(1)] * count
        add(f'effect-type-budget-{count}', record(3, extensions=[extension(4, effect_data(2, types), 2)]), expected(0, True) if count == 256 else expected(13), strings=[b'op'])
        operations = [(i, desc(), []) for i in range(count)]
        add(f'effect-budget-{count}', record(3, extensions=[extension(4, effect_data(operations=operations))]), expected(0, True) if count == 256 else expected(13), strings=[f'op-{i}'.encode() for i in range(count)])
        add(f'effect-param-budget-{count}', record(3, extensions=[extension(4, effect_data(operations=[(0, desc(), [desc(1)] * count)]))]), expected(0, True) if count <= 256 else expected(13), strings=[b'op'])
    for element, status in [(desc(7, 0), 13), (desc(0), 8), (desc(1, 0), 8), (desc(1), 0)]:
        add(f'element-kind-{len(cases)}', record(3, extensions=[extension(4, effect_data(2, [element]), 2)]), expected(status, status == 0), strings=[b'op'])
    add('backward-array-elements', record(3, extensions=[extension(4, effect_data(2, [desc(1), desc(7, 0)]), 2)]), expected(0, True), strings=[b'op'])
    for index, expected_status in [(0, 0), (1, 13), (NONE, 8)]:
        owned = record(3, functions=[(1, 0, desc(), [desc(7, index)])], extensions=[extension(4, effect_data(2, [desc(1)]), 2)])
        add(f'function-array-element-{index}', owned, expected(expected_status, expected_status == 0), strings=[b'op'], functions=[function(1)])
    for version in (1, 2):
        add(f'early-version-typed-array-{version}', record(version, functions=[(1, 0, desc(), [desc(7, 0)])]), expected(8), functions=[function(1)])
    for label, code, site, status in [('new', b'\x50\x01', 0, 0), ('empty-literal', b'\x58\x01\0\0', 0, 0),
                                       ('nonempty-literal', b'\x58\x01\x01\0', 0, 8), ('wrong-tag', b'\x50\x02', 0, 8),
                                       ('bad-prefix', b'\xff\x50\x01', 1, 8), ('operand-interior', b'\x01' + b'\0' * 8 + b'\x50\x01', 1, 8),
                                       ('outside', b'\x50\x01', 2, 13), ('unknown-tail-is-ordinary-verification', b'\x50\x01\xff', 0, 0)]:
        owned = record(3, functions=[(0, 0, desc(), [])], extensions=[extension(4, effect_data(3, [desc(1)], [(0, site, 0)]), 3)])
        add('constructor-' + label, owned, expected(status, status == 0), functions=[function(length=len(code))], code=code, strings=[b'op'])
    for count in (32768, 32769):
        code = b'\x50\x01' * count
        data = effect_data(3, [desc(1)], [(0, i * 2, 0) for i in range(count)])
        owned = record(3, functions=[(0, 0, desc(), [])], extensions=[extension(4, data, 3)])
        add(f'constructor-budget-{count}', owned, expected(0, True) if count == 32768 else expected(13), functions=[function(length=len(code))], code=code, strings=[b'op'])
    # This valid record crosses every shared v3 extension and descriptor stage.
    rich_rows = [(0, NONE, [(1, NONE, NONE)]), (2, 0, [(1, NONE, 1)])]
    rich_model = dict(rows=rich_rows, strings=[b'U', b'field', b'Variant', b'op'],
                      functions=[function(1, length=2)], code=b'\x50\x01')
    rich = record(3, [1, 0], [(1, 0, desc(), [desc(7, 0)])], paths=[[0]], extensions=[
        extension(1, union_data(1, [(2, 0, 1)])), scalar_global,
        extension(4, effect_data(3, [desc(1)], [(0, 0, 0)], [(3, desc(), [desc(7, 0)])]), 3)])
    add('complete-v3', rich, expected(0, True), **rich_model)
    for length in range(len(rich)):
        add(f'complete-v3-truncated-{length}', rich[:length], expected() if length == 0 else None, **rich_model)
    rng = random.Random(114213)
    for i in range(512):
        data = bytearray(rich)
        data[rng.randrange(len(data))] ^= 1 << rng.randrange(8)
        add(f'complete-v3-mutation-{i}', bytes(data), None, **rich_model)
    return cases


class OwnershipDeclarations(unittest.TestCase):
    def test_nested_slice_type_and_checked_refusals(self):
        from tests.native_process import run_captured
        with tempfile.TemporaryDirectory(prefix='nano-slice-context-') as name:
            work = Path(name)
            for route, command in build_routes(ROOT / 'tests/array_slice_nested.nano', work):
                self.assertEqual(checked(command), '', route)
            producers = [(ROOT / 'bin/nanoc_c', []), (ROOT / 'bin/nano_virt', [])]
            if os.environ.get('NANOLANG_TRANSLATOR_SOURCE_COMPILER'):
                producers.append((os.environ['NANOLANG_TRANSLATOR_SOURCE_COMPILER'], ['--emit-nvm']))
            invalid = ['(array_slice 1 0 1)', '(array_slice xs true 1)', '(array_slice xs 0)',
                       '(array_push ys (array_slice xs 0 1))']
            for index, expression in enumerate(invalid):
                source = work / f'bad-{index}.nano'
                source.write_text('fn main()->int{let xs:array<int> = [1] let ys:array<array<u8>> = [] '
                                  + expression + ' return 0}\nshadow main {assert true}\n')
                for producer, flags in producers:
                    output = work / 'prior'
                    output.write_bytes(b'prior slice product')
                    result = run_captured([producer, source, *flags, '-o', output], build=True, timeout=60, cwd=ROOT)
                    self.assertNotEqual(result.returncode, 0, (producer, expression, result.stdout, result.stderr))
                    self.assertNotRegex((result.stdout + result.stderr).decode(errors='replace'), r'(?i)(parse error|parsing failed|unexpected token|failed to parse)')
                    self.assertEqual(output.read_bytes(), b'prior slice product')

    def test_shared_versions_and_authority_boundaries(self):
        cases = corpus()
        oracle = ROOT / 'obj/nvm2c_ownership_reference'
        with tempfile.TemporaryDirectory(prefix='nano-ownership-declarations-') as name:
            work = Path(name)
            paths = []
            for index, (_, data, _) in enumerate(cases):
                path = work / f'case-{index}.bin'
                path.write_bytes(data)
                paths.append(path)

            def execute(command):
                values = []
                for start in range(0, len(paths), 64):
                    values += checked([*command, *paths[start:start + 64]], timeout=600).splitlines()
                self.assertEqual(len(values), len(cases))
                return values

            reference = execute([oracle])
            for (case, _, expected), actual in zip(cases, reference):
                if expected is not None:
                    with self.subTest(reference=case): self.assertEqual(actual, expected)
            routes = build_routes(ROOT / 'tests/nvm2c_ownership_driver.nano', work)
            for route, command in routes:
                values = execute(command)
                for (case, _, _), expected, actual in zip(cases, reference, values):
                    with self.subTest(route=route, case=case): self.assertEqual(actual, expected)
            print(f'I matched {len(cases)} shared ownership status/authority decisions across {len(routes)} routes; '
                  f'{sum(expected is not None for _, _, expected in cases)} explicit controls include unchanged path/extension/variant/global/effect/constructor budgets.')


if __name__ == '__main__':
    unittest.main()
