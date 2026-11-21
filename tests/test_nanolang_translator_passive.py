"""I compare full passive claims with the production C passive validator."""
from pathlib import Path
import os
import random
import shlex
import struct
import sys
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked
from tests.test_nanolang_translator_passive_calls import Assembly, call, i64, local

ROOT = Path(__file__).resolve().parents[1]


def from_module(data):
    sections = {}
    count = struct.unpack_from('<I', data, 24)[0]
    for index in range(count):
        kind, _, offset, size = struct.unpack_from('<IIQQ', data, 40 + index * 24)
        sections[kind] = data[offset:offset + size]
    signatures, cursor = [], 4
    for _ in range(struct.unpack_from('<I', sections[3])[0]):
        params, results = struct.unpack_from('<HH', sections[3], cursor)
        signatures.append((sections[3][cursor + 4:cursor + 4 + params], results))
        cursor += 4 + ((params + 3) // 4) * 4 + ((results + 3) // 4) * 4
    records = []
    count = struct.unpack_from('<I', sections[5])[0]
    for index in range(count):
        _, signature, start, length, locals_, upvalues, _, _ = struct.unpack_from('<IIQQHHHH', sections[5], 4 + index * 32)
        tags, results = signatures[signature]
        records.append(words(start, length, len(tags), locals_, upvalues, results, len(tags)) + tags)
    claim, code = sections[12], sections[6]
    return words(len(claim), count) + b''.join(records) + claim + code


def words(*values):
    return struct.pack('<' + 'I' * len(values), *values)


def node(entry, end, result, deps=(), inputs=(), effects=0, resources=0):
    return words(entry, end, result, len(deps), len(inputs), effects, resources, *deps, *inputs)


def block(nodes, entry, end, kind=1, owner=0):
    return words(kind, owner, entry, end, len(nodes)) + b''.join(nodes)


def payload(blocks, version=2):
    return words(version, len(blocks)) + b''.join(blocks)


def image(claim, functions):
    records, code = [], b''
    for body, arity, locals_, upvalues, results, tags in functions:
        records.append(words(len(code), len(body), arity, locals_, upvalues, results, len(tags)) + bytes(tags))
        code += body
    return words(len(claim), len(functions)) + b''.join(records) + claim + code


def guarded(tags=(1,), inputs=(0,), prefix=None, suffix=b'\x3d', version=2):
    if prefix is None:
        prefix = b''.join(local(16, index) + bytes([140, tag, 161]) for index, tag in enumerate(tags))
    body = b''.join(local(16, index) for index in inputs)
    body += b'\x20' * max(0, len(inputs) - 1)
    if not inputs:
        body += b'\x03\x01'
    body += local(17, len(tags))
    claim = payload([block([node(len(prefix), len(prefix) + len(body), len(tags), inputs=inputs)],
                           len(prefix), len(prefix) + len(body))], version)
    functions = [(prefix + body + suffix, len(tags), len(tags) + 1, 0, 0, tags)]
    return claim, functions


def cached_calls(reverse=False):
    roots = (1, 32) if reverse else (32, 1)
    owner = call(roots[0]) + local(17, 0) + call(roots[1]) + local(17, 1) + b'\x3d'
    functions = [(owner, 0, 2, 0, 0, ())]
    functions += [(call(index + 1) + b'\x3d', 0, 0, 0, 1, ()) for index in range(1, 65)]
    functions += [(b'\x03\x01\x3d', 0, 0, 0, 1, ())]
    claim = payload([block([node(0, 8, 0), node(8, 16, 1)], 0, 16)])
    return claim, functions


def cases(widths):
    result = []
    body = b'\x03\x01' + local(17, 0)

    def add(name, claim, functions, expected=None):
        result.append((name, image(claim, functions), expected))

    def basic(name, code=body, claim=None, expected=None, version=2, prefix=b'', suffix=b'\x3d', locals_=1):
        begin, end = len(prefix), len(prefix) + len(code)
        if claim is None:
            claim = payload([block([node(begin, end, 0)], begin, end)], version)
        add(name, claim, [(prefix + code + suffix, 0, locals_, 0, 0, ())], expected)

    add('absent-claim-empty-module', b'', [], True)
    add('absent-claim-does-not-verify-code', b'', [(b'\xff', 0, 0, 0, 0, ())], True)
    for version in (1, 2):
        basic(f'constant-{version}', expected=True, version=version)
        basic(f'end-at-function-limit-{version}', expected=True, version=version, suffix=b'')
        for kind in (0, 1, 2, 3, 0xffffffff):
            basic(f'kind-{version}-{kind}', claim=payload([block([node(0, 5, 0)], 0, 5, kind)], version), expected=kind in (1, 2))
        code = b'\0' * 65534 + body
        basic(f'node-limit-exact-{version}', code=code, expected=True, version=version)
        basic(f'node-limit-over-{version}', code=b'\0' + code, expected=version == 1, version=version)
        basic(f'owner-foreign-{version}', expected=False, version=version, suffix=b'\x3e\0\0\0\0\x3d')
        basic(f'owner-ordinary-call-{version}', expected=version == 2, version=version,
              suffix=call(0xffffffff) + b'\x3d')
        for op in (160, 161, 163, 164):
            basic(f'owner-output-assert-halt-{version}-{op}', expected=True, version=version,
                  suffix=b'\x03\x01' + bytes([op]) + b'\x3d')
    for tag in range(256):
        claim, functions = guarded(tags=(tag,))
        add(f'guard-tag-{tag}', claim, functions, tag in (1, 3, 4, 5))
    for version in (1, 2):
        claim, functions = guarded(version=version)
        add(f'external-read-version-{version}', claim, functions, version == 2)
    for name, prefix, expected in [
        ('missing', b'', False), ('nop-prefix', b'\0', False),
        ('wrong-tag', local(16, 0) + bytes([140, 4, 161]), False),
        ('wrong-parameter', local(16, 1) + bytes([140, 1, 161]), False),
        ('missing-assert', local(16, 0) + bytes([140, 1]), False),
        ('matching-guard-then-nop', local(16, 0) + bytes([140, 1, 161, 0]), True),
    ]:
        claim, functions = guarded(prefix=prefix)
        add(f'guard-prefix-{name}', claim, functions, expected)
    for name, prefix, expected in [
        ('ascending', local(16, 0) + bytes([140, 1, 161]) + local(16, 1) + bytes([140, 1, 161]), True),
        ('descending', local(16, 1) + bytes([140, 1, 161]) + local(16, 0) + bytes([140, 1, 161]), False),
        ('duplicate', (local(16, 0) + bytes([140, 1, 161])) * 2, False),
    ]:
        claim, functions = guarded(tags=(1, 1), inputs=(0, 1), prefix=prefix)
        add(f'multiple-guards-{name}', claim, functions, expected)
    claim, functions = guarded()
    base_claim, base_functions = claim, functions
    for length in range(len(claim)):
        add(f'payload-truncation-{length}', claim[:length], functions, length == 0)
    for tail in (b'\0', words(0), words(1, 2)):
        add(f'payload-trailing-{len(tail)}', claim + tail, functions, False)
    for version in (0, 3, 0xffffffff):
        add(f'unknown-version-{version}', words(version) + claim[4:], functions, False)
    for index in range(len(claim) // 4):
        for value in (0, 1, 2, 3, 0x7fffffff, 0x80000000, 0xffffffff):
            mutation = bytearray(claim)
            struct.pack_into('<I', mutation, index * 4, value)
            add(f'word-mutation-{index}-{value}', bytes(mutation), functions)
    for inputs in ((0, 0), (1, 0), (1,), (0, 2), ()):
        claim, functions = guarded(tags=(1, 1), inputs=(0, 1))
        claim = payload([block([node(12, 22, 2, inputs=inputs)], 12, 22)])
        add(f'exact-input-accounting-{inputs}', claim, functions, False)
    for name, suffix, expected in [('write-read-parameter', b'\x03\x01' + local(17, 0) + b'\x3d', False),
                                   ('write-other-parameter', b'\x03\x01' + local(17, 1) + b'\x3d', True)]:
        claim, functions = guarded(tags=(1, 1), inputs=(0,), suffix=suffix)
        add(name, claim, functions, expected)
    claim, functions = guarded()
    body0, arity, locals_, upvalues, results, tags = functions[0]
    add('missing-param-types', claim, [(body0, arity, locals_, upvalues, results, ())], False)
    for version in (1, 2):
        producer = b'\x03\x01' + local(17, 1)
        consumer = local(16, 1) + local(17, 0)
        functions = [(producer + consumer + b'\x3d', 0, 2, 0, 0, ())]
        for deps, expected in [((1,), True), ((), False), ((0,), False), ((1, 1), False), ((2,), False)]:
            claim = payload([block([node(5, 11, 0, deps=deps), node(0, 5, 1)], 0, 11, kind=2)], version)
            add(f'forward-flow-{version}-{deps}', claim, functions, expected)
        claim = payload([block([node(5, 11, 0, deps=(1,)), node(0, 5, 1)], 0, 11)], version)
        add(f'par-forbids-dependencies-{version}', claim, functions, False)
        claim = payload([block([node(0, 6, 0, deps=(1,)), node(6, 12, 1, deps=(0,))], 0, 12, kind=2)], version)
        add(f'cycle-{version}', claim, [(local(16, 1) + local(17, 0) + local(16, 0) + local(17, 1), 0, 2, 0, 0, ())], False)
        code = body + b'\x03\x01' + local(17, 1)
        for nodes, expected, name in [([node(0, 5, 0), node(5, 10, 1)], True, 'source-order'),
                                      ([node(5, 10, 1), node(0, 5, 0)], False, 'unstable-order'),
                                      ([node(0, 5, 0), node(5, 10, 0)], False, 'duplicate-result'),
                                      ([node(0, 5, 0), node(5, 10, 1, deps=(0,))], False, 'unused-dependency')]:
            add(f'independent-{version}-{name}', payload([block(nodes, 0, 10, kind=2)], version),
                [(code + b'\x3d', 0, 2, 0, 0, ())], expected)
    for branch_op in (56, 57, 58):
        a = Assembly()
        a.emit(b'\x03\x01')
        if branch_op == 56:
            a.branch(branch_op, 'store')
        else:
            a.branch(branch_op, 'right')
            a.emit(b'\x03\x00')
            a.branch(56, 'store')
            a.label('right')
            a.emit(b'\x03\x01')
        a.label('store')
        a.emit(local(17, 0))
        basic(f'node-forward-branch-{branch_op}', code=a.code(), expected=True)
        basic(f'v1-node-branch-{branch_op}', code=a.code(), expected=False, version=1)
    for name, code in [('backedge', b'\x38\0\0\0\0' + body),
                        ('operand-target', b'\x38\x01\0\0\0' + body),
                        ('exit-target', b'\x38' + struct.pack('<i', 10) + body),
                        ('unreachable-gap', b'\x03\x01\x38\x06\0\0\0\0' + local(17, 0)),
                        ('intermediate-store', body + body),
                        ('wrong-store', b'\x03\x01' + local(17, 1))]:
        basic('node-refusal-' + name, code=code, expected=False, locals_=2)
    for name, destination, expected in [('entry', 5, True), ('interior', 6, False), ('outside-module-is-ordinary-verification', 0x7fffffff, True)]:
        basic('owner-branch-' + name, prefix=b'\x38' + struct.pack('<i', destination), expected=expected)
    for reverse in (False, True):
        claim, functions = cached_calls(reverse)
        add(f'shared-cache-ordered-{reverse}', claim, functions, not reverse)
    for effect in (0, 1):
        claim = payload([block([node(0, 10, 0)], 0, 10)])
        callee = b'\xa0\x3d' if effect else local(16, 0) + b'\x3d'
        functions = [(b'\x03\x01' + call(1) + local(17, 0) + b'\x3d', 0, 1, 0, 0, ()),
                     (callee, 1, 1, 0, 1, (1,))]
        add(f'node-call-{effect}', claim, functions, not effect)
    for op, width in enumerate(widths):
        code = bytes([op]) + b'\0' * max(0, width - 1)
        for version in (1, 2):
            basic(f'owner-op-{version}-{op}', prefix=code, version=version)
        for stack in range(3):
            basic(f'node-op-{op}-stack-{stack}', code=i64() * stack + code + local(17, 0))
    rng = random.Random(114212)
    for i in range(256):
        claim = bytearray(base_claim)
        claim[rng.randrange(len(claim))] ^= 1 << rng.randrange(8)
        add(f'payload-bit-mutation-{i}', bytes(claim), base_functions)
    for i in range(128):
        claim = rng.randbytes(rng.randrange(96))
        add(f'random-record-{i}', claim, base_functions)
    return result


class PassiveRecords(unittest.TestCase):
    def test_unchanged_producer_par_and_flow(self):
        from tests.native_process import native_link_flags
        producers = [(ROOT / 'bin/nano_virt', [])]
        if os.environ.get('NANOLANG_TRANSLATOR_SOURCE_COMPILER'):
            producers.append((os.environ['NANOLANG_TRANSLATOR_SOURCE_COMPILER'], ['--emit-nvm']))
        with tempfile.TemporaryDirectory(prefix='nano-passive-originals-') as name:
            work = Path(name)
            paths = []
            for producer_index, (producer, flags) in enumerate(producers):
                for kind, expected in [('par', '13\n'), ('flow', '15\n')]:
                    stem = work / f'{kind}-{producer_index}'
                    module, source, binary = (stem.with_suffix(suffix) for suffix in ('.nvm', '.c', '.native'))
                    checked([producer, ROOT / f'tests/nanoisa/fixtures/passive_{kind}.nano', *flags, '-o', module], build=True)
                    self.assertEqual(checked([ROOT / 'obj/nvm2c_module_reference', module]).strip(), '0')
                    checked([ROOT / 'bin/nano_vm', '--verify-only', module])
                    self.assertEqual(checked([ROOT / 'bin/nano_vm', module]), expected)
                    checked([ROOT / 'bin/nvm2c', module, '-o', source], build=True)
                    checked([*shlex.split(os.environ.get('CC', 'cc')), '-std=c11', '-O1', '-g',
                             '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                             source, ROOT / 'bin/nano_aot_runtime.o', '-lm',
                             *(['-ldl', '-Wl,--export-dynamic'] if sys.platform.startswith('linux') else []),
                             *native_link_flags(), '-o', binary], build=True)
                    self.assertEqual(checked([binary]), expected)
                    path = stem.with_suffix('.bin')
                    path.write_bytes(from_module(module.read_bytes()))
                    paths.append(path)
            self.assertEqual(checked([ROOT / 'obj/nvm2c_passive_reference', *paths]).splitlines(), ['1'] * len(paths))
            for route, command in build_routes(ROOT / 'tests/nvm2c_passive_driver.nano', work):
                self.assertEqual(checked([*command, *paths]).splitlines(), ['1'] * len(paths), route)
            print(f'I retained {len(paths)} unchanged par/flow products through verified VM/native execution and five-route passive checks when both producers are selected.')

    def test_complete_passive_records(self):
        oracle = ROOT / 'obj/nvm2c_passive_reference'
        widths = list(map(int, checked([ROOT / 'obj/nvm2c_module_reference', '--widths']).splitlines()))
        corpus = cases(widths)
        with tempfile.TemporaryDirectory(prefix='nano-passive-record-') as name:
            work = Path(name)
            paths = []
            for i, (_, data, _) in enumerate(corpus):
                path = work / f'case-{i}.bin'
                path.write_bytes(data)
                paths.append(path)

            def execute(command):
                values = []
                for i in range(0, len(paths), 64):
                    values += checked([*command, *paths[i:i + 64]], timeout=600).splitlines()
                self.assertEqual(len(values), len(corpus))
                return values

            reference = execute([oracle])
            for (name, _, expected), actual in zip(corpus, reference):
                if expected is not None:
                    with self.subTest(reference=name):
                        self.assertEqual(actual, str(int(expected)))
            routes = build_routes(ROOT / 'tests/nvm2c_passive_driver.nano', work)
            for route, command in routes:
                actual = execute(command)
                for (name, _, _), expected, value in zip(corpus, reference, actual):
                    with self.subTest(route=route, case=name):
                        self.assertEqual(value, expected)
            print(f'I matched {len(corpus)} complete passive-record decisions across {len(routes)} routes; '
                  f'{sum(expected is not None for _, _, expected in corpus)} have explicit expectations.')


if __name__ == '__main__':
    unittest.main()
