"""I compare closed-call analysis with the actual existing C implementation."""
from collections import deque
from pathlib import Path
import random
import struct
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked

ROOT = Path(__file__).resolve().parents[1]


def i64(value=1):
    return b'\x01' + struct.pack('<q', value)


def local(op, index):
    return bytes([op]) + struct.pack('<H', index)


def call(index, tail=False):
    return bytes([0x33 if tail else 0x3b]) + struct.pack('<I', index)


def image(functions, root=0):
    """I retain explicit descriptor shapes independently of inferred code."""
    records, code = [], b''
    for body, arity, locals_, upvalues, results in functions:
        records.append(struct.pack('<6I', len(code), len(body), arity, locals_, upvalues, results))
        code += body
    return struct.pack('<II', root, len(functions)) + b''.join(records) + code


class Assembly:
    def __init__(self):
        self.parts = []
        self.labels = {}
        self.size = 0

    def label(self, name):
        assert name not in self.labels
        self.labels[name] = self.size

    def emit(self, code):
        self.parts.append((self.size, code, None))
        self.size += len(code)

    def branch(self, op, label):
        self.parts.append((self.size, bytes([op]), label))
        self.size += 5

    def code(self):
        return b''.join(code if label is None else code + struct.pack('<i', self.labels[label] - pc)
                        for pc, code, label in self.parts)


def join(initialize_both):
    a = Assembly()
    a.emit(b'\x03\x01')
    a.branch(0x39, 'right')
    a.emit(i64() + local(17, 0))
    a.branch(0x38, 'merge')
    a.label('right')
    if initialize_both:
        a.emit(i64(2) + local(17, 0))
    else:
        a.emit(b'\0')
    a.label('merge')
    a.emit(local(16, 0) + b'\x3d')
    return a.code()


def loop(initial):
    a = Assembly()
    if initial:
        a.emit(i64() + local(17, 0))
    a.label('head')
    a.emit(b'\x03\x01')
    a.branch(0x39, 'exit')
    a.emit(i64() + local(17, 0))
    a.branch(0x38, 'head')
    a.label('exit')
    a.emit(local(16, 0) + b'\x3d')
    return a.code()


def worklist_stress(length, inputs):
    """I inject distinct missing-local sets around a cycle with a return edge."""
    a = Assembly()
    for index in range(inputs - 1):
        a.emit(b'\x03\x01')
        a.branch(0x39, f'init-{index}')
    a.branch(0x38, f'init-{inputs - 1}')
    for index in range(inputs):
        a.label(f'init-{index}')
        for value in range(inputs):
            if value != index:
                a.emit(b'\x03\x01')
                a.emit(local(17, value))
        a.branch(0x38, f'ring-{index * length // inputs}')
    for index in range(length):
        a.label(f'ring-{index}')
        a.emit(b'\0')
    a.emit(b'\x03\x01')
    a.branch(0x39, 'return')
    a.branch(0x38, 'ring-0')
    a.label('return')
    a.emit(b'\x3d')
    return a.code()


def stress_steps(code, limit=1048577):
    """I independently count set-intersection propagation for this generated CFG."""
    lengths = {0: 1, 3: 2, 17: 3, 56: 5, 57: 5, 61: 1}
    pcs, pc = {}, 0
    while pc < len(code):
        pcs[pc] = code[pc]
        pc += lengths[code[pc]]
    known, queue, queued = {0: 0}, deque([0]), {0}
    steps = 0
    while queue:
        steps += 1
        if steps >= limit:
            return steps
        pc = queue.popleft()
        queued.remove(pc)
        op, value = pcs[pc], known[pc]
        if op == 17:
            value |= 1 << struct.unpack_from('<H', code, pc + 1)[0]
        if op == 61:
            continue
        successors = []
        if op in (56, 57):
            successors.append(pc + struct.unpack_from('<i', code, pc + 1)[0])
        if op != 56:
            successors.append(pc + lengths[op])
        for destination in successors:
            assert destination in pcs
            updated = value if destination not in known else value & known[destination]
            if destination not in known or updated != known[destination]:
                known[destination] = updated
                if destination not in queued:
                    queue.append(destination)
                    queued.add(destination)
    return steps


def cases(widths):
    result = []

    def one(name, body, expected=None, arity=0, locals_=0, upvalues=0, results=0):
        result.append((name, image([(body, arity, locals_, upvalues, results)]), expected))

    for size in range(8):
        result.append((f'short-descriptor-header-{size}', b'\0' * size, False))
    result.append(('empty-function-table', image([]), False))
    result.append(('missing-root', image([(b'\x3d', 0, 0, 0, 0)], root=1), False))
    result.append(('oversized-function-count', struct.pack('<II', 0, 2**32-1), False))
    for field in (0, 1, 2, 3, 4, 5):
        record = [0, 1, 0, 0, 0, 0]
        record[field] = 2**32-1
        result.append((f'descriptor-overflow-{field}', struct.pack('<II6I', 0, 1, *record) + b'\x3d', False))
    one('void-return', b'\x3d', True)
    one('scalar-return', i64() + b'\x3d', True, results=1)
    one('missing-result', b'\x3d', False, results=1)
    one('extra-stack', i64() + b'\x3d', False)
    one('parameter', local(16, 0) + b'\x3d', True, arity=1, locals_=1, results=1)
    one('uninitialized', local(16, 0) + b'\x3d', False, locals_=1, results=1)
    one('stored-local', i64() + local(17, 0) + local(16, 0) + b'\x3d', True, locals_=1, results=1)
    one('out-of-range-local', local(16, 1) + b'\x3d', False, arity=1, locals_=1, results=1)
    one('capture', b'\x3d', False, upvalues=1)
    one('arity-over-locals', b'\x3d', False, arity=1)
    one('maximum-arity', b'\x3d', True, arity=65535, locals_=65535)
    one('multiple-results', b'\x3d', False, results=2)
    one('zero-code', b'', False)
    one('falloff', b'\0', False)
    one('jump-self-no-return', b'\x38\0\0\0\0\x3d', False)
    one('jump-out-of-range', b'\x38\xff\xff\xff\x7f\x3d', False)
    one('jump-into-operand', b'\x38\x01\0\0\0\x3d', False)
    one('unreachable-foreign', b'\x3d\x3e\0\0\0\0', True)
    one('unreachable-bad-local', b'\x3d' + local(16, 65535), True)
    one('unreachable-undecodable', b'\x3d\xff', False)
    one('join-both-initialized', join(True), True, locals_=1, results=1)
    one('join-loses-initialization', join(False), False, locals_=1, results=1)
    one('loop-initialized', loop(True), True, locals_=1, results=1)
    one('loop-backedge-cannot-initialize-entry', loop(False), False, locals_=1, results=1)
    mismatch = Assembly()
    mismatch.emit(b'\x03\x01')
    mismatch.branch(0x39, 'merge')
    mismatch.emit(i64())
    mismatch.label('merge')
    mismatch.emit(b'\x3d')
    one('stack-join-disagreement', mismatch.code(), False)
    for index in (0, 31, 32, 63, 64, 127, 65534):
        one(f'local-set-boundary-{index}', i64() + local(17, index) + local(16, index) + b'\x3d', True, locals_=index + 1, results=1)
        one(f'local-unset-boundary-{index}', local(16, index) + b'\x3d', False, locals_=index + 1, results=1)
    for length in (63, 64, 65):
        functions = [(call(index + 1) + b'\x3d', 0, 0, 0, 0) for index in range(length - 1)]
        functions.append((b'\x3d', 0, 0, 0, 0))
        result.append((f'call-depth-{length}', image(functions), length <= 64))
    for tail in (False, True):
        suffix = b'' if tail else b'\x3d'
        result.append((f'closed-call-{tail}', image([(call(1, tail) + suffix, 0, 0, 0, 1),
                                                   (i64() + b'\x3d', 0, 0, 0, 1)]), True))
        one(f'self-recursion-{tail}', call(0, tail) + suffix, False)
        one(f'missing-callee-{tail}', call(1, tail) + suffix, False)
        result.append((f'mutual-recursion-{tail}', image([(call(1, tail) + suffix, 0, 0, 0, 0),
                                                        (call(0, tail) + suffix, 0, 0, 0, 0)]), False))
    result.append(('tail-result-disagreement', image([(call(1, True), 0, 0, 0, 0), (i64() + b'\x3d', 0, 0, 0, 1)]), False))
    result.append(('arity-consumption', image([(i64() + call(1) + b'\x3d', 0, 0, 0, 1),
                                               (local(16, 0) + b'\x3d', 1, 1, 0, 1)]), True))
    result.append(('callee-uninitialized', image([(call(1) + b'\x3d', 0, 0, 0, 1),
                                                 (local(16, 0) + b'\x3d', 0, 1, 0, 1)]), False))
    result.append(('repeated-checked-callee', image([(call(1) + call(1) + b'\x3d', 0, 0, 0, 0),
                                                   (b'\x3d', 0, 0, 0, 0)]), True))
    result.append(('missing-actual-argument', image([(call(1) + b'\x3d', 0, 0, 0, 1),
                                                    (local(16, 0) + b'\x3d', 1, 1, 0, 1)]), False))
    cached = [(call(64) + call(1) + b'\x3d', 0, 0, 0, 0)]
    cached += [(call(index + 1) + b'\x3d', 0, 0, 0, 0) for index in range(1, 64)]
    cached += [(b'\x3d', 0, 0, 0, 0)]
    result.append(('depth-before-checked-cache', image(cached), False))
    for count in (65536, 65537):
        one(f'instruction-limit-{count}', b'\0' * (count - 1) + b'\x3d', count <= 65536)
    for count in (2048, 2049):
        one(f'storage-limit-{count}', b'\0' * (count - 1) + b'\x3d', count <= 2048, locals_=65535)
    for length, inputs, expected in [(512, 8, True), (40000, 32, False)]:
        code = worklist_stress(length, inputs)
        steps = stress_steps(code)
        assert (steps <= 1048576) == expected, (length, inputs, steps)
        one(f'worklist-limit-{length}-{inputs}-steps-{steps}', code, expected, locals_=inputs)
    for op, width in enumerate(widths):
        if width:
            for stack in range(5):
                one(f'op-{op}-stack-{stack}', i64() * stack + bytes([op]) + b'\0' * (width - 1) + b'\x3d')
            for cut in range(1, width):
                one(f'truncated-{op}-{cut}', bytes([op]) + b'\0' * (cut - 1), False)
        else:
            one(f'unknown-{op}', bytes([op]), False)
    rng = random.Random(1142)
    for index in range(128):
        code = rng.randbytes(rng.randrange(32))
        one(f'random-{index}', code, None, arity=1, locals_=4, results=index % 2)
    return result


class PassiveCalls(unittest.TestCase):
    def test_closed_scalar_calls_and_budgets(self):
        reference = ROOT / 'obj/nvm2c_passive_calls_reference'
        widths = list(map(int, checked([ROOT / 'obj/nvm2c_module_reference', '--widths']).splitlines()))
        corpus = cases(widths)
        with tempfile.TemporaryDirectory(prefix='nano-passive-calls-') as name:
            work = Path(name)
            paths = []
            for index, (_, data, _) in enumerate(corpus):
                path = work / f'case-{index}.bin'
                path.write_bytes(data)
                paths.append(path)

            def execute(command):
                values = []
                for start in range(0, len(paths), 64):
                    values.extend(checked([*command, *paths[start:start + 64]], timeout=600).splitlines())
                self.assertEqual(len(values), len(corpus))
                return values

            expected = execute([reference])
            for (case, _, decision), actual in zip(corpus, expected):
                if decision is not None:
                    with self.subTest(reference=case):
                        self.assertEqual(actual, str(int(decision)))
            allowed = checked([reference, '--allowed']).splitlines()
            self.assertEqual(len(allowed), 512)
            routes = build_routes(ROOT / 'tests/nvm2c_passive_calls_driver.nano', work)
            for route, command in routes:
                self.assertEqual(checked([*command, '--allowed']).splitlines(), allowed, route)
                actual = execute(command)
                for (case, _, _), want, got in zip(corpus, expected, actual):
                    with self.subTest(route=route, case=case):
                        self.assertEqual(got, want)
            print(f'I matched {len(corpus)} closed-call cases and512 opcode-profile decisions '
                  f'across {len(routes)} routes, including actual depth/instruction/storage/worklist limits.')


if __name__ == '__main__':
    unittest.main()
