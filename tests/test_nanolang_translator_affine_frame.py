"""I compare CFG frame primitives with actual C, before function admission."""
from collections import deque
from itertools import product
from pathlib import Path
import hashlib
import json
import os
import random
import struct
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked
from tests.test_nanolang_translator_affine_graph import model
from tests.test_nanolang_translator_affine_state import array_model, union_model
from tests.test_nanolang_translator_ownership import desc, record, function, image

ROOT = Path(__file__).resolve().parents[1]
NO = 0xffffffff
UNKNOWN = 65535


def words(*values):
    return struct.pack('<'+'I'*len(values), *values)


def value(tag=1, *, observation=False, root=UNKNOWN, owned=False, layout=NO,
          variant=UNKNOWN, excluded=()):
    return (tag, int(observation), root, int(owned), layout, variant,
            bytes(int(index in excluded) for index in range(256)))


def frame(stack=(), globals_=(), handlers=(), floor=0, activation=0, commands=()):
    return (words(floor, activation, len(globals_), len(handlers), len(stack), len(commands))
            + b''.join(words(*command) for command in commands)
            + words(*globals_) + b''.join(words(*h) for h in handlers)
            + b''.join(words(*item[:6])+item[6] for item in stack))


def program(first, second=None, *, operation=1, argument=0, carrier=None):
    if carrier is None:
        carrier = model(locals_=[[desc(1), desc(1)]])
    return (words(2, operation, argument, len(carrier))+carrier+first
            + (frame() if second is None else second))


def flow_cases():
    """I derive expected relations with sets, independently of packed C masks."""
    relation = lambda bits: {(a, b) for a in range(2) for b in range(2)
                             if bits & (1 << (a*2+b))}
    encode = lambda pairs: sum(1 << (a*2+b) for a, b in pairs)
    def normalize(allowed, normal, returned):
        inputs = {a for a in range(2) if allowed & (1 << a)}
        return (inputs, {p for p in relation(normal) if p[0] in inputs},
                {p for p in relation(returned) if p[0] in inputs})
    def require(before, permitted):
        return {a for a in range(2) if all(b in permitted for x, b in before if x == a)}
    def compose(before, after):
        return {(a, c) for a, b in before for other, c in after if other == b}
    def report(flow):
        inputs, normal, returned = flow
        return f'{sum(1 << a for a in inputs)},{encode(normal)},{encode(returned)},;'
    cases = []
    for a, b in product(range(16), range(4)):
        permitted = {i for i in range(2) if b & (1 << i)}
        expected = sum(1 << i for i in require(relation(a), permitted))
        cases.append(((0, a, b, 0, 0, 0, 0), f'{expected},;'))
    for a, b in product(range(16), repeat=2):
        cases.append(((1, a, b, 0, 0, 0, 0), f'{encode(compose(relation(a), relation(b)))},;'))
    valid = set()
    for a, b, c in product(range(4), range(16), range(16)):
        item = normalize(a, b, c)
        cases.append(((2, a, b, c, 0, 0, 0), report(item)))
        valid.add((sum(1 << x for x in item[0]), encode(item[1]), encode(item[2])))
    for first, second in product(sorted(valid), repeat=2):
        a, normal_a, return_a = normalize(*first)
        b, normal_b, return_b = normalize(*second)
        permitted = a & require(normal_a, b)
        normal = compose(normal_a, normal_b)
        returned = return_a | compose(normal_a, return_b)
        sequential = (permitted, {p for p in normal if p[0] in permitted},
                      {p for p in returned if p[0] in permitted})
        joined = (a & b, {p for p in normal_a | normal_b if p[0] in a & b},
                  {p for p in return_a | return_b if p[0] in a & b})
        cases.append(((3, *first, *second), report(sequential)))
        cases.append(((4, *first, *second), report(joined)))
    # I keep individual runtime batches bounded; every normalized pair remains.
    return [(f'global-algebra-{offset:06d}', words(0, len(batch))+b''.join(words(*c) for c, _ in batch),
             ''.join(expected for _, expected in batch))
            for offset in range(0, len(cases), 512)
            for batch in [cases[offset:offset+512]]], len(cases), len(valid)


def queue_cases():
    rng = random.Random(1142002)
    cases = []
    for capacity in (1, 2, 3, 7, 64, 256, 4096):
        operations = [(1, 0)]+[(0, i) for i in range(capacity)]
        operations += [(0, 0), (0, capacity-1)]
        operations += [(1, 0)]*capacity+[(1, 0)]
        operations += [(rng.randrange(2), rng.randrange(capacity)) for _ in range(80)]
        pending = deque(); queued = set(); head = tail = 0; expected = ''
        for op, index in operations:
            if op == 0:
                ok, result = True, index
                if index not in queued:
                    pending.append(index); queued.add(index); tail = (tail+1) % capacity
            else:
                ok = bool(pending); result = pending.popleft() if ok else -1
                if ok:
                    queued.remove(result); head = (head+1) % capacity
            expected += f'{int(ok)},{result},{head},{tail},{len(pending)}:'
            expected += (f'{pending[0]},' if pending else '')+';'
        cases.append((f'bounded-fifo-{capacity}', words(1, capacity, len(operations))
                      + b''.join(words(*op) for op in operations), expected))
    return cases


def frame_cases():
    cases = []
    def add(name, a, b=None, *, valid=True, changed=False, message='', operation=1, argument=0, carrier=None):
        prefix = f'{int(valid)}{int(changed)}|{message}|'
        cases.append((name, program(a, b, operation=operation, argument=argument, carrier=carrier), prefix))
    stack_error = 'I require exact ownership and stack provenance at every join'
    handler_error = 'I require identical handler context at every control-flow join'
    add('empty-identical', frame())
    add('scalar-identical', frame([value()]), frame([value()]))
    add('scalar-unused-layout', frame([value()]), frame([value(layout=12)]))
    for item in (value(2), value(owned=True), value(observation=True, root=0)):
        add(f'stack-incompatible-{item[:6]}', frame([value()]), frame([item]), valid=False, message=stack_error)
    add('different-depth', frame([value()]), valid=False, message=stack_error)
    add('observation-same-root', frame([value(8, observation=True, root=0, layout=0)]),
        frame([value(8, observation=True, root=0, layout=0)]))
    add('observation-distinct-root', frame([value(8, observation=True, root=0, layout=0)]),
        frame([value(8, observation=True, root=1, layout=0)]), valid=False, message=stack_error)
    for tag in (8, 10):
        add(f'nominal-layout-{tag}', frame([value(tag, layout=0)]), frame([value(tag, layout=1)]), valid=False, message=stack_error)
    for a, b in product((0, 1, 63, 64, 127, 128, 191, 192, 255, UNKNOWN), repeat=2):
        add(f'union-arm-{a}-{b}', frame([value(10, layout=1, variant=a)]),
            frame([value(10, layout=1, variant=b)]), changed=(a != UNKNOWN and a != b), carrier=union_model(False))
    for bit in range(256):
        add(f'exclusion-{bit}', frame([value(10, layout=1, excluded=range(256))]),
            frame([value(10, layout=1, excluded=(bit,))]), changed=True, carrier=union_model(False))
    add('union-root-weaken', frame([value(10, layout=1, root=0)]),
        frame([value(10, layout=1, root=1)]), changed=True, carrier=union_model(False))
    add('union-root-already-unknown', frame([value(10, layout=1)]),
        frame([value(10, layout=1, root=0)]), carrier=union_model(False))
    add('failure-after-union-mutation', frame([value(10, variant=0), value()]),
        frame([value(10, variant=1), value(2)]), valid=False, message=stack_error)
    for a, b in product(range(16), repeat=2):
        add(f'global-join-{a}-{b}', frame(globals_=[a]), frame(globals_=[b]), changed=(a | b) != a)
    add('global-domain-refusal', frame(globals_=[9]), valid=False,
        message='I require one global declaration domain at every join')
    add('global-domain-after-local-weakening', frame(globals_=[9], commands=[(0, 0, 0, 0)]),
        valid=False, message='I require one global declaration domain at every join')
    add('handler-key', frame(handlers=[(1, 0)]), frame(handlers=[(2, 0)]), valid=False, message=handler_error)
    add('handler-site', frame(handlers=[(1, 0)]), frame(handlers=[(1, 1)]), valid=False, message=handler_error)
    add('handler-floor', frame(handlers=[(1, 0)], floor=1), frame(handlers=[(1, 0)]), valid=False, message=handler_error)
    add('handler-activation', frame(activation=1), valid=False, message=handler_error)
    add('handler-order', frame(handlers=[(1, 0), (2, 1)]), frame(handlers=[(2, 1), (1, 0)]), valid=False, message=handler_error)
    add('handler-before-stack', frame([value()], handlers=[(1, 0)]), frame([value(2)]), valid=False, message=handler_error)
    add('handler-bound', frame(handlers=[(i, i+1) for i in range(64)], floor=64),
        frame(handlers=[(i, i+1) for i in range(64)], floor=64))
    add('global-bound', frame(globals_=[9]*256), frame(globals_=[10]*256), changed=True)
    for initialized_a, initialized_b in product((False, True), repeat=2):
        add(f'local-initialization-{initialized_a}-{initialized_b}',
            frame(commands=[(0, 0, 0, 0)] if initialized_a else []),
            frame(commands=[(0, 0, 0, 0)] if initialized_b else []), changed=initialized_a and not initialized_b)
    owner = image(record(1, flags=[3], functions=[(2, 1, desc(), [desc(8, 0), desc(1)])]),
                  rows=[(0, NO, [(1, NO, NO)])], functions=[function(2, 1, params=(8,))])
    add('owner-exact', frame(), frame(), carrier=owner)
    add('owner-obligation-mismatch', frame(), frame(commands=[(5, 0, 0, 0)]), valid=False, message=stack_error, carrier=owner)
    held = [(6, 0, 0, 0), (8, 1, 0, 2)]
    add('reference-authority-exact', frame(commands=held), frame(commands=held), carrier=owner)
    add('reference-authority-mismatch', frame(commands=held), frame(), valid=False, message=stack_error, carrier=owner)
    for a, b, owned in ((0, 4, False), (2, 5, True), (0, 2, False)):
        add(f'array-exact-leaf-{a}-{b}', frame([value(7, layout=a, owned=owned)]),
            frame([value(7, layout=b, owned=owned)]), valid=b != 2,
            message='' if b != 2 else stack_error, carrier=array_model())
    for selected, wanted in product((0, 1, 255, UNKNOWN), (0, 1, 255)):
        add(f'refine-{selected}-{wanted}', frame([value(10, layout=1, variant=selected, excluded=(2, 3))]),
            operation=2, argument=wanted, valid=selected in (wanted, UNKNOWN), carrier=union_model(False))
    add('refine-observation-refusal', frame([value(10, observation=True)]), operation=2, valid=False)
    add('refine-empty-refusal', frame(), operation=2, valid=False)
    for root in (0, 1, 65535):
        add(f'observation-query-{root}', frame([value(8, observation=True, root=1), value(8, owned=True, root=0)]),
            operation=4, argument=root, valid=root == 1)
    add('clone-nested-provenance', frame([value(10, variant=UNKNOWN, excluded=(0, 63, 64, 127, 128, 191, 192, 255))],
        globals_=[1, 2, 3, 15], handlers=[(7, 2)], commands=[(0, 0, 0, 0)]), operation=3)
    return cases


class AffineFrame(unittest.TestCase):
    def test_complete_cfg_foundation_against_actual_c(self):
        work = Path(tempfile.mkdtemp(prefix='nano-affine-frame-'))
        flows, flow_count, flow_states = flow_cases()
        queues, frames = queue_cases(), frame_cases()
        cases = flows+queues+frames
        paths = []
        record_ = dict(scope='I check internal frame joins, FIFO and relation primitives only. '
                       'I grant no instruction/function/private/executable admission.',
                       flow_operations=flow_count, normalized_flow_states=flow_states,
                       queue_traces=len(queues), frame_cases=len(frames), corpus=[], routes=[])
        for index, (name, data, expected) in enumerate(cases):
            path = work/f'{index:04d}.bin'; path.write_bytes(data); paths.append(path)
            record_['corpus'].append(dict(name=name, path=str(path), bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(), expected=expected))
        def save(): (work/'manifest.json').write_text(json.dumps(record_, indent=2)+'\n')
        save()
        def execute(command, label):
            reports = []
            for start in range(0, len(paths), 8):
                reports += checked([*command, *paths[start:start+8]], timeout=600).splitlines()
            self.assertEqual(len(reports), len(cases), label)
            (work/(label+'.reports')).write_text('\n'.join(reports)+'\n')
            for (name, _, expected), actual in zip(cases, reports):
                with self.subTest(route=label, case=name):
                    if name.startswith(('global-algebra-', 'bounded-fifo-')):
                        self.assertEqual(actual, expected)
                    else:
                        self.assertTrue(actual.startswith(expected), (expected, actual[:200]))
            record_['routes'].append(dict(label=label, command=[str(x) for x in command],
                reports=len(reports), sha256=hashlib.sha256(('\n'.join(reports)+'\n').encode()).hexdigest()))
            save()
            return reports
        reference = Path(os.environ.get('NANOLANG_AFFINE_FRAME_REFERENCE', ROOT/'obj/nvm2c_affine_frame_reference'))
        expected = execute([reference], 'actual-c')
        for label, command in build_routes(ROOT/'tests/nvm2c_affine_frame_driver.nano', work):
            self.assertEqual(execute(command, label), expected, label)


class AffineFlow(unittest.TestCase):
    def test_all_relation_and_worklist_routes(self):
        work = Path(tempfile.mkdtemp(prefix='nano-affine-flow-'))
        flows, operation_count, states = flow_cases()
        cases = flows+queue_cases()
        record_ = dict(scope='I qualify the relation/FIFO prerequisites; the full frame/AOT blocker remains #1202.',
                       flow_operations=operation_count, normalized_flow_states=states, corpus=[], routes=[])
        paths = []
        for index, (name, data, expected) in enumerate(cases):
            path = work/f'{index:04d}.bin'; path.write_bytes(data); paths.append(path)
            record_['corpus'].append(dict(name=name, path=str(path), bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(), expected=expected))
        def save(): (work/'manifest.json').write_text(json.dumps(record_, indent=2)+'\n')
        save()
        reference = Path(os.environ.get('NANOLANG_AFFINE_FRAME_REFERENCE', ROOT/'obj/nvm2c_affine_frame_reference'))
        routes = [('actual-c', [reference])]+build_routes(ROOT/'tests/nvm2c_affine_flow_driver.nano', work)
        for label, command in routes:
            reports = []
            for start in range(0, len(paths), 8):
                reports += checked([*command, *paths[start:start+8]], timeout=600).splitlines()
            (work/(label+'.reports')).write_text('\n'.join(reports)+'\n')
            self.assertEqual(len(reports), len(cases), label)
            for (name, _, expected), actual in zip(cases, reports):
                with self.subTest(route=label, case=name): self.assertEqual(actual, expected)
            record_['routes'].append(dict(label=label, command=[str(x) for x in command],
                reports=len(reports), sha256=hashlib.sha256(('\n'.join(reports)+'\n').encode()).hexdigest()))
            save()
