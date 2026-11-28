"""I compare global frame transfer and entry preconditions with the actual C step."""
from itertools import product
from pathlib import Path
import hashlib
import json
import os
import tempfile
import unittest

from tests import nanolang_translator_routes as routes
from tests.test_nanolang_translator_affine_frame import words, value, frame, flow_cases, queue_cases, frame_cases
from tests.test_nanolang_translator_affine_graph import model
from tests.test_nanolang_translator_affine_scalar import scalar_batches, stack_text
from tests.test_nanolang_translator_affine_state import array_model, union_model, signature_model
from tests.test_nanolang_translator_ownership import desc

ROOT = Path(__file__).resolve().parents[1]
NO = 0xffffffff
SLOT = 'I require an exact declared global slot'
OWNER = 'I require explicit global owner transfers and lifetime before resource storage'
TYPE = 'I require the exact global type without an owner or observation escape'
EMPTY = 'I require a value for a global store'
FULL = 'I cannot retain a checked global load'


def preimage(relation, permitted):
    pairs = {(a, b) for a, b in product(range(2), repeat=2) if relation & (1 << (2*a+b))}
    return sum(1 << a for a in range(2) if all(b in permitted for x, b in pairs if x == a))


def initialize(relation):
    before = {(a, b) for a, b in product(range(2), repeat=2) if relation & (1 << (2*a+b))}
    pairs = {(a, 1) for a, _ in before}
    return sum(1 << (2*a+b) for a, b in pairs)


def global_cases():
    cases = []
    def add(name, opcode, values=(), *, index=0, graph=True, effect=False, declarations=None,
            allowed=None, relations=None, carrier=None, error='', result=None,
            after_allowed=None, after_relations=None):
        declarations = [(1, False, NO)] if declarations is None else declarations
        count = len(declarations)
        allowed = [3]*count if allowed is None else list(allowed)
        relations = [9]*count if relations is None else list(relations)
        carrier = model(locals_=[[desc(1), desc(1)]]) if carrier is None else carrier
        values = list(values)
        payload = (words(7, opcode, index, int(graph), int(effect), count)
                   + b''.join(words(tag, int(mutable), layout, mask)
                              for (tag, mutable, layout), mask in zip(declarations, allowed))
                   + words(len(carrier))+carrier
                   + frame(values, globals_=relations, handlers=[(5, 7)], floor=1, activation=23))
        expected = dict(valid=not error, message=error,
                        stack=stack_text(values if error or result is None else result),
                        allowed=allowed if error or after_allowed is None else after_allowed,
                        relations=relations if error or after_relations is None else after_relations)
        cases.append((name, payload, expected))
        return payload
    for op, relation, mask, mutable, effect in product((18, 19), range(16), range(4), (False, True), (False, True)):
        values = [] if op == 18 else [value(1)]
        permit = {1} if op == 18 else {0}
        after_mask = mask if effect or (op == 19 and mutable) else mask & preimage(relation, permit)
        add(f'relation-{op}-{relation}-{mask}-{mutable}-{effect}', op, values,
            declarations=[(1, mutable, NO)], relations=[relation], allowed=[mask], effect=effect,
            result=[value(1)] if op == 18 else [], after_allowed=[after_mask],
            after_relations=[relation if op == 18 else initialize(relation)])
    # Physical inner-step behavior does not admit arbitrary declaration tags publicly.
    for tag, graph, effect in product(range(256), (False, True), (False, True)):
        add(f'load-tag-{tag}-{graph}-{effect}', 18, declarations=[(tag, False, NO)], graph=graph, effect=effect,
            result=[value(tag)], error='' if graph else SLOT, after_allowed=[3 if effect else 2])
    for tag, observed, owned, graph in product(range(256), (False, True), (False, True), (False, True)):
        item = value(tag, observation=observed, owned=owned, root=7, layout=NO,
                     variant=3, excluded=(0, 63, 64, 127, 128, 191, 192, 255))
        error = SLOT if not graph else TYPE if observed or owned or tag != 1 else ''
        add(f'store-tag-{tag}-{observed}-{owned}-{graph}', 19, [item], graph=graph,
            error=error, result=[], after_allowed=[1], after_relations=[10])
    for resource, op, graph in product((False, True), (18, 19), (False, True)):
        for label, tag, layout, carrier in [
            ('record', 8, 0, signature_model(flags=3 if resource else 1)),
            ('union', 10, 1, union_model(resource)),
        ]:
            error = SLOT if not graph else OWNER if resource else ''
            add(f'{label}-{resource}-{op}-{graph}', op, [value(tag, layout=layout)] if op == 19 else [],
                declarations=[(tag, False, layout)], carrier=carrier, graph=graph,
                error=error, result=[value(tag, layout=layout)] if op == 18 else [],
                after_allowed=[2 if op == 18 else 1], after_relations=[9 if op == 18 else 10])
    for layout, op, graph in product(range(6), (18, 19), (False, True)):
        error = SLOT if not graph else OWNER if layout in (2, 3, 5) else ''
        add(f'array-{layout}-{op}-{graph}', op, [value(7, layout=layout)] if op == 19 else [],
            declarations=[(7, False, layout)], carrier=array_model(), graph=graph, error=error,
            result=[value(7, layout=layout)] if op == 18 else [], after_allowed=[2 if op == 18 else 1],
            after_relations=[9 if op == 18 else 10])
    # Global stores compare descriptor identity even when two array recipes are equivalent.
    for source, target in product((0, 1, 4), repeat=2):
        add(f'exact-array-layout-{source}-{target}', 19, [value(7, layout=source)],
            declarations=[(7, True, target)], carrier=array_model(), error='' if source == target else TYPE,
            result=[], after_relations=[10])
    for layout in (0, 1, NO):
        add(f'exact-scalar-layout-{layout}', 19, [value(1, layout=layout)],
            error='' if layout == NO else TYPE, result=[], after_allowed=[1], after_relations=[10])
    for op, index in product((18, 19), (1, 255, 256, NO)):
        add(f'slot-bound-{op}-{index}', op, [value()], index=index, error=SLOT)
    for count, op in product((0, 1, 255, 256), (18, 19)):
        selected = max(0, count-1)
        after_allowed = [3]*count
        after_relations = [9]*count
        if count:
            after_allowed[selected] = 2 if op == 18 else 1
            if op == 19: after_relations[selected] = 10
        add(f'global-count-{count}-{op}', op, [value()] if op == 19 else [], index=selected,
            declarations=[(1, False, NO)]*count, error='' if count else SLOT,
            result=[value()] if op == 18 else [], after_allowed=after_allowed, after_relations=after_relations)
    for count, effect in product((0, 1, 255, 256), (False, True)):
        values = [value(4)]*count
        add(f'load-stack-bound-{count}-{effect}', 18, values, effect=effect,
            error=FULL if count == 256 else '', result=values+[value(1)], after_allowed=[3 if effect else 2])
    add('empty-store', 19, error=EMPTY)
    add('type-before-immutable-precondition', 19, [value(4)], allowed=[3], relations=[8], error=TYPE)
    add('owner-before-empty-store', 19, carrier=signature_model(flags=3), declarations=[(8, False, 0)], error=OWNER)
    add('slot-before-owner', 18, carrier=signature_model(flags=3), declarations=[(8, False, 0)], graph=False, error=SLOT)
    good = add('valid-malformed-base', 18, result=[value(1)], after_allowed=[2])
    basic = model()
    malformed = [b'', words(7), words(8), words(7, 17, 0, 1, 0, 0),
                 words(7, 18, 0, 2, 0, 0), words(7, 18, 0, 1, 2, 0),
                 words(7, 18, 0, 1, 0, 257),
                 words(7, 18, 0, 1, 0, 1, 256, 0, NO, 3),
                 words(7, 18, 0, 1, 0, 1, 1, 2, NO, 3),
                 words(7, 18, 0, 1, 0, 1, 1, 0, NO, 4),
                 words(7, 18, 0, 1, 0, 0, len(basic))+basic+frame(globals_=[9]),
                 words(7, 18, 0, 1, 0, 1, 1, 0, NO, 3, len(basic))+basic+frame(globals_=[16]),
                 words(7, 18, 0, 1, 0, 0, 28)+words(*([0]*7)), good+b'\0']
    malformed += [good[:cut] for cut in (1, 3, 4, 8, 12, 16, 20, 23, len(good)-1)]
    cases += [(f'malformed-{i}', raw, 'D') for i, raw in enumerate(malformed)]
    return cases


class AffineGlobals(unittest.TestCase):
    def test_original_frames_and_exact_global_transfer(self):
        work = Path(tempfile.mkdtemp(prefix='nano-affine-globals-'))
        old = flow_cases()[0]+queue_cases()+frame_cases()
        self.assertEqual(len(old), 1001)
        reference = Path(os.environ.get('NANOLANG_AFFINE_GLOBALS_REFERENCE', ROOT/'obj/nvm2c_affine_globals_reference'))
        manifest = dict(scope='I check global transfer and current-function preconditions, without CFG/call/effect admission.', cases=[], routes=[])
        prepared = []
        for group, cases in [('original', old), ('globals', global_cases())]:
            paths = []
            for index, (name, raw, expected) in enumerate(cases):
                path = work/f'{group}-{index:05d}.bin';path.write_bytes(raw);paths.append(path)
                manifest['cases'].append(dict(group=group, name=name, path=str(path), sha256=hashlib.sha256(raw).hexdigest(), expected=expected))
            prepared.append((group, cases, paths))
        def save(): (work/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        save();reports = {}
        products = [('actual-c', [reference])]+routes.build_routes(ROOT/'tests/nvm2c_affine_globals_driver.nano', work)
        self.assertEqual(len(products), 6 if routes.source_compiler_command() else 4)
        for label, command in products:
            for group, cases, paths in prepared:
                chunks = ([paths[i:i+8] for i in range(0, len(paths), 8)] if group == 'original' else scalar_batches(paths))
                actual = []
                for chunk in chunks: actual += routes.checked([*command, *chunk], timeout=600 if group == 'original' else 60).splitlines()
                self.assertEqual(len(actual), len(cases), (label, group))
                raw = ('\n'.join(actual)+'\n').encode();(work/f'{group}-{label}.reports').write_bytes(raw)
                if group == 'original': self.assertEqual(hashlib.sha256(raw).hexdigest(), 'f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0')
                for (name, _, expected), text in zip(cases, actual):
                    with self.subTest(route=label, group=group, case=name):
                        if isinstance(expected, str):
                            if group == 'original' and not name.startswith(('global-algebra-', 'bounded-fifo-')):self.assertTrue(text.startswith(expected))
                            else:self.assertEqual(text, expected)
                        else:
                            valid, message, state_, allowed = text.split('|', 3)
                            self.assertEqual(valid, str(int(expected['valid'])));self.assertEqual(message, expected['message'])
                            local_, stack, relations, context = state_.split('/')
                            self.assertEqual(stack, expected['stack']);self.assertEqual(context, '1,23,5,7,')
                            self.assertEqual(relations, ''.join(f'{v},' for v in expected['relations']))
                            self.assertEqual(allowed, ''.join(f'{v},' for v in expected['allowed']))
                if group not in reports:reports[group] = actual
                else:self.assertEqual(actual, reports[group], (label, group))
                manifest['routes'].append(dict(label=label, group=group, cases=len(cases), sha256=hashlib.sha256(raw).hexdigest()));save()


if __name__ == '__main__':
    unittest.main()
