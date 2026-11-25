"""I compare all nine affine array steps with unchanged C before CFG admission."""
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
from tests.test_nanolang_translator_affine_state import array_model
from tests.test_nanolang_translator_ownership import desc, record, function, image, extension, effect_data

ROOT = Path(__file__).resolve().parents[1]
NO = 0xffffffff
TYPES = [(1, NO), (7, 0), (8, 0), (7, 2), (1, NO), (8, 0)]
RETAINED = 'I require retained element types for array flow'
OPERANDS = 'I require every array operation operand'
ARRAY = 'I require an exact array operand'
OBSERVE = 'I cannot mutate or extract through an array owner observation'
EXTRACT = 'I require explicit resource-element extraction and disposal contracts'
INDEX = 'I require an integer array index'
END = 'I require an integer slice end'
ELEMENT = 'I require the exact array element type and ownership'
LENGTH = 'I require a live owner observation for resource-array length'
CONSTRUCTOR = 'I require checked empty-array constructor declarations'
AMBIGUOUS = 'I require an unambiguous retained element type for empty array construction'
DECLARED = 'I require the exact declared empty-array element tag'
LITERAL = 'I require an exact retained array literal element type'
LITERAL_OWNER = 'I require exact array literal element ownership in source order'


def constructor_model(types=TYPES, *, selected=None, revision=3, tag=1, parameter=False):
    code = bytes([80, tag, 0, 61])
    slots = [desc(8, 0), desc(1)] if parameter else [desc(7, i) for i in range(len(types))]
    constructors = [] if selected is None else [(0, 0, selected)]
    own = record(3, flags=[3], functions=[(len(slots), int(parameter), desc(), slots)],
                 extensions=[extension(4, effect_data(revision, types=[desc(t, layout) for t, layout in types],
                                                        constructors=constructors), revision)])
    return image(own, rows=[(0, NO, [(1, NO, NO)])], strings=[b'op'],
                 functions=[function(len(slots), int(parameter), params=(8,) if parameter else (), length=len(code))], code=code)


def array_cases():
    cases = []
    def add(name, opcode, values=(), *, tag=0, count=0, function_=0, offset=0,
            carrier=None, commands=(), error='', result=None, live=None):
        carrier = array_model() if carrier is None else carrier
        values = list(values)
        payload = (words(8, opcode, tag, count, function_, offset, len(carrier))+carrier
                   + frame(values, globals_=[9], handlers=[(5, 7)], floor=1, activation=23, commands=commands))
        expected = dict(valid=not error, message=error,
                        stack=stack_text(values if error or result is None else result), live=live or {})
        cases.append((name, payload, expected));return payload
    def operands(op, receiver=None):
        receiver = value(7, layout=0) if receiver is None else receiver
        return ([receiver, value(1)] if op in (81, 83, 87) else
                [receiver, value(1), value(1)] if op in (84, 86) else [receiver])
    def output(op, layout=0):
        if op == 85:return [value(1)]
        if op in (82, 83):
            tag, slot = TYPES[layout]
            return [value(tag, layout=slot, owned=layout in (2, 3, 5))]
        return [value(7, layout=layout, owned=layout in (2, 3, 5))]
    for op in range(80, 89):
        add(f'no-array-facts-{op}', op, carrier=model(), error=RETAINED)
    for op in range(81, 88):
        required = len(operands(op))
        for count in range(required):add(f'missing-{op}-{count}', op, operands(op)[:count], error=OPERANDS)
        for tag, observed, owned in product(range(256), (False, True), (False, True)):
            receiver = value(tag, observation=observed, owned=owned, root=65535, layout=0)
            error = ARRAY if tag != 7 else (
                LENGTH if op == 85 and (owned or observed) else
                '' if op == 85 else OBSERVE if observed else EXTRACT if owned and op != 81 else '')
            add(f'receiver-{op}-{tag}-{observed}-{owned}', op, operands(op, receiver), error=error, result=output(op))
    for op, tag, observed, owned in product((83, 84, 86, 87), range(256), (False, True), (False, True)):
        values = operands(op);values[1] = value(tag, observation=observed, owned=owned)
        add(f'index-{op}-{tag}-{observed}-{owned}', op, values,
            error='' if tag == 1 and not observed and not owned else INDEX, result=output(op))
    for tag, observed, owned in product(range(256), (False, True), (False, True)):
        values = operands(86);values[2] = value(tag, observation=observed, owned=owned)
        add(f'slice-end-{tag}-{observed}-{owned}', 86, values,
            error='' if tag == 1 and not observed and not owned else END, result=output(86))
    for op, tag, observed, owned in product((81, 84), range(256), (False, True), (False, True)):
        values = operands(op);values[-1] = value(tag, observation=observed, owned=owned, layout=17, variant=3, excluded=(0, 63, 64, 255))
        add(f'element-{op}-{tag}-{observed}-{owned}', op, values,
            error='' if tag == 1 and not observed and not owned else ELEMENT, result=output(op))
    for layout, op in product(range(6), range(81, 88)):
        receiver = value(7, layout=layout, owned=layout in (2, 3, 5))
        values = operands(op, receiver)
        if op in (81, 84):
            tag, slot = TYPES[layout];values[-1] = value(tag, layout=slot, owned=layout in (2, 3, 5))
        error = LENGTH if op == 85 and receiver[3] else EXTRACT if receiver[3] and op != 81 else ''
        add(f'retained-layout-{layout}-{op}', op, values, error=error, result=output(op, layout))
    for op, layout, observed, owned in product(range(81, 88), (6, NO), (False, True), (False, True)):
        add(f'unknown-layout-{op}-{layout}-{observed}-{owned}', op,
            operands(op, value(7, layout=layout, observation=observed, owned=owned)), error=ARRAY)
    for op, revision in product((80, 88), (2, 3)):
        add(f'constructor-revision-{op}-{revision}', op, tag=1, carrier=constructor_model(revision=revision), result=[value(7, layout=4)])
    add('new-ignores-unused-literal-count', 80, tag=1, count=65535,
        carrier=constructor_model(selected=0), result=[value(7, layout=0)])
    for op, supplied in product((81, 84), (0, 1, 4)):
        values = operands(op, value(7, layout=1));values[-1] = value(7, layout=supplied)
        add(f'structural-array-element-{op}-{supplied}', op, values, error='' if supplied in (0, 4) else ELEMENT, result=output(op, 1))
    for live, mode in product((False, True), (0, 1, 2)):
        if not live and mode:continue
        commands = [(5, 0, 0, 0)] if not live else []
        if mode:commands += [(6, 0, 0, 0), (8, 1, 0, mode)]
        add(f'length-root-access-{live}-{mode}', 85, [value(7, observation=True, root=0, layout=0)],
            carrier=constructor_model(parameter=True), commands=commands,
            error='' if live and mode != 2 else LENGTH, result=[value(1)], live={0: live and mode != 2})
    for op, index in product((80, 88), (0, 1, 4)):
        add(f'explicit-empty-{op}-{index}', op, tag=TYPES[index][0], carrier=constructor_model(selected=index, tag=TYPES[index][0]), result=[value(7, layout=index)])
    for op in (80, 88):
        add(f'fallback-equivalent-empty-{op}', op, tag=1, carrier=constructor_model(), result=[value(7, layout=4)])
        add(f'fallback-ambiguous-empty-{op}', op, tag=7, carrier=constructor_model(), error=AMBIGUOUS)
        add(f'explicit-tag-mismatch-{op}', op, tag=5, carrier=constructor_model(selected=0), error=DECLARED)
        for function_, offset in [(1, 0), (NO, 0), (0, 4), (0, NO)]:
            add(f'constructor-context-{op}-{function_}-{offset}', op, tag=1, function_=function_, offset=offset,
                carrier=constructor_model(), error=CONSTRUCTOR)
    for tag in range(256):
        add(f'empty-physical-tag-{tag}', 80, tag=tag, carrier=constructor_model(),
            error='' if tag in (1, 8) else AMBIGUOUS, result=[value(7, layout=4 if tag == 1 else 5, owned=tag == 8)])
    for count in (1, 2, 255, 256):
        add(f'int-literal-{count}', 88, [value(1)]*count, tag=1, count=count, result=[value(7, layout=0)])
    for count in (1, 2, 257, 65535):
        add(f'missing-literal-{count}', 88, tag=1, count=count, error='I require all declared array literal elements')
    for first_tag, observed, owned in product(range(256), (False, True), (False, True)):
        item = value(first_tag, observation=observed, owned=owned)
        error = LITERAL if first_tag != 1 else LITERAL_OWNER if observed or owned else ''
        add(f'literal-first-{first_tag}-{observed}-{owned}', 88, [item], tag=1, count=1,
            error=error, result=[value(7, layout=0)])
    for tag, observed, owned in product(range(256), (False, True), (False, True)):
        item = value(tag, observation=observed, owned=owned, layout=17)
        error = '' if tag == 1 and not observed and not owned else LITERAL_OWNER
        add(f'literal-later-{tag}-{observed}-{owned}', 88, [value(1), item], tag=1, count=2,
            error=error, result=[value(7, layout=0)])
    add('literal-first-layout-matters', 88, [value(1, layout=17), value(1)], tag=1, count=2, error=LITERAL)
    for tag, layout, owned in [(7, 0, False), (7, 4, False), (8, 0, True), (7, 2, True)]:
        expected_layout = 1 if tag == 7 and layout in (0, 4) else 2 if tag == 8 else 3
        add(f'aggregate-literal-{tag}-{layout}-{owned}', 88, [value(tag, layout=layout, owned=owned)]*2,
            tag=tag, count=2, result=[value(7, layout=expected_layout, owned=owned)])
    for op, count in product((80, 88), (0, 1, 255, 256)):
        values = [value(4)]*count
        add(f'empty-constructor-stack-{op}-{count}', op, values, tag=1, carrier=constructor_model(selected=0),
            error='I cannot retain an array construction result' if count == 256 else '', result=values+[value(7, layout=0)])
    add('observation-before-index', 83, [value(7, observation=True, layout=0), value(4)], error=OBSERVE)
    add('extraction-before-index', 83, [value(7, owned=True, layout=0), value(4)], error=EXTRACT)
    add('slice-start-before-end', 86, [value(7, layout=0), value(4), value(4)], error=INDEX)
    add('set-index-before-element', 84, [value(7, layout=0), value(4), value(4)], error=INDEX)
    good = add('valid-malformed-base', 85, [value(7, layout=0)], result=[value(1)])
    malformed = [b'', words(8), words(9), words(8, 79, 0, 0, 0, 0, 0),
                 words(8, 85, 256, 0, 0, 0, 0), words(8, 85, 0, 65536, 0, 0, 0),
                 words(8, 85, 0, 0, 0, 0, 28)+words(*([0]*7)), good+b'\0']
    malformed += [good[:cut] for cut in (1, 3, 4, 8, 12, 16, 20, 24, 27, len(good)-1)]
    cases += [(f'malformed-{i}', raw, 'D') for i, raw in enumerate(malformed)]
    return cases


class AffineArrays(unittest.TestCase):
    def test_original_frames_and_complete_array_transfer(self):
        work = Path(tempfile.mkdtemp(prefix='nano-affine-arrays-'))
        old = flow_cases()[0]+queue_cases()+frame_cases()
        self.assertEqual(len(old), 1001)
        reference = Path(os.environ.get('NANOLANG_AFFINE_ARRAYS_REFERENCE', ROOT/'obj/nvm2c_affine_arrays_reference'))
        manifest = dict(scope='I check all nine array transfers, without profile/CFG/private/executable admission.', cases=[], routes=[])
        prepared = []
        for group, cases in [('original', old), ('arrays', array_cases())]:
            paths = []
            for index, (name, raw, expected) in enumerate(cases):
                path = work/f'{group}-{index:05d}.bin';path.write_bytes(raw);paths.append(path)
                manifest['cases'].append(dict(group=group, name=name, path=str(path), sha256=hashlib.sha256(raw).hexdigest(), expected=expected))
            prepared.append((group, cases, paths))
        def save(): (work/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        save();reports = {}
        products = [('actual-c', [reference])]+routes.build_routes(ROOT/'tests/nvm2c_affine_arrays_driver.nano', work)
        self.assertEqual(len(products), 6 if routes.source_compiler_command() else 4)
        for label, command in products:
            for group, cases, paths in prepared:
                chunks = ([paths[i:i+8] for i in range(0, len(paths), 8)] if group == 'original' else scalar_batches(paths))
                actual = []
                for chunk in chunks:actual += routes.checked([*command, *chunk], timeout=600 if group == 'original' else 60).splitlines()
                self.assertEqual(len(actual), len(cases), (label, group))
                raw = ('\n'.join(actual)+'\n').encode();(work/f'{group}-{label}.reports').write_bytes(raw)
                if group == 'original':self.assertEqual(hashlib.sha256(raw).hexdigest(), 'f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0')
                for (name, _, expected), text in zip(cases, actual):
                    with self.subTest(route=label, group=group, case=name):
                        if isinstance(expected, str):
                            if group == 'original' and not name.startswith(('global-algebra-', 'bounded-fifo-')):self.assertTrue(text.startswith(expected))
                            else:self.assertEqual(text, expected)
                        else:
                            valid, message, state_ = text.split('|', 2);local_, stack, relations, context = state_.split('/')
                            self.assertEqual(valid, str(int(expected['valid'])));self.assertEqual(message, expected['message'])
                            self.assertEqual(stack, expected['stack']);self.assertEqual(relations, '9,');self.assertEqual(context, '1,23,5,7,')
                            for index, live in expected['live'].items():self.assertEqual(local_.strip(',').split(',')[int(index)*4], str(int(live)))
                if group not in reports:reports[group] = actual
                else:self.assertEqual(actual, reports[group], (label, group))
                manifest['routes'].append(dict(label=label, group=group, cases=len(cases), sha256=hashlib.sha256(raw).hexdigest()));save()


if __name__ == '__main__':
    unittest.main()
