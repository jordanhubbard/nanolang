"""I compare scalar transfer with the actual C step, without admitting a CFG."""
from itertools import product
from pathlib import Path
import hashlib
import json
import os
import tempfile
import unittest

from tests import nanolang_translator_routes as routes
from tests.test_nanolang_translator_affine_frame import (
    words, value, frame, flow_cases, queue_cases, frame_cases,
)
from tests.test_nanolang_translator_affine_graph import model
from tests.test_nanolang_translator_ownership import desc

ROOT = Path(__file__).resolve().parents[1]
I64, U8, F64, BOOL, TEXT = 1, 2, 3, 4, 5
LEAF = ({0, 1, 2, 3, 5, 6, 7, 8, 9, 56, 57, 58, 160, 161, 163, 164}
        | set(range(32, 38)) | set(range(40, 46)) | set(range(48, 51))
        | set(range(200, 205)) | set(range(216, 222)))
SCALARS = {I64, U8, F64, BOOL}


def stack_text(values):
    return ''.join(','.join(map(str, item[:6]))+','
                   + ''.join(map(str, item[6]))+';' for item in values)


def scalar_batches(paths, max_count=32, max_bytes=8192):
    """I retain every case and isolate large frames under the same 60s limit."""
    batch = []; size = 0
    for path in paths:
        length = path.stat().st_size
        if batch and (len(batch) == max_count or size+length > max_bytes):
            yield batch
            batch = []; size = 0
        batch.append(path); size += length
    if batch:
        yield batch


def scalar_cases():
    carrier = model(locals_=[[desc(1), desc(1)]])
    cases = [(f'applicability-{op}', words(4, op), str(int(op in LEAF)))
             for op in range(256)]
    def add(name, op, values=(), *, operand=0, graph=False, result=None, error=''):
        values = list(values)
        expected = dict(valid=not error, message=error,
                        stack=stack_text(values if error or result is None else result))
        payload = frame(values, globals_=[9], handlers=[(5, 7)], floor=1, activation=23)
        cases.append((name, words(5, op, operand, int(graph), len(carrier))+carrier+payload, expected))
    sample = lambda tag, observation=0, owned=0: value(
        tag, observation=observation, owned=owned, root=7, layout=11,
        variant=3, excluded=(0, 63, 64, 127, 128, 191, 192, 255))
    flags = list(product((0, 1), repeat=2))
    for op, tag, operand in [(1, I64, 0), (2, F64, 0), (3, BOOL, 0),
                             (3, BOOL, 1), (5, 0, 0), (6, U8, 255)]:
        for count, graph in product((0, 1, 255, 256), (False, True)):
            values = [sample(1)]*count
            error = 'I cannot extend my analysis stack' if count == 256 else ''
            add(f'literal-{op}-{operand}-{count}-{graph}', op, values, operand=operand,
                graph=graph, result=values+[value(tag)] if not error else None, error=error)
    for operand, graph in product(range(2, 256), (False, True)):
        add(f'boolean-byte-{operand}-{graph}', 3, operand=operand, graph=graph,
            error='I require a Boolean literal')
    add('bad-boolean-before-full-stack', 3, [sample(1)]*256, operand=2,
        error='I require a Boolean literal')
    for op, graph in product((7, 8, 9), (False, True)):
        message = {7: 'I refuse to duplicate reference authority',
                   8: 'I require a scalar discard; an observation is not an owned consume',
                   9: 'I require scalar stack permutation'}[op]
        add(f'empty-stack-{op}-{graph}', op, graph=graph, error=message)
    add('swap-one-item', 9, [sample(1)], error='I require scalar stack permutation')
    for tag, (observed, owned), graph in product(range(256), flags, (False, True)):
        item = sample(tag, observed, owned)
        for op in (7, 8):
            error = ''
            if observed or owned:
                error = ('I refuse to duplicate reference authority' if op == 7 else
                         'I require a scalar discard; an observation is not an owned consume')
            elif tag == 5 and not graph:
                error = ('I refuse unsupported string duplication' if op == 7 else
                         'I require PRINT or PRINTLN to consume a string')
            add(f'stack-{op}-{tag}-{observed}-{owned}-{graph}', op, [item], graph=graph,
                result=[item, item] if op == 7 else [], error=error)
        for position in range(2):
            values = [sample(1), sample(1)];values[position] = item
            error = ('I require scalar stack permutation' if observed or owned else
                     'I refuse unsupported string permutation' if tag == 5 and not graph else '')
            add(f'swap-{tag}-{observed}-{owned}-{position}-{graph}', 9, values, graph=graph,
                result=list(reversed(values)), error=error)
    for tag, graph in product((0, 1, 5, 8, 255), (False, True)):
        values = [sample(1)]*255+[sample(tag)]
        error = ('I refuse unsupported string duplication' if tag == 5 and not graph else
                 'I cannot extend my analysis stack')
        add(f'full-dup-{tag}-{graph}', 7, values, graph=graph, error=error)
    for op, count in product((0, 56, 163), (0, 1, 256)):
        add(f'unchanged-{op}-{count}', op, [sample(8, 1, 1)]*count)
    for op, tag, (observed, owned), graph in product((160, 164), range(256), flags, (False, True)):
        error = ('I require string output inside an owned value-call graph' if not graph else
                 '' if tag == 5 and not observed and not owned else
                 'I require one exact immutable string print operand')
        add(f'print-{op}-{tag}-{observed}-{owned}-{graph}', op,
            [sample(tag, observed, owned)], graph=graph, result=[], error=error)
    for op in (57, 58, 161):
        error = ('I require an exact Boolean assertion condition' if op == 161 else
                 'I require an exact Boolean branch condition')
        add(f'condition-empty-{op}', op, error=error)
        for tag, (observed, owned) in product(range(256), flags):
            valid = tag == BOOL and not observed and not owned
            add(f'condition-{op}-{tag}-{observed}-{owned}', op,
                [sample(tag, observed, owned)], result=[], error='' if valid else error)
    families = [((32, 33, 34, 35, 36, 37), I64, I64, 37, 'I require integer arithmetic operands'),
                ((48, 49, 50), BOOL, BOOL, 50, 'I require Boolean operands'),
                ((200, 201, 202, 203, 204), F64, F64, 204, 'I require float arithmetic operands'),
                ((216, 217, 218, 219, 220, 221), F64, BOOL, -1, 'I require float comparison operands')]
    for ops, tag, output, unary, error in families:
        for op in ops:
            needed = 1 if op == unary else 2
            for count in range(needed):
                add(f'underflow-{op}-{count}', op, [sample(tag)]*count, error=error)
            combinations = [(item, (0, 0)) for item in flags] if needed == 1 else product(flags, repeat=2)
            for first, second in combinations:
                values = [sample(tag, *first)]
                if needed == 2: values.append(sample(tag, *second))
                valid = not any(first) and (needed == 1 or not any(second))
                add(f'flags-{op}-{first}-{second}', op, values,
                    result=[value(output)], error='' if valid else error)
        for other in range(256):
            add(f'physical-arithmetic-{ops[0]}-{other}', ops[0], [sample(other)]*2,
                result=[value(output)], error='' if other == tag else error)
    for op, graph in product(range(40, 46), (False, True)):
        add(f'comparison-empty-{op}-{graph}', op, graph=graph,
            error='I require scalar comparison operands')
        for tag in range(256):
            string = graph and op in (40, 41) and tag == 5
            valid = tag in SCALARS or string
            error = '' if valid else 'I require scalar comparison operands'
            add(f'comparison-{op}-{tag}-{graph}', op, [sample(tag)]*2,
                graph=graph, result=[value(BOOL)], error=error)
            add(f'comparison-one-{op}-{tag}-{graph}', op, [sample(tag)], graph=graph,
                error=('I require exact STRING equality operands' if string else
                       'I require matching comparison tags' if tag in SCALARS else
                       'I require scalar comparison operands'))
        for top_tag, lower_tag, top_flags, lower_flags in product((1, 2, 3, 4, 5), (1, 5), flags, flags):
            string = graph and op in (40, 41) and top_tag == 5
            top_valid = top_tag in SCALARS and not any(top_flags)
            valid = (string or top_valid) and top_tag == lower_tag and not any(top_flags+lower_flags)
            error = ('' if valid else 'I require exact STRING equality operands' if string else
                     'I require scalar comparison operands' if not top_valid else
                     'I require matching comparison tags')
            add(f'comparison-flags-{op}-{graph}-{top_tag}-{lower_tag}-{top_flags}-{lower_flags}',
                op, [sample(lower_tag, *lower_flags), sample(top_tag, *top_flags)], graph=graph,
                result=[value(BOOL)], error=error)
    valid = words(5, 1, 0, 0, len(carrier))+carrier+frame()
    malformed = [b'', words(6), words(4), words(4, 256), words(4, 0)+b'\0',
                 words(5, 256, 0, 0, 0), words(5, 1, 256, 0, 0),
                 words(5, 1, 0, 2, 0), words(5, 4, 0, 0, 0), valid+b'\0',
                 words(5, 1, 0, 0, 28)+words(*([0]*7)),
                 words(2, 3, 0, 28)+words(*([0]*7))]
    malformed += [valid[:cut] for cut in (1, 3, 4, 8, 12, 16, 19, len(valid)-1)]
    cases += [(f'malformed-{i}', data, 'D') for i, data in enumerate(malformed)]
    return cases


class AffineScalar(unittest.TestCase):
    def test_original_frames_and_complete_scalar_transfer(self):
        work = Path(tempfile.mkdtemp(prefix='nano-affine-scalar-'))
        old = flow_cases()[0]+queue_cases()+frame_cases()
        self.assertEqual(len(old), 1001)
        groups = [('original', old, 8, 600), ('scalar', scalar_cases(), 32, 60)]
        reference = Path(os.environ.get('NANOLANG_AFFINE_SCALAR_REFERENCE', ROOT/'obj/nvm2c_affine_scalar_reference'))
        report = dict(scope='I check internal scalar/frame transfer only; no graph/CFG or executable admission.',
                      groups=[], routes=[])
        expected_reports = {}
        prepared = []
        for group, cases, batch, budget in groups:
            paths = []
            for index, (name, data, expected) in enumerate(cases):
                path = work/f'{group}-{index:05d}.bin';path.write_bytes(data);paths.append(path)
                report['groups'].append(dict(group=group, name=name, path=str(path),
                    sha256=hashlib.sha256(data).hexdigest(), expected=expected))
            prepared.append((group, cases, paths, batch, budget))
        def save(): (work/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
        save()
        products = [('actual-c', [reference])]+routes.build_routes(ROOT/'tests/nvm2c_affine_scalar_driver.nano', work)
        self.assertEqual(len(products), 6 if routes.source_compiler_command() else 4)
        for label, command in products:
            for group, cases, paths, batch, budget in prepared:
                actual = []
                chunks = ([paths[start:start+batch] for start in range(0, len(paths), batch)]
                          if group == 'original' else scalar_batches(paths, batch))
                for chunk in chunks:
                    actual += routes.checked([*command, *chunk], timeout=budget).splitlines()
                self.assertEqual(len(actual), len(cases), (label, group))
                raw = ('\n'.join(actual)+'\n').encode();(work/f'{group}-{label}.reports').write_bytes(raw)
                if group == 'original':
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), 'f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0')
                for (name, _, expected), text in zip(cases, actual):
                    with self.subTest(route=label, group=group, case=name):
                        if isinstance(expected, str):
                            if group == 'original' and name.startswith(('global-algebra-', 'bounded-fifo-')):
                                self.assertEqual(text, expected)
                            elif group == 'original': self.assertTrue(text.startswith(expected), (name, text[:200]))
                            else: self.assertEqual(text, expected)
                        else:
                            flag, message, state_ = text.split('|', 2)
                            self.assertEqual(flag, str(int(expected['valid'])))
                            self.assertEqual(message, expected['message'])
                            local_, stack_, global_, context_ = state_.split('/')
                            self.assertEqual(stack_, expected['stack'])
                            self.assertEqual(global_, '9,')
                            self.assertEqual(context_, '1,23,5,7,')
                if group not in expected_reports: expected_reports[group] = actual
                else: self.assertEqual(actual, expected_reports[group], (label, group))
                report['routes'].append(dict(label=label, group=group, cases=len(cases), sha256=hashlib.sha256(raw).hexdigest()))
                save()


if __name__ == '__main__':
    unittest.main()
