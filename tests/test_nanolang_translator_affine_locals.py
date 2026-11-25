"""I compare exact local transfer with my unchanged C step, before CFG admission."""
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
from tests.test_nanolang_translator_ownership import desc, record, function, image, extension, union_data

ROOT = Path(__file__).resolve().parents[1]
MISSING = 'I refuse an observation escape or missing scalar store'
EXACT = 'I require the exact scalar local type'


def local_cases():
    cases = []
    def add(name, opcode, local=0, values=(), *, carrier=None, commands=(), graph=False,
            error='', result=None, live=None):
        carrier = model(locals_=[[desc(1)]]) if carrier is None else carrier
        values = list(values)
        payload = frame(values, globals_=[9], handlers=[(5, 7)], floor=1, activation=23, commands=commands)
        expected = dict(valid=not error, message=error,
                        stack=stack_text(values if error or result is None else result), live=live or {})
        cases.append((name, words(6, opcode, local, int(graph), len(carrier))+carrier+payload, expected))
    for tag, initialized, graph in product(range(1, 6), (False, True), (False, True)):
        carrier = model(locals_=[[desc(tag)]])
        commands = [(1 if tag == 5 else 0, 0, 0, 0)] if initialized else []
        for count in (0, 1, 255, 256):
            values = [value(4)]*count
            error = ('I require a live checked local' if not initialized else
                     'I cannot extend my analysis stack' if count == 256 else '')
            add(f'load-{tag}-{initialized}-{graph}-{count}', 16, carrier=carrier,
                values=values, commands=commands, graph=graph, error=error,
                result=values+[value(tag, root=0)], live={0: initialized})
    for declared, tag, observed, owned, graph in product(range(1, 6), range(256), (False, True), (False, True), (False, True)):
        carrier = model(locals_=[[desc(declared)]])
        item = value(tag, observation=observed, owned=owned, root=7, layout=0xffffffff,
                     excluded=(0, 63, 64, 127, 128, 191, 192, 255))
        valid = not observed and not owned and tag == declared and (tag in (1, 2, 3, 4) or graph)
        error = MISSING if observed or owned else '' if valid else EXACT
        add(f'store-{declared}-{tag}-{observed}-{owned}-{graph}', 17,
            carrier=carrier, values=[item], graph=graph, error=error, result=[], live={0: valid})
    for opcode, local in product((16, 17), (1, 255, 256, 65535)):
        add(f'local-bound-{opcode}-{local}', opcode, local, [value()],
            error='I require a live checked local' if opcode == 16 else MISSING)
    add('empty-store', 17, error=MISSING)
    # I retain copyable and resource aggregate boundaries, not scalar-only examples.
    for resource, graph in product((False, True), (False, True)):
        # A parameter gives the resource union live authority without a copy-definition shortcut.
        parameter = image(record(3, flags=[3 if resource else 1],
            functions=[(1, 1, desc(), [desc(10, 0)])],
            extensions=[extension(1, union_data(0, [(1, 0, 0)]))]),
            rows=[(2, 0, [])], strings=[b'Choice', b'Empty'],
            functions=[function(1, 1, params=(10,))])
        add(f'union-parameter-load-{resource}-{graph}', 16, carrier=parameter, graph=graph,
            error='I require an explicit move of a resource union' if resource else '',
            result=[value(10, root=0, layout=0)], live={0: True})
        carrier = union_model(resource)
        for arm in (0, 1, 2, 3, 65535):
            item = value(10, layout=1, variant=arm)
            valid = graph and not resource and arm in (0, 1, 2, 65535)
            add(f'union-store-{resource}-{graph}-{arm}', 17, 3, [item],
                carrier=carrier, graph=graph, error='' if valid else EXACT, result=[], live={3: valid})
        carrier = signature_model(parameters=[(8, 0)], flags=3 if resource else 1)
        add(f'record-parameter-load-{resource}-{graph}', 16, carrier=carrier, graph=graph,
            result=[value(8, root=0, layout=0, observation=resource)], live={0: True})
        item = value(8, layout=0)
        add(f'record-store-{resource}-{graph}', 17, values=[item], carrier=carrier, graph=graph,
            error='' if graph and not resource else EXACT, result=[], live={0: True})
    for mode, graph in product((1, 2), (False, True)):
        carrier = signature_model(parameters=[(8, mode)], flags=3)
        add(f'borrowed-record-load-{mode}-{graph}', 16, carrier=carrier, graph=graph,
            result=[value(8, root=0, observation=True)], live={0: True})
        add(f'borrowed-record-store-{mode}-{graph}', 17, values=[value(8, layout=0)],
            carrier=carrier, graph=graph, error=EXACT, live={0: True})
    for graph in (False, True):
        for source, destination in product((0, 1, 4), repeat=2):
            # Array recipes0 and4 are structurally equal INT elements; recipe1 is nested.
            item = value(7, layout=source)
            valid = graph and (source == destination or {source, destination} == {0, 4})
            add(f'array-store-{source}-{destination}-{graph}', 17, destination, [item],
                carrier=array_model(), graph=graph, error='' if valid else EXACT, result=[], live={destination: valid})
        for local in (0, 1, 4):
            add(f'array-load-{local}-{graph}', 16, local, carrier=array_model(),
                commands=[(4, local, local, 0)], graph=graph,
                result=[value(7, root=local, layout=local)], live={local: True})
    good = words(6, 16, 0, 0, len(model()))+model()+frame()
    malformed = [b'', words(7), words(6), words(6, 15, 0, 0, 0),
                 words(6, 16, 65536, 0, 0), words(6, 16, 0, 2, 0),
                 words(6, 16, 0, 0, 28)+words(*([0]*7)), good+b'\0']
    malformed += [good[:cut] for cut in (1, 3, 4, 8, 12, 16, 19, len(good)-1)]
    cases += [(f'malformed-{i}', raw, 'D') for i, raw in enumerate(malformed)]
    return cases


class AffineLocals(unittest.TestCase):
    def test_original_frames_and_exact_local_transfer(self):
        work = Path(tempfile.mkdtemp(prefix='nano-affine-locals-'))
        old = flow_cases()[0]+queue_cases()+frame_cases()
        self.assertEqual(len(old), 1001)
        reference = Path(os.environ.get('NANOLANG_AFFINE_LOCALS_REFERENCE', ROOT/'obj/nvm2c_affine_locals_reference'))
        manifest = dict(scope='I check internal local/frame transfer, without graph/CFG or executable admission.', cases=[], routes=[])
        prepared = []
        for group, cases in [('original', old), ('locals', local_cases())]:
            paths = []
            for index, (name, data, expected) in enumerate(cases):
                path = work/f'{group}-{index:05d}.bin';path.write_bytes(data);paths.append(path)
                manifest['cases'].append(dict(group=group, name=name, path=str(path), sha256=hashlib.sha256(data).hexdigest(), expected=expected))
            prepared.append((group, cases, paths))
        def save(): (work/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        save(); reports = {}
        products = [('actual-c', [reference])]+routes.build_routes(ROOT/'tests/nvm2c_affine_locals_driver.nano', work)
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
                            if group == 'original' and not name.startswith(('global-algebra-', 'bounded-fifo-')): self.assertTrue(text.startswith(expected))
                            else: self.assertEqual(text, expected)
                        else:
                            valid, message, state_ = text.split('|', 2)
                            self.assertEqual(valid, str(int(expected['valid'])));self.assertEqual(message, expected['message'])
                            local_, stack, global_, context = state_.split('/')
                            self.assertEqual(stack, expected['stack']);self.assertEqual(global_, '9,');self.assertEqual(context, '1,23,5,7,')
                            locals_ = local_.strip(',').split(',')
                            for index, live in expected['live'].items(): self.assertEqual(locals_[int(index)*4], str(int(live)))
                if group not in reports: reports[group] = actual
                else: self.assertEqual(actual, reports[group], (label, group))
                manifest['routes'].append(dict(label=label, group=group, cases=len(cases), sha256=hashlib.sha256(raw).hexdigest()));save()


if __name__ == '__main__':
    unittest.main()
