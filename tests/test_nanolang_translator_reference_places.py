"""I compare actual reference-place APIs without granting affine execution."""
from pathlib import Path
import hashlib
import itertools
import json
import os
import random
import struct
import tempfile
import unittest
from tests.nanolang_translator_routes import build_routes, checked

ROOT = Path(__file__).resolve().parents[1]
NO = 0xffffffff
LEAF = (0, True, tuple((tag, NO) for tag in (1, 2, 3, 4)))
ROWS = (LEAF, LEAF, (0, True, ((8, 0), (8, 0))), (0, True, ((8, 2),)))


def place(*, invocation=17, local=3, root=3, fields=(0, 0), referent=0,
          mode=1, present=True, fields_present=True):
    return (int(present), invocation & NO, invocation >> 32, local, root,
            int(fields_present), len(fields), referent, mode, *fields)


def carrier(rows=ROWS, left=None, right=None, *, table_present=True,
            items_present=True, authority_left=3, authority_right=3):
    words = [int(table_present), int(items_present), len(rows), authority_left, authority_right]
    for kind, present, fields in rows:
        words.extend((kind, int(present), len(fields)))
        words.extend(value for field in fields for value in field)
    words.extend(place() if left is None else left)
    words.extend(place(fields=(0, 1), mode=2) if right is None else right)
    return struct.pack('<'+'I'*len(words), *words)


def corpus():
    cases = []

    def add(name, *, expected=None, **kwargs):
        cases.append((name, carrier(**kwargs), expected))

    add('original-disjoint-siblings', expected='1100000000')
    add('original-shared-self', right=place(), expected='1111000101')
    add('original-exclusive-self', left=place(mode=2), right=place(mode=2), expected='1111111111')
    add('original-mixed-self', right=place(mode=2), expected='1111110111')
    add('original-nonleaf-prefix', left=place(fields=(0,), referent=2), expected={0: '0', 1: '1', 2: '1', 4: '1', 7: '1'})
    add('original-relabeled-whole-root', left=place(root=123, fields=()), expected={0: '0', 2: '1', 7: '1'})
    for key, value in [('invocation', 18), ('local', 4)]:
        add('original-distinct-'+key, right=place(**{key: value}), expected='1100000000')
    for key, value in [('present', False), ('fields_present', False), ('mode', 0), ('invocation', 0)]:
        add('original-invalid-descriptor-'+key, left=place(**{key: value}), expected='0111111111')
    for key in ('table_present', 'items_present'):
        add('original-absent-'+key, **{key: False}, expected='0000000000')
    add('original-wrong-authority', authority_left=1, expected={0: '0', 1: '1'})
    add('original-wrong-referent', left=place(referent=1), expected={0: '0', 1: '1'})
    add('original-bad-path', left=place(fields=(0, 2)), expected={0: '0'})
    for nested in (3, NO):
        add('original-invalid-root-edge-'+str(nested), rows=(*ROWS[:3], (0, True, ((8, nested),))), expected={0: '0', 1: '0'})
    for tag in range(256):
        rows = ((0, True, ((tag, NO),)),)
        p = place(root=0, fields=())
        add('scalar-tag-'+str(tag), rows=rows, left=p, right=p,
            authority_left=0, authority_right=0, expected={0: str(int(tag in (1, 2, 3, 4))), 1: str(int(tag in (1, 2, 3, 4)))})
    # I reject an unselected non-scalar sibling, not just the selected referent.
    for tag, nested in ((5, NO), (7, NO), (8, 2), (8, NO), (1, 0)):
        rows = (*ROWS[:2], (0, True, ((8, 0), (tag, nested))), ROWS[3])
        add(f'unselected-sibling-{tag}-{nested}', rows=rows, left=place(), right=place(), expected={0: '0', 1: '0'})
    for index in range(4):
        for kind in (1, 2, 3, 255):
            rows = list(ROWS); _, present, fields = rows[index]; rows[index] = (kind, present, fields)
            # Nominal row1 is unreachable and must not poison row0's root.
            add(f'layout-kind-{index}-{kind}', rows=rows, expected={0: str(int(index == 1)), 1: str(int(index == 1))})
        rows = list(ROWS); kind, _, fields = rows[index]; rows[index] = (kind, False, fields)
        add(f'absent-layout-fields-{index}', rows=rows, expected={0: str(int(index == 1)), 1: str(int(index == 1))})
    # Identity and conflict queries deliberately do not trust claimed layouts.
    identities = (0, 1, 0xffffffff, 1 << 32, (1 << 63)-1, 1 << 63, (1 << 64)-1)
    paths = ((), (0,), (0, 0), (0, 1), (1,), (65535,))
    for invocation, local, path, mode in itertools.product(identities, (0, 3, 65535), paths, (0, 1, 2, 3, NO)):
        add(f'identity-{invocation}-{local}-{path}-{mode}', right=place(invocation=invocation, local=local, fields=path, mode=mode))
    for left_mode, right_mode, left_path, right_path in itertools.product((1, 2), (1, 2), paths, paths):
        p = place(fields=left_path, mode=left_mode, root=NO, referent=NO)
        q = place(fields=right_path, mode=right_mode, root=123, referent=456)
        common = min(len(left_path), len(right_path))
        overlap = left_path[:common] == right_path[:common]
        bits = '00'+str(int(overlap))*2+str(int(overlap and 2 in (left_mode, right_mode)))*2
        bits += str(int(overlap and left_mode == 2))+str(int(overlap))
        bits += str(int(overlap and right_mode == 2))+str(int(overlap))
        add(f'prefix-modes-{left_mode}-{right_mode}-{left_path}-{right_path}', left=p, right=q, expected=bits)
    for count in (0, 1, 255, 65535):
        for present in (False, True):
            rows = ((0, present, ((1, NO),)*count),)
            p = place(root=0, fields=(), fields_present=False)
            add(f'field-count-{count}-{present}', rows=rows, left=p, right=p,
                authority_left=0, authority_right=0, expected={0: str(int(present or count == 0))})
    for depth in (1, 2, 32, 256, 1024):
        rows = [LEAF]+[(0, True, ((8, i-1), (8, i-1))) for i in range(1, depth+1)]
        p = place(root=depth, fields=(0,)*depth)
        add(f'deep-shared-dag-{depth}', rows=rows, left=p, right=p,
            authority_left=depth, authority_right=depth, expected='1111000101')
        # One forbidden leaf poisons both aliases no matter how deep the path.
        rows[0] = (0, True, ((5, NO),))
        add(f'deep-shared-dag-string-{depth}', rows=rows, left=p, right=p,
            authority_left=depth, authority_right=depth, expected='0011000101')
    add('path-count-65535', left=place(fields=(0,)*65535), expected={0: '0'})
    rng = random.Random(1142011)
    for number in range(512):
        rows = tuple((rng.choice((0, 0, 0, 2)), rng.randrange(5) != 0,
                      tuple((rng.randrange(16), rng.choice((NO, rng.randrange(8)))) for _ in range(rng.randrange(5))))
                     for _ in range(rng.randrange(8)))
        def arbitrary():
            return place(invocation=rng.choice(identities), local=rng.choice((0, 3, 65535)),
                         root=rng.choice((0, 3, NO)), fields=tuple(rng.randrange(5) for _ in range(rng.randrange(5))),
                         mode=rng.choice((0, 1, 2, 3)), present=rng.randrange(5) != 0,
                         fields_present=rng.randrange(5) != 0, referent=rng.randrange(8))
        add(f'random-descriptor-{number}', rows=rows, left=arbitrary(), right=arbitrary())
    valid = carrier()
    for length in (*range(20), 31, len(valid)-1):
        cases.append((f'truncated-carrier-{length}', valid[:length], 'D'))
    cases.append(('trailing-carrier', valid+b'\0', 'D'))
    cases.append(('carrier-count-bound', struct.pack('<5I', 1, 1, 65537, 0, 0), 'D'))
    return cases


class ReferencePlaces(unittest.TestCase):
    def test_actual_places_and_conflicts(self):
        (ROOT/'.test_output').mkdir(exist_ok=True)
        requested = os.environ.get('NANOLANG_REFERENCE_PLACES_PRODUCTS')
        if requested:
            work = Path(requested); work.mkdir(parents=True, exist_ok=False)
        else:
            work = Path(tempfile.mkdtemp(prefix='reference-places-', dir=ROOT/'.test_output'))
        print(f'I retain reference-place products at {work}', flush=True)
        cases = corpus(); paths = []
        record = dict(cases=len(cases), independent_expectations=sum(w is not None for _, _, w in cases),
                      inputs=[], routes=[], limits='I check places only; affine state/bytecode and runtime admission remain open.')
        for index, (name, data, want) in enumerate(cases):
            p = work/f'case-{index}.bin'; p.write_bytes(data); paths.append(p)
            record['inputs'].append(dict(name=name, file=p.name, expected=want, sha256=hashlib.sha256(data).hexdigest()))
        def save(): (work/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
        save()
        def execute(command, name):
            reports = []
            for i in range(0, len(paths), 32):
                reports.extend(checked([*command, *paths[i:i+32]], timeout=600).splitlines())
            self.assertEqual(len(reports), len(cases), name)
            (work/(name+'.reports')).write_text('\n'.join(reports)+'\n')
            return reports
        oracle = Path(os.environ.get('NANOLANG_REFERENCE_PLACES_REFERENCE', ROOT/'obj/nvm2c_reference_places_reference'))
        expected = execute([oracle], 'actual-c')
        for (name, _, want), got in zip(cases, expected):
            if isinstance(want, dict):
                for index, value in want.items():
                    with self.subTest(expectation=name, bit=index): self.assertEqual(got[index], value)
            elif want is not None:
                with self.subTest(expectation=name): self.assertEqual(got, want)
        for name, command in build_routes(ROOT/'tests/nvm2c_reference_places_driver.nano', work):
            reports = execute(command, name)
            for (case, _, _), want, got in zip(cases, expected, reports):
                with self.subTest(route=name, case=case): self.assertEqual(got, want)
            record['routes'].append(dict(name=name, reports=len(reports), command=list(map(str, command))))
            save()
        print(f'I match {len(cases)} actual reference-place reports through {len(record["routes"])} routes.', flush=True)
