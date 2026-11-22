"""I compare all bounded value-call edges with my actual C graph checker."""
from pathlib import Path
import hashlib
import json
import os
import random
import struct
import tempfile
import unittest
from tests.nanolang_translator_routes import build_routes, checked
from tests.test_nanolang_translator_ownership import desc, record, function, image, extension, effect_data, union_data

ROOT = Path(__file__).resolve().parents[1]


def program(model, entry=0, present=True, functions_present=True):
    return struct.pack('<III', entry, int(present), int(functions_present))+model


def model(bodies=(b'\x3d',), *, locals_=None, parameters=None, results=None,
          result_counts=None, upvalues=None):
    count = len(bodies)
    locals_ = locals_ if locals_ is not None else [[] for _ in bodies]
    parameters = parameters if parameters is not None else [() for _ in bodies]
    results = results if results is not None else [1]+[0]*(count-1)
    result_counts = result_counts if result_counts is not None else [int(tag != 0) for tag in results]
    upvalues = upvalues if upvalues is not None else [0]*count
    declarations = [(len(slots), len(params), desc(tag), slots)
                    for slots, params, tag in zip(locals_, parameters, results)]
    offset = 0
    functions = []
    for body, slots, params, tag, result_count, captures in zip(
            bodies, locals_, parameters, results, result_counts, upvalues):
        entry = bytearray(function(len(slots), len(params), result_count, tag,
                                   params=params, start=offset, length=len(body)))
        struct.pack_into('<I', entry, 16, captures)
        functions.append(bytes(entry))
        offset += len(body)
    return image(record(1, functions=declarations), functions=functions, code=b''.join(bodies))


def call(target, opcode=59):
    return bytes([opcode])+struct.pack('<I', target)


def signature_model(slot, *, flags=(), rows=(), extensions=(), strings=(), version=1, parameter=True):
 root=call(1);body=b'\x3d';locals_=[slot] if parameter else []
 own=record(version,flags=flags,functions=[(0,0,desc(1),[]),(len(locals_),len(locals_),desc() if parameter else slot,locals_)],extensions=extensions)
 f=[function(0,results=1,result_tag=1,length=len(root)),function(len(locals_),len(locals_),results=0 if parameter else 1,result_tag=0 if parameter else slot[0],params=(slot[0],) if parameter else (),start=len(root),length=1)]
 return image(own,rows=rows,strings=strings,functions=f,code=root+body)

def signature_graphs():
 out=[]
 for flags in range(4):
  for parameter in (False,True):
   out.append((f'record-flags-{flags}-parameter-{parameter}',program(signature_model(desc(8,0),flags=[flags],rows=[(0,0xffffffff,[(1,0xffffffff,0xffffffff)])],parameter=parameter)),bool(flags&1)))
 rows=[(0,0xffffffff,[(1,0xffffffff,0xffffffff)]),(2,2,[(8,0,3)])]
 extensions=[extension(1,union_data(1,[(0,0,1),(1,1,0)]))]
 for parameter in (False,True):
  out.append((f'resource-union-parameter-{parameter}',program(signature_model(desc(10,1),flags=[3,3],rows=rows,extensions=extensions,strings=[b'Some',b'Empty',b'Choice',b'value'],version=3,parameter=parameter)),True))
 for parameter in (False,True):
  out.append((f'array-int-parameter-{parameter}',program(signature_model(desc(7,0),extensions=[extension(4,effect_data(2,types=[desc(1)]),2)],strings=[b'op'],version=3,parameter=parameter)),True))
 out.append(('borrowed-record-parameter',program(signature_model(desc(8,0,1),flags=[3],rows=rows[:1])),False))
 return out


def corpus():
    cases = []

    def add(name, raw, expected):
        cases.append((name, raw, str(expected) if expected == 'D' else str(int(expected))))

    # A graph deliberately does not check stack depth or resource resolution.
    add('entry-ret-needs-stack-analysis', program(model()), True)
    add('entry-nop-fallthrough-needs-body-analysis', program(model((b'\x00',))), True)
    add('entry-string-needs-type-analysis', program(model((b'\x04\0\0\0\0\x3d',))), True)
    add('missing-module', program(model(), present=False), False)
    add('missing-function-array', program(model(), functions_present=False), False)
    for entry in (1, 0xffffffff):
        add(f'wrong-entry-{entry}', program(model(), entry=entry), False)
    add('no-functions', program(model(())), False)
    for count in (8, 9):
        add(f'function-bound-{count}', program(model((b'\x3d',)*count)), count == 8)
    for tag in range(11):
        add(f'entry-result-tag-{tag}', program(model(results=[tag])), tag in (1, 2, 4))
    for count in (0, 1, 2):
        add(f'entry-result-count-{count}', program(model(result_counts=[count])), count == 1)
    add('entry-has-parameter', program(model(locals_=[[desc(1)]], parameters=[(1,)])), False)
    for count in (255, 256, 257):
        add(f'local-bound-{count}', program(model(locals_=[[desc(1)]*count])), count <= 256)
    for count in (4095, 4096, 4097):
        add(f'instruction-bound-{count}', program(model((b'\x00'*(count-1)+b'\x3d',))), count <= 4096)
    add('code-byte-bound', program(model((b'\x00'*131073,))), False)
    for captures in (1, 65535):
        add(f'entry-captures-{captures}', program(model(upvalues=[captures])), False)
    for target in (0, 1, 2, 0xffffffff):
        add(f'direct-target-{target}', program(model((call(target), b'\x3d'))), target == 1)
        add(f'dead-direct-target-{target}', program(model((b'\x3d'+call(target), b'\x3d'))), target == 1)
    for label, body in (
        ('call-ref', call(1, 15)+b'\0\0'),
        ('tail-call', call(1, 51)),
        ('indirect', b'\x3c\0\0\0\0'),
        ('module-call', b'\x3f'+b'\0'*12),
        ('extern-call', call(0, 62)),
    ):
        for prefix in (b'', b'\x3d'):
            add(label+'-dead-'+str(bool(prefix)), program(model((prefix+body, b'\x3d'))), False)
    for label, body in (('empty', b''), ('truncated-call', b'\x3b\1'), ('unknown', b'\xff'),
                        ('branch-interior', b'\x38\1\0\0\0')):
        add(label, program(model((body,))), False)
    for tag in (0, 1, 2, 3, 4, 5, 7, 8, 10):
        add(f'helper-value-parameter-{tag}', program(model((call(1), b'\x3d'),
            locals_=[[], [desc(tag)]], parameters=[(), (tag,)])), tag in (1, 2, 4, 5))
        add(f'helper-value-result-{tag}', program(model((call(1), b'\x3d'), results=[1,tag])), tag in (0,1,2,4))
    # I exhaust every graph among two helpers; edges to entry zero refuse above.
    edge_slots = [(source, target) for source in range(3) for target in (1, 2)]
    for bits in range(1 << len(edge_slots)):
        selected = {edge for index, edge in enumerate(edge_slots) if bits & (1 << index)}
        closure = set(selected)
        for middle in range(3):
            for source in range(3):
                for target in range(3):
                    if (source,middle) in closure and (middle,target) in closure:
                        closure.add((source,target))
        bodies = tuple(b''.join(call(target) for a,target in sorted(selected) if a == source)+b'\x3d'
                       for source in range(3))
        add(f'exhaustive-three-function-graph-{bits}', program(model(bodies)),
            not any((index,index) in closure for index in range(3)))
    rng = random.Random(11421183)
    for index in range(64):
        count = rng.randint(1, 8)
        edges = {(source,target) for source in range(count) for target in range(1,count)
                 if rng.randrange(5) == 0}
        # An independent DFS checks cycles, rather than copying transitive closure.
        visiting, done = set(), set()
        def acyclic(node):
            if node in visiting: return False
            if node in done: return True
            visiting.add(node)
            if not all(acyclic(target) for source,target in edges if source == node): return False
            visiting.remove(node); done.add(node); return True
        expected = all(acyclic(node) for node in range(count))
        bodies = tuple(b''.join(call(target) for a,target in sorted(edges) if a == source)+b'\x3d'
                       for source in range(count))
        add(f'random-eight-function-graph-{index}', program(model(bodies)), expected)
    for name, raw, expected in signature_graphs(): add(name,raw,expected)
    add('short-carrier', b'', 'D')
    add('invalid-presence', struct.pack('<III', 0, 2, 1)+model(), 'D')
    add('truncated-model', program(b'\0'*27), 'D')
    return cases


class AffineGraph(unittest.TestCase):
    def test_complete_bounded_graph_against_actual_c(self):
        work = Path(tempfile.mkdtemp(prefix='nano-affine-graph-'))
        cases = corpus()
        paths = []
        record_ = dict(scope='I check value-call-graph declarations/edges only, including dead code. '
            'I grant no stack, lifetime, global/effect, private-profile or executable admission.',
            corpus=[], routes=[])
        for index, (label, data, expected) in enumerate(cases):
            path = work/f'{index:04d}.bin'; path.write_bytes(data); paths.append(path)
            record_['corpus'].append(dict(name=label, path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                                          bytes=len(data), expected=expected))
        def save(): (work/'manifest.json').write_text(json.dumps(record_, indent=2)+'\n')
        save()
        def execute(command, label):
            reports = []
            for start in range(0, len(paths), 16):
                reports += checked([*command, *paths[start:start+16]], timeout=600).splitlines()
            self.assertEqual(len(reports), len(cases), label)
            (work/(label+'.reports')).write_text('\n'.join(reports)+'\n')
            for (name, _, expected), actual in zip(cases, reports):
                with self.subTest(route=label, case=name): self.assertEqual(actual, expected)
            return reports
        reference = Path(os.environ.get('NANOLANG_AFFINE_GRAPH_REFERENCE', ROOT/'obj/nvm2c_affine_graph_reference'))
        expected = execute([reference], 'actual-c')
        for label, command in build_routes(ROOT/'tests/nvm2c_affine_graph_driver.nano', work):
            actual = execute(command, label)
            self.assertEqual(actual, expected, label)
            record_['routes'].append(dict(name=label, reports=len(actual), command=list(map(str,command))))
            save()
        print(f'I match {len(cases)} actual affine value-call graphs through {len(record_["routes"])} routes; retained {work}.', flush=True)


if __name__ == '__main__':
    unittest.main()
