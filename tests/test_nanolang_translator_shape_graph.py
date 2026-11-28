"""I compare exact representation graphs with the actual unmodified C graph."""
from pathlib import Path
import itertools
import os
import random
import shlex
import struct
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked, source_compiler_command
from tests.native_process import native_link_flags

ROOT = Path(__file__).resolve().parents[1]
KINDS = ('unknown','int','string','array','record','map','optional','bool','float',
         'numeric','variant-scalar','variant-int-array','function','u8','bstring')
PREFIX = ''.join(f'K|{i}|{name}\n' for i,name in enumerate(KINDS))


class Program:
    def __init__(self, name, mode=0):
        self.name, self.mode, self.ops, self.nodes = name, mode, [], 0
    def op(self, code, a=0, b=0, c=0, want=None):
        self.ops.append((code,a,b,c,0 if want is None else want,int(want is not None)))
        return self
    def new(self, kind):
        self.nodes += 1
        self.op(0,kind,want=self.nodes)
        return self.nodes
    def child(self, node, index):
        self.nodes += 1
        self.op(3,node,index,want=self.nodes)
        return self.nodes
    def join(self, left, right, success=1):
        self.op(5,left,right,want=success)
    def reset(self):
        self.op(11,want=0);self.nodes=0


def recursive(p):
    record, array = p.new(4), p.new(3)
    p.join(p.child(record,0),array)
    p.join(p.child(array,0),record)
    return record


def corpus():
    cases = []
    for left,right in itertools.product(range(15),repeat=2):
        p=Program(f'kind-{left}-{right}');a=p.new(left);b=p.new(right)
        p.op(6,a,21845,want=1).op(6,b,43690,want=1)
        success=int(left==0 or right==0 or left==right)
        p.join(a,b,success);p.op(1,b,want=1 if success else 0)
        if success:p.op(7,a,want=65535)
        else:p.op(0,1,want=0).op(13,want=0)
        cases.append(p)
    for kind,index in itertools.product(range(15),(0,1,3,0xffffffff)):
        p=Program(f'projection-{kind}-{index}');a=p.new(kind)
        allowed=kind in (0,4) or (kind==5 and index<2) or (kind in (3,6) and index==0)
        p.op(4,a,index,want=0)
        if allowed:
            child=p.child(a,index);p.op(3,a,index,want=child).op(4,a,index,want=child)
        else:p.op(13,want=0).op(3,a,index,want=0)
        cases.append(p)
    p=Program('cycles-and-shared-children');a=recursive(p);b=recursive(p);p.join(a,b)
    first,second=p.child(a,1),p.child(b,2);p.join(first,second);p.join(first,p.new(2))
    p.op(4,b,1,want=first).op(4,a,2,want=first);p.join(a,b);cases.append(p)
    for target in (1,3):
        p=Program(f'projection-unify-{target}');a=p.new(0);p.child(a,3);b=p.new(target)
        p.join(a,b,0);p.op(1,0,want=0).op(0,15,want=0);cases.append(p)
    for malformed in (15,0xffffffff):
        p=Program(f'poison-reset-{malformed}');p.new(7);p.op(0,malformed,want=0).op(1,0,want=0).op(13,want=0)
        p.reset();p.new(2);p.op(2,1,want=2).op(13,want=1);cases.append(p)
    for mask in [0,65535,21845,43690,*[1<<i for i in range(16)]]:
        p=Program(f'mask-{mask}');a=p.new(0);b=p.new(0);p.op(6,a,mask,want=1).op(6,b,1,want=1);p.join(a,b)
        p.op(7,b,want=mask|1).op(2,b,want=0);cases.append(p)
    for absent in (0,1):
        p=Program(f'callable-{absent}');a=p.new(0)
        for target in (17,0xffffffff,0,3,17):p.op(8,a,target,want=1)
        p.op(9,a,want=4)
        for index in range(4):p.op(10,a,index,1,want=1)
        p.op(10,a,0,0,want=0) if absent else p.op(10,a,4,1,want=0)
        p.op(9,a,want=0);cases.append(p)
    for kind in range(15):
        p=Program(f'callable-kind-{kind}');a=p.new(kind)
        p.op(8,a,0xffffffff,want=int(kind in (0,12)));cases.append(p)
    for high in (1,0x80000000,0xffffffff):
        p=Program(f'callable-wide-index-{high}');a=p.new(12);p.op(8,a,0,want=1).op(12,a,0,high,want=0);cases.append(p)
    p=Program('callable-projection-refusal');a=p.new(0);p.child(a,0);p.op(8,a,1,want=0);cases.append(p)
    p=Program('callable-exact-join');a=p.new(12);b=p.new(12)
    for target in (9,1,9):p.op(8,a,target,want=1)
    for target in (1,0,0xffffffff):p.op(8,b,target,want=1)
    p.join(b,a);p.op(9,a,want=4)
    for index in range(4):p.op(10,a,index,1,want=1)
    cases.append(p)
    for conflict in (0,1):
        p=Program(f'map-{conflict}');maps=[p.new(5),p.new(5)]
        for side,node in enumerate(maps):
            p.join(p.child(node,0),p.new(2));p.join(p.child(node,1),p.new(2 if conflict and side else 1))
        p.join(*maps,success=1-conflict);cases.append(p)
    for bad in (0,0xffffffff):
        for operation in (1,2,3,4,5,6,7,8,9,10):
            p=Program(f'invalid-id-{operation}-{bad}');p.new(0);p.op(operation,bad,0,0,want=0).op(13,want=0);cases.append(p)
    rng=random.Random(1142)
    for number in range(64):
        p=Program(f'rank-and-mask-{number}')
        for _ in range(8):p.new(0)
        for _ in range(16):
            a,b=rng.randrange(1,9),rng.randrange(1,9);p.op(6,a,rng.randrange(65536),want=1);p.join(a,b)
        for node in range(1,9):p.op(1,node).op(7,node)
        cases.append(p)
    return cases


def deep(conflict):
    p=Program(f'original-deep-10000-{conflict}',mode=1);roots=[]
    for side in range(2):
        node=p.new(4);roots.append(node)
        for _ in range(10000):
            next_node=p.new(4);p.join(p.child(node,0),next_node);node=next_node
        leaf=p.new(2 if side and conflict else 1);p.join(p.child(node,65534),leaf)
    p.join(*roots,success=1-conflict)
    p.op(1,roots[0],want=0 if conflict else roots[0]).op(1,roots[1],want=0 if conflict else roots[0])
    if conflict:p.op(0,1,want=0)
    assert p.nodes==40006
    return p


def wide():
    p=Program('original-wide-300',mode=1);roots=[p.new(4),p.new(4)]
    for side in range(2):
        for field in range(300):
            value=p.new(2 if field%2 else 1);p.join(p.child(roots[side],field),value)
    p.join(*roots)
    for field in range(300):p.op(4,roots[0],field)
    assert p.nodes==1202
    return p


def packet(cases):
    data=bytearray(struct.pack('<I',len(cases)))
    for p in cases:
        data.extend(struct.pack('<II',p.mode,len(p.ops)))
        for op in p.ops:data.extend(struct.pack('<6I',*op))
    return bytes(data)


class ShapeGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory(prefix='nano-shape-graph-')
        cls.addClassCleanup(cls.workspace.cleanup);cls.directory=Path(cls.workspace.name)
        cls.fixture=ROOT/'tests/nvm2c_shape_graph_driver.nano'
        cls.routes=build_routes(cls.fixture,cls.directory)
        cls.references=[]
        for mode,flags in [('ordinary',['-O2']),('sanitized',['-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all'])]:
            binary=cls.directory/('reference-'+mode)
            checked([*shlex.split(os.environ.get('CC','cc')),'-std=c11','-Wall','-Wextra','-Werror',*flags,
                     ROOT/'tests/nvm2c_shape_graph_reference.c',*native_link_flags(),'-o',binary],build=True)
            cls.references.append(binary)

    def compare(self,cases,label):
        path=self.directory/(label+'.bin');path.write_bytes(packet(cases))
        baseline=checked([self.references[0],path])
        self.assertTrue(baseline.startswith(PREFIX))
        self.assertNotIn('ASSERT|',baseline)
        self.assertEqual(sum(line.startswith('Z|') for line in baseline.splitlines()),len(cases))
        for route in [[self.references[1]],*[command for _,command in self.routes]]:
            self.assertEqual(checked([*route,path]),baseline)
        return baseline

    def test_exact_graph_matrix(self):
        cases=corpus()
        self.assertEqual(len(cases),418)
        baseline=self.compare(cases,'matrix')
        self.assertIn('I found conflicting aggregate shape kinds int/string at nodes 1/2',baseline)
        self.assertIn('N|1|1|7|0|0||0|\n',self.compare([Program('shape-bool').op(0,7,want=1)],'namespace'))
        self.assertEqual(KINDS[4],'record');self.assertEqual(KINDS[7],'bool')

    def test_original_deep_and_wide_bounds(self):
        for case in [deep(0),deep(1),wide()]:
            baseline=self.compare([case],case.name)
            self.assertIn(f'|{case.nodes}|',baseline)
            if 'deep-10000-1' in case.name:self.assertIn('conflicting aggregate shape kinds int/string',baseline)

    def test_original_c_baseline(self):
        binary=self.directory/'original-c-tests'
        checked([*shlex.split(os.environ.get('CC','cc')),'-std=c11','-O1','-g','-fsanitize=address,undefined',
                 '-fno-sanitize-recover=all','-I',ROOT/'src/nanoisa',ROOT/'tests/nanoisa/test_nvm2c_shape.c',
                 ROOT/'src/nanoisa/nvm2c_shape.c',*native_link_flags(),'-o',binary],build=True)
        checked([binary])

    def test_malformed_packets(self):
        good=packet([Program('one').op(0,4,want=1)])
        cases=[b'',b'\0',good[:-1],good+b'\0',struct.pack('<III',1,2,0),struct.pack('<I',0xffffffff)]
        for offset,value in [(12,14),(12+20,2),(12+8,65536)]:
            raw=bytearray(good);struct.pack_into('<I',raw,offset,value)
            if offset==20:struct.pack_into('<I',raw,12,6)
            cases.append(bytes(raw))
        for index,raw in enumerate(cases):
            path=self.directory/f'bad-{index}.bin';path.write_bytes(raw)
            for route in [[b] for b in self.references]+[command for _,command in self.routes]:
                self.assertEqual(checked([*route,path]),'INVALID\n')

    def test_imported_shadows(self):
        binary=self.directory/'imported-native'
        checked([ROOT/'bin/nanoc_c',cls_fixture:=self.fixture,'--test-imports','-o',binary],build=True)
        self.assertEqual(checked([binary]),'')
        producers=[('c',[ROOT/'bin/nano_virt'])]
        if source_compiler_command():producers.append(('nanolang',source_compiler_command()))
        for name,command in producers:
            module=self.directory/('imported-'+name+'.nvm')
            checked([*command,cls_fixture,'--test-imports','--emit-nvm','-o',module],build=True)
            checked([ROOT/'bin/nano_vm','--verify-only',module])
            self.assertEqual(checked([ROOT/'bin/nano_vm',module,'--']),'')
