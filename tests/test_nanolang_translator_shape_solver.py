"""I retain every original C shape assertion and replay its observed APIs in NanoLang."""
from pathlib import Path
import itertools
import os
import shlex
import struct
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked, source_compiler_command
from tests.native_process import native_link_flags
from tests.test_nanolang_translator_shape_graph import Program, packet, PREFIX

ROOT = Path(__file__).resolve().parents[1]


def directional_cases():
    cases=[]
    for operation,left,right in itertools.product((14,15,16),range(15),range(15)):
        p=Program(f'flow-{operation}-{left}-{right}');a=p.new(left);b=p.new(right)
        if left==12:p.op(8,a,17,want=1)
        if right==12:p.op(8,b,31,want=1)
        p.op(6,a,21845,want=1).op(6,b,43690,want=1)
        p.op(operation,a,b,want=1).op(20).op(21,want=0)
        cases.append(p)
    for operation in range(14,20):
        for bad in (0,0xffffffff):
            for position in (0,1):
                p=Program(f'bad-id-{operation}-{bad}-{position}');p.new(0)
                p.op(operation,bad if position==0 else 1,bad if position else 1,want=0)
                p.op(20,want=0).op(21,want=0);cases.append(p)
    p=Program('final-unknown-variant');a=p.new(0);b=p.new(10)
    p.op(14,a,b,want=1).op(20,want=0).op(13,want=0).op(21,want=0);cases.append(p)
    p=Program('final-unknown-integer-array');a=p.new(3);b=p.new(11)
    p.op(14,a,b,want=1).op(20,want=0).op(13,want=0).op(21,want=0);cases.append(p)
    p=Program('independent-callable-producers');a=p.new(12);b=p.new(12);joined=p.new(0)
    p.op(8,a,2,want=1).op(8,b,9,want=1).op(14,a,joined,want=1).op(14,b,joined,want=1)
    p.op(20,want=1).op(9,a,want=1).op(9,b,want=1).op(9,joined,want=2)
    p.op(27,a,0,want=2).op(27,b,0,want=9).op(27,joined,1,want=9)
    p.op(20,want=1).op(23,want=2).op(21,want=0);cases.append(p)
    for k in (1,2,7,8):
        for exact in (0,1):
            p=Program(f'late-optional-{k}-{exact}');source=p.new(k);target=p.new(k if exact else 0)
            p.op(14,source,target,want=1).op(20,want=1)
            optional=p.new(6);payload=p.child(optional,0);p.join(payload,source)
            p.op(14,optional,target,want=1).op(20,want=1-exact).op(21,want=0);cases.append(p)
    return cases


class ShapeSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory(prefix='nano-shape-solver-')
        cls.addClassCleanup(cls.workspace.cleanup);cls.directory=Path(cls.workspace.name)
        cls.fixture=ROOT/'tests/nvm2c_shape_solver_driver.nano'
        cls.routes=build_routes(cls.fixture,cls.directory)
        cls.references=[];cls.tracers=[]
        for mode,flags in [('ordinary',['-O2']),('sanitized',['-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all'])]:
            for kind,source in [('reference','nvm2c_shape_solver_reference.c'),('original-trace','nvm2c_shape_solver_original_trace.c')]:
                binary=cls.directory/(kind+'-'+mode)
                checked([*shlex.split(os.environ.get('CC','cc')),'-std=c11','-Wall','-Wextra','-Werror',*flags,
                         '-I',ROOT/'src/nanoisa',ROOT/'tests'/source,*native_link_flags(),'-o',binary],build=True)
                (cls.references if kind=='reference' else cls.tracers).append(binary)

    def compare(self,path,expected_cases):
        baseline=checked([self.references[0],path])
        self.assertTrue(baseline.startswith(PREFIX));self.assertNotIn('ASSERT|',baseline)
        self.assertEqual(sum(s.startswith('Z|') for s in baseline.splitlines()),expected_cases)
        for command in [[self.references[1]],*[cmd for _,cmd in self.routes]]:
            self.assertEqual(checked([*command,path]),baseline)
        return baseline

    def test_original_api_trace(self):
        for index,tracer in enumerate(self.tracers):
            path=self.directory/f'original-{index}.bin'
            report=checked([tracer,path]);self.assertRegex(report,r'^Shape constraints: [1-9][0-9]* passed, 0 failed\n$')
            raw=path.read_bytes();count=struct.unpack_from('<I',raw)[0];self.assertGreater(count,150)
            self.assertLessEqual(len(raw),16777216)
            cursor=4;operations=0;histogram={}
            for _ in range(count):
                mode,size=struct.unpack_from('<II',raw,cursor);cursor+=8;self.assertEqual(mode,1)
                for _ in range(size):
                    opcode,a,b,c,want,check=struct.unpack_from('<6I',raw,cursor);cursor+=24
                    self.assertEqual(check,1);operations+=1;histogram[opcode]=histogram.get(opcode,0)+1
            self.assertEqual(cursor,len(raw));self.assertGreater(operations,120000)
            self.assertTrue(set(range(14,28))<=histogram.keys())
            baseline=self.compare(path,count)
            self.assertIn('D|',baseline);self.assertIn('I|',baseline)
            self.assertIn('N|40006|',baseline)
            self.assertIn('W|',baseline);self.assertIn('R|',baseline)
            self.assertEqual(sum(s.startswith('P|') for s in baseline.splitlines()),histogram[21])
            self.assertIn('I found conflicting aggregate shape kinds optional/string at nodes 1/2',baseline)

    def test_named_directional_countercases(self):
        cases=directional_cases();self.assertEqual(len(cases),710)
        path=self.directory/'directional.bin';path.write_bytes(packet(cases))
        baseline=self.compare(path,len(cases))
        self.assertIn('I require proved scalar producers for variant scalar storage',baseline)
        self.assertIn('I require an exact integer element for variant array storage',baseline)
        self.assertIn('I cannot widen an exactly constrained',baseline)
        self.assertIn('I cannot convert aggregate storage',baseline)

    def test_boxed_fields_below_exact_shared_record_views(self):
        cases=[]
        for kind,alias,wrong,late in itertools.product((1,2,3,5,7,8,12,13,14),
                                                       (False,True),(False,True),(False,True)):
            p=Program(f'boxed-field-{kind}-{alias}-{wrong}-{late}')
            source=p.new(6);target=p.new(6)
            left=p.child(source,0);right=p.child(target,0)
            if alias:
                p.join(left,p.new(3));p.join(right,p.new(3))
                left=p.child(left,0);right=p.child(right,0)
            p.join(left,p.new(4));p.join(right,p.new(4))
            present=p.child(left,0);boxed=p.child(right,0)
            if late:p.op(14,source,target,want=1)
            p.join(present,p.new(kind));p.join(boxed,p.new(6))
            payload=p.child(boxed,0)
            p.join(payload,p.new((2 if kind==1 else 1) if wrong else kind))
            if kind in (3,5):
                p.join(p.child(present,0),p.new(1))
                if not wrong:p.join(p.child(payload,0),p.new(1))
            if kind==12:p.op(8,present,17,want=1)
            if not late:p.op(14,source,target,want=1)
            accepted=int(alias and not wrong)
            p.op(20,want=accepted)
            if accepted:
                p.op(2,present,want=kind).op(2,boxed,want=6).op(2,payload,want=kind)
                if kind==12:p.op(9,payload,want=1).op(27,payload,0,want=17)
                p.op(20,want=1)
            else:p.op(13,want=0)
            cases.append(p)
        self.assertEqual(len(cases),72)
        path=self.directory/'boxed-shared-record-views.bin';path.write_bytes(packet(cases))
        self.compare(path,len(cases))

    def test_malformed_solver_packets(self):
        good=packet([Program('one').op(0,4,want=1)])
        cases=[b'',b'\0',good[:-1],good+b'\0',struct.pack('<III',1,2,0),struct.pack('<I',0xffffffff)]
        for offset,value in [(12,28),(32,2),(20,65536)]:
            raw=bytearray(good);struct.pack_into('<I',raw,offset,value)
            if offset==20:struct.pack_into('<I',raw,12,6)
            cases.append(bytes(raw))
        for index,raw in enumerate(cases):
            path=self.directory/f'bad-{index}.bin';path.write_bytes(raw)
            for command in [[b] for b in self.references]+[cmd for _,cmd in self.routes]:
                self.assertEqual(checked([*command,path]),'INVALID\n')

    def test_imported_shadows(self):
        binary=self.directory/'imported-native'
        checked([ROOT/'bin/nanoc_c',self.fixture,'--test-imports','-o',binary],build=True)
        self.assertEqual(checked([binary]),'')
        for name,command in [('c',[ROOT/'bin/nano_virt']),('nanolang',source_compiler_command())]:
            if not command:continue
            module=self.directory/('imported-'+name+'.nvm')
            checked([*command,self.fixture,'--test-imports','--emit-nvm','-o',module],build=True)
            checked([ROOT/'bin/nano_vm','--verify-only',module])
            self.assertEqual(checked([ROOT/'bin/nano_vm',module,'--']),'')
