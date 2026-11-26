"""I compare complete handler records/maps with the unchanged shared C helper."""
from pathlib import Path
import os
import random
import shlex
import struct
import tempfile
import unittest
import yaml
from tests.nanolang_translator_routes import build_routes, checked, source_compiler_command
from tests.native_process import native_link_flags

ROOT = Path(__file__).resolve().parents[1]
OPS = {v['name']: v for v in yaml.safe_load((ROOT/'spec/nanoisa.yaml').read_bytes())['legacy_opcodes']}
FMT = {'U8':'B','U16':'H','U32':'I','I32':'i','I64':'q','F64':'Q'}

def instruction(name, *args):
    row = OPS[name]
    assert len(args) == len(row['operands'])
    return bytes([row['code']])+b''.join(struct.pack('<'+FMT[k], v) for k,v in zip(row['operands'],args))

def packet(code, functions=None, names=(b'operation',)):
    if functions is None:
        functions = [(0,len(code),0)]
    return (struct.pack('<II',len(functions),len(names))
            + b''.join(struct.pack('<III',*f) for f in functions)
            + b''.join(struct.pack('<I',len(n))+n for n in names)+code)

def report(lengths, handlers):
    parts = [f'1:{len(handlers)}']
    for f,operation,target,start,count,live in handlers:
        parts.append('|'+','.join(map(str,(f,operation,target,start,count)))+':'
                     +''.join('1' if i in live else '0' for i in range(lengths[f]+1)))
    return ''.join(parts)

def cases():
    rows = []
    def add(label, code, want=None, **kwargs):
        rows.append((label,packet(code,**kwargs),want))
    hp = lambda name,target,start=0,count=0: instruction('HANDLER_PUSH',name,target,start,count)
    add('no-functions',b'\xff', '1:0', functions=[])
    add('empty-body',b'', '1:0')
    for op in ('RET','HALT','EFFECT_RESUME','TAIL_CALL'):
        raw = instruction(op,*([0]*len(OPS[op]['operands'])))
        code = hp(0,13)+raw+instruction('NOP')
        add('terminal-'+op,code,report([len(code)],[(0,0,13,0,0,{13})]))
    code=hp(0,13)+instruction('NOP')
    add('terminal-boundary-fallthrough',code,report([14],[(0,0,13,0,0,{13,14})]))
    code=hp(0,13)+hp(0,14)+instruction('HALT')+instruction('NOP')+instruction('RET')
    add('nested-registration-fallthrough',code,report([len(code)],[(0,0,13,0,0,{13,26}),(0,0,27,0,0,{27,28})]))
    # The arm's conditional visits both edges, loops back once, and reaches the end.
    code=hp(0,13)+instruction('JMP_TRUE',6)+instruction('NOP')+instruction('JMP_FALSE',-6)+instruction('NOP')
    add('conditional-cycle',code,report([len(code)],[(0,0,13,0,0,{13,18,19,24,25})]))
    code=hp(2,13,2)+instruction('HALT')+hp(1,13)+instruction('NOP')
    add('ordered-other-function-duplicate-name',code,
        report([14,14],[(0,0,13,2,0,{13}),(1,1,13,0,0,{13,14})]),
        functions=[(0,14,2),(14,14,0)],names=[b'a',b'b',b'a\0suffix'])
    # Same body twice retains function ordering and local indexes, not shared maps.
    code=hp(1,13,65535,0)+instruction('RET')
    add('duplicate-body-max-locals',code,report([14,14],[(0,0,13,65535,0,{13}),(1,0,13,65535,0,{13})]),
        functions=[(0,14,65535)]*2,names=[b'',b'\0tail'])
    for name,target,start,count,locals_ in [(1,13,0,0,0),(0,14,0,0,0),(0,-1,0,0,0),(0,1,0,0,0),
                                            (0,13,1,0,0),(0,13,65535,1,65535),(0,13,0,1,0)]:
        code=hp(name,target,start,count)+instruction('HALT')
        add(f'bad-handler-{name}-{target}-{start}-{count}',code,'0',functions=[(0,14,locals_)])
    code=hp(0,0)+instruction('HALT')
    add('registration-self-cycle',code,report([14],[(0,0,0,0,0,{0,13})]))
    code=hp(0,13)+instruction('JMP',5)
    add('jump-terminal',code,report([18],[(0,0,13,0,0,{13,18})]))
    for name,row in sorted(OPS.items(),key=lambda p:p[1]['code']):
        values=[0]*len(row['operands'])
        if name=='HANDLER_PUSH':values=[0,13,0,0]
        raw=instruction(name,*values)
        code=hp(0,13)+raw+instruction('NOP')
        add('every-arm-opcode-'+name,code)
        add('every-dead-opcode-'+name,instruction('HALT')+raw)
        for cut in range(1,len(raw)):
            add(f'truncated-{name}-{cut}',hp(0,13)+raw[:cut],'0')
    for op in (245,246,253,254,255):
        # Some primary values may become defined; actual schema owns that choice.
        if op not in {v['code'] for v in OPS.values()}:
            add('reserved-'+str(op),hp(0,13)+bytes([op]),'0')
    for code in [b'\xff',instruction('JMP',1),instruction('CALL',1),instruction('TAIL_CALL',1),hp(0,13)]:
        add('bad-other-'+code.hex(),instruction('HALT')+code,'1:0' if code in (instruction('CALL',1),instruction('TAIL_CALL',1)) else '0',functions=[(0,1,0),(1,len(code),0)])
    for op in ('CALL','TAIL_CALL'):
        code=instruction(op,2)
        add('out-of-range-other-'+op,instruction('HALT')+code,'0',functions=[(0,1,0),(1,len(code),0)])
    for functions in [[(1,1,0)],[(0,2,0)],[(0xffffffff,0,0)],[(0,1,65536)]]:
        add('bad-metadata-'+str(len(rows)),b'\xa3','0',functions=functions)
    add('uncovered-bytes',b'\xff\xa3\xff','1:0',functions=[(1,1,0)])
    rng=random.Random(1142165)
    for k in range(128):
        # I derive expected byte maps from the generated labelled graph before encoding.
        n=rng.randrange(1,25);kinds=[rng.choice(['NOP','JMP','JMP_TRUE','JMP_FALSE','RET','EFFECT_RESUME']) for _ in range(n)]
        widths=[1 if op in ('NOP','RET','EFFECT_RESUME') else 5 for op in kinds]
        offsets=[13]
        for width in widths:offsets.append(offsets[-1]+width)
        targets=[rng.randrange(n+1) for _ in kinds]
        entry=rng.randrange(n);code=hp(0,offsets[entry]);graph={}
        for i,op in enumerate(kinds):
            graph[i]=[] if op in ('RET','EFFECT_RESUME') else ([targets[i]] if op=='JMP' else [i+1,targets[i]] if op.startswith('JMP') else [i+1])
            code+=instruction(op,offsets[targets[i]]-offsets[i]) if widths[i]==5 else instruction(op)
        seen=set();pending=[entry]
        while pending:
            i=pending.pop()
            if i in seen:continue
            seen.add(i);pending.extend(graph.get(i,[]))
        add(f'random-labelled-cfg-{k}',code,report([len(code)],[(0,0,offsets[entry],0,0,{offsets[i] for i in seen})]))
    rows += [('short-packet',b'','0'),('huge-count',struct.pack('<II',0xffffffff,0),'0'),
             ('huge-name-count',struct.pack('<II',0,0xffffffff),'0'),
             ('short-name',struct.pack('<III',0,1,2)+b'x','0')]
    return rows

class EffectFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory(prefix='nano-effect-flow-');cls.addClassCleanup(cls.workspace.cleanup)
        cls.directory=Path(cls.workspace.name);cls.fixture=ROOT/'tests/nvm2c_effect_flow_driver.nano'
        cls.routes=build_routes(cls.fixture,cls.directory);cls.references=[]
        for mode,flags in [('ordinary',['-O2']),('sanitized',['-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all'])]:
            binary=cls.directory/('reference-'+mode)
            checked([*shlex.split(os.environ.get('CC','cc')),'-std=c11','-Wall','-Wextra','-Werror',*flags,
                ROOT/'tests/nvm2c_effect_flow_reference.c',ROOT/'src/nanoisa/isa.c',ROOT/'src/nanovm/vm_decode.c',
                *native_link_flags(),'-o',binary],build=True);cls.references.append(binary)
    def test_complete_handler_matrix(self):
        rows=cases();paths=[]
        for index,(label,data,want) in enumerate(rows):
            path=self.directory/f'{index:04d}-{label}.bin';path.write_bytes(data);paths.append(path)
        for start in range(0,len(rows),128):
            chunk=paths[start:start+128];expected=checked([self.references[0],*chunk]);values=expected.splitlines()
            self.assertEqual(len(values),len(chunk))
            for (label,_,want),value in zip(rows[start:start+128],values):
                if want is not None:self.assertEqual(value,want,label)
            for command in [[self.references[1]],*[cmd for _,cmd in self.routes]]:
                self.assertEqual(checked([*command,*chunk]),expected)
    def test_imported_shadows(self):
        binary=self.directory/'imported-native';checked([ROOT/'bin/nanoc_c',self.fixture,'--test-imports','-o',binary],build=True)
        checked([binary])
        for name,command in [('c',[ROOT/'bin/nano_virt']),('nanolang',source_compiler_command())]:
            if not command:continue
            module=self.directory/('imported-'+name+'.nvm');checked([*command,self.fixture,'--test-imports','--emit-nvm','-o',module],build=True)
            checked([ROOT/'bin/nano_vm','--verify-only',module]);checked([ROOT/'bin/nano_vm',module,'--'])
