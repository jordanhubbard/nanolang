"""I compare the complete bounded C decision without granting executable admission."""
from pathlib import Path
import ast
import random
import re
import shlex
import os
import struct
import tempfile
import unittest
import yaml
from tests.nanolang_translator_routes import build_routes, checked, source_compiler_command
from tests.native_process import native_link_flags

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=yaml.safe_load((ROOT/'spec/nanoisa.yaml').read_bytes())
OPS={v['name']:v for v in SCHEMA['legacy_opcodes']}
WIDTH={'U8':1,'U16':2,'U32':4,'I32':4,'I64':8,'F64':8}
FMT={'U8':'B','U16':'H','U32':'I','I32':'i','I64':'q','F64':'Q'}

def instruction(name,*args):
    row=OPS[name]
    return bytes([row['code']])+b''.join(struct.pack('<'+FMT[k],v) for k,v in zip(row['operands'],args))

def packet(code,functions=None,imports=(),current=0,query=0,locals_=0,results=0):
    if functions is None:functions=[(0,len(code),0,locals_,results)]
    return (struct.pack('<4I',len(functions),len(imports),current,query)
            +b''.join(struct.pack('<5I',*f) for f in functions)
            +b''.join(struct.pack('<2I',*f) for f in imports)+code)

def conditional(prefix):
    # The fallthrough NOP is dead exactly when the condition is known true.
    return prefix+instruction('JMP_TRUE',6)+instruction('NOP')+instruction('HALT')

def cases():
    result=[]
    def add(label,code,query,want=None,**kw):result.append((label,packet(code,query=query,**kw),want))
    for boolean in (0,1,2,255):
        code=conditional(instruction('PUSH_BOOL',boolean));add(f'boolean-{boolean}',code,7,int(boolean!=0))
    for value in (0,1,-1,-2**63,2**63-1):
        code=conditional(instruction('PUSH_I64',value));add(f'integer-is-not-bool-{value}',code,14,0)
    for left in (-2**63,-1,0,1,2**63-1):
        for right in (-2**63,-1,0,1,2**63-1):
            for op in ('LT','I64_LT_S','EQ','NE'):
                prefix=instruction('PUSH_I64',left)+instruction('PUSH_I64',right)+instruction(op)
                want={'LT':left<right,'I64_LT_S':left<right,'EQ':left==right,'NE':left!=right}[op]
                add(f'{op}-{left}-{right}',conditional(prefix),len(prefix)+5,int(want))
    for op,row in sorted(OPS.items(),key=lambda p:p[1]['code']):
        raw=instruction(op,*([0]*len(row['operands'])))
        forbidden=(11<=row['code']<=15 or 22<=row['code']<=31 or 145<=row['code']<=152
                   or 165<=row['code']<=168 or 182<=row['code']<=185 or row['code']==107)
        code=instruction('HALT')+raw
        add(f'every-op-dead-{op}',code,1,0 if forbidden else None)
        # A nonselected body is globally decoded for exclusions, but its bad
        # branch/call targets are not selected-function decode obligations.
        code=instruction('HALT')+instruction('NOP')+raw
        add(f'every-op-other-{op}',code,1,0 if forbidden else 1,
            functions=[(0,2,0,0,0),(2,len(raw),0,0,0)])
        for cut in range(1,len(raw)):
            code=instruction('HALT')+instruction('NOP')+raw[:cut]
            add(f'other-truncated-{op}-{cut}',code,1,0,functions=[(0,2,0,0,0),(2,cut,0,0,0)])
    for op in ('ARR_PUSH','ARR_POP','ARR_REMOVE','ARR_SET','CALL','TAIL_CALL','CALL_INDIRECT','CALL_EXTERN','CALL_MODULE'):
        prefix=instruction('ARR_LITERAL',1,0)+instruction('STORE_LOCAL',0)+instruction('LOAD_LOCAL',0)+instruction('STORE_LOCAL',1)
        prefix+=instruction('LOAD_LOCAL',0)
        if op=='ARR_PUSH':prefix+=instruction('PUSH_I64',7)
        if op=='ARR_REMOVE':prefix+=instruction('PUSH_I64',0)
        if op=='ARR_SET':prefix+=instruction('PUSH_I64',0)+instruction('PUSH_I64',7)
        if op=='CALL_INDIRECT':prefix+=instruction('FUNCREF',0)
        if op in ('CALL','TAIL_CALL','CALL_EXTERN'):raw=instruction(op,0)
        elif op=='CALL_INDIRECT':raw=instruction(op,0,0)
        elif op=='CALL_MODULE':raw=instruction(op,0,0,0,0)
        else:raw=instruction(op)
        prefix+=raw
        prefix+=instruction('LOAD_LOCAL',1)+instruction('ARR_LEN')+instruction('PUSH_I64',0)+instruction('EQ')
        code=conditional(prefix)
        add(f'alias-invalidated-{op}',code,len(prefix)+5,int(op=='TAIL_CALL'),locals_=2,imports=[(0,0)])
    prefix=instruction('ARR_LITERAL',1,0)+instruction('STORE_LOCAL',0)+instruction('LOAD_LOCAL',0)+instruction('ARR_LEN')
    prefix+=instruction('PUSH_I64',0)+instruction('EQ')
    add('array-length-retained',conditional(prefix),len(prefix)+5,1,locals_=1)
    for count in (0,1,3,16):
        prefix=instruction('PUSH_I64',7)*count+instruction('ARR_LITERAL',1,count)+instruction('ARR_LEN')+instruction('PUSH_I64',count)+instruction('EQ')
        add(f'array-literal-length-{count}',conditional(prefix),len(prefix)+5,1)
    for code in (b'',b'\xff',instruction('POP'),instruction('PICK',0),instruction('JMP',1),instruction('CALL',99),instruction('CALL_EXTERN',99)):
        add('decode-stack-refusal-'+code.hex(),code,0,0)
    for code in (instruction('RET')+b'\0',instruction('HALT')+b'\0',instruction('JMP',5)+b'\0'):
        for query in range(len(code)+2):add(f'query-{code.hex()}-{query}',code,query,None)
    for bad in (0,0xffffffff):
        add(f'current-{bad}',b'\xa3\0',1,1 if bad==0 else 0,current=bad)
    add('bad-other-branch-is-not-resolved',b'\xa3\0'+instruction('JMP',1),1,1,functions=[(0,2,0,0,0),(2,5,0,0,0)])
    add('bad-other-call-is-not-resolved',b'\xa3\0'+instruction('CALL',0xffffffff),1,1,functions=[(0,2,0,0,0),(2,5,0,0,0)])
    add('uncovered-bytes-are-not-scanned',b'\xff\xa3\0\xff',1,1,functions=[(1,2,0,0,0)])
    rng=random.Random(114205)
    choices=['NOP','PUSH_I64','PUSH_BOOL','POP','DUP','SWAP','ROT3','EQ','NE','LT','I64_LT_S','NOT','LOAD_LOCAL','STORE_LOCAL','ARR_LEN']
    for i in range(80):
        code=b''
        for _ in range(rng.randrange(1,30)):
            name=rng.choice(choices);args=[]
            for kind in OPS[name]['operands']:args.append(rng.randrange(4) if kind!='I64' else rng.randrange(-3,4))
            code+=instruction(name,*args)
        code+=instruction('HALT')+instruction('NOP');add(f'random-{i}',code,len(code)-1,None,locals_=3)
    # I retain the earlier matrix as an exact prefix and complete metadata/join
    # countercases before accepting the component.
    for size in range(16):result.append((f'short-packet-{size}',bytes(size),0))
    for count,foreign in [(0xffffffff,0),(0,0xffffffff),(1,0)]:
        result.append((f'table-overrun-{count}-{foreign}',struct.pack('<4I',count,foreign,0,0),0))
    result.append(('no-function-table',packet(b'\xa3\0',functions=[]),0))
    for field,value in [(0,0xffffffff),(1,0xffffffff),(2,65536),(3,65536),(4,256)]:
        entry=[0,2,0,0,0];entry[field]=value
        add(f'function-domain-{field}',b'\xa3\0',1,0,functions=[tuple(entry)])
    for foreign in [[(65536,0)],[(0,256)]]:
        add(f'import-domain-{foreign[0]}',b'\xa3\0',1,0,imports=foreign)
    add('other-locals-do-not-use-selected-limit',b'\xa3\0',1,1,functions=[(0,2,0,0,0),(2,0,0,1025,255)])
    for arm in (0,1):
        prefix=instruction('LOAD_LOCAL',0)+instruction('JMP_TRUE',12)+instruction('PUSH_BOOL',1)+instruction('JMP',7)+instruction('PUSH_BOOL',arm)
        add(f'join-constants-{arm}',conditional(prefix),len(prefix)+5,arm,locals_=1)
    code=instruction('LOAD_LOCAL',0)+instruction('JMP_TRUE',19)+instruction('PUSH_I64',7)+instruction('JMP',5)+instruction('HALT')+instruction('NOP')
    add('incompatible-stack-join',code,len(code)-1,0,locals_=1)
    prefix=instruction('ARR_LITERAL',1,0)+instruction('STORE_LOCAL',0)+instruction('LOAD_LOCAL',0)+instruction('ARR_LEN')+instruction('STORE_LOCAL',1)
    prefix+=instruction('LOAD_LOCAL',0)+instruction('PUSH_I64',7)+instruction('ARR_PUSH')+instruction('POP')+instruction('LOAD_LOCAL',1)+instruction('PUSH_I64',0)+instruction('EQ')
    add('derived-integer-survives-array-mutation',conditional(prefix),len(prefix)+5,1,locals_=2)
    for value in (0,1,2,255):
        prefix=instruction('PUSH_BOOL',value)+instruction('NOT');add(f'not-bool-{value}',conditional(prefix),len(prefix)+5,int(value==0))
    sequences=[('dup',instruction('PUSH_BOOL',1)+instruction('DUP')+instruction('EQ'),1),
      ('swap',instruction('PUSH_I64',1)+instruction('PUSH_I64',2)+instruction('SWAP')+instruction('LT'),0),
      ('rot3',instruction('PUSH_I64',1)+instruction('PUSH_I64',2)+instruction('PUSH_I64',3)+instruction('ROT3')+instruction('POP')+instruction('LT'),0),
      ('pick',instruction('PUSH_I64',1)+instruction('PUSH_I64',2)+instruction('PICK',1)+instruction('SWAP')+instruction('LT'),1),
      ('normalized-bools',instruction('PUSH_BOOL',255)+instruction('PUSH_BOOL',2)+instruction('EQ'),1),
      ('unknown-string-equality',instruction('PUSH_STR',0)+instruction('PUSH_STR',0)+instruction('EQ'),0),
      ('unmodeled-arithmetic',instruction('PUSH_I64',1)+instruction('PUSH_I64',1)+instruction('I64_SUB')+instruction('PUSH_I64',0)+instruction('EQ'),0)]
    for label,prefix,want in sequences:add(label,conditional(prefix),len(prefix)+5,want)
    for query,want in ((5,1),(6,0),(4,0)):
        add(f'terminal-branch-query-{query}',instruction('JMP',6)+instruction('NOP'),query,want)
    for name,row in sorted(OPS.items(),key=lambda p:p[1]['code']):
        raw=instruction(name,*([0]*len(row['operands'])))
        code=instruction('PUSH_I64',0)*16+raw+instruction('HALT')+instruction('NOP')
        add(f'every-op-live-{name}',code,len(code)-1,None,locals_=1,imports=[(0,0)])
    for op in sorted(set(range(256))-{row['code'] for row in OPS.values()}):
        add(f'unknown-primary-selected-{op}',bytes([163,op]),1,0)
        add(f'unknown-primary-other-{op}',bytes([163,0,op]),1,0,functions=[(0,2,0,0,0),(2,1,0,0,0)])
    for name,prefix,kwargs in [
        ('array-new-remains-unknown',instruction('ARR_NEW',1)+instruction('ARR_LEN')+instruction('PUSH_I64',0)+instruction('EQ'),{}),
        ('declared-bool-is-not-constant',instruction('CALL_EXTERN',0),{'imports':[(0,4)]}),
        ('typed-equality-is-not-folded',instruction('PUSH_I64',1)+instruction('PUSH_I64',1)+instruction('I64_EQ'),{})]:
        add(name,conditional(prefix),len(prefix)+5,0,**kwargs)
    return result

def reverse_chain(extra_block,n=722):
    # Block i executes i times, including its first/final stable round.
    blocks=[];starts={};position=5
    for i in range(1,n+1):
        starts[i]=position
        width=1 if i==1 else 5
        if i==extra_block:width+=1
        blocks.append((i,width));position+=width
    code=instruction('JMP',starts[n])
    for i,width in blocks:
        if i==extra_block:code+=instruction('NOP')
        if i==1:code+=instruction('HALT')
        else:code+=instruction('JMP',starts[i-1]-len(code))
    return code+instruction('NOP'),position

def scan_visits(code):
    # I interpret this fixture's encoded CFG independently of its construction
    # formula, with no visit cap. No constants, branches or joins are guessed.
    pcs=[];edges={};pc=0
    while pc<len(code):
        pcs.append(pc);op=code[pc]
        if op==0:edges[pc]=pc+1;pc+=1
        elif op==163:edges[pc]=None;pc+=1
        elif op==56:edges[pc]=pc+struct.unpack_from('<i',code,pc+1)[0];pc+=5
        else:raise AssertionError(op)
    boundaries=set(pcs)|{len(code)};assert all(v is None or v in boundaries for v in edges.values())
    seen={0};visits=0
    while True:
        changed=False
        for pc in pcs:
            if pc not in seen:continue
            visits+=1;target=edges[pc]
            if target is not None and target not in seen:seen.add(target);changed=True
        if not changed:return visits

def frame_charge(code,locals_):
    pc=0;stack=0;charge=0
    while pc<len(code):
        charge+=locals_+stack;op=code[pc]
        if op==163:return charge
        if op==0:pc+=1
        elif op==1:stack+=1;pc+=9
        else:raise AssertionError(op)
    raise AssertionError('I require the fixture HALT.')

def resource_cases():
    cases=[]
    for count in (65535,65536,65537):
        code=instruction('HALT')+instruction('NOP')*(count-1)
        cases.append((f'instructions-{count}',packet(code,query=1),int(count<=65536)))
    for local in (1023,1024,1025):
        cases.append((f'locals-{local}',packet(b'\xa3\0',query=1,locals_=local),int(local<=1024)))
    # 1025 frames * 1023 locals = 1048575 constants. The last one/two stack
    # slots give the exact 16MiB boundary and one constant beyond it.
    for extra in range(3):
        code=instruction('NOP')*1024 if extra==0 else instruction('NOP')*(1024-extra)+instruction('PUSH_I64',7)+instruction('NOP')*(extra-1)
        code+=instruction('HALT');query=len(code);code+=instruction('NOP')
        assert frame_charge(code,1023)==1048575+extra
        cases.append((f'frame-constants-{1048575+extra}',packet(code,query=query,locals_=1023),int(extra<=1)))
    # I retain both wrongly named draft packets, correcting their expectation
    # only after an uncapped encoded-CFG count proves they are below the cap.
    for block,old_name in [(419,262144),(420,262145)]:
        code,query=reverse_chain(block,n=721);actual=scan_visits(code)
        assert actual==261002+block and actual<262144
        cases.append((f'draft-visits-{old_name}-actual-{actual}',packet(code,query=query),1))
    for block in (419,420):
        code,query=reverse_chain(block)
        expected_visits=722*725//2+block
        assert scan_visits(code)==expected_visits and expected_visits in (262144,262145)
        cases.append((f'visits-{expected_visits}',packet(code,query=query),int(expected_visits<=262144)))
    return cases

class ConstantReachability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory(prefix='nano-constant-reachability-');cls.addClassCleanup(cls.workspace.cleanup)
        cls.directory=Path(cls.workspace.name);cls.fixture=ROOT/'tests/nvm2c_constant_reachability_driver.nano'
        cls.routes=build_routes(cls.fixture,cls.directory);cls.references=[]
        for mode,flags in [('ordinary',['-O2']),('sanitized',['-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all'])]:
            binary=cls.directory/('reference-'+mode)
            checked([*shlex.split(os.environ.get('CC','cc')),'-std=c11','-Wall','-Wextra','-Werror',*flags,
                ROOT/'tests/nvm2c_constant_reachability_reference.c',ROOT/'src/nanoisa/isa.c',ROOT/'src/nanovm/vm_decode.c',
                *native_link_flags(),'-o',binary],build=True);cls.references.append(binary)
    def compare(self,rows,batch=16):
        paths=[]
        for i,(label,data,want) in enumerate(rows):
            path=self.directory/(label+'.bin');path.write_bytes(data);paths.append(path)
        for start in range(0,len(rows),batch):
            chunk=paths[start:start+batch];expected=checked([self.references[0],*chunk]);values=expected.splitlines()
            self.assertEqual(len(values),len(chunk))
            for (label,_,want),value in zip(rows[start:start+batch],values):
                self.assertIn(value,('0','1'))
                if want is not None:self.assertEqual(value,str(want),label)
            for command in [[self.references[1]],*[cmd for _,cmd in self.routes]]:
                self.assertEqual(checked([*command,*chunk]),expected)
    def test_full_decision_matrix(self):self.compare(cases())
    def test_original_classifier_inputs(self):
        source=(ROOT/'tests/nanoisa/test_nvm2c.c').read_text();source=source.split('static void test_classifier_unreachable_and_invalid_joins(void) {',1)[1]
        literals=source.split('const char *sources[] = {',1)[1].split('};',1)[0]
        groups=re.findall(r'((?:"(?:[^"\\]|\\.)*"\s*)+)(?:,|$)',literals.strip());self.assertEqual(len(groups),3)
        rows=[]
        for i,group in enumerate(groups):
            text=''.join(ast.literal_eval(s) for s in re.findall(r'"(?:[^"\\]|\\.)*"',group))
            assembly=self.directory/f'original-{i}.nasm';module=self.directory/f'original-{i}.nvm';assembly.write_text(text)
            checked([ROOT/'bin/nanoisa','asm',assembly,'-o',module],build=True);raw=module.read_bytes()
            self.assertEqual(raw[:4],b'NVM\x02');directory,count=struct.unpack_from('<II',raw,20);sections={}
            for k in range(count):
                kind,flags,at,size=struct.unpack_from('<IIQQ',raw,directory+k*24);sections[kind]=raw[at:at+size]
            fn=sections[5];self.assertEqual(struct.unpack_from('<I',fn)[0],1)
            _,signature,begin,length,locals_,upvalues,maximum,flags=struct.unpack_from('<IIQQHHHH',fn,4)
            signatures=sections[3];arity,results=struct.unpack_from('<HH',signatures,4);self.assertEqual(signature,0)
            code=bytearray(sections[6]);pc=0;constants=0
            while pc<len(code):
                op=next(v for v in OPS.values() if v['code']==code[pc]);width=1+sum(WIDTH[k] for k in op['operands'])
                if op['name']=='PUSH_I64':
                    constants+=1
                    if i and constants==2:
                        code[pc:pc+width]=bytes(width)
                        if i==2:code[pc:pc+5]=instruction('PUSH_STR',0)
                        break
                pc+=width
            for query in range(length+1):
                want=int(i==0 and query in (10,11))
                rows.append((f'original-{i}-query-{query}',packet(bytes(code),functions=[(begin,length,arity,locals_,results)],query=query),want))
        self.compare(rows)
    def test_exact_resource_bounds(self):self.compare(resource_cases(),batch=1)
    def test_imported_shadows(self):
        binary=self.directory/'imported-native';checked([ROOT/'bin/nanoc_c',self.fixture,'--test-imports','-o',binary],build=True)
        checked([binary])
        for name,command in [('c',[ROOT/'bin/nano_virt']),('nanolang',source_compiler_command())]:
            if not command:continue
            module=self.directory/('imported-'+name+'.nvm');checked([*command,self.fixture,'--test-imports','--emit-nvm','-o',module],build=True)
            checked([ROOT/'bin/nano_vm','--verify-only',module]);checked([ROOT/'bin/nano_vm',module,'--'])
