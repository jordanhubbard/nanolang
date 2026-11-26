"""I exercise the whole callable collector/linker and retain its original C suite."""
from pathlib import Path
import os
import random
import re
import shlex
import struct
import sys
import tempfile
import unittest
import yaml
from tests.nanolang_translator_routes import build_routes, checked, source_compiler_command
from tests.native_process import native_link_flags

ROOT=Path(__file__).resolve().parents[1]
OPS={v['name']:v for v in yaml.safe_load((ROOT/'spec/nanoisa.yaml').read_bytes())['legacy_opcodes']}
FMT={'U8':'B','U16':'H','U32':'I','I32':'i','I64':'q','F64':'Q'}

def instruction(name,*args):
    row=OPS[name];assert len(args)==len(row['operands']),(name,args,row['operands'])
    return bytes([row['code']])+b''.join(struct.pack('<'+FMT[k],v) for k,v in zip(row['operands'],args))

def packet(code,functions=None,names=(),imports=()):
    if functions is None:functions=[(0,len(code),0,0,0,0)]
    return (struct.pack('<III',len(functions),len(names),len(imports))
            +b''.join(struct.pack('<6I',*f) for f in functions)
            +b''.join(struct.pack('<5I',library,symbol,len(types),result,int(present))
                       +(b''.join(struct.pack('<I',t) for t in types) if present else b'')
                       for library,symbol,types,result,present in imports)
            +b''.join(struct.pack('<I',len(n))+n for n in names)+code)

def diagnostics(value):
    # C argument evaluation can allocate graph node IDs in a different order.
    # I preserve the complete raw report and compare every diagnostic byte except
    # those private IDs. Function indexes/byte offsets/counts remain exact.
    if value.startswith(('0|I cannot convert aggregate storage ',
                         '0|I cannot widen an exactly constrained ',
                         '0|I found conflicting aggregate shape kinds ')):
        value=re.sub(r' at nodes \d+/\d+', ' at nodes ID/ID', value)
        value=re.sub(r' \(conversion \d+/\d+\)', ' (conversion ID/ID)',value)
    return value

def reports(value):
    # I retain both the unresolved marker and ordered module-local targets.
    if not value.startswith('1|'):return None
    result={}
    for part in value.split('|')[1:]:
        position,mask,targets=part.split(':');f,pc=map(int,position.split(','))
        result[f,pc]=(int(mask),None if targets=='-' else tuple(map(int,targets.split(',')[1:])))
    return result

def labelled(items):
    offsets={};position=0
    for item in items:
        if isinstance(item,str):offsets[item]=position
        else:position+=len(instruction(item[0],*([0]*len(item[1:]))))
    code=b''
    for item in items:
        if isinstance(item,str):continue
        args=[offsets[v]-len(code) if isinstance(v,str) else v for v in item[1:]]
        code+=instruction(item[0],*args)
    return code,offsets

def cases():
    rows=[]
    def add(name,code,want=None,functions=None,**kwargs):rows.append((name,packet(code,functions,**kwargs),want))
    add('no-functions',b'',('error','I require callable module functions'),functions=[])
    add('empty-body',b'',('ok',))
    for op,row in sorted(OPS.items(),key=lambda p:p[1]['code']):
        raw=instruction(op,*([0]*len(row['operands'])))
        add('live-opcode-'+op,raw+instruction('HALT'))
        add('dead-opcode-'+op,instruction('HALT')+raw)
        pops=row.get('pops',-1)
        if 0<=pops<=3:
            add('live-stack-opcode-'+op,instruction('PUSH_VOID')*pops+raw+instruction('RET'))
        for cut in range(1,len(raw)):
            add(f'truncated-{op}-{cut}',instruction('HALT')+raw[:cut],('error',None))
    for index in (0,1,4095,4096,0xffffffff):
        raw=instruction('LOAD_GLOBAL',index)+instruction('POP')+instruction('RET')
        add(f'global-boundary-{index}',raw,('error',None) if index>=4096 else ('ok',))
    for code in (instruction('POP'),instruction('LOAD_LOCAL',1),instruction('LOAD_UPVALUE',1,0),
                 instruction('PICK',0),instruction('CALL_EXTERN',0),instruction('FUNCREF',99),
                 instruction('JMP',1),instruction('HANDLER_PUSH',0,13,0,0),b'\xff'):
        add('refusal-'+code.hex(),code,('error',None))
    add('join-mismatch',instruction('PUSH_BOOL',0)+instruction('JMP_FALSE',14)+instruction('PUSH_I64',1)+instruction('RET'),('error','I require equal callable stack heights at joins'))
    for arity,results in ((0,0),(1,0),(0,1),(1,1),(0,2)):
        prefix=instruction('PUSH_I64',7)*arity+instruction('FUNCREF',1)
        raw=prefix+instruction('CALL_INDIRECT',arity,results)+instruction('HALT')
        leaf=instruction('PUSH_I64',8)+instruction('RET')
        add(f'indirect-signature-{arity}-{results}',raw+leaf,
            ('callee',len(prefix),(1,)) if (arity,results)==(0,1) else ('error',None),
            functions=[(0,len(raw),0,0,0,0),(len(raw),len(leaf),0,0,0,1)])
    # An unknown/nonfunction callee remains unresolved; no invented target is admitted.
    raw=instruction('PUSH_I64',9)+instruction('CALL_INDIRECT',0,0)+instruction('RET')
    add('unresolved-scalar',raw,('unresolved',9))
    # Globals and closure captures are collected before later bodies add producers.
    first=instruction('FUNCREF',2)+instruction('STORE_GLOBAL',0)+instruction('PUSH_I64',7)+instruction('LOAD_GLOBAL',0)+instruction('CALL_INDIRECT',1,1)+instruction('POP')+instruction('RET')
    leaf=instruction('LOAD_LOCAL',0)+instruction('RET')
    add('target-zero-and-late-global',first+leaf+leaf,('callee',len(instruction('FUNCREF',2)+instruction('STORE_GLOBAL',0)+instruction('PUSH_I64',7)+instruction('LOAD_GLOBAL',0)),(2,)),functions=[(0,len(first),0,0,0,0),(len(first),len(leaf),1,1,0,1),(len(first)+len(leaf),len(leaf),1,1,0,1)])
    # I transmit producer tags through record fields, including tags with no exact shape.
    for op,args,mask in [('PUSH_I64',(7,),2),('PUSH_F64',(0,),8),('PUSH_BOOL',(1,),16),('PUSH_STR',(0,),32),('PUSH_U8',(255,),4),('PUSH_VOID',(),1)]:
        prefix=instruction(op,*args)+instruction('AGG_PACK',0,0,0,1)
        raw=prefix+instruction('AGG_GET',0)+instruction('POP')+instruction('RET')
        add('projection-'+op,raw,('projection',len(prefix),mask),names=[b'hello'])
    for symbol,parameter,result,want in [('vm_array_sort',7,7,128),('vm_bytes_from_string',5,7,4),('vm_string_from_bytes',7,5,32)]:
        for present in (False,True):
            for library in (b'',b'foreign'):
                prefix=instruction('PUSH_STR',0)+instruction('CALL_EXTERN',0)
                if symbol!='vm_string_from_bytes':
                    prefix+=instruction('PUSH_I64',0)+instruction('ARR_GET')
                raw=prefix+instruction('AGG_PACK',0,0,0,1)+instruction('AGG_GET',0)+instruction('POP')+instruction('RET')
                expected=want if present and not library else (32768 if symbol=='vm_string_from_bytes' else 0)
                if symbol=='vm_array_sort':expected=0
                add(f'import-{symbol}-{present}-{bool(library)}',raw,('projection',len(prefix)+len(instruction('AGG_PACK',0,0,0,1)),expected),
                    names=[b'x',library,symbol.encode()],imports=[(1,2,[parameter],result,present)])
    for handler_count in (0,1):
        for perform_name in (0,2):
            main,labels=labelled([
                ('HANDLER_PUSH',1,'arm',0,handler_count),('FUNCREF',1),('PERFORM',perform_name,1),
                'call',('CALL_INDIRECT',0,0),('HANDLER_POP',1),('RET',),
                'arm',('LOAD_LOCAL',0),('EFFECT_RESUME',)])
            add(f'effect-argument-result-{handler_count}-{perform_name}',main+instruction('RET'),
                ('callee',labels['call'],(1,)) if handler_count==1 and perform_name==0 else ('unresolved',labels['call']),
                functions=[(0,len(main),0,1,0,0),(len(main),1,0,0,0,0)],names=[b'op',b'op\0tail',b'other'])
    main,labels=labelled([
        ('HANDLER_PUSH',0,'first-arm',0,0),('HANDLER_PUSH',0,'second-arm',0,0),('PERFORM',0,0),
        'call',('CALL_INDIRECT',0,0),('HANDLER_POP',1),('HANDLER_POP',1),('RET',),
        'first-arm',('FUNCREF',1),('EFFECT_RESUME',),'second-arm',('FUNCREF',2),('EFFECT_RESUME',)])
    add('effect-multiple-arm-results',main+instruction('RET')*2,('callee',labels['call'],(1,2)),
        functions=[(0,len(main),0,0,0,0),(len(main),1,0,0,0,0),(len(main)+1,1,0,0,0,0)],names=[b'op'])
    main,labels=labelled([
        ('HANDLER_PUSH',0,'arm',0,1),('FUNCREF',2),('CALL',1),'call',('CALL_INDIRECT',0,0),
        ('HANDLER_POP',1),('RET',),'arm',('LOAD_LOCAL',0),('EFFECT_RESUME',)])
    performer=instruction('LOAD_LOCAL',0)+instruction('PERFORM',0,1)+instruction('RET')
    add('effect-cross-function-perform',main+performer+instruction('RET'),('callee',labels['call'],(2,)),
        functions=[(0,len(main),0,1,0,0),(len(main),len(performer),1,1,0,1),(len(main)+len(performer),1,0,0,0,0)],names=[b'op'])
    for functions in [[(1,1,0,0,0,0)],[(0,2,0,0,0,0)],[(0,1,1,0,0,0)]]:
        add('metadata-'+str(len(rows)),instruction('RET'),('error',None),functions=functions)
    rng=random.Random(114212)
    for i in range(100):
        # Each branch stores an independently selected function into a shared global.
        left,right=rng.randrange(1,5),rng.randrange(1,5)
        prefix=instruction('PUSH_BOOL',rng.randrange(2))+instruction('JMP_FALSE',20)
        prefix+=instruction('FUNCREF',left)+instruction('STORE_GLOBAL',0)+instruction('JMP',15)
        prefix+=instruction('FUNCREF',right)+instruction('STORE_GLOBAL',0)
        main=prefix+instruction('LOAD_GLOBAL',0)+instruction('CALL_INDIRECT',0,0)+instruction('RET')
        bodies=main+instruction('RET')*4
        fns=[(0,len(main),0,0,0,0)]+[(len(main)+k,1,0,0,0,0) for k in range(4)]
        add(f'random-global-join-{i}',bodies,('callee',len(prefix)+5,tuple(sorted({left,right}))),functions=fns)
    rows.extend([('short-packet',b'',('packet',)),('oversized-count',struct.pack('<III',0xffffffff,0,0),('packet',))])
    return rows

class CallableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory(prefix='nano-callables-');cls.addClassCleanup(cls.workspace.cleanup)
        cls.directory=Path(cls.workspace.name);cls.fixture=ROOT/'tests/nvm2c_callables_driver.nano'
        cls.routes=build_routes(cls.fixture,cls.directory);cls.references=[]
        for mode,flags in [('ordinary',['-O2']),('sanitized',['-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all'])]:
            binary=cls.directory/('reference-'+mode)
            checked([*shlex.split(os.environ.get('CC','cc')),'-std=c11','-Wall','-Wextra','-Werror','-D_GNU_SOURCE','-ffunction-sections','-fdata-sections',*flags,
                ROOT/'tests/nvm2c_callables_reference.c',ROOT/'src/nanoisa/nvm2c_callables.c',ROOT/'src/nanoisa/nvm2c_shape.c',ROOT/'src/nanoisa/isa.c',
                ROOT/'src/nanovm/vm_decode.c',ROOT/'src/nanoisa/nvm_format.c',
                *(['-Wl,--gc-sections'] if sys.platform.startswith('linux') else ['-Wl,-dead_strip']),
                *native_link_flags(),'-o',binary],build=True);cls.references.append(binary)
    def compare(self,rows):
        paths=[]
        for i,(label,data,want) in enumerate(rows):
            path=self.directory/f'{label}.bin';path.write_bytes(data);paths.append(path)
        for start in range(0,len(rows),64):
            chunk=paths[start:start+64];expected=checked([self.references[0],*chunk]);values=expected.splitlines()
            self.assertEqual(len(values),len(chunk))
            for (label,_,want),value in zip(rows[start:start+64],values):
                if want is None:continue
                if want[0]=='packet':self.assertEqual(value,'P',label)
                elif want[0]=='error':
                    self.assertTrue(value.startswith('0|'),(label,value))
                    if want[1] is not None:self.assertEqual(value,'0|'+want[1],label)
                elif want[0]=='ok':self.assertTrue(value.startswith('1|'),(label,value))
                else:
                    observed=reports(value);self.assertIsNotNone(observed,(label,value))
                    mask,targets=observed[0,want[1]]
                    if want[0]=='projection':self.assertEqual(mask,want[2],label)
                    elif want[0]=='callee':self.assertEqual(targets,want[2],label)
                    elif want[0]=='unresolved':self.assertIsNone(targets,label)
            for command in [[self.references[1]],*[cmd for _,cmd in self.routes]]:
                actual=checked([*command,*chunk]).splitlines()
                self.assertEqual(list(map(diagnostics,actual)),list(map(diagnostics,values)))
    def test_full_collector_matrix(self):self.compare(cases())
    def test_original_callable_suite(self):
        make=(ROOT/'Makefile.gnu').read_text().replace('\\\n',' ')
        source_line=re.search(r'^NANOISA_SOURCES = (.*)$',make,re.M)[1]
        objects=[ROOT/'obj/nanoisa'/(Path(v).stem+'.o') for v in source_line.split()]
        objects += [ROOT/'obj/nanoisa/nanoisa_facade.o',ROOT/'obj/nanovm/vm_decode.o',ROOT/'obj/nanovm/vm_dispatch.o',ROOT/'lib/libnano_compiler_inputs.a',ROOT/'obj/utf8.o']
        for p in objects:self.assertTrue(p.is_file(),p)
        binary=self.directory/'original-callables';corpus=self.directory/'original.packets'
        checked([*shlex.split(os.environ.get('CC','cc')),'-std=c11','-Wall','-Wextra','-D_GNU_SOURCE','-Isrc','-Isrc/nanoisa',
            ROOT/'tests/nvm2c_callables_original_trace.c',*objects,'-lm','-lcrypto',*native_link_flags(),'-o',binary],build=True)
        output=checked([binary,corpus]);self.assertIn('505 passed, 0 failed',output)
        raw=corpus.read_bytes();position=0;rows=[]
        while position<len(raw):
            length=struct.unpack_from('<I',raw,position)[0];position+=4;self.assertLessEqual(length,len(raw)-position)
            rows.append((f'original-{len(rows)}',raw[position:position+length],None));position+=length
        self.assertGreater(len(rows),15);self.compare(rows)
    def test_imported_shadows(self):
        binary=self.directory/'imported-native';checked([ROOT/'bin/nanoc_c',self.fixture,'--test-imports','-o',binary],build=True);checked([binary])
        for name,command in [('c',[ROOT/'bin/nano_virt']),('nanolang',source_compiler_command())]:
            if not command:continue
            module=self.directory/('imported-'+name+'.nvm');checked([*command,self.fixture,'--test-imports','--emit-nvm','-o',module],build=True)
            checked([ROOT/'bin/nano_vm','--verify-only',module]);checked([ROOT/'bin/nano_vm',module,'--'])
