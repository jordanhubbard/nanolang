"""I compare preflight boundaries; the C body continues with its original visit bound."""
from itertools import product
from pathlib import Path
import hashlib
import json
import os
import struct
import tempfile
import unittest
from tests import nanolang_translator_routes as routes
from tests.test_nanolang_translator_affine_frame import words, flow_cases, queue_cases, frame_cases
from tests.test_nanolang_translator_affine_scalar import scalar_batches
from tests.test_nanolang_translator_ownership import desc, record, function, image, extension, effect_data, globals_data

ROOT=Path(__file__).resolve().parents[1]
NO=0xffffffff
DECL='I require checked ownership declarations'
SIZE='I exceeded my bounded affine analysis size'
COUNT='I exceeded my bounded affine instruction count'
DECODE='I require decodable function control flow'
PROFILE='I require a connected affine instruction contract'
LITERAL='I require string literals inside an owned value-call graph'
OUTPUT='I require string output inside an owned value-call graph'
EFFECT='I require a retained exact effect signature at every operation'
SLOTS='I require bounded handler parameter slots'
TYPES='I require exact declared handler parameter types'
GLOBAL='I require an exact declared global slot'
GRAPH='I require a bounded acyclic value graph for global flow'
CAPACITY='I require exact instruction fact capacity'
CALLER='I require exact caller-origin parameter authority'


def preflight_model(code=b'\x3d',*,slots=(),parameters=0,globals_=0,effect=None,result=1,own=None,declared_length=None,declared_locals=None):
    extensions=[]
    if globals_:extensions.append(extension(3,globals_data([struct.pack('<BBHI',1,1,0,NO)]*globals_)))
    if effect is not None:extensions.append(extension(4,effect_data(1,operations=[(0,desc(1),[desc(t) for t in effect])])))
    locals_=len(slots) if declared_locals is None else declared_locals
    ownership=record(3 if extensions else 1,functions=[(len(slots),parameters,desc(result),[desc(*s) for s in slots])],extensions=extensions) if own is None else own
    return image(ownership,strings=[b'op'],functions=[function(locals_,parameters,results=int(result!=0),result_tag=result,
        params=[s[0] for s in slots[:parameters]],length=len(code) if declared_length is None else declared_length)],code=code)


def caller_model(mode):
    return image(record(1,flags=[3],functions=[(2,1,desc(),[desc(8,0),desc(1)]),(1,1,desc(),[desc(8,0,mode)])]),
        rows=[(0,NO,[(1,NO,NO)])],functions=[function(2,1,params=[8],length=1),function(1,1,params=[8],start=1,length=1)],code=b'\xa3\x3d')


def preflight_cases():
    cases=[]
    def add(name,carrier=None,*,present=True,functions_present=True,function_=0,caller=False,reference=0,valid=True,graph=True,effect_mode=False,
            facts=False,fact_function=0,capacity=0,borrow=0,globals_=0,error='',offset=0,count=1,slots=(),parameters=0,bound=False,functions=1):
        if carrier is None:carrier=preflight_model(globals_=globals_)
        raw=words(16,int(present),int(functions_present),function_,int(caller),reference,int(valid),int(graph),int(effect_mode),int(facts),fact_function,capacity,borrow,len(carrier))+carrier
        proceed=not error
        result=lambda e='',o=0:f'0,{o},0,0,{e}'
        states=[0]*8;results=[result()]*8;allowed=[[0]*globals_ for _ in range(8)]
        entered=present and functions_present and function_<functions and valid and (not globals_ or graph)
        if graph and not caller and function_<8:
            states[function_]=1 if proceed else 2
            if not proceed:results[function_]=result(error,offset)
            if entered:allowed[function_]=[3]*globals_
        nums=lambda values:''.join(f'{int(n)},' for n in values)
        text=f'{int(proceed)}|'+result(error,offset)+f'|{int(valid)},{int(graph)},{int(effect_mode)},{globals_}|'
        text+='1,1,4294967295,;'*globals_+'|'
        for f in range(8):text+=f'{states[f]}~'+results[f]+'~'+nums(allowed[f])+'/'+nums([0]*globals_)+'/#'
        if not proceed:text+='|-'
        else:
            limit=count*(2*len(slots)+4*globals_+1+256*257)
            text+='|'+nums([limit,count,0,0,0,0])+'|'+nums([9]*globals_)+'|'
            for i,(tag,layout,mode) in enumerate(slots):
                live=i<parameters and not(bound and mode)
                parameter=i<parameters and mode!=0
                read=parameter;write=parameter and mode==2
                text+=nums([live,tag if live else 0,mode if live else 0,65535,int(mode==0),tag if mode==0 else 0,layout if mode==0 else 0,
                    parameter,tag if parameter else 0,layout if parameter else 0,mode if parameter else 0,
                    read,1 if read else 0,write,1 if write else 0])+';'
        cases.append((name,raw,text))
    for present,functions_present,valid in product((False,True),repeat=3):
        if present and functions_present and valid:continue
        add(f'entry-{present}-{functions_present}-{valid}',present=present,functions_present=functions_present,valid=valid,error=DECL)
    for f in (1,7,8,NO):add(f'function-{f}',function_=f,error=DECL)
    for graph,effect_mode,globals_ in product((False,True),(False,True),(0,1,3)):
        # HALT is a complete terminal body; these manually supplied private context flags do not establish graph validity.
        add(f'context-{graph}-{effect_mode}-{globals_}',preflight_model(b'\xa3',globals_=globals_),graph=graph,effect_mode=effect_mode,globals_=globals_,error=GRAPH if globals_ and not graph else '')
    add('local-size',preflight_model(declared_locals=257),error=SIZE)
    add('code-size',preflight_model(declared_length=131073),error=SIZE)
    add('missing-owned-declarations',preflight_model(own=b''),error=DECL)
    for name,body in [('unknown',b'\xff'),('short',b'\x01'),('interior',b'\x38'+words(2)+b'\x3d')]:
        add('decode-'+name,preflight_model(body),error=DECODE)
    add('empty-code',preflight_model(b''),error=COUNT)
    add('too-many-instructions',preflight_model(b'\0'*4097),error=COUNT)
    add('maximum-instructions',preflight_model(b'\0'*4095+b'\xa3'),count=4096)
    for opcode,error in [(64,PROFILE),(4,LITERAL),(160,OUTPUT),(164,OUTPUT)]:
        body=b'\0'+bytes([opcode])+(words(0) if opcode==4 else b'')+b'\x3d'
        add(f'profile-{opcode}',preflight_model(body),graph=False,error=error,offset=1)
    add('profile-before-global',preflight_model(b'\x12'+words(0)+b'\x40\x3d'),error=PROFILE,offset=5)
    add('effect-before-global',preflight_model(b'\xa7'+words(0)+b'\0\0\x12'+words(0)+b'\x3d'),error=EFFECT)
    add('global-before-effect',preflight_model(b'\0\x12'+words(0)+b'\xa7'+words(0)+b'\0\0\x3d'),error=GLOBAL,offset=1)
    for opcode,slot,globals_ in product((18,19),(0,1,NO),(0,1)):
        body=b'\0'+bytes([opcode])+words(slot)+b'\xa3'
        add(f'global-{opcode}-{slot}-{globals_}',preflight_model(body,globals_=globals_),globals_=globals_,count=3,error=GLOBAL if slot>=globals_ else '',offset=1 if slot>=globals_ else 0)
    for enabled,fact_function,capacity in product((False,True),(0,1),(0,1,2,NO)):
        add(f'facts-{enabled}-{fact_function}-{capacity}',facts=enabled,fact_function=fact_function,capacity=capacity,error=CAPACITY if enabled and fact_function==0 and capacity!=1 else '')
    for declared,actual,first,arity in product((1,2,3,4),(1,2,3,4),(0,1),(0,1,2)):
        slots=((declared,NO,0),)
        body=b'\xa5'+words(0)+struct.pack('<iHH',13,first,arity)+b'\x3d'
        error=EFFECT if arity!=1 else SLOTS if first else TYPES if declared!=actual else ''
        add(f'handler-{declared}-{actual}-{first}-{arity}',preflight_model(body,slots=slots,effect=[actual]),slots=slots,count=2,error=error)
    for actual,arity in product((1,2,3,4),(0,1,2)):
        body=b'\xa7'+words(0)+struct.pack('<H',arity)+b'\x3d'
        add(f'perform-{actual}-{arity}',preflight_model(body,effect=[actual]),count=2,error=EFFECT if arity!=1 else '')
    for mode,borrow,reference in product((1,2),(0,1,2),(0,1,2,NO)):
        accepted=reference==1 and borrow!=0 and (mode==1 or borrow==2)
        add(f'caller-{mode}-{borrow}-{reference}',caller_model(mode),function_=1,functions=2,caller=True,reference=reference,borrow=borrow,
            slots=((8,0,mode),),parameters=1,bound=True,error='' if accepted else CALLER)
    for mode in (1,2):
        add(f'caller-capacity-bypass-{mode}',caller_model(mode),function_=1,functions=2,caller=True,reference=1,borrow=2,
            facts=True,fact_function=1,capacity=0,slots=((8,0,mode),),parameters=1,bound=True)
    good=cases[0][1]
    malformed=[b'',words(16),good+b'\0']+[good[:n] for n in (1,4,8,12,16,20,24,28,32,36,40,44,48,52,55,len(good)-1)]
    for index in (0,1,3,5,6,7,8,11):
        bad=bytearray(good);bad[4+index*4:8+index*4]=words(3);malformed.append(bytes(bad))
    cases.extend((f'malformed-{i}',raw,'D') for i,raw in enumerate(malformed))
    return cases


class AffinePreflight(unittest.TestCase):
    def test_original_frames_and_fresh_preflight(self):
        work=Path(tempfile.mkdtemp(prefix='nano-affine-preflight-'))
        old=flow_cases()[0]+queue_cases()+frame_cases();self.assertEqual(len(old),1001)
        reference=Path(os.environ.get('NANOLANG_AFFINE_PREFLIGHT_REFERENCE',ROOT/'obj/nvm2c_affine_preflight_reference'))
        manifest=dict(scope='I compare preflight and separate public-C entry queries; my C body continues with its original limit. No pure body/callee/whole-CFG proof.',cases=[],routes=[]);prepared=[]
        for group,cases in [('original',old),('preflight',preflight_cases())]:
            paths=[]
            for index,(name,raw,expected) in enumerate(cases):
                path=work/f'{group}-{index:05d}.bin';path.write_bytes(raw);paths.append(path)
                manifest['cases'].append(dict(group=group,name=name,path=str(path),sha256=hashlib.sha256(raw).hexdigest(),expected=expected))
            prepared.append((group,cases,paths))
        def save():(work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        save();reports={};products=[('actual-c',[reference])]+routes.build_routes(ROOT/'tests/nvm2c_affine_preflight_driver.nano',work)
        self.assertEqual(len(products),6 if routes.source_compiler_command() else 4)
        for label,command in products:
            for group,cases,paths in prepared:
                chunks=([paths[i:i+8] for i in range(0,len(paths),8)] if group=='original' else scalar_batches(paths));actual=[]
                for chunk in chunks:actual+=routes.checked([*command,*chunk],timeout=600 if group=='original' else 60).splitlines()
                self.assertEqual(len(actual),len(cases),(label,group));raw=('\n'.join(actual)+'\n').encode();(work/f'{group}-{label}.reports').write_bytes(raw)
                if group=='original':self.assertEqual(hashlib.sha256(raw).hexdigest(),'f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0')
                for (name,_,expected),text in zip(cases,actual):
                    with self.subTest(route=label,group=group,case=name):
                        if group=='original' and not name.startswith(('global-algebra-','bounded-fifo-')):self.assertTrue(text.startswith(expected))
                        else:self.assertEqual(text,expected)
                if group not in reports:reports[group]=actual
                else:self.assertEqual(actual,reports[group],(label,group))
                manifest['routes'].append(dict(label=label,group=group,cases=len(cases),sha256=hashlib.sha256(raw).hexdigest()));save()


if __name__=='__main__':unittest.main()
