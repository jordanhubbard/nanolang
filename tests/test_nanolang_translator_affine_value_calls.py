"""I compare value-call preparation and summary application, without deriving NanoLang callee proofs."""
from itertools import product
from pathlib import Path
import hashlib
import json
import os
import struct
import tempfile
import unittest

from tests import nanolang_translator_routes as routes
from tests.test_nanolang_translator_affine_frame import words, value, frame, flow_cases, queue_cases, frame_cases
from tests.test_nanolang_translator_affine_scalar import scalar_batches, stack_text
from tests.test_nanolang_translator_affine_owners import authority, authority_text, reference
from tests.test_nanolang_translator_ownership import desc, record, function, image, extension, effect_data, union_data, globals_data

ROOT=Path(__file__).resolve().parents[1]
NO=0xffffffff
CALL='I require a checked acyclic owned value call'
SHAPE='I require a complete consuming argument list'
ARGUMENT='I require exact positional consuming argument types'
RESOLUTION='I require complete consuming-helper owner resolution'
LOCAL='I cannot suspend live local owners across an escaping effect call'
STAGED='I cannot suspend staged owners across an escaping effect call'
ESCAPE='I cannot retain escaping operation identities'
RESULT='I cannot retain a checked value-call result'
STRINGS=[b'op',b'op',b'other',b'Choice',b'Some',b'Empty',b'value',b'Record']


def call_model(parameters=(),result=(1,NO),*,globals_=1,resource=False,arrays=False,caller=0,target=1,functions=2,
               slots=(),caller_parameters=0,extra_locals=0,body=None):
    rows=[(0,7,[(1,NO,6)]),(2,3,[(8,0,6)])]
    types=[desc(1),desc(1),desc(8,0),desc(7,2),desc(10,1),desc(10,1)] if arrays else []
    extensions=[extension(1,union_data(1,[(4,0,1),(5,1,0)]))]
    if globals_:extensions.append(extension(3,globals_data([struct.pack('<BBHI',1,1,0,NO)]*globals_)))
    revision=2 if arrays else 1
    extensions.append(extension(4,effect_data(revision,types=types,operations=[(0,desc(1),[])]),revision))
    selected=target if 0<target<functions and target!=caller else 1 if caller!=1 else 0
    declarations=[];entries=[];code=b''
    for index in range(functions):
        local=[];params=0;ret=(1,NO);instructions=b'\xa3' if index==0 else b'\x3d'
        if index==caller:local=[desc(*s) for s in slots];params=caller_parameters
        if index==selected:
            local=[desc(*s) for s in parameters]+[desc(1)]*extra_locals;params=len(parameters);ret=result
            instructions=b'\x3d' if body is None else body
        declarations.append((len(local),params,desc(*ret),local))
        entries.append(function(len(local),params,results=int(ret[0]!=0),result_tag=ret[0],
                                params=tuple(s[0] for s in (parameters if index==selected else slots[:params])),
                                start=len(code),length=len(instructions)))
        code+=instructions
    return image(record(3,flags=[3 if resource else 1]*2,functions=declarations,extensions=extensions),
                 rows=rows,strings=STRINGS,functions=entries,code=code)


def compose(before,after):
    a={(x,y) for x,y in product(range(2),repeat=2) if before&(1<<(2*x+y))}
    b={(x,y) for x,y in product(range(2),repeat=2) if after&(1<<(2*x+y))}
    return sum(1<<(2*x+z) for x,z in product(range(2),repeat=2) if any((x,y) in a and (y,z) in b for y in range(2)))


def require(relation,permitted):
    pairs={(x,y) for x,y in product(range(2),repeat=2) if relation&(1<<(2*x+y))}
    return sum(1<<x for x in range(2) if all(permitted&(1<<y) for a,y in pairs if a==x))


def value_call_cases():
    cases=[]
    def add(name,values=(),*,parameters=(),return_type=(1,NO),target=1,caller=0,functions=2,graph=True,effect=False,
            cached=True,summary_ok=True,allowed=None,relations=None,callee_allowed=None,callee_returns=None,
            escaped=(),callee_keys=(),handlers=(),floor=0,activation=23,resource=False,arrays=False,
            slots=(),caller_parameters=0,commands=(),state=None,body=None,extra_locals=0,error='',derived=False):
        allowed=[3] if allowed is None else list(allowed);count=len(allowed)
        relations=[9]*count if relations is None else list(relations)
        callee_allowed=[3]*count if callee_allowed is None else list(callee_allowed)
        callee_returns=[9]*count if callee_returns is None else list(callee_returns)
        assert len(relations)==len(callee_allowed)==len(callee_returns)==count
        carrier=call_model(parameters,return_type,globals_=count,resource=resource,arrays=arrays,caller=caller,
                           target=target,functions=functions,slots=slots,caller_parameters=caller_parameters,
                           extra_locals=extra_locals,body=body)
        raw=(words(13,target,caller,int(graph),int(effect),int(cached),int(summary_ok),count,len(escaped),len(callee_keys),len(carrier))
             +carrier+b''.join(words(a,b,c) for a,b,c in zip(allowed,callee_allowed,callee_returns))+words(*escaped)+words(*callee_keys)
             +frame(values,globals_=relations,handlers=handlers,floor=floor,activation=activation,commands=commands))
        result=list(values);after_relations=list(relations);after_allowed=list(allowed);keys=list(escaped)
        if not error:
            if parameters:result=result[:-len(parameters)]
            tag,layout=return_type
            if tag:
                owned=resource and (tag in (8,10) or tag==7 and layout>=2)
                result.append(value(tag,layout=layout,owned=owned))
            if not effect:
                after_allowed=[a&require(r,b) for a,r,b in zip(allowed,relations,callee_allowed)]
                after_relations=[compose(r,b) for r,b in zip(relations,callee_returns)]
            for key in callee_keys:
                def same(a,b):return a<len(STRINGS) and b<len(STRINGS) and len(STRINGS[a])==len(STRINGS[b]) and STRINGS[a].split(b'\0',1)[0]==STRINGS[b].split(b'\0',1)[0]
                if any(same(key,h) for h,_ in handlers) or any(same(key,old) for old in keys):continue
                assert len(keys)<256;keys.append(key)
        state=authority(len(slots),caller_parameters) if state is None else state
        numbers=lambda values:''.join(f'{int(n)},' for n in values)
        context=numbers((floor,activation))+''.join(f'{key},{site},' for key,site in handlers)
        want=dict(valid=not error,message=error,stack=stack_text(result),globals=numbers(after_relations),
                  context=context,authority=authority_text(state),allowed=numbers(after_allowed),keys=numbers(keys),derived=derived)
        cases.append((name,raw,want));return raw
    for graph,target,functions in product((False,True),(0,1,7,8,NO),(2,8)):
        add(f'graph-target-{graph}-{target}-{functions}',graph=graph,target=target,functions=functions,
            error='' if graph and 0<target<functions else CALL)
    for declared,tag,observed,owned in product(range(1,6),range(256),(False,True),(False,True)):
        error=SHAPE if declared==3 else '' if tag==declared and not observed and not owned else ARGUMENT
        add(f'positional-{declared}-{tag}-{observed}-{owned}',[value(tag,observation=observed,owned=owned,layout=17,root=7,variant=3)],
            parameters=[(declared,NO)],error=error)
    for count in (0,1,7,8,9):
        add(f'parameter-count-{count}',[value(1)]*count,parameters=[(1,NO)]*count,error=SHAPE if count>8 else '')
    add('missing-argument',parameters=[(1,NO)],error=SHAPE)
    for result in (0,1,2,3,4,5):
        for count in (0,255,256):
            add(f'result-{result}-prefix-{count}',[value(1)]*count,return_type=(result,NO),
                error=SHAPE if result in (3,5) else RESULT if result and count==256 else '')
    for relation,caller_allowed,callee_allowed,callee_returns,effect in product(range(16),range(4),range(4),range(16),(False,True)):
        add(f'global-compose-{relation}-{caller_allowed}-{callee_allowed}-{callee_returns}-{effect}',allowed=[caller_allowed],relations=[relation],
            callee_allowed=[callee_allowed],callee_returns=[callee_returns],effect=effect)
    for count in (0,1,255,256):
        add(f'global-count-{count}',allowed=[3]*count,callee_allowed=[2]*count,callee_returns=[10]*count)
    for ok,escaping in product((False,True),(False,True)):
        add(f'callee-status-{ok}-{escaping}',summary_ok=ok,callee_keys=[0] if escaping else [],
            error='' if ok else RESOLUTION)
    for tag,observed,owned,escaping in product(range(256),(False,True),(False,True),(False,True)):
        add(f'prefix-{tag}-{observed}-{owned}-{escaping}',[value(tag,observation=observed,owned=owned,root=7,layout=17,variant=3)],
            callee_keys=[0] if escaping else [],error=STAGED if escaping and (observed or owned) else '')
    for held,moved,escaping in product((False,True),(False,True),(False,True)):
        commands=([(5,0,0,0)] if moved else [])+([(6,0,0,0)] if held else [])
        state=authority(2,1,region=int(held));state['live'][0]=not moved
        error=LOCAL if escaping and (not moved or held) else ''
        add(f'caller-local-{held}-{moved}-{escaping}',caller=1,target=2,functions=3,
            resource=True,slots=[(8,0),(8,0)],caller_parameters=1,commands=commands,state=state,
            callee_keys=[0] if escaping else [],error=error)
    add('local-before-staged',[value(8,owned=True,layout=0)],caller=1,target=2,functions=3,
        resource=True,slots=[(8,0)],caller_parameters=1,callee_keys=[0],error=LOCAL)
    add('arguments-before-callee-status',[value(4)],parameters=[(1,NO)],summary_ok=False,error=ARGUMENT)
    add('callee-status-before-local',caller=1,target=2,functions=3,resource=True,slots=[(8,0)],caller_parameters=1,
        summary_ok=False,callee_keys=[0],error=RESOLUTION)
    for held,callee_keys,count in product((None,0,1,2),([], [0], [0,1], [0,3]),(0,255,256)):
        handlers=[] if held is None else [(held,5)]
        new=0
        for key in callee_keys:
            if key in (0,1) and held in (0,1):continue
            if key in (0,1) and new:continue
            new+=1
        error=ESCAPE if count+new>256 else ''
        add(f'escape-merge-{held}-{callee_keys}-{count}',escaped=[2]*count,callee_keys=callee_keys,handlers=handlers,error=error)
    for resource,tag,layout in product((False,True),(8,10),(0,1)):
        if tag==8 and layout!=0 or tag==10 and layout!=1:continue
        for physical,observed,owned in product((0,1,NO),(False,True),(False,True)):
            error='' if physical==layout and not observed and owned==resource else ARGUMENT
            add(f'aggregate-argument-{resource}-{tag}-{physical}-{observed}-{owned}',[value(tag,layout=physical,observation=observed,owned=owned)],
                parameters=[(tag,layout)],resource=resource,error=error)
        add(f'aggregate-result-{resource}-{tag}',return_type=(tag,layout),resource=resource)
    for resource,declared,actual in product((False,True),range(6),range(6)):
        same=(declared in (0,1) and actual in (0,1)) or (declared in (4,5) and actual in (4,5)) or declared==actual
        add(f'array-argument-{resource}-{declared}-{actual}',[value(7,layout=actual,owned=resource and declared>=2)],
            parameters=[(7,declared)],resource=resource,arrays=True,error='' if same else ARGUMENT)
    for resource,layout in product((False,True),range(6)):add(f'array-result-{resource}-{layout}',return_type=(7,layout),resource=resource,arrays=True)
    # My uncached C controls independently verify every supplied final summary.
    push=lambda n:b'\x01'+struct.pack('<q',n)
    for globals_ in (0,1):
        add(f'uncached-return-{globals_}',cached=False,allowed=[3]*globals_,body=push(7)+b'\x3d',derived=True)
        add(f'uncached-missing-result-{globals_}',cached=False,allowed=[3]*globals_,body=b'\x3d',summary_ok=False,
            callee_returns=[0]*globals_,error=RESOLUTION,derived=True)
    add('uncached-global-read',cached=False,body=b'\x12'+words(0)+b'\x3d',callee_allowed=[2],derived=True)
    add('uncached-global-store',cached=False,body=push(7)+b'\x13'+words(0)+push(9)+b'\x3d',callee_returns=[10],derived=True)
    add('uncached-perform',cached=False,effect=True,body=b'\xa7'+words(1)+struct.pack('<H',0)+b'\x3d',callee_keys=[0],derived=True)
    add('uncached-owned-consumption',[value(8,owned=True,layout=0)],cached=False,resource=True,parameters=[(8,0)],
        body=b'\x0e'+struct.pack('<H',0)+b'\x3d',derived=True)
    add('uncached-copyable-record',[value(8,layout=0)],cached=False,parameters=[(8,0)],
        body=b'\x10'+struct.pack('<H',0)+b'\xfc'+struct.pack('<H',0)+b'\x3d',derived=True)
    add('uncached-recursion-refusal',cached=False,body=b'\x3b'+words(1)+b'\x3d',summary_ok=False,
        callee_returns=[0],error=RESOLUTION,derived=True)
    add('uncached-local-size-refusal',cached=False,extra_locals=257,body=push(7)+b'\x3d',summary_ok=False,
        callee_returns=[0],error=RESOLUTION,derived=True)
    add('uncached-no-result-value',cached=False,return_type=(0,NO),body=b'\x3d',derived=True)
    good=add('valid-malformed-base')
    malformed=[b'',words(13),good+b'\0']+[good[:n] for n in (1,4,8,12,16,20,24,28,32,36,40,43,len(good)-1)]
    for index,number in ((1,8),(2,2),(3,2),(4,2),(5,2),(6,257),(7,257),(8,257)):
        bad=bytearray(good);bad[4+index*4:8+index*4]=words(number);malformed.append(bytes(bad))
    cases += [(f'malformed-{i}',raw,'D') for i,raw in enumerate(malformed)]
    return cases


class AffineValueCalls(unittest.TestCase):
    def test_original_frames_and_complete_value_call_summary_application(self):
        work=Path(tempfile.mkdtemp(prefix='nano-affine-value-calls-'))
        old=flow_cases()[0]+queue_cases()+frame_cases();self.assertEqual(len(old),1001)
        reference_=Path(os.environ.get('NANOLANG_AFFINE_VALUE_CALLS_REFERENCE',ROOT/'obj/nvm2c_affine_value_calls_reference'))
        manifest=dict(scope='I check value-call preparation/summary application and C-only uncached callee controls; no NanoLang callee analysis, without graph/CFG/executable admission.',cases=[],routes=[]);prepared=[]
        for group,cases in [('original',old),('calls',value_call_cases())]:
            paths=[]
            for index,(name,raw,e) in enumerate(cases):
                path=work/f'{group}-{index:05d}.bin';path.write_bytes(raw);paths.append(path)
                manifest['cases'].append(dict(group=group,name=name,path=str(path),sha256=hashlib.sha256(raw).hexdigest(),expected=e))
            prepared.append((group,cases,paths))
        def save():(work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        save();reports={};products=[('actual-c',[reference_])]+routes.build_routes(ROOT/'tests/nvm2c_affine_value_calls_driver.nano',work)
        self.assertEqual(len(products),6 if routes.source_compiler_command() else 4)
        for label,command in products:
            for group,cases,paths in prepared:
                chunks=([paths[i:i+8] for i in range(0,len(paths),8)] if group=='original' else scalar_batches(paths));actual=[]
                for chunk in chunks:actual+=routes.checked([*command,*chunk],timeout=600 if group=='original' else 60).splitlines()
                self.assertEqual(len(actual),len(cases),(label,group));raw=('\n'.join(actual)+'\n').encode();(work/f'{group}-{label}.reports').write_bytes(raw)
                if group=='original':self.assertEqual(hashlib.sha256(raw).hexdigest(),'f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0')
                for (name,_,e),text in zip(cases,actual):
                    with self.subTest(route=label,group=group,case=name):
                        if isinstance(e,str):
                            if group=='original' and not name.startswith(('global-algebra-','bounded-fifo-')):self.assertTrue(text.startswith(e))
                            else:self.assertEqual(text,e)
                        else:
                            valid,message,frame_,authority_,allowed,keys,derived=text.split('|',6);local,stack,globals_,context=frame_.split('/')
                            self.assertEqual(valid,str(int(e['valid'])));self.assertEqual(message,e['message']);self.assertEqual(stack,e['stack'])
                            self.assertEqual(globals_,e['globals']);self.assertEqual(context,e['context']);self.assertEqual(authority_,e['authority']);self.assertEqual(keys,e['keys']);self.assertEqual(allowed,e['allowed']);self.assertEqual(derived,str(int(e['derived'])))
                if group not in reports:reports[group]=actual
                else:self.assertEqual(actual,reports[group],(label,group))
                manifest['routes'].append(dict(label=label,group=group,cases=len(cases),sha256=hashlib.sha256(raw).hexdigest()));save()


if __name__=='__main__':unittest.main()
