"""I compare effect frame steps and bounded key/escape identity with unchanged C."""
from itertools import product
from pathlib import Path
import hashlib
import json
import os
import tempfile
import unittest

from tests import nanolang_translator_routes as routes
from tests.test_nanolang_translator_affine_frame import words, value, frame, flow_cases, queue_cases, frame_cases
from tests.test_nanolang_translator_affine_scalar import scalar_batches, stack_text
from tests.test_nanolang_translator_affine_owners import authority, authority_text, reference
from tests.test_nanolang_translator_ownership import desc, record, function, image, extension, effect_data, union_data

ROOT=Path(__file__).resolve().parents[1]
NO=0xffffffff
SHAPE='I require the exact retained performed signature'
ARGUMENT='I require exact positional effect argument ownership'
STAGED='I cannot abandon a staged owner across an effect'
LOCAL='I cannot abandon a local owner or borrow across an effect'
ESCAPE='I cannot retain escaping operation identities'
RESULT='I cannot retain a resumed effect value'
POP='I cannot pop absent or inherited handlers'


def effect_model(parameters=None,result=(1,NO),*,resource=False,slots=(),parameter_count=0,arrays=False):
    parameters=[(1,NO)] if parameters is None else parameters
    strings=[b'op',b'op',b'other',b'Choice',b'Some',b'Empty',b'value',b'Record']
    rows=[(0,7,[(1,NO,6)]),(2,3,[(8,0,6)])]
    types=[desc(1),desc(1),desc(8,0),desc(7,2),desc(10,1),desc(10,1)] if arrays else []
    revision=2 if arrays else 1
    effects=effect_data(revision,types=types,operations=[(0,desc(*result),[desc(*p) for p in parameters])])
    own=record(3,flags=[3 if resource else 1]*2,
               functions=[(len(slots),parameter_count,desc(),[desc(*slot) for slot in slots])],
               extensions=[extension(1,union_data(1,[(4,0,1),(5,1,0)])),extension(4,effects,revision)])
    return image(own,rows=rows,strings=strings,
                 functions=[function(len(slots),parameter_count,params=tuple(s[0] for s in slots[:parameter_count]),length=1)],code=b'\xa3')


def identity_model(strings):
    return image(record(1,functions=[(0,0,desc(),[])]),strings=strings,
                 functions=[function(length=1)],code=b'\xa3')


def c_key(a,b):
    return len(a)==len(b) and a.split(b'\0',1)[0]==b.split(b'\0',1)[0]


def effect_cases():
    cases=[]
    def add(name,opcode,values=(),*,a=0,b=0,function_=0,present=True,escaped=(),after_keys=None,
            handlers=(),floor=0,activation=23,after_handlers=None,carrier=None,commands=(),state=None,
            error='',valid=None,result=None):
        carrier=effect_model() if carrier is None else carrier
        state=authority(0,0) if state is None else state
        valid=not error if valid is None else valid
        raw=(words(11,opcode,a,b,function_,int(present),len(escaped),len(carrier))+carrier+words(*escaped)
             +frame(values,globals_=[9],handlers=handlers,floor=floor,activation=activation,commands=commands))
        after_handlers=handlers if not valid or after_handlers is None else after_handlers
        keys=escaped if not valid or after_keys is None else after_keys
        context=''.join(f'{n},' for n in (floor,activation))+''.join(f'{key},{site},' for key,site in after_handlers)
        want=dict(valid=valid,message=error,stack=stack_text(values if not valid or result is None else result),
                  authority=authority_text(state),keys=''.join(f'{n},' for n in keys),context=context)
        cases.append((name,raw,want));return raw
    # I compare key helper behavior only inside its nonnull-module contract.
    strings=[b'',b'a',b'a',b'ab',b'a\0x',b'a\0y',b'b\0x',b'a\0',b'a\0\0',b'\xff',b'\xff']
    carrier=identity_model(strings)
    for a,b in product([*range(len(strings)),len(strings),NO],repeat=2):
        expected=a<len(strings) and b<len(strings) and c_key(strings[a],strings[b])
        cases.append((f'key-{a}-{b}',words(12,1,a,b,len(carrier))+carrier,str(int(expected))))
    # I seed bounded escape tables to cover every private defensive branch.
    for function_,key,count,handler_key in product((0,7,8,NO),(0,1,2,NO),(0,1,255,256),(None,0,1,2,NO)):
        escaped=[2]*count;handlers=[] if handler_key is None else [(handler_key,5)]
        caught=key in (0,1) and handler_key in (0,1) or key==2 and handler_key==2
        duplicate=count>0 and key==2
        valid=caught or function_<8 and (duplicate or count<256)
        after=escaped if caught or duplicate else [*escaped,key]
        add(f'escape-{function_}-{key}-{count}-{handler_key}',0,a=key,function_=function_,escaped=escaped,
            handlers=handlers,valid=valid,after_keys=after)
    for a,b in product(range(len(strings)),repeat=2):
        same=c_key(strings[a],strings[b])
        add(f'escape-c-string-{a}-{b}',0,a=a,escaped=[b],carrier=carrier,
            after_keys=[b] if same else [b,a])
    for count in (0,1,63,64):
        handlers=[(2,i) for i in range(count)]
        for floor in sorted({0,count//2,count}):
            for removed in (0,1,63,64,65,65535):
                valid=removed<=count-floor
                add(f'handler-pop-{count}-{floor}-{removed}',166,a=removed,handlers=handlers,floor=floor,
                    error='' if valid else POP,after_handlers=handlers[:count-removed] if removed else handlers)
    for tag,observed,owned in product(range(256),(False,True),(False,True)):
        item=value(tag,observation=observed,owned=owned,root=7,layout=17,variant=3,excluded=(0,63,64,255))
        add(f'handler-push-step-{tag}-{observed}-{owned}',165,[item],a=NO,b=65535,function_=NO,
            escaped=[2],handlers=[(2,i) for i in range(64)],floor=64)
    for declared,tag,observed,owned in product(range(1,6),range(256),(False,True),(False,True)):
        valid=declared==tag and not observed and not owned
        item=value(tag,observation=observed,owned=owned,root=7,layout=17,variant=3,excluded=(0,255))
        add(f'performed-argument-{declared}-{tag}-{observed}-{owned}',167,[item],b=1,
            carrier=effect_model(parameters=[(declared,NO)]),error='' if valid else ARGUMENT,
            after_keys=[0],result=[value(1)])
    for tag,observed,owned in product(range(256),(False,True),(False,True)):
        item=value(tag,observation=observed,owned=owned,root=7,layout=17,variant=3)
        add(f'staged-prefix-{tag}-{observed}-{owned}',167,[item,value(1)],b=1,
            error=STAGED if observed or owned else '',after_keys=[0],result=[item,value(1)])
    for key,argc,present in product((0,1,2,8,NO),(0,1,2,256,257,65535),(False,True)):
        valid=present and key in (0,1) and argc==1
        add(f'signature-{key}-{argc}-{present}',167,[value(1)],a=key,b=argc,present=present,
            error='' if valid else SHAPE,after_keys=[0],result=[value(1)])
    add('missing-argument',167,b=1,error=SHAPE)
    add('argument-before-staged',167,[value(8,owned=True,layout=0),value(4)],b=1,error=ARGUMENT)
    add('shape-before-staged',167,[value(8,owned=True,layout=0)],b=2,error=SHAPE)
    for count in (0,1,255,256,257):
        if count==257:
            raw=add('invalid-257-parameter-model',167,[value(1)]*256,b=256,carrier=effect_model(parameters=[(1,NO)]*257))
            cases[-1]=(cases[-1][0],raw,'D');continue
        add(f'parameter-bound-{count}',167,[value(1)]*count,b=count,carrier=effect_model(parameters=[(1,NO)]*count),
            after_keys=[0],result=[value(1)])
    for result_tag in (0,1,2,3,4,5):
        for count in (0,255,256):
            add(f'result-{result_tag}-prefix-{count}',167,[value(1)]*count,b=0,
                carrier=effect_model(parameters=[],result=(result_tag,NO)),error=RESULT if count==256 else '',
                after_keys=[0],result=[value(1)]*count+[value(result_tag)])
    for tag,layout,resource in product((8,10),(0,1),(False,True)):
        if tag==8 and layout!=0 or tag==10 and layout!=1:continue
        for physical,owned,observed in product((0,1,NO),(False,True),(False,True)):
            valid=physical==layout and owned==resource and not observed
            add(f'aggregate-argument-{tag}-{layout}-{resource}-{physical}-{owned}-{observed}',167,
                [value(tag,layout=physical,owned=owned,observation=observed)],b=1,
                carrier=effect_model(parameters=[(tag,layout)],resource=resource),error='' if valid else ARGUMENT,
                after_keys=[0],result=[value(1)])
        add(f'aggregate-result-{tag}-{layout}-{resource}',167,carrier=effect_model(parameters=[],result=(tag,layout),resource=resource),
            after_keys=[0],result=[value(tag,layout=layout,owned=resource)])
    for resource,layout in product((False,True),range(6)):
        for actual in range(6):
            owned=resource and layout>=2
            valid=(layout in (0,1) and actual in (0,1)) or (layout in (4,5) and actual in (4,5)) or layout==actual
            add(f'array-argument-{resource}-{layout}-{actual}',167,[value(7,layout=actual,owned=owned)],b=1,
                carrier=effect_model(parameters=[(7,layout)],resource=resource,arrays=True),error='' if valid else ARGUMENT,
                after_keys=[0],result=[value(1)])
        add(f'array-result-{resource}-{layout}',167,carrier=effect_model(parameters=[],result=(7,layout),resource=resource,arrays=True),
            after_keys=[0],result=[value(7,layout=layout,owned=resource and layout>=2)])
    for moved,region in product((False,True),(False,True)):
        commands=([(5,0,0,0)] if moved else [])+([(6,0,0,0)] if region else [])
        state=authority(2,1,region=int(region));state['live'][0]=not moved
        add(f'local-owner-{moved}-{region}',167,carrier=effect_model(parameters=[],resource=True,slots=[(8,0),(8,0)],parameter_count=1),
            commands=commands,state=state,error='' if moved and not region else LOCAL,after_keys=[0],result=[value(1)])
    state=authority(2,1,region=1);state['references'][1]=reference(mode=2)
    add('live-borrow-obligation',167,carrier=effect_model(parameters=[],resource=True,slots=[(8,0),(8,0)],parameter_count=1),
        commands=[(6,0,0,0),(8,1,0,2)],state=state,error=LOCAL)
    add('staged-before-local',167,[value(8,owned=True,layout=0)],carrier=effect_model(parameters=[],resource=True,slots=[(8,0)],parameter_count=1),
        state=authority(1,1),error=STAGED)
    for function_,handler,count in product((0,7,8,NO),(None,0,1,2),(0,255,256)):
        escaped=[2]*count;handlers=[] if handler is None else [(handler,5)]
        valid=handler in (0,1) or function_<8 and count<256
        add(f'perform-escape-{function_}-{handler}-{count}',167,[value(1)],a=1,b=1,function_=function_,escaped=escaped,
            handlers=handlers,error='' if valid else ESCAPE,after_keys=escaped if handler in (0,1) else [*escaped,0],result=[value(1)])
    good=add('valid-malformed-base',167,[value(1)],b=1,after_keys=[0],result=[value(1)])
    malformed=[b'',words(11),words(12),good+b'\0']+[good[:n] for n in (1,4,8,12,16,20,24,28,31,len(good)-1)]
    for index,number in ((0,168),(2,65536),(4,2),(5,257)):
        bad=bytearray(good);bad[4+index*4:8+index*4]=words(number);malformed.append(bytes(bad))
    missing_context=words(12,0,0,0,len(carrier))+carrier;malformed.append(missing_context)
    cases += [(f'malformed-{i}',raw,'D') for i,raw in enumerate(malformed)]
    return cases


class AffineEffects(unittest.TestCase):
    def test_original_frames_and_complete_effect_transfer(self):
        work=Path(tempfile.mkdtemp(prefix='nano-affine-effects-'))
        old=flow_cases()[0]+queue_cases()+frame_cases();self.assertEqual(len(old),1001)
        reference_=Path(os.environ.get('NANOLANG_AFFINE_EFFECTS_REFERENCE',ROOT/'obj/nvm2c_affine_effects_reference'))
        manifest=dict(scope='I check three effect steps, key identity and mutable caller/escape isolation, without graph/CFG/executable admission.',cases=[],routes=[]);prepared=[]
        for group,cases in [('original',old),('effects',effect_cases())]:
            paths=[]
            for index,(name,raw,e) in enumerate(cases):
                path=work/f'{group}-{index:05d}.bin';path.write_bytes(raw);paths.append(path)
                manifest['cases'].append(dict(group=group,name=name,path=str(path),sha256=hashlib.sha256(raw).hexdigest(),expected=e))
            prepared.append((group,cases,paths))
        def save():(work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        save();reports={};products=[('actual-c',[reference_])]+routes.build_routes(ROOT/'tests/nvm2c_affine_effects_driver.nano',work)
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
                            valid,message,frame_,authority_,keys=text.split('|',4);local,stack,globals_,context=frame_.split('/')
                            self.assertEqual(valid,str(int(e['valid'])));self.assertEqual(message,e['message']);self.assertEqual(stack,e['stack'])
                            self.assertEqual(globals_,'9,');self.assertEqual(context,e['context']);self.assertEqual(authority_,e['authority']);self.assertEqual(keys,e['keys'])
                if group not in reports:reports[group]=actual
                else:self.assertEqual(actual,reports[group],(label,group))
                manifest['routes'].append(dict(label=label,group=group,cases=len(cases),sha256=hashlib.sha256(raw).hexdigest()));save()


if __name__=='__main__':unittest.main()
