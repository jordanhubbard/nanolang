"""I compare exact aggregate construction, projection, match and tag steps."""
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
from tests.test_nanolang_translator_affine_owners import owner_model, authority, authority_text, reference
from tests.test_nanolang_translator_affine_state import union_model
from tests.test_nanolang_translator_ownership import desc, record, function, image, extension, union_data

ROOT=Path(__file__).resolve().parents[1]
NO, UNKNOWN=0xffffffff,65535
RECORD='I require an exact complete copyable record constructor'
RECORD_FIELD='I require exact copyable record fields without owned authority'
KIND='I require explicit owned construction for records'
UNION='I require an exact retained scalar-union constructor shape'
UNION_FACT='I require exact scalar-union payload facts'
UNION_FIELD='I require exact scalar-union fields in declaration order'
MATCH='I require an exact scalar-union match source'
MATCH_VARIANT='I require a match variant within the concrete union'
TAG='I require an exact scalar-union tag source'
PROJECT_RECORD='I require an exact ordinary record projection'
PROVEN='I require a proven scalar-union variant before projection'
PROJECT_UNION='I require an exact scalar-union payload field'
DESTRUCTIVE='I require destructive unpack for aggregate payloads'
OBSERVE='I require a checked scalar field observation'


def scalar_union(tag=1, fields=1):
    strings=[b'Leaf',b'Choice',b'Some',b'Empty',b'Pair',b'value']
    rows=[(0,0,[(1,NO,5)]),(2,1,[(tag,NO,5)]*fields)]
    variants=[(2,0,fields),(3,0,0)]
    own=record(3,flags=[1,1],functions=[(4,1,desc(),[desc(8,0),desc(10,1),desc(1),desc(1)])],
               extensions=[extension(1,union_data(1,variants))])
    return image(own,rows=rows,strings=strings,functions=[function(4,1,params=(8,),length=1)],code=b'\x3d')


def aggregate_cases():
    cases=[]
    def add(name,opcode,values=(),*,a=0,b=0,c=0,d=0,graph=False,carrier=None,commands=(),state=None,error='',result=None):
        carrier=owner_model(resource=False) if carrier is None else carrier
        state=authority() if state is None else state
        raw=(words(10,opcode,int(graph),a,b,c,d,len(carrier))+carrier
             +frame(values,globals_=[9],handlers=[(5,7)],floor=1,activation=23,commands=commands))
        want=dict(valid=not error,message=error,stack=stack_text(values if error or result is None else result),authority=authority_text(state))
        cases.append((name,raw,want));return raw
    # I check all physical tags/authority flags at each consumer's distinct guard.
    for opcode,tag,observed,owned in product((97,107,252,254),range(256),(False,True),(False,True)):
        item=value(tag,observation=observed,owned=owned,root=0,layout=0,variant=0,excluded=(0,63,64,255))
        if opcode==97:
            error='' if observed and not owned else OBSERVE;out=[value(1)]
        elif opcode==107:
            error=MATCH if tag!=10 or observed else MATCH_VARIANT;out=None
        elif opcode==254:
            error='' if tag==10 and not observed and not owned else TAG;out=[value(1)]
        else:
            error=PROVEN if tag==10 and not observed and not owned and item[5]==UNKNOWN else (
                PROJECT_UNION if tag==10 and not observed and not owned else
                '' if not owned and (observed or tag==8) else OBSERVE)
            out=[value(1)]
        add(f'physical-{opcode}-{tag}-{observed}-{owned}',opcode,[item],error=error,result=out)
    for opcode in (97,107,252,254):
        add(f'empty-{opcode}',opcode,error=MATCH if opcode==107 else TAG if opcode==254 else OBSERVE)
    for tag,physical,observed,owned in product(range(1,6),range(256),(False,True),(False,True)):
        item=value(physical,observation=observed,owned=owned,layout=NO,root=0,variant=2,excluded=(0,255))
        valid=tag==physical and not observed and not owned
        add(f'record-field-{tag}-{physical}-{observed}-{owned}',251,[item],d=1,graph=True,
            carrier=owner_model(tag=tag,resource=False),error='' if valid else RECORD_FIELD,result=[value(8,layout=0)])
    for graph,layout,variant,count in product((False,True),(0,1,NO),(0,1,65535),(0,1,2,256,257,65535)):
        good=graph and layout==0 and variant==0 and count==1
        add(f'record-shape-{graph}-{layout}-{variant}-{count}',251,[value(1)],b=layout,c=variant,d=count,graph=graph,
            error='' if good else RECORD,result=[value(8,layout=0)])
    for kind in range(2,256):add(f'aggregate-kind-{kind}',251,a=kind,error=KIND)
    for fields in (0,1,2,255,256,257):
        values=[value(1)]*min(fields,256)
        add(f'record-fields-bound-{fields}',251,values,d=fields,graph=True,carrier=owner_model(fields=fields,resource=False),
            error=RECORD if fields>256 else '',result=[value(8,layout=0)])
    add('empty-record-full-stack',251,[value(1)]*256,graph=True,carrier=owner_model(fields=0,resource=False),error='I cannot retain a copyable record')
    for layout in (0,1,NO):
        for physical in (0,1,NO):
            add(f'record-physical-layout-{layout}-{physical}',251,[value(1,layout=physical)],b=layout,d=1,graph=True,
                error=RECORD if layout!=0 else RECORD_FIELD if physical!=NO else '',result=[value(8,layout=0)])
    # I preserve graph-independent union construction and exact known variant output.
    for graph,tag,arm,count in product((False,True),range(1,6),(0,1,2,65535),(0,1,2)):
        valid=arm in (0,1) and count==(1 if arm==0 else 0)
        values=[value(tag)] if arm!=1 else []
        add(f'union-shape-{graph}-{tag}-{arm}-{count}',251,values,a=1,b=0,c=arm,d=count,graph=graph,
            carrier=scalar_union(tag),error='' if valid else UNION,result=[value(10,layout=1,variant=arm)])
    for tag,physical,observed,owned in product(range(1,6),range(256),(False,True),(False,True)):
        item=value(physical,observation=observed,owned=owned,layout=17,root=7,variant=3)
        add(f'union-field-{tag}-{physical}-{observed}-{owned}',251,[item],a=1,b=0,d=1,
            carrier=scalar_union(tag),error='' if physical==tag and not observed and not owned else UNION_FIELD,
            result=[value(10,layout=1,variant=0)])
    for owned_model,owned_value,layout,arm in product((False,True),(False,True),(0,1,NO),(0,1,2,3)):
        values=[] if arm==1 else [value(8,owned=owned_value,layout=layout)]*(2 if arm==2 else 1)
        error=UNION if arm==3 else UNION_FIELD if arm!=1 and (layout!=0 or owned_value!=owned_model) else ''
        add(f'union-aggregate-fields-{owned_model}-{owned_value}-{layout}-{arm}',251,values,a=1,c=arm,d=len(values),
            carrier=union_model(owned_model),state=authority(7,0),error=error,result=[value(10,owned=owned_model,layout=1,variant=arm)])
    for fields in (0,1,255,256,257):
        values=[value(1)]*min(fields,256)
        add(f'union-field-bound-{fields}',251,values,a=1,d=fields,carrier=scalar_union(fields=fields),
            error=UNION if fields>256 else '',result=[value(10,layout=1,variant=0)])
    add('empty-union-full-stack',251,[value(1)]*256,a=1,d=0,carrier=scalar_union(fields=0),error='I cannot retain a checked scalar-union value')
    for graph,layout,arm in product((False,True),(0,1,2,3,NO),(0,1,2,3,65535)):
        item=value(10,owned=True,layout=layout,variant=2,root=3,excluded=(0,1,2,255))
        valid=(layout==1 and arm<3) or (layout==2 and arm==0)
        add(f'match-layout-ordinal-{graph}-{layout}-{arm}',107,[item],a=arm,graph=graph,carrier=union_model(True),
            state=authority(7,0),error='' if valid else MATCH_VARIANT)
    for tag,arm,field in product(range(1,6),(0,1,2,65535),(0,1,65535)):
        error=PROVEN if arm==UNKNOWN else PROJECT_UNION if arm not in (0,1) or arm==1 or field!=0 else ''
        add(f'union-project-{tag}-{arm}-{field}',252,[value(10,layout=1,variant=arm)],a=field,
            carrier=scalar_union(tag),error=error,result=[value(tag)])
    for arm,field in product((0,1,2,65535),(0,1,65535)):
        error=PROVEN if arm==UNKNOWN else PROJECT_UNION if arm==1 or field>=(2 if arm==2 else 1) else DESTRUCTIVE
        add(f'union-aggregate-project-{arm}-{field}',252,[value(10,layout=1,variant=arm)],a=field,
            carrier=union_model(False),state=authority(7,0),error=error)
    for graph,tag,field,held in product((False,True),range(1,6),(0,1,65535),(0,1,2)):
        commands=[] if held==0 else [(6,0,0,0),(8,1,0,held)]
        state=authority()
        if held:
            state['control'][0]=1;state['references'][1]=reference(mode=held)
        valid=field==0 and held!=2 and (tag!=5 or graph)
        for opcode in (97,252):
            add(f'observation-{opcode}-{graph}-{tag}-{field}-{held}',opcode,[value(8,observation=True,root=0,layout=0)],a=field,
                graph=graph,carrier=owner_model(tag=tag),commands=commands,state=state,error='' if valid else OBSERVE,result=[value(tag)])
            if tag==5 and held:cases[-1]=(cases[-1][0],cases[-1][1],'D')
    for field,layout in product((0,1,65535),(0,1,NO)):
        add(f'record-project-{field}-{layout}',252,[value(8,layout=layout)],a=field,
            error='' if field==0 and layout==0 else PROJECT_RECORD,result=[value(1)])
    good=add('valid-malformed-base',254,[value(10,layout=1)],carrier=scalar_union(),result=[value(1)])
    malformed=[b'',words(10),good+b'\0']+[good[:n] for n in (1,4,8,12,16,20,24,28,31,len(good)-1)]
    for index,number in ((0,253),(1,2),(2,65536),(4,65536),(5,65536)):
        bad=bytearray(good);bad[4+index*4:8+index*4]=words(number);malformed.append(bytes(bad))
    cases += [(f'malformed-{i}',raw,'D') for i,raw in enumerate(malformed)]
    return cases


class AffineAggregates(unittest.TestCase):
    def test_original_frames_and_complete_aggregate_transfer(self):
        work=Path(tempfile.mkdtemp(prefix='nano-affine-aggregates-'))
        old=flow_cases()[0]+queue_cases()+frame_cases();self.assertEqual(len(old),1001)
        reference_=Path(os.environ.get('NANOLANG_AFFINE_AGGREGATES_REFERENCE',ROOT/'obj/nvm2c_affine_aggregates_reference'))
        manifest=dict(scope='I check five aggregate steps and mutable caller isolation, without graph/CFG/executable admission.',cases=[],routes=[]);prepared=[]
        for group,cases in [('original',old),('aggregates',aggregate_cases())]:
            paths=[]
            for index,(name,raw,e) in enumerate(cases):
                path=work/f'{group}-{index:05d}.bin';path.write_bytes(raw);paths.append(path)
                manifest['cases'].append(dict(group=group,name=name,path=str(path),sha256=hashlib.sha256(raw).hexdigest(),expected=e))
            prepared.append((group,cases,paths))
        def save():(work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        save();reports={};products=[('actual-c',[reference_])]+routes.build_routes(ROOT/'tests/nvm2c_affine_aggregates_driver.nano',work)
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
                            valid,message,frame_,authority_=text.split('|',3);local,stack,globals_,context=frame_.split('/')
                            self.assertEqual(valid,str(int(e['valid'])));self.assertEqual(message,e['message']);self.assertEqual(stack,e['stack'])
                            self.assertEqual(globals_,'9,');self.assertEqual(context,'1,23,5,7,');self.assertEqual(authority_,e['authority'])
                if group not in reports:reports[group]=actual
                else:self.assertEqual(actual,reports[group],(label,group))
                manifest['routes'].append(dict(label=label,group=group,cases=len(cases),sha256=hashlib.sha256(raw).hexdigest()));save()


if __name__=='__main__':unittest.main()
