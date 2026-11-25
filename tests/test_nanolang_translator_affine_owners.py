"""I compare complete owner/reference state transitions without CFG admission."""
from itertools import product
from pathlib import Path
import copy
import hashlib
import json
import os
import tempfile
import unittest

from tests import nanolang_translator_routes as routes
from tests.test_nanolang_translator_affine_frame import words, value, frame, flow_cases, queue_cases, frame_cases
from tests.test_nanolang_translator_affine_scalar import scalar_batches, stack_text
from tests.test_nanolang_translator_affine_state import union_model
from tests.test_nanolang_translator_ownership import desc, record, function, image, extension, effect_data

ROOT = Path(__file__).resolve().parents[1]
NO, UNKNOWN = 0xffffffff, 65535
REGION_BEGIN = 'I cannot begin another reference region'
REGION_END = 'I need a live reference region to end'
REBORROW = 'I require permitted parent authority and a fresh reborrow in a deeper region'
PATH = 'I require a retained bounded ownership path'
OBSERVE_BORROW = 'I cannot borrow exclusively while an owner observation is on my stack'
BORROW = 'I require an available scalar-leaf resource owner and reference slot in a live region'
FIELD = 'I require a live permitted scalar reference field'
FIELD_TYPE = 'I require the exact scalar reference field type'
MOVE_OBSERVE = 'I cannot move while an owner observation is on my stack'
MOVE = 'I require a live unheld owner for an explicit move'
TOKEN = 'I require an owned token for an explicit store'
DESTINATION = 'I require an exact available owner destination'
ARM = "I require the stored union's exact selected arm"
PACK = 'I require every declared field for owned construction'
PACK_FIELD = 'I require exact scalar or owned fields in declaration order'
UNPACK_OBSERVE = 'I cannot unpack while an owner observation is on my stack'
UNPACK = 'I require an intact owner and space for all unpacked fields'


def owner_model(tag=1, *, fields=1, parameters=1, locals_=4, resource=True, nested=False, mixed=False, paths=()):
    rows = [(0, NO, [(tag, NO, NO)]*fields)]
    root = 0
    if nested or mixed:
        root = 1
        rows += [(0, NO, [(8, 0, NO), (4, NO, NO) if mixed else (8, 0, NO)])]
    slots = [desc(8, root), desc(8, 0)]+[desc(1)]*max(0, locals_-2)
    slots = slots[:locals_]
    own = record(2, flags=[3 if resource else 1]*len(rows), functions=[(locals_, parameters, desc(), slots)], paths=paths)
    return image(own, rows=rows, functions=[function(locals_, parameters, params=(8,)*parameters, length=1)], code=b'\x3d')


def array_owner_model():
    types = [desc(1), desc(7, 0), desc(8, 0), desc(7, 2), desc(1), desc(8, 0)]
    slots = [desc(7, 2), desc(7, 5), desc(1)]
    own = record(3, flags=[3], functions=[(3, 1, desc(), slots)],
                 extensions=[extension(4, effect_data(2, types=types), 2)])
    return image(own, rows=[(0, NO, [(1, NO, NO)])], strings=[b'op'],
                 functions=[function(3, 1, params=(7,), length=1)], code=b'\x3d')


def authority(count=4, parameters=1, *, region=0):
    return dict(control=[region, 0, 0, 0, 0, 0, 0, 1, 0], live=[i < parameters for i in range(count)],
                variants=[UNKNOWN]*count, references=[(0,)*10 for _ in range(count)], origins=[])


def reference(region=1, mode=1, *, root=0, layout=0, referent=0, path=(), parent=NO):
    return (1, region, parent, 1, 0, root, layout, referent, mode, len(path), *path)


def authority_text(state):
    numbers = lambda values: ''.join(f'{int(n)},' for n in values)
    return (numbers(state['control'])+'/'+''.join(f'{int(live)},{arm},;' for live, arm in zip(state['live'], state['variants']))
            +'/'+''.join(numbers(ref)+';' for ref in state['references'])
            +'/'+''.join(numbers(origin)+';' for origin in state['origins']))


def expected(*, error='', stack=None, live=None, variants=None, refs=None, region=None):
    return dict(error=error, stack=stack, live=live or {}, variants=variants or {}, refs=refs or {}, region=region)


def owner_cases():
    cases = []
    def add(name, instructions, expectations, *, carrier=None, state=None, values=(), commands=(), region_override=None):
        carrier = owner_model() if carrier is None else carrier
        state = authority() if state is None else copy.deepcopy(state)
        current = list(values);reports = []
        for e in expectations:
            if not e['error']:
                if e['stack'] is not None:current = list(e['stack'])
                for n, flag in e['live'].items():state['live'][n] = flag
                for n, arm in e['variants'].items():state['variants'][n] = arm
                for n, ref in e['refs'].items():state['references'][n] = ref
                if e['region'] is not None:state['control'][0] = e['region']
            reports.append(dict(valid=not e['error'], message=e['error'], stack=stack_text(current), authority=authority_text(state)))
        assert len(instructions)==len(reports)
        payload = (words(9, len(instructions), int(region_override is not None), region_override or 0, len(carrier))+carrier
                   + frame(values, globals_=[9], handlers=[(5, 7)], floor=1, activation=23, commands=commands)
                   + b''.join(words(*instruction) for instruction in instructions))
        cases.append((name, payload, reports));return payload
    op = lambda code,a=0,b=0,c=0: (code,a,b,c)
    for code, region in product((22, 23), (0, 1, 2, NO-1, NO)):
        error = REGION_BEGIN if code == 22 and region == NO else REGION_END if code == 23 and region == 0 else ''
        add(f'region-bound-{code}-{region}', [op(code)], [expected(error=error, region=region+(1 if code == 22 else -1))],
            state=authority(region=region), region_override=region)
    add('nested-empty-regions', [op(22),op(22),op(23),op(23),op(23)],
        [expected(region=1),expected(region=2),expected(region=1),expected(region=0),expected(error=REGION_END)])
    for code, tag, observed, owned in product((24,25), range(256), (False,True), (False,True)):
        error = OBSERVE_BORROW if code == 25 and observed else ''
        add(f'borrow-stack-{code}-{tag}-{observed}-{owned}', [op(code,1,0)],
            [expected(error=error,refs={1:reference(mode=1 if code == 24 else 2)})],
            values=[value(tag,observation=observed,owned=owned,root=0)], commands=[(6,0,0,0)], state=authority(region=1))
    for code, slot, root in product((24,25), (0,1,3,4,65535), (0,1,3,4,65535)):
        valid = slot < 4 and root == 0
        add(f'borrow-bounds-{code}-{slot}-{root}', [op(code,slot,root)],
            [expected(error='' if valid else BORROW,refs={slot:reference(mode=1 if code == 24 else 2)} if valid else {})],
            commands=[(6,0,0,0)],state=authority(region=1))
    for code in (24,25):
        add(f'borrow-needs-region-{code}',[op(code,1,0)],[expected(error=BORROW)])
        moved = authority(region=1);moved['live'][0] = False
        add(f'borrow-moved-{code}',[op(code,1,0)],[expected(error=BORROW)],commands=[(5,0,0,0),(6,0,0,0)],state=moved)
    for parent_mode, child_mode, deeper in product((1,2),(1,2),(False,True)):
        commands=[(6,0,0,0),(8,1,0,parent_mode)]+([(6,0,0,0)] if deeper else [])
        initial=authority(region=2 if deeper else 1);initial['references'][1]=reference(mode=parent_mode)
        valid=deeper and (child_mode==1 or parent_mode==2)
        add(f'reborrow-{parent_mode}-{child_mode}-{deeper}',[op(30 if child_mode==1 else 31,2,1)],
            [expected(error='' if valid else REBORROW,refs={2:reference(region=2,mode=child_mode,parent=1)} if valid else {})],commands=commands,state=initial)
    add('exclusive-shared-child-lifetime',[op(22),op(25,1,0),op(22),op(30,2,1),op(27,1,0),op(23),op(26,1,0),op(23),op(26,1,0)],
        [expected(region=1),expected(refs={1:reference(mode=2)}),expected(region=2),
         expected(refs={2:reference(region=2,mode=1,parent=1)}),expected(error=FIELD),
         expected(region=1,refs={2:(0,)*10}),expected(stack=[value(1)]),
         expected(region=0,refs={1:(0,)*10}),expected(error=FIELD)])
    path_model=owner_model(nested=True,paths=[(0,),(1,),(0,0),()])
    for code, path in product((28,29),(0,1,2,3,4,NO)):
        valid=path in (0,1)
        error='' if valid else BORROW if path in (2,3) else PATH
        add(f'path-borrow-{code}-{path}',[op(code,1,0,path)],
            [expected(error=error,refs={1:reference(mode=1 if code==28 else 2,layout=1,path=(path,))} if valid else {})],
            carrier=path_model,commands=[(6,0,0,0)],state=authority(region=1))
    add('path-error-before-observation',[op(29,1,0,NO)],[expected(error=PATH)],carrier=path_model,
        values=[value(8,observation=True,root=0,layout=1)],commands=[(6,0,0,0)],state=authority(region=1))
    add('disjoint-exclusive-paths',[op(22),op(29,1,0,0),op(29,2,0,1),op(26,1,0),op(26,2,0),op(23)],
        [expected(region=1),expected(refs={1:reference(mode=2,layout=1,path=(0,))}),
         expected(refs={2:reference(mode=2,layout=1,path=(1,))}),expected(stack=[value(1)]),
         expected(stack=[value(1),value(1)]),expected(region=0,refs={1:(0,)*10,2:(0,)*10})],carrier=path_model)
    for scalar_tag, actual_tag, observed, owned in product(range(1,5),range(256),(False,True),(False,True)):
        initial=authority(region=1);initial['references'][1]=reference(mode=2)
        error='' if scalar_tag==actual_tag and not observed and not owned else FIELD_TYPE
        item=value(actual_tag,observation=observed,owned=owned,root=7,layout=17,variant=3,excluded=(0,63,64,255))
        add(f'ref-set-{scalar_tag}-{actual_tag}-{observed}-{owned}',[op(27,1,0)],
            [expected(error=error,stack=[])],carrier=owner_model(tag=scalar_tag),commands=[(6,0,0,0),(8,1,0,2)],state=initial,values=[item])
    for scalar_tag, mode, field in product(range(1,5),(1,2),(0,1,65535)):
        initial=authority(region=1);initial['references'][1]=reference(mode=mode)
        add(f'ref-get-{scalar_tag}-{mode}-{field}',[op(26,1,field)],
            [expected(error='' if field==0 else FIELD,stack=[value(scalar_tag)])],
            carrier=owner_model(tag=scalar_tag),commands=[(6,0,0,0),(8,1,0,mode)],state=initial)
        add(f'ref-set-permission-{scalar_tag}-{mode}-{field}',[op(27,1,field)],
            [expected(error='' if field==0 and mode==2 else FIELD,stack=[])],
            carrier=owner_model(tag=scalar_tag),commands=[(6,0,0,0),(8,1,0,mode)],state=initial,values=[value(scalar_tag)])
    add('reference-field-before-stack-type',[op(27,1,1)],[expected(error=FIELD)],
        commands=[(6,0,0,0),(8,1,0,2)],state={**authority(region=1),'references':[(0,)*10,reference(mode=2),(0,)*10,(0,)*10]})
    add('owner-round-trip',[op(11,0),op(12,1),op(14,1),op(13,0),op(12,0)],
        [expected(stack=[value(8,owned=True,layout=0)],live={0:False}),expected(stack=[],live={1:True}),
         expected(stack=[value(1)],live={1:False}),expected(stack=[value(8,owned=True,layout=0)]),expected(stack=[],live={0:True})])
    for code, slot in product((11,14),(0,1,2,3,4,65535)):
        valid=slot==0
        add(f'owner-source-{code}-{slot}',[op(code,slot)],
            [expected(error='' if valid else MOVE if code==11 else UNPACK,
                      stack=[value(8,owned=True,layout=0)] if code==11 else [value(1)],live={0:False} if valid else {})])
    for code, tag, observed, owned in product((11,14),range(256),(False,True),(False,True)):
        item=value(tag,observation=observed,owned=owned,root=0)
        error=(MOVE_OBSERVE if code==11 else UNPACK_OBSERVE) if observed else ''
        add(f'owner-observation-{code}-{tag}-{observed}-{owned}',[op(code,0)],
            [expected(error=error,stack=[item,value(8,owned=True,layout=0)] if code==11 else [item,value(1)],live={0:False})],values=[item])
    for tag, observed, owned in product(range(256),(False,True),(False,True)):
        item=value(tag,observation=observed,owned=owned,layout=0)
        error=TOKEN if not owned else '' if tag==8 else DESTINATION
        add(f'owner-store-{tag}-{observed}-{owned}',[op(12,1)],
            [expected(error=error,stack=[],live={1:True})],values=[item])
    for slot in (0,2,3,4,65535):
        add(f'owner-store-destination-{slot}',[op(12,slot)],[expected(error=DESTINATION)],values=[value(8,owned=True,layout=0)])
    add('owner-store-empty',[op(12,1)],[expected(error=TOKEN)])
    for scalar_tag, actual_tag, observed, owned in product(range(1,6),range(256),(False,True),(False,True)):
        item=value(actual_tag,observation=observed,owned=owned,layout=17)
        error='' if scalar_tag==actual_tag and not observed and not owned else PACK_FIELD
        add(f'owner-pack-{scalar_tag}-{actual_tag}-{observed}-{owned}',[op(13,0)],
            [expected(error=error,stack=[value(8,owned=True,layout=0)])],carrier=owner_model(tag=scalar_tag),values=[item])
    for fields in (0,1,2,255,256,257):
        values=[value(1)]*min(fields,256)
        error=PACK if fields>256 else ''
        add(f'pack-field-count-{fields}',[op(13,0)],[expected(error=error,stack=[value(8,owned=True,layout=0)])],carrier=owner_model(fields=fields),values=values)
    for count in (0,1,255,256):
        values=[value(4)]*count
        add(f'move-stack-{count}',[op(11,0)],
            [expected(error='I cannot extend my owned analysis stack' if count==256 else '',stack=values+[value(8,owned=True,layout=0)],live={0:False})],values=values)
        add(f'unpack-stack-{count}',[op(14,0)],
            [expected(error=UNPACK if count==256 else '',stack=values+[value(1)],live={0:False})],values=values)
    add('empty-pack-full-stack',[op(13,0)],[expected(error='I cannot extend my owned analysis stack')],carrier=owner_model(fields=0),values=[value(1)]*256)
    add('empty-unpack-full-stack',[op(14,0)],[expected(live={0:False})],carrier=owner_model(fields=0),values=[value(1)]*256)
    for arm in (0,1,2,3,UNKNOWN):
        initial=authority(7,0);item=value(10,owned=True,layout=1,variant=arm)
        error=ARM if arm==3 else ''
        unpack_error=UNPACK if arm==UNKNOWN else ''
        fields=1 if arm==0 else 0 if arm==1 else 2
        add(f'union-store-unpack-{arm}',[op(12,3),op(14,3)],
            [expected(error=error,stack=[],live={3:True},variants={3:arm}),
             expected(error=UNPACK if arm==3 else unpack_error,stack=[value(8,owned=True,layout=0)]*fields,live={3:False},variants={3:UNKNOWN})],
            carrier=union_model(True),state=initial,values=[item])
    add('union-selected-round-trip',[op(12,3),op(11,3),op(12,4),op(14,4),op(12,1),op(14,1)],
        [expected(stack=[],live={3:True},variants={3:0}),expected(stack=[value(10,owned=True,layout=1,variant=0)],live={3:False},variants={3:UNKNOWN}),
         expected(stack=[],live={4:True},variants={4:0}),expected(stack=[value(8,owned=True,layout=0)],live={4:False},variants={4:UNKNOWN}),
         expected(stack=[],live={1:True}),expected(stack=[value(1)],live={1:False})],
        carrier=union_model(True),state=authority(7,0),values=[value(10,owned=True,layout=1,variant=0)])
    add('structurally-equal-owned-array-store',[op(11,0),op(12,1),op(14,1),op(11,1)],
        [expected(stack=[value(7,owned=True,layout=2)],live={0:False}),expected(stack=[],live={1:True}),
         expected(error=UNPACK),expected(stack=[value(7,owned=True,layout=5)],live={1:False})],
        carrier=array_owner_model(),state=authority(3,1))
    for fields in (0,1,2):
        add(f'unpack-field-order-{fields}',[op(14,0)],[expected(stack=[value(1)]*fields,live={0:False})],carrier=owner_model(fields=fields))
    add('mixed-unpack-order',[op(14,0)],[expected(stack=[value(8,owned=True,layout=0),value(4)],live={0:False})],carrier=owner_model(mixed=True))
    add('mixed-pack-order',[op(13,1)],[expected(stack=[value(8,owned=True,layout=1)])],carrier=owner_model(mixed=True),values=[value(8,owned=True,layout=0),value(4)])
    add('mixed-pack-wrong-order',[op(13,1)],[expected(error=PACK_FIELD)],carrier=owner_model(mixed=True),values=[value(4),value(8,owned=True,layout=0)])
    good=add('valid-malformed-base',[op(22)],[expected(region=1)])
    malformed=[b'',words(9),words(10),words(9,0,0,0,0),words(9,65,0,0,0),words(9,1,2,0,0),good+b'\0']
    malformed += [good[:cut] for cut in (1,3,4,8,12,16,19,len(good)-1)]
    for instruction in [op(10),op(32),op(11,65536),op(24,1,65536)]:malformed.append(good[:-16]+words(*instruction))
    cases += [(f'malformed-{i}',raw,'D') for i,raw in enumerate(malformed)]
    return cases


class AffineOwners(unittest.TestCase):
    def test_original_frames_and_complete_owner_reference_transfer(self):
        work=Path(tempfile.mkdtemp(prefix='nano-affine-owners-'))
        old=flow_cases()[0]+queue_cases()+frame_cases();self.assertEqual(len(old),1001)
        reference_=Path(os.environ.get('NANOLANG_AFFINE_OWNERS_REFERENCE',ROOT/'obj/nvm2c_affine_owners_reference'))
        manifest=dict(scope='I check fourteen owner/reference steps and mutable authority, without graph/CFG/executable admission.',cases=[],routes=[]);prepared=[]
        for group,cases in [('original',old),('owners',owner_cases())]:
            paths=[]
            for index,(name,raw,e) in enumerate(cases):
                path=work/f'{group}-{index:05d}.bin';path.write_bytes(raw);paths.append(path)
                manifest['cases'].append(dict(group=group,name=name,path=str(path),sha256=hashlib.sha256(raw).hexdigest(),expected=e))
            prepared.append((group,cases,paths))
        def save():(work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        save();reports={};products=[('actual-c',[reference_])]+routes.build_routes(ROOT/'tests/nvm2c_affine_owners_driver.nano',work)
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
                            observed=text.split('~');self.assertEqual(len(observed),len(e))
                            for result,want in zip(observed,e):
                                valid,message,frame_,authority_=result.split('|',3);local,stack,globals_,context=frame_.split('/')
                                self.assertEqual(valid,str(int(want['valid'])));self.assertEqual(message,want['message']);self.assertEqual(stack,want['stack'])
                                self.assertEqual(globals_,'9,');self.assertEqual(context,'1,23,5,7,');self.assertEqual(authority_,want['authority'])
                if group not in reports:reports[group]=actual
                else:self.assertEqual(actual,reports[group],(label,group))
                manifest['routes'].append(dict(label=label,group=group,cases=len(cases),sha256=hashlib.sha256(raw).hexdigest()));save()


if __name__=='__main__':unittest.main()
