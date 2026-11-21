"""I compare complete private declaration inventories and non-admitting statuses."""
from pathlib import Path
import hashlib
import json
import os
import random
import struct
import tempfile
import unittest
from tests.nanolang_translator_routes import build_routes, checked, source_compiler_command
from tests.native_process import run_captured
from tests.test_nanolang_translator_ownership import corpus as shared_corpus, image, record, desc, extension, function, union_data, u32, NONE
from tests.test_nanolang_translator_payloads import table, layout, change
ROOT=Path(__file__).resolve().parents[1]


def array_data(types=(),bindings=()):
    return u32(len(types))+b''.join(desc(tag,ref) for tag,ref in types)+u32(len(bindings))+b''.join(struct.pack('<IHHI',owner,field,0,target) for owner,field,target in bindings)


def map_data(scalars=(1,),pairs=((0,0),),bindings=((0,0,0),)):
    return u32(len(scalars))+b''.join(desc(tag) for tag in scalars)+u32(len(pairs))+b''.join(struct.pack('<II',*pair) for pair in pairs)+u32(len(bindings))+b''.join(struct.pack('<IHHI',owner,field,0,target) for owner,field,target in bindings)


def corpus(large=True):
    cases=[]
    def add(name,owned,expected=None,foreign=(0,0,0,0),**model):
        cases.append((name,struct.pack('<4I',*foreign)+image(owned,**model),expected))
    scalar=[(0,NONE,[(1,NONE,NONE)])]
    array=[(0,NONE,[(7,NONE,NONE)])]
    map_rows=[(0,NONE,[(13,NONE,NONE)])]
    arr=extension(2,array_data([(1,NONE)],[(0,0,0)]))
    mp=extension(5,map_data())
    for name,data,_ in shared_corpus():cases.append(('shared-'+name,bytes(16)+data,None))
    add('absent',b'',{1:1,2:1,3:0},raw_layouts=b'')
    add('missing-owned',b'',{1:1,2:1,3:2})
    for kind, valid_enums in ((1, 0), (3, 1)):
        for enums in (0, 1, 2):
            status = 1 if enums == valid_enums else 2
            add(f'wire-kind-{kind}-enums-{enums}', record(flags=[0]),
                {1:status, 2:status, 3:0}, rows=[(kind,NONE,[])], counts=(0,enums,0))

    for version in (1,2):
        add(f'scalar-v{version}',record(version,[1]),{1:0,2:0,3:0},rows=scalar)
    add('ordinary-array',record(3,[1],extensions=[arr]),{1:0,2:0,3:0},rows=array)
    add('missing-array-binding-extension',record(1,[1]),{1:1,2:1,3:0},rows=array)
    add('present-empty-array-binding-extension',record(3,[1],extensions=[extension(2,array_data())]),{1:2,2:2,3:0},rows=array)
    add('empty-array-table',record(3,[1],extensions=[extension(2,array_data())]),{1:0,2:0,3:0},rows=scalar)
    add('ordinary-map',record(3,[1],extensions=[mp]),{1:2,2:2,3:1},rows=map_rows)
    add('missing-map-extension',record(1,[1]),{1:2,2:2,3:2},rows=map_rows)
    names=[b'Choice',b'Only',b'value']
    union=[(2,0,[(1,NONE,2)])]
    unions=extension(1,union_data(0,[(1,0,1)]))
    add('mixed-scalar-union',record(3,[0],extensions=[unions]),{1:1,2:0,3:0},rows=union,strings=names)
    add('mixed-union-complete-flag',record(3,[1],extensions=[unions]),{1:2,2:2,3:0},rows=union,strings=names)
    hybrid=[(0,NONE,[(7,NONE,NONE),(13,NONE,NONE)]),(2,0,[(1,NONE,2)])]
    hybrid_owned=record(3,[1,0],extensions=[extension(1,union_data(1,[(1,0,1)])),arr,extension(5,map_data(bindings=[(0,1,0)]))])
    add('array-map-union-hybrid',hybrid_owned,{1:2,2:2,3:1},rows=hybrid,strings=names)
    for slot in range(4):
        foreign=[0]*4;foreign[slot]=1
        add(f'foreign-array-{slot}',record(3,[1],extensions=[arr]),{1:1,2:1,3:0},rows=array,foreign=foreign)
        add(f'foreign-map-{slot}',record(3,[1],extensions=[mp]),{1:2,2:2,3:2},rows=map_rows,foreign=foreign)
    for tag in range(256):
        data=array_data([(tag,NONE)],[(0,0,0)])
        status=0 if 1<=tag<=5 else 2
        add(f'array-element-tag-{tag}',record(3,[1],extensions=[extension(2,data)]),{1:status,2:status,3:0},rows=array)
        data=map_data(scalars=[tag])
        add(f'map-scalar-tag-{tag}',record(3,[1],extensions=[extension(5,data)]),{1:2,2:2,3:(1 if 1<=tag<=5 else 2)},rows=map_rows)
    for n in (1,2,3,4,63,64,65):
        types=[(1,NONE)]+[(7,i-1) for i in range(1,n)]
        status=0 if n<=3 else (1 if n<=64 else 3)
        add(f'array-depth-{n}',record(3,[1],extensions=[extension(2,array_data(types,[(0,0,n-1)]))]),{1:status,2:status,3:0},rows=array)
        # Forward order and cached depths must give the same classification.
        types=[(7,i+1) for i in range(n-1)]+[(1,NONE)]
        add(f'array-forward-depth-{n}',record(3,[1],extensions=[extension(2,array_data(types,[(0,0,0)]))]),{1:status,2:status,3:0},rows=array)
    for n in (1,2,64,65):
        types=[(7,(i+1)%n) for i in range(n)]
        status=3 if n>64 else 2
        add(f'array-cycle-{n}',record(3,[1],extensions=[extension(2,array_data(types,[(0,0,0)]))]),{1:status,2:status,3:0},rows=array)
    for n in (4095,4096,4097):
        data=array_data([(1,NONE)]*n)
        add(f'types-{n}',record(3,[1],extensions=[extension(2,data)]),{1:(0 if n<=4096 else 3),2:(0 if n<=4096 else 3),3:0},rows=scalar)
    borrowed=[(0,NONE,[(5,NONE,NONE)])]
    borrowed_function=[function(1,1,params=[8])]
    borrowed_owned=record(1,[3],functions=[(1,1,desc(),[desc(8,0,1)])])
    add('ambiguous-borrow',borrowed_owned,{1:1,2:1,3:0},rows=borrowed,functions=borrowed_function)
    add('ambiguous-borrow-before-later-error',borrowed_owned+b'\0',{1:1,2:2,3:2},rows=borrowed,functions=borrowed_function)
    for locals_,arity,results,tag,params,expected in [(1,0,0,0,(),1),(1,1,0,0,(13,),2),(0,0,1,13,(),2)]:
        descriptors=[desc(13)]*locals_
        returned=desc(13) if results else desc()
        add(f'map-function-{locals_}-{arity}-{results}',record(3,[1],functions=[(locals_,arity,returned,descriptors)],extensions=[mp]),{1:2,2:2,3:expected},rows=map_rows,functions=[function(locals_,arity,results,tag,params)])
    for n in (5,6):
        add(f'map-scalar-count-{n}',record(3,[1],extensions=[extension(5,map_data(scalars=list(range(1,n+1))))]),{1:2,2:2,3:(2 if n==5 else 3)},rows=map_rows)
    all_pairs=[(k,v) for k in range(5) for v in range(5)]
    all_maps=[(0,NONE,[(13,NONE,NONE)]*25)]
    add('all-25-map-pairs',record(3,[1],extensions=[extension(5,map_data(range(1,6),all_pairs,[(0,i,i) for i in range(25)]))]),{1:2,2:2,3:1},rows=all_maps)
    add('26-map-pairs',record(3,[1],extensions=[extension(5,map_data(range(1,6),all_pairs+[(0,0)]))]),{1:2,2:2,3:3},rows=map_rows)
    for name,data,rows_,strings in [('array',record(3,[1],extensions=[arr]),array,()),('map',record(3,[1],extensions=[mp]),map_rows,()),('mixed',hybrid_owned,hybrid,names)]:
        for n in range(len(data)):add(f'{name}-truncation-{n}',data[:n],rows=rows_,strings=strings)
        rng=random.Random(114230+len(data))
        for n in range(256):
            mutated=bytearray(data);mutated[rng.randrange(len(mutated))]^=1<<rng.randrange(8)
            add(f'{name}-mutation-{n}',bytes(mutated),rows=rows_,strings=strings)
    for locals_,expected in ((31774,0),(31775,3)):
        owned=record(1,functions=[(locals_,0,desc(),[desc()]*locals_)])
        add(f'descriptor-work-{locals_}',owned,{1:expected,2:expected,3:0},functions=[function(locals_)])
    # I solve actual precharged work, then use extra ARRAY visits to cross it.
    n,d=next((n,(1048575-9-2*n)//37) for n in range(2,4097) if (1048575-9-2*n)%37==0)
    for extra in (0,1,2):
        types=[(1,NONE)]*n
        for i in range(extra):types[i+1]=(7,0)
        add(f'exact-work-{1048575+extra}',record(3,[1],functions=[(d-1,0,desc(),[desc()]*(d-1))],extensions=[extension(2,array_data(types))]),{1:(0 if extra<=1 else 3),2:(0 if extra<=1 else 3),3:0},rows=scalar,functions=[function(d-1)])
    if large:
        fields=[(7,NONE,NONE)]*65535
        bigrows=[(0,NONE,fields),(0,NONE,[(7,NONE,NONE)])]
        bindings=[(0,i,0) for i in range(65535)]+[(1,0,0)]
        add('array-bindings-65536',record(3,[1,1],extensions=[extension(2,array_data([(1,NONE)],bindings))]),{1:0,2:0,3:0},rows=bigrows)
        add('array-bindings-65537',record(3,[1],extensions=[extension(2,u32(1)+desc(1)+u32(65537))]),{1:3,2:3,3:0},rows=array)
        bigrows=[(0,NONE,[(13,NONE,NONE)]*65535),(0,NONE,[(13,NONE,NONE)])]
        add('map-bindings-65536',record(3,[1,1],extensions=[extension(5,map_data(bindings=bindings))]),{1:2,2:2,3:1},rows=bigrows)
        add('map-bindings-65537',record(3,[1],extensions=[extension(5,u32(1)+desc(1)+u32(1)+bytes(8)+u32(65537))]),{1:2,2:2,3:3},rows=map_rows)
        for size,status in ((1398078,2),(1398079,3)):
            add(f'exact-map-work-payload-{size}',record(3,[1],extensions=[extension(5,bytes(size))]),{1:2,2:2,3:status},rows=map_rows)
        for size,status in ((16777216,2),(16777217,3)):
            add(f'ownership-byte-budget-{size}',record()+bytes(size-len(record())),{1:status,2:status,3:3})
    return cases


def producer_cases(work, root=None):
    root=Path(root) if root is not None else Path(tempfile.mkdtemp(prefix='declaration-producer-products-',dir=ROOT/'.test_output'))
    root.mkdir(parents=True,exist_ok=True)
    print('I retain actual producer modules/models at',root,flush=True)
    producers=[('c',[ROOT/'bin/nano_virt'])]
    source_command = source_compiler_command()
    if source_command:
        producers.append(('nanolang',source_command))
    cases=[];records=[]
    fixtures=[('array_record_fields.nano',0,''),('empty_array_append.nano',0,'first\nsecond\n'),
              ('affine_scalar_union_instances.nano',0,'right\nsemantic-source-parity-pass\n'),
              ('string_int_maps.nano',0,''),('string_string_maps.nano',0,''),
              ('map_nested_u8_borrowed.nano',0,''),('owned_mixed/basic.nano',7,'')]
    for source,expected_exit,expected_stdout in fixtures:
        fixture=ROOT/'tests/nanoisa/fixtures'/source
        for label,compiler in producers:
            stem=label+'-'+source.replace('/','_').removesuffix('.nano')
            module=root/(stem+'.nvm');model=root/(stem+'.bin')
            command=[*compiler,fixture,'--emit-nvm','-o',module]
            checked(command,build=True,timeout=180)
            result=run_captured([ROOT/'bin/nano_vm',module],cwd=ROOT,timeout=120)
            assert result.returncode==expected_exit,(source,label,result.returncode,result.stderr.decode(errors='replace'))
            assert result.stdout.decode()==expected_stdout,(source,label,result.stdout)
            checked([ROOT/'obj/nvm2c_declaration_reference','--extract',module,model],timeout=120)
            data=model.read_bytes()
            cases.append(('producer-'+stem,data,None))
            records.append(dict(name=stem,source=str(fixture.relative_to(ROOT)),source_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),module=module.name,module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),model=model.name,model_sha256=hashlib.sha256(data).hexdigest(),vm_exit=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode(errors='replace')))
            (root/'records.json').write_text(json.dumps(records,indent=2)+'\n')
    return cases


class Declarations(unittest.TestCase):
    def test_actual_provider_sanitizers(self):
        cases=corpus()
        with tempfile.TemporaryDirectory(prefix='nano-declaration-sanitizers-') as directory:
            paths=[]
            for i,(_,data,_) in enumerate(cases):
                p=Path(directory)/str(i);p.write_bytes(data);paths.append(p)
            for profile in (1,2,3):
                for i in range(0,len(paths),32):
                    args=[str(profile),*paths[i:i+32]]
                    actual=checked([ROOT/'obj/nvm2c_declaration_reference_asan',*args],timeout=600)
                    expected=checked([ROOT/'obj/nvm2c_declaration_reference',*args],timeout=600)
                    self.assertEqual(actual,expected)
        print(f'I matched {len(cases)} reports per profile through the instrumented declaration/layout/retained/reference/cursor/ISA providers; no allocation denial was injected.')

    def test_private_profiles_and_complete_queries(self):
        cases=corpus()
        with tempfile.TemporaryDirectory(prefix='nano-declaration-profiles-') as directory:
            work=Path(directory);paths=[]
            cases+=producer_cases(work)
            for i,(_,data,_) in enumerate(cases):
                p=work/f'case-{i}.bin';p.write_bytes(data);paths.append(p)
            routes=build_routes(ROOT/'tests/nvm2c_declaration_driver.nano',work)
            def execute(command,profile):
                values=[]
                for i in range(0,len(paths),32):values+=checked([*command,str(profile),*paths[i:i+32]],timeout=600).splitlines()
                self.assertEqual(len(values),len(cases));return values
            for profile in (1,2,3):
                reference=execute([ROOT/'obj/nvm2c_declaration_reference'],profile)
                for (name,_,expected),value in zip(cases,reference):
                    if expected is not None:
                        with self.subTest(oracle=name,profile=profile):self.assertEqual(int(value.split('|')[0]),expected[profile])
                for route,command in routes:
                    actual=execute(command,profile)
                    for (name,_,_),expected,value in zip(cases,reference,actual):
                        with self.subTest(route=route,profile=profile,case=name):self.assertEqual(value,expected)
            print(f'I matched {len(cases)} complete declaration/query reports per profile across three profiles and {len(routes)} routes; {sum(e is not None for _,_,e in cases)*3} explicit status controls retain original count/depth/work limits.')


if __name__=='__main__':unittest.main()
