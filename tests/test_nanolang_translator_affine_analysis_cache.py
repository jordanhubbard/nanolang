"""I compare cache-hit/recursion/early-refusal paths, without a fresh callee walk."""
from itertools import product
from pathlib import Path
import hashlib
import json
import os
import tempfile
import unittest
from tests import nanolang_translator_routes as routes
from tests.test_nanolang_translator_affine_frame import words, flow_cases, queue_cases, frame_cases
from tests.test_nanolang_translator_affine_scalar import scalar_batches
from tests.test_nanolang_translator_affine_call_context import context_model

ROOT=Path(__file__).resolve().parents[1]
NO=0xffffffff
DECLARATIONS='I require checked ownership declarations'
GLOBALS='I require a bounded acyclic value graph for global flow'
RECURSION='I refuse recursive owned value analysis'


def cache_cases():
    cases=[]
    def add(name,*,present=True,functions_present=True,functions=2,function=1,caller=False,globals_valid=True,
            graph=True,effect=False,globals_=1,status=2,ok=True,offset=42,reachable=7,visits=11,allowed=1,returns=6,keys=2,outside=False):
        missing=not present or not functions_present or function>=functions or not globals_valid
        global_refusal=bool(globals_) and not graph
        assert outside or missing or global_refusal or graph and not caller
        carrier=context_model([b'\x3d']*functions,[(1,1,0,NO)]*globals_)
        raw=words(15,int(present),int(functions_present),function,int(caller),int(globals_valid),int(graph),int(effect),status,int(ok),offset,reachable,visits,allowed,returns,keys,len(carrier))+carrier
        if outside:cases.append((name,raw,'D'));return raw
        states=[status]*8
        results=[(ok,offset,reachable,visits,f'cached-{f}') for f in range(8)]
        cached=not missing and not global_refusal and status==2
        result=results[function] if cached else (False,0,0,0,DECLARATIONS if missing else GLOBALS if global_refusal else RECURSION)
        if not cached and graph and not caller and function<8:states[function]=2;results[function]=result
        result_text=lambda r:f'{int(r[0])},{r[1]},{r[2]},{r[3]},{r[4]}'
        numbers=lambda row:''.join(f'{n},' for n in row)
        text=f'{int(cached)}|'+result_text(result)+f'|{int(globals_valid)},{int(graph)},{int(effect)},{globals_}|'
        text+='1,1,4294967295,;'*globals_+'|'
        for f in range(8):
            a=[(allowed+f+g)%4 for g in range(globals_)];r=[(returns+f+g)%16 for g in range(globals_)];k=[100*f+i for i in range(keys)]
            text+=f'{states[f]}~'+result_text(results[f])+'~'+numbers(a)+'/'+numbers(r)+'/'+numbers(k)+'#'
        cases.append((name,raw,text));return raw
    for present,functions_present,functions,function,caller,valid,graph,globals_,status,ok in product((False,True),(False,True),(2,8),(0,1,2,7,8,NO),(False,True),(False,True),(False,True),(0,1),(1,2),(False,True)):
        missing=not present or not functions_present or function>=functions or not valid
        global_refusal=bool(globals_) and not graph
        if not (missing or global_refusal or graph and not caller):continue
        add(f'guards-{present}-{functions_present}-{functions}-{function}-{caller}-{valid}-{graph}-{globals_}-{status}-{ok}',
            present=present,functions_present=functions_present,functions=functions,function=function,caller=caller,
            globals_valid=valid,graph=graph,globals_=globals_,status=status,ok=ok)
    for allowed,returns,status,effect in product(range(4),range(16),(1,2),(False,True)):
        add(f'vectors-{allowed}-{returns}-{status}-{effect}',allowed=allowed,returns=returns,status=status,effect=effect)
    for globals_,keys,status in product((0,1,255,256),(0,1,255,256),(1,2)):
        add(f'capacity-{globals_}-{keys}-{status}',globals_=globals_,keys=keys,status=status)
    for offset,reachable,visits,ok in product((0,NO),(0,NO),(0,NO),(False,True)):
        add(f'result-fields-{offset}-{reachable}-{visits}-{ok}',offset=offset,reachable=reachable,visits=visits,ok=ok)
    add('fresh-body-outside-protocol',graph=False,globals_=0,outside=True)
    add('caller-binding-outside-protocol',caller=True,outside=True)
    good=add('valid-malformed-base')
    malformed=[b'',words(15),good+b'\0']+[good[:n] for n in (1,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,67,len(good)-1)]
    for index,number in [(0,2),(1,2),(3,2),(4,2),(5,2),(6,2),(7,0),(7,3),(8,2),(12,4),(13,16),(14,257)]:
        bad=bytearray(good);bad[4+index*4:8+index*4]=words(number);malformed.append(bytes(bad))
    cases.extend((f'malformed-{i}',raw,'D') for i,raw in enumerate(malformed))
    return cases


class AffineAnalysisCache(unittest.TestCase):
    def test_original_frames_and_completed_cache_paths(self):
        work=Path(tempfile.mkdtemp(prefix='nano-affine-analysis-cache-'))
        old=flow_cases()[0]+queue_cases()+frame_cases();self.assertEqual(len(old),1001)
        reference=Path(os.environ.get('NANOLANG_AFFINE_ANALYSIS_CACHE_REFERENCE',ROOT/'obj/nvm2c_affine_analysis_cache_reference'))
        manifest=dict(scope='I compare completed/recursive/early-refusal cache paths; fresh body/caller-binding/whole-walk derivation remain outside this protocol.',cases=[],routes=[]);prepared=[]
        for group,cases in [('original',old),('cache',cache_cases())]:
            paths=[]
            for index,(name,raw,expected) in enumerate(cases):
                path=work/f'{group}-{index:05d}.bin';path.write_bytes(raw);paths.append(path)
                manifest['cases'].append(dict(group=group,name=name,path=str(path),sha256=hashlib.sha256(raw).hexdigest(),expected=expected))
            prepared.append((group,cases,paths))
        def save():(work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        save();reports={};products=[('actual-c',[reference])]+routes.build_routes(ROOT/'tests/nvm2c_affine_analysis_cache_driver.nano',work)
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
