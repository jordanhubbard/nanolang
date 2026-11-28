"""I compare public affine results and staged facts with unchanged C."""
from itertools import product
from pathlib import Path
import hashlib,json,os,random,struct,tempfile,unittest
from tests import nanolang_translator_routes as routes
from tests.test_nanolang_translator_affine_frame import words,flow_cases,queue_cases,frame_cases
from tests.test_nanolang_translator_affine_analysis import analysis_cases,definition,module,assemble,i64,u16,u32,instruction_count
from tests.test_nanolang_translator_ownership import desc,extension,globals_data

ROOT=Path(__file__).resolve().parents[1]
NO=0xffffffff
BUFFER='I require a bounded instruction fact buffer'
EFFECT_GLOBAL='I require global initialization before reads and no repeated immutable store across effects'

def public_cases():
    cases=[]
    def add(name,carrier,*,mode=1,function=0,present=True,capacity=1,seeds=2,limit=262144,expected=None):
        want={} if expected is None else dict(expected)
        if mode!=1:want['sentinels']=True
        cases.append((name,words(18,mode,function,int(present),capacity,seeds,limit,len(carrier))+carrier,want))
    for name,raw,expected in analysis_cases():
        if expected.get('malformed'):continue
        _,function,facts,capacity,seeds,size=struct.unpack('<6I',raw[:24]);carrier=raw[24:];assert len(carrier)==size
        for mode in (0,1):
            want={k:v for k,v in expected.items() if k in ['result','ok','facts','sentinels']}
            if mode==0:
                want.pop('facts',None)
                if name.startswith('fact-capacity-'):want=dict(result=(True,0,2,2,''))
            else:
                if capacity==0 or capacity>4096:want=dict(result=(False,0,0,0,BUFFER),sentinels=True)
                if name=='without-facts':want=dict(result=(True,0,2,2,''),facts=[(0,1,0,0,0),(9,1,1,1,0)])
            if name=='effect-global-private-boundary':want=dict(result=(False,0,0,1,EFFECT_GLOBAL),sentinels=True)
            add(f'body-{mode}-{name}',carrier,mode=mode,function=function,capacity=capacity,seeds=seeds,expected=want)
    good=module([definition(i64()+b'\x3d')])
    for present,capacity,function in product((False,True),(0,1,2,3,4096,4097,NO),(0,1,NO)):
        invalid=not present or not capacity or capacity>4096
        add(f'buffer-guards-{present}-{capacity}-{function}',good,present=present,capacity=capacity,function=function,
            expected=dict(result=(False,0,0,0,BUFFER),sentinels=True) if invalid else {})
    operations=[(0,desc(1),[])]
    def effect_model(arm,normal=None,*,aliases=(b'op',b'op'),key=0,callee=False):
        if normal is None:normal=b'\xa7'+words(key)+b'\0\0\x08'+u32(18,0)+u16(166,1)+b'\x3d'
        code,labels=assemble([('handler',0,'arm',0,0),normal,'arm',arm])
        defs=[definition(code)]
        if callee:defs.append(definition(b'\xa7'+words(key)+b'\0\0\x3d'))
        return module(defs,strings=aliases,globals_=1,effects=operations),labels,code
    for key,store,resume in product((0,1),(False,True),(False,True)):
        arm=(i64(7)+u32(19,0) if store else b'')+i64(8)+(b'\xa8' if resume else b'\x3d')
        carrier,labels,code=effect_model(arm,key=key)
        ok=store or not resume
        for mode in (0,1,2):
            want=dict(ok=ok)
            if not ok:want=dict(result=(False,21,0,6,EFFECT_GLOBAL),sentinels=True)
            add(f'effect-store-resume-{key}-{store}-{resume}-{mode}',carrier,mode=mode,capacity=instruction_count(code),expected=want)
    for resume in (False,True):
        normal=u32(59,1)+b'\x08'+u32(18,0)+u16(166,1)+b'\x3d'
        arm=i64(9)+(b'\xa8' if resume else b'\x3d')
        carrier,labels,code=effect_model(arm,normal=normal,callee=True)
        for mode in (0,1,2):
            add(f'lexical-owner-return-{resume}-{mode}',carrier,mode=mode,capacity=instruction_count(code),expected=dict(ok=not resume,sentinels=resume))
    # An escaping perform is terminal to the global proof, so its syntactic continuation is unreachable.
    escaping=b'\xa7'+words(0)+b'\0\0\x08'+u32(18,0)+b'\x3d'
    carrier=module([definition(escaping)],strings=[b'op'],globals_=1,effects=operations)
    for mode in (0,1,2):add(f'escaping-terminal-{mode}',carrier,mode=mode,capacity=4,expected=dict(ok=True))
    # Reentrant operations grow exact dynamic activations until the runtime depth trap.
    recursive_arm=b'\xa7'+words(0)+b'\0\0\xa8'
    carrier,labels,code=effect_model(recursive_arm)
    for mode in (0,1,2):add(f'runtime-depth-terminal-{mode}',carrier,mode=mode,capacity=instruction_count(code),expected=dict(ok=True))
    for limit in (0,1,2,3,7,31,32,33,63,64):
        error='I cannot retain bounded effect/global activation states' if limit==0 else 'I exceeded my effect/global control-state storage budget'
        add('bounded-reentrant-'+str(limit),carrier,mode=2,limit=limit,expected=dict(ok=False,error=error))
    # Public wrappers ignore the diagnostic budget field and always use their original bound.
    for mode in (0,1):add(f'public-original-bound-{mode}',carrier,mode=mode,limit=0,capacity=instruction_count(code),expected=dict(ok=True))
    chain=module([definition(u32(59,1)+b'\x3d'),definition(u32(59,2)+b'\x3d'),definition(i64()+b'\x3d')])
    add('activation-limit',chain,mode=2,limit=2,expected=dict(result=(False,0,0,2,'I cannot retain bounded effect/global activation states')))
    for op,code,error in [
        (0,b'\0','I require an exact effect/global continuation'),
        (166,u16(166,1)+b'\xa3','I cannot pop an absent or inherited effect/global handler'),
        (168,b'\xa8','I require an active effect/global resumption'),
        (18,u32(18,1)+b'\xa3','I require a declared effect/global slot'),
        (56,b'\x38'+words(5),'I require an exact effect/global branch target')]:
        add(f'direct-guard-{op}',module([definition(code)],globals_=1),mode=2,expected=dict(result=(False,0,0,1,error)))
    for branch in (57,58):
        code,labels=assemble([b'\x03\1',(branch,0,'read'),i64(),u32(19,0),'read',u32(18,0),b'\x3d'])
        for mode in (0,1,2):add(f'joined-global-{branch}-{mode}',module([definition(code)],globals_=1),mode=mode,capacity=instruction_count(code),expected=dict(ok=False,sentinels=True))
    immutable=[extension(3,globals_data([struct.pack('<BBHI',1,0,0,NO)]))]
    code,labels=assemble(['loop',u32(19,0),(56,0,'loop')])
    add('immutable-canonical-store-requeue',module([definition(code)],extra=immutable),mode=2,expected=dict(result=(False,0,0,3,EFFECT_GLOBAL)))
    code,labels=assemble([('handler',0,'arm',0,0),'loop',i64(),u32(19,0),b'\xa7'+words(0)+b'\0\0\x08',(56,0,'loop'),'arm',i64(),b'\xa8'])
    carrier=module([definition(code)],extra=immutable,strings=[b'op'],effects=operations)
    for mode in (0,1):add(f'immutable-resumed-store-{mode}',carrier,mode=mode,capacity=instruction_count(code),expected=dict(ok=False,error=EFFECT_GLOBAL,sentinels=True))
    rng=random.Random(20261125)
    tuples=[(0,0,0,0,0),(NO,0,NO,NO,NO),(1,2,3,4,5)]+[tuple(rng.randrange(2**32) for _ in range(5)) for _ in range(128)]
    for index,values in enumerate(tuples):
        value=2166136261
        for item in values:value=((value^item)*16777619)%(2**32)
        cases.append((f'hash-{index}',words(19,*values),dict(hash=value%65536)))
    good_raw=next(raw for name,raw,_ in cases if name=='body-1-scalar-return')
    malformed=[b'',words(18),good_raw+b'\0']+[good_raw[:n] for n in (1,4,8,12,16,20,24,28,31,len(good_raw)-1)]
    for index,value in [(0,3),(2,2),(4,4097)]:
        bad=bytearray(good_raw);bad[4+index*4:8+index*4]=words(value);malformed.append(bytes(bad))
    malformed+=[words(19),words(19,1,2,3,4),words(19,1,2,3,4,5,6)]
    cases.extend((f'malformed-{i}',raw,dict(malformed=True)) for i,raw in enumerate(malformed))
    return cases

def inspect_report(text,expected):
    if expected.get('malformed'):assert text=='D';return
    if 'hash' in expected:assert text==str(expected['hash'])+',';return
    result,fact_rows=text.split('|');ok,offset,reachable,visits,message=result.split(',',4)
    value=(ok=='1',int(offset),int(reachable),int(visits),message);assert value[0]==(message=='')
    if 'result' in expected:assert value==tuple(expected['result']),(value,expected['result'])
    if 'ok' in expected:assert value[0]==expected['ok'],(value,expected)
    if 'error' in expected:assert message==expected['error']
    count,rest=fact_rows.split(',',1);facts=[tuple(int(n) for n in row.split(',') if n) for row in rest.split(';') if row];assert len(facts)==int(count)
    if 'facts' in expected:assert facts==[tuple(item) for item in expected['facts']]
    if expected.get('sentinels'):assert facts==[(1234567,1,7,99,11)]*2

class AffinePublic(unittest.TestCase):
    def test_original_frames_and_public_effect_globals(self):
        work=Path(tempfile.mkdtemp(prefix='nano-affine-public-'))
        old=flow_cases()[0]+queue_cases()+frame_cases();self.assertEqual(len(old),1001)
        reference=Path(os.environ.get('NANOLANG_AFFINE_PUBLIC_REFERENCE',ROOT/'obj/nvm2c_affine_public_reference'))
        manifest=dict(scope='I compare both public checkers, bounded private effect/global diagnostics and staged facts.',cases=[],routes=[]);prepared=[]
        for group,cases in [('original',old),('public',public_cases())]:
            paths=[]
            for index,(name,raw,expected) in enumerate(cases):
                path=work/f'{group}-{index:05d}.bin';path.write_bytes(raw);paths.append(path)
                manifest['cases'].append(dict(group=group,name=name,path=str(path),sha256=hashlib.sha256(raw).hexdigest(),expected=expected))
            prepared.append((group,cases,paths))
        def save():(work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        save();reports={};products=[('actual-c',[reference])]+routes.build_routes(ROOT/'tests/nvm2c_affine_public_driver.nano',work)
        self.assertEqual(len(products),6 if routes.source_compiler_command() else 4)
        for label,command in products:
            for group,cases,paths in prepared:
                chunks=[paths[i:i+8] for i in range(0,len(paths),8)] if group=='original' else [[p] for p in paths]
                actual=[]
                for chunk in chunks:actual+=routes.checked([*command,*chunk],timeout=600 if group=='original' else 60).splitlines()
                self.assertEqual(len(actual),len(cases),(label,group));raw=('\n'.join(actual)+'\n').encode();(work/f'{group}-{label}.reports').write_bytes(raw)
                if group=='original':self.assertEqual(hashlib.sha256(raw).hexdigest(),'f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0')
                for (name,_,expected),text in zip(cases,actual):
                    with self.subTest(route=label,group=group,case=name):
                        if group=='public':inspect_report(text,expected)
                        elif name.startswith(('global-algebra-','bounded-fifo-')):self.assertEqual(text,expected)
                        else:self.assertTrue(text.startswith(expected))
                if group not in reports:reports[group]=actual
                else:self.assertEqual(actual,reports[group],(label,group))
                manifest['routes'].append(dict(label=label,group=group,cases=len(cases),sha256=hashlib.sha256(raw).hexdigest()));save()

if __name__=='__main__':unittest.main()
