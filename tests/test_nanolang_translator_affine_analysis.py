"""I compare complete private body results, derived caches and converged facts."""
from pathlib import Path
import hashlib,json,os,re,struct,tempfile,unittest
from tests import nanolang_translator_routes as routes
from tests.test_nanolang_translator_affine_frame import words,flow_cases,queue_cases,frame_cases
from tests.test_nanolang_translator_ownership import desc,record,function,image,extension,effect_data,globals_data,union_data

ROOT=Path(__file__).resolve().parents[1]
NO=0xffffffff

def i64(value=0):return b'\x01'+struct.pack('<q',value)
def u16(op,*values):return bytes([op])+struct.pack('<'+'H'*len(values),*values)
def u32(op,value):return bytes([op])+words(value)
def definition(code,slots=(),parameters=0,result=(1,NO)):
    return dict(code=code,slots=slots,parameters=parameters,result=result)

def module(definitions,*,rows=(),flags=(),strings=(),globals_=0,effects=None,extra=()):
    extensions=list(extra)
    if globals_:extensions.append(extension(3,globals_data([struct.pack('<BBHI',1,1,0,NO)]*globals_)))
    if effects is not None:extensions.append(extension(4,effect_data(1,operations=effects)))
    declarations=[];entries=[];code=b''
    for d in definitions:
        slots=d['slots'];parameters=d['parameters'];tag,layout=d['result']
        declarations.append((len(slots),parameters,desc(tag,layout),[desc(*s) for s in slots]))
        entries.append(function(len(slots),parameters,results=int(tag!=0),result_tag=tag,params=[s[0] for s in slots[:parameters]],start=len(code),length=len(d['code'])))
        code+=d['code']
    return image(record(3 if extensions else 1,flags=flags,functions=declarations,extensions=extensions),rows=rows,strings=strings,functions=entries,code=code)

def assemble(parts):
    labels={};offset=0
    for part in parts:
        if isinstance(part,str):labels[part]=offset
        else:offset+=len(part) if isinstance(part,bytes) else 13 if part[0]=='handler' else 7 if part[0]=='match' else 5
    code=b''
    for part in parts:
        if isinstance(part,str):continue
        if isinstance(part,bytes):code+=part;continue
        relative=labels[part[2]]-len(code)
        if part[0]=='handler':code+=b'\xa5'+words(part[1])+struct.pack('<iHH',relative,part[3],part[4])
        elif part[0]=='match':code+=b'\x6b'+struct.pack('<Hi',part[1],relative)
        else:code+=bytes([part[0]])+struct.pack('<i',relative)
    return code,labels

def widths():
    text=(ROOT/'src_nano/nvm2c/isa_metadata.nano').read_text();array=re.search(r'nvm2c_primary_widths: array<int> = \[([^]]+)\]',text,re.S)[1]
    return [int(n) for n in re.findall(r'\d+',array)]

def instruction_count(code):
    sizes=widths();offset=0;count=0
    while offset<len(code):
        size=sizes[code[offset]]
        if not size or offset+size>len(code):return 0
        offset+=size;count+=1
    return count

def analysis_cases():
    cases=[]
    def add(name,definitions,*,function_=0,facts=True,capacity=None,seeds=2,expected=None,**kwargs):
        carrier=module(definitions,**kwargs)
        if capacity is None:capacity=instruction_count(definitions[function_]['code']) if function_<len(definitions) else 0
        raw=words(17,function_,int(facts),capacity,seeds,len(carrier))+carrier
        cases.append((name,raw,expected or {}))
    good=definition(i64()+b'\x3d')
    add('scalar-return',[good],expected=dict(result=(True,0,2,2,''),statuses={0:2},facts=[(0,True,0,0,0),(9,True,1,1,0)]))
    add('missing-return-operand',[definition(b'\x3d')],expected=dict(result=(False,0,1,1,'I require an exact declared result and no live owned obligations'),sentinels=True))
    add('closed-self-loop',[definition(b'\x38'+words(0))],expected=dict(result=(True,0,1,1,'')))
    add('fallthrough',[definition(b'\0')],expected=dict(result=(False,0,1,1,'I require an explicit return rather than fallthrough'),sentinels=True))
    add('empty-code',[definition(b'')],expected=dict(result=(False,0,0,0,'I exceeded my bounded affine instruction count'),sentinels=True))
    for capacity in (0,1,3,NO):add('fact-capacity-'+str(capacity),[good],capacity=capacity,expected=dict(result=(False,0,0,0,'I require exact instruction fact capacity'),sentinels=True))
    add('without-facts',[good],facts=False,expected=dict(result=(True,0,2,2,''),sentinels=True))
    code,labels=assemble([b'\x03\1',(57,0,'then'),i64(1),(56,0,'join'),'then',i64(2),'join',b'\x3d'])
    add('diamond',[definition(code)],expected=dict(result=(True,0,6,6,'')))
    code,labels=assemble([b'\x03\1',(57,0,'then'),i64(1),'jump',(56,0,'join'),'then',b'\x03\1','join',b'\0\x3d'])
    add('join-type-conflict',[definition(code)],expected=dict(result=(False,labels['jump'],6,6,'I require exact ownership and stack provenance at every join'),sentinels=True))
    code,labels=assemble([b'\x03\1',(57,0,'init'),b'\0\0\0\0',(56,0,'join'),'init',i64(2),u16(17,0),'join',u16(16,0),b'\x08',i64(),b'\x3d'])
    add('requeue-initialization-loss',[definition(code,[(1,NO,0)])],expected=dict(ok=False,sentinels=True))
    for depth in (2,8,9):
        defs=[definition(u32(59,i+1)+b'\x3d') for i in range(depth-1)]+[good]
        add('callee-chain-'+str(depth),defs,expected=dict(ok=depth<=8))
    add('cached-callee',[definition(u32(59,1)+b'\x08'+u32(59,1)+b'\x3d'),good],expected=dict(result=(True,0,4,4,''),statuses={0:2,1:2}))
    add('failed-callee',[definition(u32(59,1)+b'\x3d'),definition(b'\x3d')],expected=dict(result=(False,0,1,1,'I require complete consuming-helper owner resolution'),statuses={0:2,1:2},sentinels=True))
    add('positional-callee',[definition(i64(3)+u32(59,1)+b'\x3d'),definition(u16(16,0)+b'\x3d',[(1,NO,0)],1)],expected=dict(result=(True,0,3,3,''),statuses={0:2,1:2}))
    for helper_mode,borrow_mode,reference in [(1,1,1),(1,2,1),(2,2,1),(2,1,1),(1,2,0)]:
        root=i64()+u32(13,0)+u16(12,0)+b'\x16'+u16(24 if borrow_mode==1 else 25,1,0)+b'\x0f'+words(1)+struct.pack('<H',reference)+b'\x17'+u16(14,0)+b'\x08\x3d'
        defs=[definition(root,[(8,0,0),(1,NO,0)]),definition(u16(26,0,0)+b'\x3d',[(8,0,helper_mode)],1)]
        add(f'reference-call-{helper_mode}-{borrow_mode}-{reference}',defs,rows=[(0,NO,[(1,NO,NO)])],flags=[3],expected=dict(ok=reference==1 and (helper_mode==1 or borrow_mode==2)))
    for alias in (0,1):
        code,labels=assemble([('handler',alias,'arm',0,0),b'\xa7'+words(alias)+b'\0\0',u16(166,1),b'\x3d','arm',i64(8),b'\xa8'])
        add(f'handled-effect-alias-{alias}',[definition(code)],strings=[b'op',b'op'],effects=[(0,desc(1),[])],expected=dict(result=(True,0,6,6,''),keys={0:[]}))
    add('escaping-effect',[definition(b'\xa7'+words(0)+b'\0\0\x3d')],strings=[b'op'],effects=[(0,desc(1),[])],expected=dict(result=(True,0,2,2,''),keys={0:[0]}))
    add('derived-callee-escape',[definition(u32(59,1)+b'\x3d'),definition(b'\xa7'+words(0)+b'\0\0\x3d')],strings=[b'op'],effects=[(0,desc(1),[])],expected=dict(ok=True,keys={0:[0],1:[0]},statuses={0:2,1:2}))
    add('effect-global-private-boundary',[definition(u32(18,0)+b'\x08\xa7'+words(0)+b'\0\0\x3d')],globals_=1,strings=[b'op'],effects=[(0,desc(1),[])],expected=dict(ok=True,allowed={0:[3]},keys={0:[0]}))
    add('uninitialized-global-read',[definition(u32(18,0)+b'\x3d')],globals_=1,expected=dict(result=(False,5,2,2,'I require global initialization before reads and no repeated immutable store'),allowed={0:[2]},sentinels=True))
    add('initialized-global-read',[definition(i64()+u32(19,0)+u32(18,0)+b'\x3d')],globals_=1,expected=dict(ok=True,allowed={0:[3]},returns={0:[10]}))
    full=i64()*256
    add('late-global-load-cache',[definition(full+u32(18,0)+b'\x3d')],globals_=1,expected=dict(result=(False,len(full),257,257,'I cannot retain a checked global load'),allowed={0:[2]},sentinels=True))
    add('late-effect-cache',[definition(full+b'\xa7'+words(0)+b'\0\0\x3d')],strings=[b'op'],effects=[(0,desc(1),[])],expected=dict(result=(False,len(full),257,257,'I cannot retain a resumed effect value'),keys={0:[0]},sentinels=True))
    add('late-value-call-cache',[definition(full+u32(59,1)+b'\x3d'),definition(u32(18,0)+b'\x3d')],globals_=1,expected=dict(result=(False,len(full),257,257,'I cannot retain a checked value-call result'),allowed={0:[2],1:[2]},statuses={0:2,1:2},sentinels=True))
    add('owned-pack-unpack',[definition(i64()+u32(13,0)+u16(12,0)+u16(14,0)+b'\x3d',[(8,0,0)])],rows=[(0,NO,[(1,NO,NO)])],flags=[3],expected=dict(ok=True))
    types=[desc(1)]
    add('array-length-empty-effect-table-refusal',[definition(bytes([80,1,85,61]),[(7,0,0)])],extra=[extension(4,effect_data(2,types=types,operations=[]),2)],expected=dict(result=(False,0,0,0,'I require checked ownership declarations'),sentinels=True))
    add('array-length',[definition(bytes([80,1,85,61]),[(7,0,0)])],strings=[b'op'],extra=[extension(4,effect_data(2,types=types),2)],expected=dict(ok=True))
    variants=extension(1,union_data(0,[(1,0,0),(2,0,0)]))
    pack=lambda variant:b'\xfb'+bytes([1])+words(0)+struct.pack('<HH',variant,0)
    code,labels=assemble([pack(0),('match',0,'hit'),b'\x08',i64(2),b'\x3d','hit',b'\x08',i64(1),b'\x3d'])
    add('constructed-union-dead-edge',[definition(code)],rows=[(2,0,[])],flags=[1],strings=[b'U',b'A',b'B'],extra=[variants],expected=dict(ok=True))
    code,labels=assemble([u32(59,1),('match',0,'a'),('match',1,'b'),b'\x08\x03\1\x3d','a',b'\x08',i64(1),b'\x3d','b',b'\x08',i64(2),b'\x3d'])
    add('unknown-union-exhaustion',[definition(code),definition(pack(0)+b'\x3d',result=(10,0))],rows=[(2,0,[])],flags=[1],strings=[b'U',b'A',b'B'],extra=[variants],expected=dict(ok=True))
    sizes=widths()
    for opcode in range(256):
        body=bytes([opcode])+b'\0'*max(0,sizes[opcode]-1)+b'\xa3'
        add(f'primary-{opcode}',[definition(body)])
    good_raw=cases[0][1]
    malformed=[b'',words(17),good_raw+b'\0']+[good_raw[:n] for n in (1,4,8,12,16,20,23,len(good_raw)-1)]
    for index,value in [(1,2),(3,4097)]:
        bad=bytearray(good_raw);bad[4+index*4:8+index*4]=words(value);malformed.append(bytes(bad))
    cases.extend((f'malformed-{i}',raw,dict(malformed=True)) for i,raw in enumerate(malformed))
    return cases

def inspect_report(text,expected):
    if expected.get('malformed'):assert text=='D';return
    result,flags,global_rows,cache_rows,fact_rows=text.split('|')
    ok,offset,reachable,visits,message=result.split(',',4);value=(ok=='1',int(offset),int(reachable),int(visits),message)
    assert value[0]==(message=='')
    if 'result' in expected:assert value==tuple(expected['result']),(value,expected['result'])
    if 'ok' in expected:assert value[0]==expected['ok'],(value,expected)
    entries=[row.split('~') for row in cache_rows.split('#') if row];assert len(entries)==8
    for f,status in expected.get('statuses',{}).items():assert int(entries[int(f)][0])==status
    for key,column in [('allowed',0),('returns',1),('keys',2)]:
        for f,wanted in expected.get(key,{}).items():
            f=int(f)
            assert [int(n) for n in entries[f][2].split('/')[column].split(',') if n]==wanted,(key,f,entries[f],wanted)
    count,rest=fact_rows.split(',',1);facts=[tuple(int(n) for n in row.split(',') if n) for row in rest.split(';') if row];assert len(facts)==int(count)
    if 'facts' in expected:assert facts==[tuple(item) for item in expected['facts']]
    if expected.get('sentinels'):assert facts==[(1234567,1,7,99,11)]*2


class AffineAnalysis(unittest.TestCase):
    def test_original_frames_and_complete_private_body(self):
        work=Path(tempfile.mkdtemp(prefix='nano-affine-analysis-'))
        old=flow_cases()[0]+queue_cases()+frame_cases();self.assertEqual(len(old),1001)
        reference=Path(os.environ.get('NANOLANG_AFFINE_ANALYSIS_REFERENCE',ROOT/'obj/nvm2c_affine_analysis_reference'))
        manifest=dict(scope='I compare private body results/cache/facts; public effect-global proof and product admission remain separate.',cases=[],routes=[]);prepared=[]
        for group,cases in [('original',old),('analysis',analysis_cases())]:
            paths=[]
            for index,(name,raw,expected) in enumerate(cases):
                path=work/f'{group}-{index:05d}.bin';path.write_bytes(raw);paths.append(path)
                manifest['cases'].append(dict(group=group,name=name,path=str(path),sha256=hashlib.sha256(raw).hexdigest(),expected=expected))
            prepared.append((group,cases,paths))
        def save():(work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        save();reports={};products=[('actual-c',[reference])]+routes.build_routes(ROOT/'tests/nvm2c_affine_analysis_driver.nano',work)
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
                        if group=='analysis':inspect_report(text,expected)
                        elif name.startswith(('global-algebra-','bounded-fifo-')):self.assertEqual(text,expected)
                        else:self.assertTrue(text.startswith(expected))
                if group not in reports:reports[group]=actual
                else:self.assertEqual(actual,reports[group],(label,group))
                manifest['routes'].append(dict(label=label,group=group,cases=len(cases),sha256=hashlib.sha256(raw).hexdigest()));save()

if __name__=='__main__':unittest.main()
