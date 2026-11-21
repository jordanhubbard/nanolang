"""I compare actual rich service adapters without granting module execution."""
from pathlib import Path
import copy,json,os,random,struct,tempfile,unittest
from tests.nanolang_translator_routes import build_routes,checked
from tests.test_nanolang_translator_service_nominal import model,producer_cases
from tests.test_nanolang_translator_services import fixed,multi
from tests.test_nanolang_translator_payloads import layout,table
NONE=0xffffffff
ROOT=Path(__file__).resolve().parents[1]


def view(cats=(1,),legacy=False):
    source=model(list(cats));sigs=[];functions=[];imports=[]
    for raw in source['functions']:
        start,length,arity,locals_,upvalues,results,tag,n=struct.unpack_from('<8I',raw)
        functions.append([len(sigs),locals_,start,length]);sigs.append([arity,results,raw[32:32+n],bytes([tag])*results])
    for mod,sym,kind,arity,result,params in source['imports']:
        imports.append([mod,sym,len(sigs),kind]);sigs.append([arity,1,params,bytes([result])])
    profile=2 if cats==(1,) else 3 if cats==(2,) else 4 if cats==(3,) else 5
    service=multi(source['bindings']) if profile==5 else fixed(profile,source['bindings'][0][1][:len(source['imports'])],source['bindings'][0][2][:len(source['rows'])])
    if legacy:service=fixed(1,list(range(5)),[])
    return dict(features=512,links=0,callbacks=0,constants=[[5,n] for n in source['names']],signatures=sigs,functions=functions,imports=imports,layouts=table(*(layout(k,f,n) for k,n,f in source['rows'])),code=source['code'],service=service,ownership=source['ownership'])


def packet(m):
    const=m['constants'];sigs=m['signatures'];funcs=m['functions'];imports=m['imports']
    result=struct.pack('<11I',m['features'],m['links'],m['callbacks'],len(const),len(sigs),len(m['layouts']),len(funcs),len(imports),len(m['code']),len(m['service']),len(m['ownership']))
    for tag,data in const:result+=struct.pack('<II',tag,len(data))+data
    for arity,results,params,returns in sigs:result+=struct.pack('<4I',arity,results,len(params),len(returns))+params+returns
    for sig,locals_,start,length in funcs:result+=struct.pack('<IIQQ',sig,locals_,start,length)
    for row in imports:result+=struct.pack('<4I',*row)
    return result+m['layouts']+m['code']+m['service']+m['ownership']


def from_module(data):
    sections={}
    for i in range(struct.unpack_from('<I',data,24)[0]):
        kind,_,start,size=struct.unpack_from('<IIQQ',data,40+24*i);sections[kind]=data[start:start+size]
    constants=[];raw=sections.get(2,bytes(4));p=4
    for i in range(struct.unpack_from('<I',raw)[0]):
        tag=raw[p];size=struct.unpack_from('<I',raw,p+4)[0];constants.append([tag,raw[p+8:p+8+size]]);p=(p+8+size+3)&~3
    signatures=[];raw=sections.get(3,bytes(4));p=4
    for i in range(struct.unpack_from('<I',raw)[0]):
        arity,results=struct.unpack_from('<HH',raw,p);p+=4;params=raw[p:p+arity];p=(p+arity+3)&~3;returns=raw[p:p+results];p=(p+results+3)&~3
        signatures.append([arity,results,params,returns])
    functions=[];raw=sections.get(5,bytes(4))
    for i in range(struct.unpack_from('<I',raw)[0]):
        name,sig,start,length,locals_,upvalues,stack,_=struct.unpack_from('<IIQQHHHH',raw,4+32*i);functions.append([sig,locals_,start,length])
    raw=sections.get(8,bytes(4));imports=[list(struct.unpack_from('<IIIB3x',raw,4+16*i)) for i in range(struct.unpack_from('<I',raw)[0])]
    return dict(features=struct.unpack_from('<I',data,8)[0],links=struct.unpack_from('<I',sections.get(9,bytes(4)))[0],callbacks=struct.unpack_from('<I',sections.get(11,bytes(4)))[0],constants=constants,signatures=signatures,functions=functions,imports=imports,layouts=sections.get(4,bytes(4)),code=sections.get(6,b''),service=sections.get(14,b''),ownership=sections.get(13,b''))


def corpus():
    cases=[]
    def add(name,m,status=None,route=0):cases.append((route,name,packet(m),status))
    for cats in ((1,),(2,),(3,),(1,2,1,3)):
        base=view(cats);label='-'.join(map(str,cats));status=8 if cats==(3,) else 0
        add(label+'-shared',base,status);add(label+'-private',base,0 if cats==(3,) else 8,1)
        route=1 if cats==(3,) else 0
        add(label+'-exact-selected',base,0,route)
        for feature in (0,1,511,512,513,NONE):
            m=copy.deepcopy(base);m['features']=feature;add(label+f'-feature-{feature}',m,0,route)
        for i in range(len(base['service'])):
            m=copy.deepcopy(base);m['service']=m['service'][:i];add(label+f'-truncation-{i}',m,8,route)
        for tag in range(256):
            m=copy.deepcopy(base);m['constants'][0][0]=tag;add(label+f'-constant-{tag}',m,0 if tag==5 else 8,route)
            m=copy.deepcopy(base);sig=m['imports'][0][2];m['signatures'][sig][3]=bytes([tag]);add(label+f'-result-{tag}',m,0 if tag==10 else 8,route)
        for i in range(len(base['imports'])):
            for field,value in [(0,NONE),(1,NONE),(2,NONE),(3,0)]:
                m=copy.deepcopy(base);m['imports'][i][field]=value;add(label+f'-import-{i}-{field}',m,8,route)
        for index in (0,NONE,len(base['signatures'])):
            m=copy.deepcopy(base);m['functions'][0][0]=index;add(label+f'-function-signature-{index}',m,0 if index==0 else 8,route)
        for count in (0,2,255,256,65535):
            m=copy.deepcopy(base);m['signatures'][0][1]=count;m['signatures'][0][3]=bytes([1])*count
            add(label+f'-function-results-{count}',m,8,route)
        for params,results in ((b'',b'\x01'),(b'\x08',b'')):
            m=copy.deepcopy(base);m['signatures'][0][2]=params;m['signatures'][0][3]=results
            add(label+f'-function-missing-storage-{len(params)}-{len(results)}',m,8,route)
        for key in ('links','callbacks'):
            m=copy.deepcopy(base);m[key]=1;add(label+'-'+key,m,8,route)
        m=copy.deepcopy(base);m['constants'][0][0]=1
        raw=bytearray(m['layouts']);struct.pack_into('<I',raw,4+8+8+4,1);m['layouts']=bytes(raw)
        add(label+'-edge-before-constant',m,13,route)
        raw[4]=255;m['layouts']=bytes(raw);add(label+'-kind-before-edge',m,8,route)
        m=copy.deepcopy(base);m['ownership']=b'';m['constants'][0][0]=1;add(label+'-missing-owner-before-constant',m,8,route)
        # Unused rich signatures are not a whole-module validation claim.
        m=copy.deepcopy(base);m['signatures'].append([65535,65535,b'',b'']);add(label+'-unused-missing-signature-storage',m,0,route)
        rng=random.Random(114222+len(base['layouts']))
        for i in range(100):
            m=copy.deepcopy(base);raw=bytearray(m['service']);raw[rng.randrange(len(raw))]^=1<<rng.randrange(8);m['service']=bytes(raw);add(label+f'-wire-mutation-{i}',m,None,route)
    legacy=view(legacy=True)
    add('legacy-valid',legacy,0)
    for tag in range(256):
        m=copy.deepcopy(legacy);m['constants'][m['imports'][0][0]][0]=tag;add(f'legacy-module-tag-{tag}',m,0 if tag==5 else 8)
        m=copy.deepcopy(legacy);m['constants'].append([tag,b'ignored']);add(f'legacy-unused-constant-{tag}',m,0)
    for i in range(5):
        m=copy.deepcopy(legacy);m['imports'][i][2]=NONE;add(f'legacy-signature-index-{i}',m,13)
        if i<4:
            m=copy.deepcopy(legacy);m['imports'][i][1]=NONE;m['imports'][i+1][2]=NONE;add(f'legacy-entry-order-{i}',m,8)
    for results in (0,2,65535):
        m=copy.deepcopy(legacy);sig=m['imports'][0][2];m['signatures'][sig][1]=results;m['signatures'][sig][3]=bytes([10])*results;add(f'legacy-result-count-{results}',m,8)
    m=copy.deepcopy(legacy);m['functions'][0][0]=NONE;m['ownership']=b'';m['layouts']=bytes(4);add('legacy-ignores-function-ownership-layout',m,0)
    for flags in (0,512):
        m=copy.deepcopy(legacy);m['service']=b'';m['imports']=[];m['features']=flags;m['code']=b'';m['functions']=[]
        add(f'absent-feature-{flags}',m,8 if flags else 0)
    for op in range(256):
        for start,length in ((0,1),(0,0),(1,1),(2**64-1,1),(0,2**64-1)):
            m=copy.deepcopy(legacy);m['service']=b'';m['imports']=[];m['features']=0;m['code']=bytes([op]);m['functions']=[[NONE,0,start,length]]
            add(f'wide-scan-{op}-{start}-{length}',m,8 if start==0 and length==1 and 145<=op<=152 else 0)
    for count in (7,8,65536,65537):
        m=view();m['layouts']=table(*(layout(1,[]) for _ in range(count)))
        add(f'layout-count-{count}',m,8)
    for count in (0,4,5,320,321):
        m=view();m['imports']=(m['imports']*65)[:count];add(f'import-count-{count}',m,0 if count==5 else 8)
    m=view((1,)*64);add('all64-instances',m,0)
    return cases


class ServiceV2(unittest.TestCase):
    def compare(self,commands):
        cases=corpus()
        with tempfile.TemporaryDirectory(prefix='nano-service-v2-') as directory:
            work=Path(directory);products=work/'products';producer_cases(products)
            for record in json.loads((products/'records.json').read_text()):
                m=from_module((products/record['module']).read_bytes());profile=record['profile'];cases.append((1 if profile==4 else 0,'producer-'+str(profile)+'-'+record['producer'],packet(m),0))
            groups={}
            for i,(route,name,data,status) in enumerate(cases):
                path=work/f'{i}.bin';path.write_bytes(data);groups.setdefault(route,[]).append((name,path,status))
            routes=commands(work)
            for mode,items in groups.items():
                paths=[p for _,p,_ in items]
                def execute(command):
                    output=[]
                    for i in range(0,len(paths),32):output+=checked([*command,str(mode),*paths[i:i+32]],timeout=600).splitlines()
                    self.assertEqual(len(output),len(items));return output
                reference=execute([ROOT/'obj/nvm2c_service_v2_reference'])
                for (name,_,status),report in zip(items,reference):
                    if status is not None:
                        with self.subTest(oracle=name):self.assertEqual(int(report.split('|')[0]),status)
                    if name.startswith('wide-scan-'):
                        with self.subTest(classification=name):self.assertEqual(report,f'{status}|0,{int(status==8)}')
                for route,command in routes:
                    actual=execute(command)
                    for (name,_,_),expected,value in zip(items,reference,actual):
                        with self.subTest(route=route,case=name):self.assertEqual(value,expected)
        return len(cases),len(routes)
    def test_rich_adapters_and_classification(self):
        count,routes=self.compare(lambda work:build_routes(ROOT/'tests/nvm2c_service_v2_driver.nano',work));print(f'I matched {count} complete rich-adapter reports across {routes} routes.')
    def test_actual_rich_provider_sanitizers(self):
        count,_=self.compare(lambda work:[('actual-rich-asan-ubsan',[ROOT/'obj/nvm2c_service_v2_reference_asan'])]);print(f'I matched {count} reports through actual instrumented rich providers.')


if __name__=='__main__':unittest.main()
