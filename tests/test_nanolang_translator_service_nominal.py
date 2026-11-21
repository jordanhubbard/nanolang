"""I compare complete nominal metadata plans without granting service execution."""
from pathlib import Path
import copy,hashlib,json,os,random,struct,tempfile,unittest
from tests.nanolang_translator_routes import build_routes,checked,source_compiler_command
from tests.test_nanolang_translator_ownership import image,record,desc,function,u32,NONE
from tests.test_nanolang_translator_services import fixed,multi
from tests.test_nanolang_translator_payloads import table,layout
ROOT=Path(__file__).resolve().parents[1]
DOCS={i:json.loads((ROOT/'tests/fixtures'/('nsi_'+name+'_plan.json')).read_text()) for i,name in [(1,'file'),(2,'socket'),(3,'websocket')]}


def model(catalogs,interleave=False,reverse=False,extra=0):
    names=[]
    def name(text):
        value=text.encode() if isinstance(text,str) else text
        if value not in names:names.append(value)
        return names.index(value)
    order=[(i,j) for i,c in enumerate(catalogs) for j in range(len(DOCS[c]['types']))]
    if interleave:order.sort(key=lambda item:(item[1],item[0]))
    positions={key:extra+i for i,key in enumerate(order)}
    def member(i,id):
        if not id:return 0,NONE
        if id in ('nsi:core/int','nsi:core/bool','nsi:core/string'):return {'nsi:core/int':1,'nsi:core/bool':4,'nsi:core/string':5}[id],NONE
        types=DOCS[catalogs[i]]['types'];j=next(j for j,t in enumerate(types) if t['id']==id)
        return (10 if types[j]['kind']=='variant' else 8),positions[i,j]
    rows=[(1,NONE,[])]*extra;flags=[0]*extra
    for i,j in order:
        t=DOCS[catalogs[i]]['types'][j];members=t.get('fields',t.get('cases',[]))
        rows.append((2 if t['kind']=='variant' else 0,name(t['id']),[(*member(i,m.get('type')),name(m['id'])) for m in members]))
        flags.append(3 if j in (0,3) else 1)
    imports=[];bindings=[]
    for i,cat in enumerate(catalogs):
        indices=[]
        for method in DOCS[cat]['methods']:
            params=method['params'][:-1]
            indices.append(len(imports));imports.append([name(DOCS[cat]['interface']['id']),name(method['id']),3,len(params),10,bytes(member(i,p['type'])[0] for p in params)])
        bindings.append([cat,indices+[NONE]*(5-len(indices)),[positions[i,j] for j in range(len(DOCS[cat]['types']))]+[NONE]*(9-len(DOCS[cat]['types']))])
    if reverse:
        imports.reverse()
        for cat,indices,layouts in bindings:
            for i,index in enumerate(indices):
                if index!=NONE:indices[i]=len(imports)-1-index
    functions=[function(arity=1,locals_=3,results=1,result_tag=1,params=[8],start=NONE,length=NONE)]
    own=record(1,flags,[(3,1,desc(1),[desc(8,positions[0,0],2),desc(1),desc(10,positions[0,3])])])
    return dict(names=names,rows=rows,flags=flags,functions=functions,ownership=own,imports=imports,bindings=bindings,code=b'\xff',refs=0,callbacks=0)


def packet(m,profile):
    imports=m['imports']
    service=m.get('service')
    if service is None:
        if profile==5:service=multi(m['bindings'])
        else:
            cat,methods,layouts=m['bindings'][0];service=fixed(profile,methods[:len(DOCS[cat]['methods'])],layouts[:len(DOCS[cat]['types'])])
    prefix=struct.pack('<4I',len(service),len(imports),m['refs'],m['callbacks'])
    for module,symbol,kind,arity,result,params in imports:prefix+=struct.pack('<6I',module,symbol,kind,arity,result,len(params))+params
    return prefix+service+image(m['ownership'],rows=m['rows'],strings=m['names'],functions=m['functions'],code=m['code'],raw_layouts=m.get('layouts'),counts=m.get('counts',tuple(sum(row[0]==kind for row in m['rows']) for kind in (0,3,2))))


def corpus():
    cases=[]
    def add(p,name,m,expected=None):cases.append((p,name,packet(m,p),expected))
    for profile,cats in [(2,[1]),(3,[2]),(4,[3]),(5,[1,2,1,3])]:
        base=model(cats)
        add(profile,'valid',base,0)
        add(profile,'interleaved-reversed',model(cats,True,True,3),0)
        m=model(cats,extra=1);m['rows'][0]=(3,NONE,[]);add(profile,'unknown-enum',m,0)
        wire=fixed(profile,base['bindings'][0][1][:len(DOCS[cats[0]]['methods'])],base['bindings'][0][2][:len(DOCS[cats[0]]['types'])]) if profile<5 else multi(base['bindings'])
        layouts=table(*(layout(kind,fields,name) for kind,name,fields in base['rows']))
        for label,data,key in [('service',wire,'service'),('layouts',layouts,'layouts'),('ownership',base['ownership'],'ownership')]:
            for n in range(len(data)):
                m=copy.deepcopy(base);m[key]=data[:n];add(profile,f'{label}-truncation-{n}',m,1)
            m=copy.deepcopy(base);m[key]=data+b'\0';add(profile,label+'-tail',m,1)
            rng=random.Random(114270+profile+len(data))
            for n in range(64):
                value=bytearray(data);value[rng.randrange(len(value))]^=1<<rng.randrange(8)
                m=copy.deepcopy(base);m[key]=bytes(value);add(profile,f'{label}-mutation-{n}',m)
        for key in ('refs','callbacks'):
            m=copy.deepcopy(base);m[key]=1;m['layouts']=u32(65537);add(profile,key+'-before-limit',m,1)
        m=copy.deepcopy(base);m['layouts']=u32(65537);m['ownership']=b'';add(profile,'layout-limit-before-ownership',m,2)
        for i,im in enumerate(base['imports']):
            for field,value in [(0,NONE),(1,NONE),(2,0),(3,im[3]+1),(4,1)]:
                m=copy.deepcopy(base);m['imports'][i][field]=value
                if field==3:m['imports'][i][5]=bytes(value)
                add(profile,f'import-{i}-{field}',m,1)
            if im[3]:
                m=copy.deepcopy(base);m['imports'][i][5]=b'';add(profile,f'import-{i}-missing-tags',m,1)
        for tag in range(256):
            m=copy.deepcopy(base);m['imports'][0][4]=tag;add(profile,f'import-result-{tag}',m,0 if tag==10 else 1)
            i=next(i for i,entry in enumerate(base['imports']) if entry[3]);m=copy.deepcopy(base);tags=bytearray(m['imports'][i][5]);tags[0]=tag;m['imports'][i][5]=bytes(tags)
            add(profile,f'import-param-{tag}',m,0 if tag==base['imports'][i][5][0] else 1)
        for index,(kind,name,fields) in enumerate(base['rows']):
            m=copy.deepcopy(base);m['rows'][index]=(kind,NONE,fields);add(profile,f'layout-name-{index}',m,1)
            for j,(tag,nested,field_name) in enumerate(fields):
                for component,value in [(0,16),(1,index),(2,NONE)]:
                    m=copy.deepcopy(base);f=list(m['rows'][index][2][j]);f[component]=value;m['rows'][index][2][j]=tuple(f);add(profile,f'field-{index}-{j}-{component}',m,1)
        for flags in (0,1,2,3,255):
            m=copy.deepcopy(base);f=base['flags'].copy();f[0]=flags
            m['ownership']=record(1,f,[(3,1,desc(1),[desc(8,base['bindings'][0][2][0],2),desc(1),desc(10,base['bindings'][0][2][3])])]);add(profile,f'owner-flags-{flags}',m,0 if flags==3 else 1)
        for tag in range(256):
            m=copy.deepcopy(base);m['ownership']=record(1,base['flags'],[(3,1,desc(1),[desc(8,base['bindings'][0][2][0],2),desc(tag),desc(10,base['bindings'][0][2][3])])]);add(profile,f'unresolved-local-tag-{tag}',m,0 if tag<16 else 1)
        for mode in range(4):
            m=copy.deepcopy(base);m['ownership']=record(1,base['flags'],[(3,1,desc(1),[desc(8,base['bindings'][0][2][0],mode),desc(1),desc(10,base['bindings'][0][2][3])])]);add(profile,f'owner-parameter-mode-{mode}',m,0 if mode<3 else 1)
        for version in (0,2,3):
            m=copy.deepcopy(base);m['ownership']=struct.pack('<I',version)+m['ownership'][4:];add(profile,f'ownership-version-{version}',m,1)
        m=copy.deepcopy(base);m['functions']=[];m['ownership']=record(1,m['flags']);add(profile,'no-functions',m,0)
        for result_tag,descriptor_tag,status in [(255,0,0),(1,0,0),(0,1,1)]:
            m=copy.deepcopy(base);m['functions']=[function(arity=1,locals_=3,results=0,result_tag=result_tag,params=[8])]
            m['ownership']=record(1,m['flags'],[(3,1,desc(descriptor_tag),[desc(8,m['bindings'][0][2][0],2),desc(1),desc(10,m['bindings'][0][2][3])])]);add(profile,f'void-result-{result_tag}-{descriptor_tag}',m,status)
        for extra_kind in range(4):
            m=model(cats,extra=1);m['rows'][0]=(extra_kind,NONE,[]);add(profile,f'unknown-kind-{extra_kind}',m,0)
        for index,value in [(0,0),(0,1),(0,16),(1,1),(2,1),(3,1)]:
            m=model(cats,extra=1);m['rows'][0]=(1,NONE,[(1,NONE,NONE)])
            raw=bytearray(table(*(layout(kind,fields,name) for kind,name,fields in m['rows'])));raw[12+index]=value;m['layouts']=bytes(raw)
            add(profile,f'unknown-member-padding-tag-{index}-{value}',m,0 if index==0 and value<16 else 1)
        m=copy.deepcopy(base);m['counts']=(0,0,0);add(profile,'kind-counts',m,1)
        m=copy.deepcopy(base);m['names'][base['rows'][0][1]]+=b'\0';add(profile,'embedded-zero-name',m,1)
    m=model([1,2,3]*21+[1],True,True);add(5,'all64-instances',m,0)
    m=model([1]);m['rows'] += [(1,NONE,[])]*(65536-len(m['rows']));m['flags'] += [0]*(65536-len(m['flags']));m['ownership']=record(1,m['flags'],[(3,1,desc(1),[desc(8,0,2),desc(1),desc(10,3)])]);add(2,'all65536-layouts',m,0)
    # Same catalog IDs in different instances do not authorize cross-instance edges.
    m=model([1,1],True);index=m['bindings'][1][2][3];kind,name,fields=m['rows'][index];fields[0]=(8,m['bindings'][0][2][0],fields[0][2]);add(5,'cross-instance-nominal-edge',m,1)
    return cases


def producer_cases(root=None):
    from tests import test_service_drivers as file_source
    from tests import test_socket_service_source as socket_source
    from tests import test_websocket_service_drivers as websocket_source
    from tests import test_mixed_websocket_source as mixed_source
    root=Path(root) if root is not None else Path(tempfile.mkdtemp(prefix='service-nominal-products-',dir=ROOT/'.test_output'))
    root.mkdir(parents=True,exist_ok=True)
    producers=[('c',[ROOT/'bin/nano_virt'])]
    source = source_compiler_command()
    if source:producers.append(('nanolang',source))
    cases=[];records=[]
    for profile,catalog in [(2,'file'),(3,'socket'),(4,'websocket'),(5,'mixed')]:
        work=root/catalog;work.mkdir(exist_ok=True)
        if profile==5:
            path,deps,flags,_=mixed_source.graph(work)
        else:
            (work/'interface.nsi.json').write_bytes((ROOT/f'tests/fixtures/nsi_{catalog}_plan.json').read_bytes())
            path=work/'source.nano';deps=[]
            if profile==2:text=file_source.DECL+file_source.MAIN+'shadow main {assert (== (main) 7)}\n';flags=['--allow-temporary-files']
            elif profile==3:
                text=socket_source.DECL+socket_source.ADDRESS+'fn main()->int {let value:Endpoint=(address) return value.port}\nshadow main {assert (== (main) 12345)}\n';flags=['--allow-tcp-connections']
            else:text=websocket_source.DECL+websocket_source.MESSAGE+websocket_source.MAIN+'shadow main {assert (== (main) 7)}\n';flags=['--allow-websocket-connections']
            path.write_text(text)
        for label,compiler in producers:
            module=work/(label+'.nvm');carrier=work/(label+'.bin')
            checked([*compiler,path,*flags,'--emit-nvm','-o',module],build=True,timeout=180)
            checked([ROOT/'obj/nvm2c_service_nominal_reference','--extract',str(profile),module,carrier])
            cases.append((profile,'producer-'+catalog+'-'+label,carrier.read_bytes(),0))
            records.append(dict(profile=profile,producer=label,module=str(module.relative_to(root)),module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),carrier=str(carrier.relative_to(root)),carrier_sha256=hashlib.sha256(carrier.read_bytes()).hexdigest(),sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,*deps]},metadata_only=True))
            (root/'records.json').write_text(json.dumps(records,indent=2)+'\n')
    print('I retain actual service source products at',root,flush=True)
    return cases


class ServiceNominal(unittest.TestCase):
    def compare(self,commands):
        cases=corpus()+producer_cases()
        with tempfile.TemporaryDirectory(prefix='nano-service-nominal-') as directory:
            work=Path(directory);groups={}
            for i,(profile,name,data,status) in enumerate(cases):
                path=work/f'{i}.bin';path.write_bytes(data);groups.setdefault(profile,[]).append((name,path,status))
            routes=commands(work)
            expected=checked([ROOT/'obj/nvm2c_service_catalog_reference'])
            for route,command in routes:
                if route!='actual-nominal-asan-ubsan':self.assertEqual(checked([*command,'--catalog'],timeout=600),expected,route)
            for profile,items in groups.items():
                paths=[p for _,p,_ in items]
                def execute(command):
                    values=[]
                    for offset in range(0,len(paths),32):values+=checked([*command,str(profile),*paths[offset:offset+32]],timeout=600).splitlines()
                    self.assertEqual(len(values),len(items));return values
                reference=execute([ROOT/'obj/nvm2c_service_nominal_reference'])
                for (name,_,status),value in zip(items,reference):
                    if status is not None:
                        with self.subTest(oracle=name,profile=profile):self.assertEqual(int(value.split('|')[0]),status)
                for route,command in routes:
                    actual=execute(command)
                    for (name,_,_),expected,value in zip(items,reference,actual):
                        with self.subTest(route=route,profile=profile,case=name):self.assertEqual(value,expected)
        return len(cases),len(routes)

    def test_complete_plan_views_and_catalog(self):
        count,routes=self.compare(lambda work:build_routes(ROOT/'tests/nvm2c_service_nominal_driver.nano',work))
        print(f'I compared {count} complete nominal metadata/query reports across four profiles and {routes} routes; no execution authority follows.')

    def test_actual_nominal_provider_sanitizers(self):
        count,_=self.compare(lambda work:[('actual-nominal-asan-ubsan',[ROOT/'obj/nvm2c_service_nominal_reference_asan'])])
        print(f'I compared {count} reports through my instrumented nominal C providers, including unchanged-output getter controls.')


if __name__=='__main__':unittest.main()
