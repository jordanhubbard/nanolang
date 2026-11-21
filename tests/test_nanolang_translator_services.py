"""I compare all raw service values and canonical bytes, never service authority."""
from pathlib import Path
import random,struct,tempfile,unittest
from tests.nanolang_translator_routes import build_routes,checked
ROOT=Path(__file__).resolve().parents[1]
NONE=0xffffffff
METHODS={1:5,2:5,3:4}
TYPES={1:8,2:9,3:7}


def row(catalog,base=0):
    return (catalog,list(range(base,base+METHODS.get(catalog,5)))+[NONE]*(5-METHODS.get(catalog,5)),
            list(range(base,base+TYPES.get(catalog,9)))+[NONE]*(9-TYPES.get(catalog,9)))


def fixed(profile,imports=None,layouts=None):
    catalog=1 if profile==1 else profile-1
    methods=5 if profile==1 else METHODS[catalog]
    types=0 if profile==1 else TYPES[catalog]
    imports=list(range(methods)) if imports is None else imports
    layouts=list(range(types)) if layouts is None else layouts
    return struct.pack('<HHIII',1 if profile==1 else 2,catalog,methods,types,0)+b''.join(struct.pack('<II',i,index) for values in (imports,layouts) for i,index in enumerate(values))


def multi(rows):
    return struct.pack('<HHIII',3,0,len(rows),64,0)+b''.join(struct.pack('<IHH14I',i,catalog,0,*imports,*layouts) for i,(catalog,imports,layouts) in enumerate(rows))


def values(rows):
    return struct.pack('<I',len(rows))+b''.join(struct.pack('<15I',catalog,*imports,*layouts) for catalog,imports,layouts in rows)


def changed(data,offset,fmt,value):
    result=bytearray(data);struct.pack_into(fmt,result,offset,value);return bytes(result)


def corpus():
    cases=[]
    def add(profile,mode,name,data,expected=None):cases.append((profile,mode,name,data,expected))
    mixed=[row(1,0),row(2,20),row(3,40)]
    bases={p:fixed(p) if p<5 else multi(mixed) for p in range(1,6)}
    for profile,base in bases.items():
        add(profile,'wire','valid',base,0)
        for n in range(len(base)):
            add(profile,'wire',f'truncation-{n}',base[:n],2)
        for tail in (b'\0',b'\xff'*16):add(profile,'wire',f'tail-{len(tail)}',base+tail,2)
        for other,data in bases.items():add(profile,'wire',f'other-profile-{other}',data)
        for offset in (0,2,4,8,12):
            for octet in range(256):
                data=changed(base,offset,'<B',octet)
                if offset==0:expected=0 if octet==(1 if profile==1 else 3 if profile==5 else 2) else 3
                elif offset==2:expected=0 if octet==(0 if profile==5 else 1 if profile==1 else profile-1) else 6 if profile==5 else 4
                elif offset==4:
                    if profile==5:expected=0 if octet==3 else 5 if octet==0 or octet>64 else 2
                    else:expected=0 if octet==(4 if profile==4 else 5) else 5
                elif offset==8:
                    if profile==5:expected=0 if octet==64 else 2
                    elif profile==1:expected=0 if octet==0 else 6
                    else:expected=0 if octet==TYPES[profile-1] else 5
                else:expected=0 if octet==0 else 6
                add(profile,'wire',f'header-{offset}-{octet}',data,expected)
        for offset in range(len(base)):
            for bit in (1,128):add(profile,'wire',f'bit-{offset}-{bit}',changed(base,offset,'<B',base[offset]^bit))
        rng=random.Random(114260+profile)
        for n in range(128):
            data=bytearray(base)
            for _ in range(1+n%3):data[rng.randrange(len(data))]^=1<<rng.randrange(8)
            add(profile,'wire',f'mutation-{n}',bytes(data))
        if profile<5:
            methods=4 if profile==4 else 5;types=0 if profile==1 else TYPES[profile-1]
            for i in range(methods+types):
                add(profile,'wire',f'ordinal-{i}',changed(base,16+8*i,'<I',NONE),7)
                add(profile,'wire',f'index-sentinel-{i}',changed(base,20+8*i,'<I',NONE),8)
            add(profile,'wire','ordinal-before-index',changed(changed(base,20,'<I',NONE),16+8*(methods+types-1),'<I',NONE),7)
            add(profile,'wire','same-index-across-tables',fixed(profile),0)
            add(profile,'wire','unsigned-extrema',fixed(profile,[NONE-1-i for i in range(methods)],[NONE-1-i for i in range(types)]),0)
            for i in range(methods+types):
                candidate=row(1 if profile==1 else profile-1)
                catalog,imports,layouts=candidate;imports=imports.copy();layouts=layouts.copy()
                if i<methods:imports[i]=NONE
                else:layouts[i-methods]=NONE
                add(profile,'value',f'value-index-{i}',values([(catalog,imports,layouts)]),8)
            add(profile,'value','valid-value',values([row(1 if profile==1 else profile-1)]),0)
            add(profile,'value','duplicate-import',values([(1,[0,1,2,3,0],list(range(9)))]),8 if methods==5 else 0)
            if types:
                layouts=list(range(9));layouts[types-1]=0
                add(profile,'value','duplicate-layout',values([(2,list(range(5)),layouts)]),8)
        else:
            for count in (0,1,2,3,63,64):
                rows=[row(1+i%3,20*i) for i in range(count)]
                add(5,'wire',f'instance-count-{count}',multi(rows),5 if count==0 else 0)
                add(5,'value',f'value-instance-count-{count}',values(rows),5 if count==0 else 0)
            add(5,'value','value-instance-count-65',struct.pack('<I',65),5)
            add(5,'wire','above-wire-cap',base+bytes(4113-len(base)),2)
            for count in (65,NONE):add(5,'wire',f'header-count-{count}',changed(base,4,'<I',count),5)
            for catalog in (0,1,2,3,4,65535):
                candidate=row(catalog)
                add(5,'wire',f'catalog-{catalog}',multi([candidate]),0 if catalog in (1,2,3) else 4)
                add(5,'value',f'value-catalog-{catalog}',values([candidate]),0 if catalog in (1,2,3) else 4)
            for i in range(3):
                offset=16+64*i
                add(5,'wire',f'instance-ordinal-{i}',changed(base,offset,'<I',NONE),7)
                add(5,'wire',f'instance-reserved-{i}',changed(base,offset+6,'<H',1),6)
                for slot in range(14):
                    active=slot<METHODS[mixed[i][0]] if slot<5 else slot-5<TYPES[mixed[i][0]]
                    data=changed(base,offset+8+4*slot,'<I',NONE if active else 0)
                    add(5,'wire',f'instance-slot-{i}-{slot}',data,8 if active else 6)
                    rows=[(c,a.copy(),b.copy()) for c,a,b in mixed]
                    rows[i][1 if slot<5 else 2][slot if slot<5 else slot-5]=NONE if active else 0
                    add(5,'value',f'value-slot-{i}-{slot}',values(rows),8 if active else 6)
            add(5,'wire','later-ordinal-before-first-catalog',changed(changed(base,20,'<H',0),80,'<I',NONE),7)
            add(5,'wire','later-reserved-before-first-index',changed(changed(base,24,'<I',NONE),86,'<H',1),6)
            for table in (0,1):
                rows=[(c,a.copy(),b.copy()) for c,a,b in mixed]
                rows[2][1+table][0]=rows[0][1+table][0]
                add(5,'wire',f'cross-instance-table-{table}',multi(rows),8)
                add(5,'value',f'value-cross-instance-table-{table}',values(rows),8)
            # Semantic checks are per instance; catalog errors in a later row
            # cannot hide an earlier duplicate. Framing checks still run first.
            rows=[row(1),row(0,20)];rows[0][1][1]=rows[0][1][0]
            add(5,'wire','earlier-duplicate-before-later-catalog',multi(rows),8)
            add(5,'value','value-earlier-duplicate-before-later-catalog',values(rows),8)
            rows=[row(1)];rows[0][1][0]=NONE;rows[0][2][8]=0
            add(5,'wire','unused-layout-before-active-index',multi(rows),6)
            add(5,'value','value-unused-layout-before-active-index',values(rows),6)
    return cases


class ServiceWire(unittest.TestCase):
    def compare(self,commands):
        cases=corpus()
        with tempfile.TemporaryDirectory(prefix='nano-service-wire-') as directory:
            work=Path(directory);groups={}
            for i,(profile,mode,name,data,expected) in enumerate(cases):
                path=work/f'{i}.bin';path.write_bytes(data)
                groups.setdefault((profile,mode),[]).append((name,path,expected))
            for (profile,mode),group in groups.items():
                paths=[p for _,p,_ in group]
                def execute(command):
                    result=[]
                    for offset in range(0,len(paths),32):result+=checked([*command,str(profile),mode,*paths[offset:offset+32]],timeout=600).splitlines()
                    self.assertEqual(len(result),len(group));return result
                reference=execute([ROOT/'obj/nvm2c_service_reference'])
                for (name,_,expected),report in zip(group,reference):
                    if expected is not None:
                        with self.subTest(profile=profile,mode=mode,oracle=name):self.assertEqual(int(report.split('|')[0]),expected)
                for route,command in commands(work):
                    actual=execute(command)
                    for (name,_,_),expected,report in zip(group,reference,actual):
                        with self.subTest(route=route,profile=profile,mode=mode,case=name):self.assertEqual(report,expected)
        return len(cases)

    def test_complete_codecs_and_values(self):
        cached=[]
        def routes(work):
            if not cached:cached.extend(build_routes(ROOT/'tests/nvm2c_service_driver.nano',work))
            return cached
        count=self.compare(routes)
        print(f'I matched {count} raw service decode/value/check/canonical-byte reports across five profiles and {len(cached)} routes; no module or host authority follows.')

    def test_actual_codec_provider_sanitizers(self):
        count=self.compare(lambda _: [('actual-codec-asan-ubsan',[ROOT/'obj/nvm2c_service_reference_asan'])])
        print(f'I matched {count} reports through all five instrumented C codec providers, including failure-atomic and short-buffer controls.')


if __name__=='__main__':unittest.main()
