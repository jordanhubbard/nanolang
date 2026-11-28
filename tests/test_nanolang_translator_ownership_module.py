"""I compare shared ownership-bearing module decisions without granting execution."""
from pathlib import Path
import json,random,struct,tempfile,unittest
from tests.nanolang_translator_routes import build_routes,checked
from tests.test_nanolang_translator_module import packet
from tests.test_nanolang_translator_payloads import table,constant,signature,layout
from tests.test_nanolang_translator_ownership import corpus as declarations,record,extension,desc,u32,NONE
from tests.test_nanolang_translator_passive import block,node,payload
ROOT=Path(__file__).resolve().parents[1]


def private_layout(data):
    if len(data)<4:return False
    count=struct.unpack_from('<I',data)[0];cursor=4
    if count>(len(data)-4)//8:return False
    for _ in range(count):
        if cursor+8>len(data):return False
        n=struct.unpack_from('<H',data,cursor+2)[0];cursor+=8
        if n>(len(data)-cursor)//12:return False
        for _ in range(n):
            if data[cursor] in (7,13):return True
            cursor+=12
    return False


def wrapped(data):
    layout_size,owned_size,strings,functions,*_=struct.unpack_from('<7I',data)
    cursor=28;names=[];shapes=[];entries=[]
    for _ in range(strings):
        n=struct.unpack_from('<I',data,cursor)[0];cursor+=4
        names.append(data[cursor:cursor+n]);cursor+=n
    for i in range(functions):
        start,length,arity,locals_,upvalues,results,result_tag,count=struct.unpack_from('<8I',data,cursor);cursor+=32
        tags=data[cursor:cursor+count] if count else bytes(arity);cursor+=count
        shapes.append(signature(tags,[result_tag]*results))
        entries.append(struct.pack('<IIQQHHHH',len(names)+i,i,start,length,locals_,upvalues,0,0))
    layouts=data[cursor:cursor+layout_size];cursor+=layout_size
    owned=data[cursor:cursor+owned_size];cursor+=owned_size
    constants=table(*(constant(5,name) for name in names+[f'__function_{i}'.encode() for i in range(functions)]))
    parts=[(2,constants),(3,table(*shapes)),(4,layouts),(5,table(*entries)),(6,data[cursor:])]
    if owned_size:parts.append((13,owned))
    # ARRAY/MAP layout profiles remain a quantified pending boundary. Explicit
    # kind5 selection is separately classified by the actual C routing oracle.
    return packet(parts,features=128+(256 if owned_size else 0)),private_layout(layouts)


def corpus():
    cases=[]
    for name,data,_ in declarations():
        module,pending=wrapped(data)
        cases.append(('declaration-'+name,module,None,pending,bytes(16)+data))
    base={2:table(constant(5,b'main')),3:table(signature([],[])),4:table(),5:table(struct.pack('<IIQQHHHH',0,0,0,0,0,0,0,0)),6:b'',13:record(1,functions=[(0,0,desc(),[])])}
    def add(name,changes=None,expected=None,features=384,sections=None,pending=False):
        values=base|(changes or {})
        cases.append((name,packet(list(values.items()) if sections is None else sections,features=features),expected,pending,None))
    add('ordinary-empty-retained',expected=0)
    add('missing-ownership-bit',expected=14,features=128)
    add('missing-retained-empty-layout',expected=8,features=256)
    add('empty-ownership', {13:b''},9)
    add('truncated-ownership',{13:b'\1'},13)
    add('invalid-owner-routing-version',{13:record(4)},13)
    add('unverified-opcode-is-not-execution',{5:table(struct.pack('<IIQQHHHH',0,0,0,1,0,0,0,0)),6:b'\xff'},0)
    for offset in (2**32-1,2**32,2**63):
        add(f'debug-narrow-{offset}',{10:table(struct.pack('<QII',offset,0,0))},0 if offset<NONE+1 else 13,features=392)
    add('nonstring-before-ownership',{2:table(constant(1,bytes(8))),13:b'\1'},8)
    add('duplicate-before-ownership',{2:table(constant(5,b'main'),constant(5,b'main')),13:b'\1'},13)
    add('references-before-ownership',{5:table(struct.pack('<IIQQHHHH',1,0,0,0,0,0,0,0)),13:b'\1'},13)
    add('retained-permitted-tail',{4:table()+b'tail'},0)
    for owned_first in (False,True):
        sections=[(13,b''),(2,b'')]
        add('empty-section-order-'+str(owned_first),expected=9 if owned_first else 1,sections=sections if owned_first else list(reversed(sections)))
    # The original passive node supplies a separate valid/invalid co-profile.
    code=b'\x03\1\x11\0\0\x3d'
    good=payload([block([node(0,5,0)],0,5)])
    f=table(struct.pack('<IIQQHHHH',0,0,0,len(code),1,0,0,0))
    owned=record(1,functions=[(1,0,desc(),[desc(2)])])
    add('valid-owned-and-passive',{5:f,6:code,12:good,13:owned},0,features=448)
    add('ownership-before-bad-passive',{5:f,6:code,12:b'bad',13:b'\1'},13,features=448)
    add('bad-passive-after-ownership',{5:f,6:code,12:b'bad',13:owned},13,features=448)
    add('passive-feature-before-ownership',{5:f,6:code,12:good,13:b'\1'},14)
    for n in range(len(owned)):
        add(f'owned-truncation-{n}',{5:f,6:code,13:owned[:n]})
    rng=random.Random(114251)
    for n in range(128):
        value=bytearray(owned);value[rng.randrange(len(value))]^=1<<rng.randrange(8)
        add(f'owned-mutation-{n}',{5:f,6:code,13:bytes(value)})
    # Explicitly selected kind5 remains private even without a map field.
    add('selected-kind5-pending',{13:record(3,functions=[(0,0,desc(),[])],extensions=[extension(5,b'')])},pending=True)
    return cases


class OwnershipModules(unittest.TestCase):
    def test_shared_conversion_and_exact_claims(self):
        cases=corpus()
        with tempfile.TemporaryDirectory(prefix='nano-ownership-modules-') as directory:
            work=Path(directory);paths=[]
            from tests.test_nanolang_translator_declarations import producer_cases
            # I prepare my generated parent when this gate runs alone.
            (ROOT/'.test_output').mkdir(exist_ok=True)
            products=Path(tempfile.mkdtemp(prefix='ownership-module-products-',dir=ROOT/'.test_output'))
            models=producer_cases(work,products)
            records=json.loads((products/'records.json').read_text())
            self.assertEqual(len(models),len(records))
            for (name,carrier,_),record in zip(models,records):
                self.assertEqual(carrier[:16],bytes(16),'I require a separate routing carrier for foreign context.')
                _,private=wrapped(carrier[16:])
                cases.append((name,(products/record['module']).read_bytes(),0,private,carrier))
            for i,(_,data,_,_,_) in enumerate(cases):
                p=work/f'case-{i}.nvm';p.write_bytes(data);paths.append(p)
            allowed=[]
            routing_paths=[]
            for i,(_,_,_,private,carrier) in enumerate(cases):
                allowed.append(private)
                if carrier is not None:
                    path=work/f'route-{i}.bin';path.write_bytes(carrier)
                    routing_paths.append((i,path))
            for offset in range(0,len(routing_paths),32):
                batch=routing_paths[offset:offset+32]
                reports=checked([ROOT/'obj/nvm2c_declaration_reference','0',*[path for _,path in batch]],timeout=600).splitlines()
                self.assertEqual(len(reports),len(batch))
                for (i,_),report in zip(batch,reports):
                    self.assertTrue(report.startswith('R|'))
                    allowed[i] |= int(report[2:].split(',')[2])==1
            def execute(command):
                values=[]
                for i in range(0,len(paths),32):values+=checked([*command,'--claims',*paths[i:i+32]],timeout=600).splitlines()
                self.assertEqual(len(values),len(cases));return values
            reference=execute([ROOT/'obj/nvm2c_module_reference'])
            for (name,_,expected,_,_),value in zip(cases,reference):
                if expected is not None:
                    with self.subTest(oracle=name):self.assertEqual(int(value.split(':')[0]),expected)
            pending_counts=[]
            for route,command in build_routes(ROOT/'tests/nvm2c_module_driver.nano',work):
                actual=execute(command);pending=0
                for (name,_,_,_,_),private,expected,value in zip(cases,allowed,reference,actual):
                    with self.subTest(route=route,case=name):
                        if value.startswith('-1:'):
                            # Version3 kind5 can select a private profile without
                            # an array/map layout. Quantify that separate route.
                            self.assertTrue(private,name)
                            self.assertEqual(value,'-1:0::0:');pending+=1
                        else:self.assertEqual(value,expected)
                pending_counts.append(pending)
            self.assertEqual(len(set(pending_counts)),1)
            print(f'I compared {len(cases)} full ownership module decisions and exact ownership/passive bytes across {len(pending_counts)} routes; {pending_counts[0]} private-profile inputs remain pending, not successful admission.')


if __name__=='__main__':unittest.main()
