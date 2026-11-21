"""I compare profile routing, never executable admission, with the actual C selectors."""
from pathlib import Path
import random
import struct
import tempfile
import unittest
from tests.nanolang_translator_routes import build_routes, checked
from tests.test_nanolang_translator_declarations import corpus as declarations, producer_cases
from tests.test_nanolang_translator_ownership import image, record, extension, function, desc, NONE, u32
from tests.test_nanolang_translator_payloads import table, layout, change
ROOT=Path(__file__).resolve().parents[1]


def corpus():
    cases=[('declaration-'+name,data,None) for name,data,_ in declarations()]
    def add(name,owned,expected=None,foreign=(0,0,0,0),**model):
        cases.append((name,struct.pack('<4I',*foreign)+image(owned,**model),expected))
    arr=[(0,NONE,[(7,NONE,NONE)])]
    maps=[(0,NONE,[(13,NONE,NONE)])]
    scalar=[(0,NONE,[(1,NONE,NONE)])]
    def report(service=False,arrays=False,maps=0,owners=0,mixed=False):
        return f'R|{int(service)},{int(arrays)},{maps},{owners},{int(mixed)},'
    for presence in range(4):
        add(f'empty-presence-{presence}',b'',report(maps=4 if presence else 0,owners=2 if presence&2 else 0),raw_layouts=b'',foreign=(0,presence,0,0))
        add(f'empty-service-{presence}',b'',report(True,maps=4 if presence else 0),raw_layouts=b'',foreign=(0,presence,1,0))
    for version in (1,2,3):
        for flag in range(256):
            resources=bool(flag&2)
            ordinary=version==3 and not resources
            mixed=version in (1,2) and resources
            owner=0 if not resources else (1 if flag==3 and version!=3 else 2)
            framing=flag<=3
            add(f'array-version-{version}-flag-{flag}',record(version,[flag],extensions=[extension(2,u32(0)+u32(0))] if version==3 else []),
                report(arrays=ordinary,maps=0 if framing else 4,owners=owner,mixed=mixed),rows=arr)
    for encoded in range(257):
        pending=encoded==4
        add(f'import-kind-{encoded}',record(1,[3]),report(pending,owners=0 if pending else 1,mixed=not pending),rows=arr,foreign=(encoded,0,0,0))
    for opcode in range(256):
        pending=145<=opcode<=152
        add(f'file-primary-{opcode}',record(1,[3],functions=[(0,0,desc(),[])]),report(pending,owners=0 if pending else 1,mixed=not pending),rows=arr,functions=[function(length=1)],code=bytes([opcode]))
    for start,length,code,pending in [(1,1,b'\0\x91',True),(2,1,b'\x91',False),(0,0,b'\x91',False),(NONE,NONE,b'\x91',False),(0,9,b'\x01\x91'+bytes(7),False),(0,2,b'\xff\x91',False)]:
        add(f'file-span-{start}-{length}-{code.hex()}',record(1,[3],functions=[(0,0,desc(),[])]),report(pending,owners=0 if pending else 1,mixed=not pending),rows=arr,functions=[function(start=start,length=length)],code=code)
    map_owned=record(3,[1],extensions=[extension(5,u32(0))])
    add('map-valid-framing',map_owned,report(maps=1),rows=maps)
    add('map-service-keeps-selected',map_owned,report(True,maps=1),rows=maps,foreign=(0,0,1,0))
    add('map-field-missing-ownership',b'',report(maps=2),rows=maps)
    add('map-kind-without-field',map_owned,report(maps=1),rows=scalar)
    for n in range(len(map_owned)):
        # Expectations for partial frames come from the actual reference.
        add(f'map-kind-truncation-{n}',map_owned[:n],rows=scalar)
    for byte in (1,128,255):
        add(f'map-bad-layout-reserved-{byte}',map_owned,report(maps=2),raw_layouts=change(table(layout(fields=[(13,NONE,NONE)])),5,'<B',byte))
        add(f'map-later-layout-malformed-{byte}',map_owned,report(maps=2),raw_layouts=table(layout(fields=[(13,NONE,NONE)]))+bytes([byte]))
    for revision in (0,2,65535):
        add(f'map-revision-{revision}',record(3,[1],extensions=[extension(5,b'',revision)]),report(maps=2),rows=scalar)
    early=struct.pack('<II',3,1)+bytes([1])
    add('ordinary-array-before-missing-functions',early,report(arrays=True,maps=4),rows=arr)
    early_resource=struct.pack('<II',1,1)+bytes([3])
    add('mixed-before-missing-functions',early_resource,report(maps=4,owners=2,mixed=True),rows=arr)
    bad=struct.pack('<I',1)+struct.pack('<BBHI',255,255,2,NONE)+struct.pack('<B3xII',7,NONE,NONE)
    add('ordinary-array-before-malformed-tail',record(3,[1],extensions=[extension(2,b'')]),report(arrays=True,maps=4),raw_layouts=bad)
    bad_map=change(bad,12,'<B',13)
    add('map-before-malformed-layout-tail',record(1,[1]),report(maps=2),raw_layouts=bad_map)
    for version,flags in [(1,[3]),(2,[3]),(3,[1])]:
        base=record(version,flags,extensions=[extension(2,u32(0)+u32(0))] if version==3 else [])
        rng=random.Random(114240+version)
        for n in range(128):
            data=bytearray(base);data[rng.randrange(len(data))]^=1<<rng.randrange(8)
            add(f'ownership-route-mutation-{version}-{n}',bytes(data),rows=arr)
    add('kind5-inside-span-above-byte-cap',map_owned+bytes(16777217-len(map_owned)),report(maps=3),rows=scalar)
    distant=record(3,extensions=[extension(5,b'')],raw_paths=bytes(16777216))
    add('kind5-after-bounded-prefix',distant,report(maps=4))
    add('map-field-inside-span-above-byte-cap',record(1,[1]),report(maps=3),raw_layouts=table(layout(fields=[(13,NONE,NONE)]))+bytes(16777217-24))
    return cases


class Routing(unittest.TestCase):
    def test_actual_classifiers_and_quarantine(self):
        cases=corpus()
        with tempfile.TemporaryDirectory(prefix='nano-profile-routing-') as directory:
            work=Path(directory);cases+=producer_cases(work);paths=[]
            for i,(_,data,_) in enumerate(cases):
                p=work/f'case-{i}.bin';p.write_bytes(data);paths.append(p)
            def execute(command):
                values=[]
                for i in range(0,len(paths),32):values+=checked([*command,'0',*paths[i:i+32]],timeout=600).splitlines()
                self.assertEqual(len(values),len(cases));return values
            reference=execute([ROOT/'obj/nvm2c_declaration_reference'])
            for (name,_,expected),value in zip(cases,reference):
                if expected is not None:
                    with self.subTest(oracle=name):self.assertEqual(value,expected)
            routes=build_routes(ROOT/'tests/nvm2c_declaration_driver.nano',work)
            for route,command in routes:
                actual=execute(command)
                for (name,_,_),expected,value in zip(cases,reference,actual):
                    with self.subTest(route=route,case=name):self.assertEqual(value,expected)
            print(f'I matched {len(cases)} five-selector routing reports across {len(routes)} routes; {sum(e is not None for _,_,e in cases)} explicit reports retain service/File priority, early identities and byte/count limits. No routing result grants execution authority.')


if __name__=='__main__':unittest.main()
