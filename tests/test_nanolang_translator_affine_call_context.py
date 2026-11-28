"""I check analysis context initialization without proving a callee body."""
from itertools import product
from pathlib import Path
import hashlib
import json
import os
import struct
import tempfile
import unittest
from tests import nanolang_translator_routes as routes
from tests.test_nanolang_translator_affine_frame import words, flow_cases, queue_cases, frame_cases
from tests.test_nanolang_translator_affine_scalar import scalar_batches
from tests.test_nanolang_translator_ownership import desc, record, function, image, extension, globals_data

ROOT=Path(__file__).resolve().parents[1]
NO=0xffffffff
EFFECTS={165:b'\xa5'+words(0)+struct.pack('<iHH',0,0,0),166:b'\xa6\0\0',167:b'\xa7'+words(0)+b'\0\0',168:b'\xa8'}


def context_model(bodies=(b'\x3d',),globals_=(),*,entry_tag=1,upvalues=0,locals_=0,own=None,globals_extension=False):
    count=len(bodies);extensions=[]
    if globals_ or globals_extension:extensions=[extension(3,globals_data([struct.pack('<BBHI',*g) for g in globals_]))]
    declarations=[(locals_ if i==0 else 0,0,desc(entry_tag if i==0 else 1),[desc(1)]*(locals_ if i==0 else 0)) for i in range(count)]
    owned=record(3 if extensions else 1,functions=declarations,extensions=extensions) if own is None else own
    rows=[];offset=0
    for i,body in enumerate(bodies):
        f=bytearray(function(locals_ if i==0 else 0,results=1,result_tag=entry_tag if i==0 else 1,start=offset,length=len(body)))
        if i==0:struct.pack_into('<I',f,16,upvalues)
        rows.append(bytes(f));offset+=len(body)
    return image(owned,functions=rows,code=b''.join(bodies))


def context_cases():
    cases=[]
    def add(name,carrier,*,entry=0,present=True,valid=True,graph=True,effect=False,globals_=()):
        raw=words(14,entry,int(present),len(carrier))+carrier
        rows=''.join(f'{tag},{mutable},{layout},;' for tag,mutable,reserved,layout in globals_)
        expected=f'{int(valid)},{int(graph)},{int(effect)},{len(globals_)}|'+rows
        cases.append((name,raw,expected));return raw
    for count,global_count,opcode,where,dead,present,entry in product((1,2,8,9),(0,1),(0,165,166,167,168),(0,-1),(False,True),(False,True),(0,1)):
        body=(b'\x3d' if dead else b'')+(b'\0' if opcode==0 else EFFECTS[opcode])+b'\x3d'
        bodies=[b'\x3d']*count;bodies[where]=body
        globals_=[(1,1,0,NO)]*global_count
        add(f'scan-{count}-{global_count}-{opcode}-{where}-{dead}-{present}-{entry}',context_model(bodies,globals_),
            entry=entry,present=present,valid=present,graph=present and count<=8 and entry==0,
            effect=present and count<=8 and entry==0 and global_count>0 and opcode in EFFECTS,globals_=globals_ if present else ())
    for count in (0,1,255,256,257):
        globals_=[(1,int(i%2),0,NO) for i in range(count)]
        valid=count<=256
        add(f'global-count-{count}',context_model((EFFECTS[168]+b'\x3d',),globals_),valid=valid,graph=valid,
            effect=valid and count>0,globals_=globals_ if valid else ())
    add('empty-global-extension',context_model(globals_extension=True),valid=False,graph=False)
    for tag,mutable,reserved,layout in product(range(12),(0,1,2),(0,1),(0,NO)):
        globals_=[(tag,mutable,reserved,layout)];valid=tag in (1,2,3,4,5) and mutable<2 and reserved==0 and layout==NO
        add(f'global-row-{tag}-{mutable}-{reserved}-{layout}',context_model(globals_=globals_),valid=valid,graph=valid,globals_=globals_ if valid else ())
    for tag in (1,2,3,4,5):
        globals_=[(tag,0,0,NO),(tag,1,0,NO)]
        add(f'preserved-global-rows-{tag}',context_model(globals_=globals_),globals_=globals_)
    for count in (0,1,8,9):add(f'function-count-{count}',context_model([b'\x3d']*count),graph=0<count<=8)
    for captures in (1,65535):add(f'capture-{captures}',context_model(upvalues=captures),graph=False)
    for locals_ in (255,256,257):add(f'local-size-{locals_}',context_model(locals_=locals_),graph=locals_<=256)
    for count in (4095,4096,4097):add(f'instruction-size-{count}',context_model((b'\0'*(count-1)+b'\x3d',)),graph=count<=4096)
    for label,body in [('empty',b''),('truncated-call',b'\x3b\1'),('unknown',b'\xff'),('branch-interior',b'\x38\1\0\0\0'),('cycle',b'\x3b'+words(1)+b'\x3d'),('reference-call',b'\x0f'+words(1)+b'\0\0')]:
        add('invalid-graph-'+label,context_model((b'\x3d',body),[(1,1,0,NO)]),graph=False,globals_=[(1,1,0,NO)])
    for size in (0,1,2,3,4,8,12,16,24):
        add(f'invalid-ownership-{size}',context_model(own=b'\xff'*size),valid=size==0,graph=False)
    good=add('valid-malformed-base',context_model())
    malformed=[b'',words(14),good+b'\0',words(14,0,2,len(good)-16)+good[16:]]+[good[:n] for n in (1,4,8,12,15,16,len(good)-1)]
    cases.extend((f'malformed-{i}',raw,'D') for i,raw in enumerate(malformed))
    return cases


class AffineCallContext(unittest.TestCase):
    def test_original_frames_and_complete_call_context_initialization(self):
        work=Path(tempfile.mkdtemp(prefix='nano-affine-call-context-'))
        old=flow_cases()[0]+queue_cases()+frame_cases();self.assertEqual(len(old),1001)
        reference=Path(os.environ.get('NANOLANG_AFFINE_CALL_CONTEXT_REFERENCE',ROOT/'obj/nvm2c_affine_call_context_reference'))
        manifest=dict(scope='I initialize private call-analysis flags/global declarations only; no callee/CFG/executable authority.',cases=[],routes=[]);prepared=[]
        for group,cases in [('original',old),('context',context_cases())]:
            paths=[]
            for index,(name,raw,expected) in enumerate(cases):
                path=work/f'{group}-{index:05d}.bin';path.write_bytes(raw);paths.append(path)
                manifest['cases'].append(dict(group=group,name=name,path=str(path),sha256=hashlib.sha256(raw).hexdigest(),expected=expected))
            prepared.append((group,cases,paths))
        def save():(work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        save();reports={};products=[('actual-c',[reference])]+routes.build_routes(ROOT/'tests/nvm2c_affine_call_context_driver.nano',work)
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
