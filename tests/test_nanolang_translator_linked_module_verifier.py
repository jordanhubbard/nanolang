"""I compare real V2 linked roots and complete flat graphs through public APIs."""
from contextlib import nullcontext
from pathlib import Path
import hashlib
import json
import os
import struct
import tempfile
import unittest
from tests.nanolang_translator_routes import build_routes, checked, source_compiler_command
from tests.native_process import run_captured
from tests.test_nanolang_translator_module_verifier import assembled, NONE
from tests.test_nanolang_translator_function_body import corpus as body_corpus
from tests.test_nanolang_translator_stack_heights import ins, OPS
from tests.test_nanolang_translator_service_modules import corpus as service_corpus
from tests.test_nanolang_translator_module import corpus as core_corpus

ROOT=Path(__file__).resolve().parents[1]


def carrier(root, links=()):
    return struct.pack('<II',len(root),len(links))+root+b''.join(
        struct.pack('<II',int(resolved),len(data))+data for resolved,data in links)


def inherited():
    cases=[]
    for name,data,_ in body_corpus():
        if len(data)<40: continue
        start,span,count,current,foreign,linked,strings,records,enums,unions=struct.unpack_from('<10I',data)
        if not linked: continue
        # These two legacy structs cannot be represented by a converted V2 signature.
        if name in ('invalid-link-arity','invalid-link-results'): continue
        cursor=40
        functions=[struct.unpack_from('<5I',data,cursor+i*20) for i in range(count)]
        cursor+=count*20
        imports=[struct.unpack_from('<2I',data,cursor+i*8) for i in range(foreign)]
        cursor+=foreign*8
        links=[]
        for i in range(linked):
            resolved,n=struct.unpack_from('<2I',data,cursor);cursor+=8
            shapes=[struct.unpack_from('<2I',data,cursor+j*8) for j in range(n)];cursor+=n*8
            target=assembled([ins('LOAD_GLOBAL',0)*r+ins('RET') for a,r in shapes],
                             params=[[0]*a for a,r in shapes],returns=[[1]*r for a,r in shapes],entry=NONE)
            links.append((resolved,target))
        original=data[cursor:]
        pool=bytearray(original);spans=[];bodies=[]
        for i,(a,r,tag,local,up) in enumerate(functions):
            if i==current: spans.append((start,span));bodies.append(original)
            else:
                code=ins('LOAD_GLOBAL',0)*r+ins('RET');spans.append((len(pool),len(code)));pool+=code;bodies.append(code)
        root=assembled(bodies,params=[[0]*f[0] for f in functions],returns=[[f[2]]*f[1] for f in functions],
                       locals_=[f[3] if i==current else max(f[0],f[3]) for i,f in enumerate(functions)],
                       upvalues=[f[4] for f in functions],spans=spans,pool=bytes(pool),
                       strings=[f'name-{i}'.encode() for i in range(strings)],records=records,enums=enums,unions=unions,
                       imports=[([0]*a,[] if t==0 else [t],0,0,0) for a,t in imports])
        cases.append(('inherited-'+name,carrier(root,links),None,None,False))
    return cases


def corpus():
    cases=inherited()
    def add(name,root,links=(),want=None,graph=None,private=False):
        cases.append((name,carrier(root,links),want,graph,private))
    empty=assembled();target=assembled([ins('RET')]);bad=assembled([b'\xff'])
    add('empty',empty,want='1:1:0',graph='-1|1:1:0')
    add('unused-unresolved-garbage',empty,[(False,b'garbage')],want='1:1:0',graph='-1|1:1:0')
    add('root-good-target-bad',empty,[(True,bad)],want='1:1:0',graph='1|D')
    add('root-good-second-target-bad',empty,[(False,b''),(True,bad)],want='1:1:0',graph='2|D')
    add('root-good-empty-result-target',empty,[(True,assembled(returns=[[1]]))],want='1:1:0',graph='1|0:I require a body for function[0] with 1 declared result(s).')
    add('root-bad-target-bad',bad,[(True,bad)],want='D',graph='0|D')
    add('target-loader-before-root-body',bad,[(True,b'')],want='R:1',graph='1|R:1')
    add('root-loader-first',b'',[(True,b'')],want='R:1',graph='0|R:1')
    add('target-declaration-before-root-body',bad,[(True,assembled([b'\xff'],declared=[1]))],want='D',graph='1|D')
    add('root-declaration-before-target-loader',assembled([ins('POP')],declared=[1]),[(True,b'')],
        want='0:function[0] stack underflow at offset 0 (POP needs 1, has 0)')
    for resolved in (False,True):
        for module in (0,1,63,64,0xffffffff):
            for function in (0,1,0xffffffff):
                root=assembled([ins('CALL_MODULE',module,function,0,0)+ins('RET')])
                add(f'call-{resolved}-{module}-{function}',root,[(resolved,target)])
    for op,item in OPS.items():
        if not (11 <= item['code'] <= 14 or 22 <= item['code'] <= 31): continue
        code=ins(op)
        add('owned-root-'+op,assembled([code]),[(False,b'')],want='0:I refuse linked ownership execution contracts',graph='0|0:I refuse linked ownership execution contracts')
        add('owned-target-'+op,empty,[(True,assembled([code]))],want='0:I refuse linked ownership execution contracts',graph='0|0:I refuse linked ownership execution contracts')
        if len(code)>1:
            add('truncated-target-'+op,empty,[(True,assembled([code[:-1]]))],want='1:1:0',graph='1|D')
        add('unknown-before-owned-'+op,empty,[(True,assembled([b'\xff'+code]))],want='1:1:0',graph='1|D')
    for current in (0,1):
        bodies=[b'\xff',ins('REGION_BEGIN')]
        if current: bodies.reverse()
        add(f'owned-later-function-{current}',empty,[(True,assembled(bodies))],want='0:I refuse linked ownership execution contracts')
    for name,data,_ in service_corpus():
        if name.endswith('-valid'):
            add('root-service-'+name,data,[(False,b'')],want='0:I refuse service contracts before mixed execution selection',graph='0|0:I refuse service contracts before mixed execution selection')
            add('linked-service-'+name,assembled(returns=[[1]]),[(True,data)],
                want='0:I refuse linked service contracts before reviewed dispatch admission',graph='0|0:I refuse linked service contracts before reviewed dispatch admission')
    for name,data,_ in core_corpus()[1]:
        add('private-root-'+name,data,[(True,target)],private=True)
        add('private-target-'+name,empty,[(True,data)],private=True)
    for raw in (b'',b'\0'*7,struct.pack('<II',0,65),struct.pack('<II',0,0)+b'x'):
        cases.append(('bad-carrier-'+str(len(cases)),raw,'X','X',False))
    return cases


class NanoLangTranslatorLinkedModuleVerifier(unittest.TestCase):
    def test_public_linked_and_complete_graph(self):
        retained=os.environ.get('NANOLANG_LINKED_VERIFIER_PRODUCTS')
        with (nullcontext(retained) if retained else tempfile.TemporaryDirectory(prefix='nano-linked-verifier-')) as name:
            directory=Path(name);directory.mkdir(parents=True,exist_ok=True)
            inputs=directory/'inputs';inputs.mkdir(exist_ok=False)
            cases=corpus();paths=[]
            for i,case in enumerate(cases):
                path=inputs/f'{i}.bin';path.write_bytes(case[1]);paths.append(path)
            routes=build_routes(ROOT/'tests/nvm2c_linked_module_verifier_driver.nano',directory)
            oracle=ROOT/'obj/nvm2c_linked_module_verifier_reference'
            commands=directory/'commands';commands.mkdir();index=[];counts={}
            def reports(command,selected,graph):
                rows=[]
                for start in range(0,len(selected),32):
                    args=[*command,*(['--graph'] if graph else []),*selected[start:start+32]]
                    done=run_captured(args,timeout=600,cwd=ROOT)
                    number=len(index);(commands/f'{number}.stdout').write_bytes(done.stdout);(commands/f'{number}.stderr').write_bytes(done.stderr)
                    index.append(dict(argv=list(map(str,args)),returncode=done.returncode))
                    (directory/'commands.json').write_text(json.dumps(index,indent=2)+'\n')
                    self.assertEqual(done.returncode,0,done.stderr.decode(errors='replace'));rows+=done.stdout.decode().splitlines()
                self.assertEqual(len(rows),len(selected));return rows
            for graph,column,mode in [(False,2,'linked'),(True,3,'graph')]:
                reference=reports([oracle],paths,graph)
                (directory/f'{mode}-reference.txt').write_text('\n'.join(reference)+'\n')
                for case,value in zip(cases,reference):
                    if case[column] is not None:
                        with self.subTest(reference=case[0],mode=mode): self.assertEqual(value,case[column])
                baseline=None
                for route,command in routes:
                    actual=reports(command,paths,graph);(directory/f'{mode}-{route}.txt').write_text('\n'.join(actual)+'\n')
                    for case,want,value in zip(cases,reference,actual):
                        with self.subTest(route=route,case=case[0],mode=mode):
                            if case[4]: self.assertEqual(value.split('|')[-1],'A')
                            else: self.assertEqual(value,want)
                    if baseline is None: baseline=actual
                    else: self.assertEqual(actual,baseline,route)
                counts[mode]={value:reference.count(value) for value in sorted(set(reference))}
            producers=[([ROOT/'bin/nano_virt'],[])]
            source=source_compiler_command()
            if source: producers.append((source,['--emit-nvm']))
            products=[]
            for i,(producer,flags) in enumerate(producers):
                for original in ['examples/language/nl_functions_basic.nano','examples/language/nl_control_if_while.nano','examples/cross_backend/hello_cross_backend.nano']:
                    path=directory/f'original-{i}-{Path(original).stem}.nvm'
                    checked([*producer,ROOT/original,*flags,'-o',path],build=True)
                    raw=path.read_bytes();entry=struct.unpack_from('<I',raw,28)[0]
                    self.assertNotEqual(entry,NONE)
                    root=assembled([ins('CALL_MODULE',0,entry,0,1)+ins('RET')],returns=[[1]])
                    linked=directory/f'original-{i}-{Path(original).stem}.bin'
                    linked.write_bytes(carrier(root,[(True,raw)]));products.append(linked)
            for graph,mode in [(False,'linked'),(True,'graph')]:
                expected=reports([oracle],products,graph)
                self.assertTrue(all(value.startswith('-1|1:' if graph else '1:') for value in expected))
                (directory/f'original-{mode}-reference.txt').write_text('\n'.join(expected)+'\n')
                for route,command in routes:
                    actual=reports(command,products,graph)
                    (directory/f'original-{mode}-{route}.txt').write_text('\n'.join(actual)+'\n')
                    self.assertEqual(actual,expected,(mode,route))
            (directory/'manifest.json').write_text(json.dumps(dict(cases=len(cases),inherited=sum(c[0].startswith('inherited-') for c in cases),
                routes=[r for r,_ in routes],reports=counts,original_products=len(products),excluded_nonexpressible_legacy_contexts=2,explicit_private_deferrals=sum(c[4] for c in cases),independent_expectations={mode:sum(c[col] is not None for c in cases) for mode,col in [('linked',2),('graph',3)]},inputs=[dict(name=c[0],file=p.name,sha256=hashlib.sha256(c[1]).hexdigest()) for c,p in zip(cases,paths)]),indent=2)+'\n')
