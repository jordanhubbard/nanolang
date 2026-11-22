"""I compare the ordinary shared body and retain an explicit affine boundary."""
from pathlib import Path
import hashlib
import json
import os
import random
import struct
import tempfile
import unittest
from tests.nanolang_translator_routes import build_routes
from tests.native_process import run_captured
from tests.test_nanolang_translator_stack_heights import ins,depth,OPS,corpus as height_corpus
from tests.test_nanolang_translator_record_stack import Assembly

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=(0,0,0,4,4)  # arity, results, tag, locals, upvalues


def carrier(code=b'', *, functions=(DEFAULT,), imports=(), linked=(), current=0,
            strings=4, records=4, enums=4, unions=4, start=0, length=None):
    return (struct.pack('<10I',start,len(code)-start if length is None else length,
                        len(functions),current,len(imports),len(linked),strings,records,enums,unions)
            +b''.join(struct.pack('<5I',*f) for f in functions)
            +b''.join(struct.pack('<2I',*f) for f in imports)
            +b''.join(struct.pack('<2I',resolved,len(functions))+b''.join(struct.pack('<2I',*f) for f in functions) for resolved,functions in linked)+code)


def corpus():
    cases=[]
    def add(name,code=b'',expected=None,**kwargs):cases.append((name,carrier(code,**kwargs),expected))
    halt=ins('HALT');push=ins('PUSH_VOID');integer=ins('PUSH_I64');string=ins('PUSH_STR')
    for name,data,want in height_corpus():
        start,length,count,current,locals_,foreign=struct.unpack_from('<6I',data)
        signatures=[struct.unpack_from('<2I',data,24+i*8) for i in range(count)]
        imports=[struct.unpack_from('<2I',data,24+count*8+i*8) for i in range(foreign)]
        code=data[24+(count+foreign)*8:]
        functions=tuple((arity,results,1 if results else 0,locals_ if i==current else 0,0) for i,(arity,results) in enumerate(signatures))
        add('height-composition-'+name,code,functions=functions,imports=imports,current=current,start=start,length=length)
    by_code={item['code']:name for name,item in OPS.items()}
    for opcode in range(256):
        encoded=ins(by_code[opcode]) if opcode in by_code else bytes([opcode])
        for have in (0,1,2,3):
            add(f'primary-{opcode}-{have}',push*have+encoded+halt,imports=((0,0),))
    for name in ('LOAD_GLOBAL','STORE_GLOBAL','LOAD_LOCAL','STORE_LOCAL','LOAD_UPVALUE','STORE_UPVALUE',
                 'STRUCT_NEW','STRUCT_LITERAL','ENUM_VAL','UNION_CONSTRUCT','PUSH_STR','CALL_EXTERN','CLOSURE_NEW','FUNCREF'):
        kinds=OPS[name].get('operands',[])
        for index in (0,1,3,4,255,4095,4096,65534,65535,65536,0xffffffff):
            if kinds[0]=='U16' and index>65535:continue
            args=[index]+[0]*(len(kinds)-1)
            add(f'operand-{name}-{index}',push*3+ins(name,*args)+halt,imports=((0,0),))
            add(f'dead-operand-{name}-{index}',halt+ins(name,*args),imports=((0,0),))
    for name in ('LOAD_UPVALUE','STORE_UPVALUE'):
        for dep in (0,1,65535):
            for index in (0,3,4,65535):
                add(f'upvalue-{name}-{dep}-{index}',push+ins(name,dep,index)+halt)
    for name,key in [('STRUCT_NEW','records'),('STRUCT_LITERAL','records'),('ENUM_VAL','enums'),('UNION_CONSTRUCT','unions')]:
        for count in (0,1,65536,0xffffffff):
            for index in (0,1,65535,0xffffffff):
                args=[index]+[0]*(len(OPS[name].get('operands',[]))-1)
                add(f'optional-table-{name}-{count}-{index}',ins(name,*args)+halt,**{key:count})
    for name in ('ARR_NEW','TYPE_CHECK','ARR_LITERAL'):
        for tag in range(256):
            label='OP_ARR_LITERAL' if name=='ARR_LITERAL' else name
            description='element' if name=='ARR_LITERAL' else 'type'
            want=f'1:{1 if name=="TYPE_CHECK" else 2}' if tag<16 else f'0:function[0] {label} at offset 1: {description} tag {tag} >= TAG_COUNT 16'
            add(f'tag-{name}-{tag}',push+ins(name,*([tag]+[0]*(len(OPS[name].get('operands',[]))-1)))+halt,expected=want)
    for key in (0,1,15,16,255):
        for value in (0,1,15,16,255):
            want='1:1' if key<16 and value<16 else f'0:function[0] OP_HM_NEW at offset 0: {"key" if key>=16 else "value"} tag {key if key>=16 else value} >= TAG_COUNT 16'
            add(f'map-tags-{key}-{value}',ins('HM_NEW',key,value)+halt,expected=want)
    for kind in (0,1,2,3,255):
        for layout in (0,3,4,0xffffffff):
            for count in (0,4):
                add(f'aggregate-{kind}-{layout}-{count}',ins('AGG_PACK',kind,layout,0,0)+halt,records=count,unions=count)
    add('aggregate-kind-priority',ins('AGG_PACK',255,0xffffffff,0,0)+halt,
        expected='0:function[0] AGG_PACK at offset 0: invalid kind 255')
    for result in (0,1,2):
        for tag in (0,1,2):
            functions=((0,1,1,0,0),(0,result,tag,0,0))
            want='1:1' if result==1 and tag==1 else '0:function[0] OP_TAIL_CALL at offset 0 has incompatible result signature'
            add(f'tail-signature-{result}-{tag}',ins('TAIL_CALL',1),functions=functions,expected=want)
    for imports in ((),((0,0),),((1,1),)):
        for index in (0,1,0xffffffff):add(f'import-index-{len(imports)}-{index}',ins('CALL_EXTERN',index)+halt,imports=imports)
    for captured in (0,1,4,65535):
        for declared in (0,1,4,65535):
            add(f'closure-shape-{captured}-{declared}',depth(captured)+ins('CLOSURE_NEW',1,captured)+halt,
                functions=(DEFAULT,(0,0,0,0,declared)))
    for module in (0,1,63,64,0xffffffff):
        for function in (0,1,65535,0xffffffff):
            for linked in ((),((0,()),),((1,()),),((1,((0,0),)),),((1,((0,0),(1,1))),)):
                add(f'linked-target-{len(cases)}',ins('CALL_MODULE',module,function,0,0)+halt,linked=linked)
    for actual_arity in (0,1,65535):
        for actual_results in (0,1,255):
            for declared_arity in (0,1,65535):
                for declared_results in (0,1,255,65535):
                    # I prefill through an indirect stack effect; a linked
                    # module call here would itself trigger shape checks.
                    prefill=push+ins('CALL_INDIRECT',0,declared_arity) if declared_arity else b''
                    site=f'0:function[0] OP_CALL_MODULE at offset {len(prefill)}'
                    want=f'1:{max(declared_arity,declared_results)}'
                    if declared_arity!=actual_arity:want=f'{site} declares arity {declared_arity} but linked callee takes {actual_arity}'
                    elif declared_results!=actual_results:want=f'{site} declares {declared_results} results but linked callee returns {actual_results}'
                    add(f'linked-shape-{actual_arity}-{actual_results}-{declared_arity}-{declared_results}',
                        prefill+ins('CALL_MODULE',0,0,declared_arity,declared_results)+halt,
                        linked=((1,((actual_arity,actual_results),)),),functions=((0,0,0,0,0),),expected=want)
    add('linked-unresolved-priority',ins('CALL_MODULE',0,0xffffffff,65535,65535)+halt,linked=((0,()),),
        expected='0:function[0] OP_CALL_MODULE at offset 0: linked module 0 is unresolved')
    add('linked-arity-priority',ins('CALL_MODULE',0,0,65535,65535)+halt,linked=((1,((1,1),)),),
        expected='0:function[0] OP_CALL_MODULE at offset 0 declares arity 65535 but linked callee takes 1')
    add('linked-results-priority',ins('CALL_MODULE',0,0,1,65535)+halt,linked=((1,((1,1),)),),
        expected='0:function[0] OP_CALL_MODULE at offset 0 declares 65535 results but linked callee returns 1')
    width=len(ins('HANDLER_PUSH'))
    for name in (0,3,4,0xffffffff):
        for base in (0,3,4,65535):
            for params in (0,1,4,65535):
                for target in (0,width,width+1):
                    want='1:0'
                    if name>=4 or base+params>4:want='0:I require valid handler names and parameter slots.'
                    elif target==width+1:want='0:I require an executable handler arm.'
                    add(f'handler-{name}-{base}-{params}-{target}',ins('HANDLER_PUSH',name,target,base,params)+halt,expected=want)
    add('handler-name-before-arm',ins('HANDLER_PUSH',4,width,0,0),
        expected='0:I require valid handler names and parameter slots.')
    add('handler-executable-arm',ins('HANDLER_PUSH',0,width,0,0),expected='0:I require an executable handler arm.')
    for name in (0,3,4,0xffffffff):
        add(f'effect-name-{name}',ins('PERFORM',name,0)+halt)
    affine=[name for name,item in OPS.items() if 11<=item['code']<=14 or 22<=item['code']<=31]
    for name in affine:
        add('affine-'+name,ins(name)+halt,expected='A')
        add('dead-affine-'+name,halt+ins(name),expected='A')
        add('ordinary-before-affine-'+name,ins('LOAD_LOCAL',65535)+ins(name)+halt,
            expected='0:function[0] LOAD_LOCAL at offset 0: slot 65535 >= local_count 4')
        add('affine-before-ordinary-'+name,ins(name)+ins('LOAD_LOCAL',65535)+halt,expected='A')
    typed_bad=string+integer+ins('I64_ADD')+halt
    add('typed-mismatch',typed_bad,expected='0:function[0] I64_ADD at offset 14 expects int but the operand is string')
    add('height-before-type',typed_bad[:-1]+ins('POP')+ins('POP')+halt,
        expected='0:function[0] stack underflow at offset 16 (POP needs 1, has 0)')
    add('late-operand-before-type',typed_bad+ins('LOAD_LOCAL',65535),
        expected='0:function[0] LOAD_LOCAL at offset 16: slot 65535 >= local_count 4')
    add('late-operand-before-height',ins('POP')+halt+ins('LOAD_LOCAL',65535),
        expected='0:function[0] LOAD_LOCAL at offset 2: slot 65535 >= local_count 4')
    add('decode-before-operands',ins('LOAD_LOCAL',65535)+b'\xff',expected='D')
    add('nonzero-type-offset',b'\xff'+typed_bad+b'\xff',start=1,length=len(typed_bad),
        expected='0:function[0] I64_ADD at offset 14 expects int but the operand is string')
    add('nonzero-operand-offset',b'\xff'+ins('LOAD_LOCAL',65535)+halt+b'\xff',start=1,length=4,
        expected='0:function[0] LOAD_LOCAL at offset 1: slot 65535 >= local_count 4')
    add('generic-add',string+integer+ins('ADD')+halt,expected='1:2')
    add('valid-typed-add',integer+integer+ins('I64_ADD')+halt,expected='1:2')
    for size in (256,257):
        expected='0:function[0] I64_ADD at offset 27 expects int but the operand is string' if size==256 else '1:257'
        add(f'optional-types-{size}',depth(size-2)+typed_bad,expected=expected,functions=((0,0,0,0,0),))
    for field in range(5):
        f=list(DEFAULT);f[field]=65536 if field in (0,3,4) else 256
        add('invalid-function-field-'+str(field),halt,functions=(tuple(f),),expected='D')
    add('invalid-link-resolved',halt,linked=((2,()),),expected='D')
    add('invalid-link-arity',halt,linked=((1,((65536,0),)),),expected='D')
    add('invalid-link-results',halt,linked=((1,((0,256),)),),expected='D')
    add('current-65535',ins('LOAD_LOCAL',65535)+halt,current=65535,functions=(DEFAULT,)*65536,
        expected='0:function[65535] LOAD_LOCAL at offset 0: slot 65535 >= local_count 4')
    rng=random.Random(1142007);pool=['NOP','PUSH_I64','PUSH_STR','PUSH_BOOL','PUSH_U8','POP','DUP','SWAP','I64_ADD','F64_ADD',
                                  'GC_RETAIN','GC_RELEASE','LOAD_LOCAL','STORE_LOCAL','CALL','TAIL_CALL','CALL_EXTERN','RET','HALT']
    for number in range(1024):
        a=Assembly();length=rng.randrange(1,49)
        for index in range(length):
            a.label(str(index))
            if rng.randrange(4)==0:a.branch(rng.choice([OPS[n]['code'] for n in ('JMP','JMP_TRUE','JMP_FALSE')]),str(rng.randrange(length+1)))
            else:a.emit(ins(rng.choice(pool)))
        a.label(str(length));a.emit(halt)
        add(f'random-body-{number}',a.finish(),functions=((0,rng.randrange(3),rng.randrange(3),rng.choice((0,1,4,65535)),0),),imports=((rng.randrange(3),rng.randrange(3)),))
    return cases


class FunctionBody(unittest.TestCase):
    def test_ordinary_body_composition(self):
        (ROOT/'.test_output').mkdir(exist_ok=True)
        work=Path(tempfile.mkdtemp(prefix='function-body-',dir=ROOT/'.test_output'))
        print(f'I retain ordinary function-body products at {work}',flush=True)
        oracle=Path(os.environ.get('NANOLANG_FUNCTION_BODY_REFERENCE',ROOT/'obj/nvm2c_function_body_reference'))
        cases=corpus();paths=[]
        record=dict(cases=[dict(name=n,expected=e,sha256=hashlib.sha256(d).hexdigest()) for n,d,e in cases],routes=[])
        (work/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
        for index,(_,data,_) in enumerate(cases):
            path=work/f'case-{index}.bin';path.write_bytes(data);paths.append(path)
        def execute(command,name):
            reports=[];errors=[]
            for i in range(0,len(paths),32):
                r=run_captured([*command,*paths[i:i+32]],cwd=ROOT,timeout=600)
                reports+=r.stdout.decode().splitlines();errors.append(r.stderr.decode())
                (work/(name+'.reports')).write_text('\n'.join(reports)+'\n');(work/(name+'.stderr')).write_text(''.join(errors))
                self.assertEqual(r.returncode,0,errors[-1])
            self.assertEqual(len(reports),len(cases));return reports
        expected=execute([oracle],'actual-c')
        self.assertEqual([(n,w,g) for (n,_,w),g in zip(cases,expected) if w is not None and w!=g],[])
        record['decisions']={kind:sum(x=='A' if kind=='affine-deferred' else x=='D' if kind=='decode-precondition' else x.startswith('1:') if kind=='passed' else x.startswith('0:') for x in expected) for kind in ('affine-deferred','decode-precondition','passed','ordinary-refused')}
        (work/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
        for name,command in build_routes(ROOT/'tests/nvm2c_function_body_driver.nano',work):
            actual=execute(command,name)
            for (case,_,_),want,got in zip(cases,expected,actual):
                with self.subTest(route=name,case=case):self.assertEqual(got,want)
            record['routes'].append(dict(name=name,command=list(map(str,command)),reports=len(actual)))
            (work/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
        print(f'I matched {len(cases)} ordinary body reports per route, including {sum(w is not None for _,_,w in cases)} explicit expectations; decisions={record["decisions"]}.',flush=True)
