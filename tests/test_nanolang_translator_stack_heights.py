"""I compare ordinary heights, explicit balances and exact shared diagnostics."""
from pathlib import Path
import hashlib
import json
import os
import random
import struct
import tempfile
import unittest
import yaml
from tests.nanolang_translator_routes import build_routes, checked
from tests.test_nanolang_translator_record_stack import Assembly

ROOT = Path(__file__).resolve().parents[1]
OPS = {x['name']: x for x in yaml.safe_load((ROOT/'spec/nanoisa.yaml').read_text())['legacy_opcodes']}
FORMATS = dict(U8='B', U16='H', U32='I', I32='i', I64='q', F64='d')


def ins(name, *values):
    item = OPS[name]; kinds = item.get('operands', [])
    if not values: values = (0,)*len(kinds)
    assert len(values) == len(kinds)
    return bytes([item['code']])+struct.pack('<'+''.join(FORMATS[k] for k in kinds), *values)


def depth(count):
    # I describe an encoded external-module call's stack effect only. No
    # module/host execution or full structural acceptance follows from it.
    return ins('CALL_MODULE', 0, 0, 0, count)


def carrier(code=b'', *, signatures=((0,0),), imports=(), current=0, locals_=0, start=0, length=None):
    return (struct.pack('<6I',start,len(code)-start if length is None else length,
                        len(signatures),current,locals_,len(imports))
            + b''.join(struct.pack('<II',*s) for s in (*signatures,*imports))+code)


def corpus():
    cases=[]
    def add(name, code=b'', expected=None, **kwargs):
        cases.append((name,carrier(code,**kwargs),expected))
    halt=ins('HALT');ret=ins('RET');pop=ins('POP');push=ins('PUSH_VOID')
    # All primary bytes, including undefined opcodes and unreachable malformed
    # bytes, are decoded before this helper walks reachable instructions.
    for opcode in range(256):
        name=next((name for name,item in OPS.items() if item['code']==opcode),None)
        encoded=ins(name) if name else bytes([opcode])
        for height in (0,1,2,3,16,65535):
            add(f'primary-{opcode}-depth-{height}',depth(height)+encoded+halt,imports=((0,0),))
    for pushes in range(8):
        for pops in range(9):
            for results in range(3):
                code=push*pushes+pop*pops+ret
                want=f'1:{pushes}'
                if pops>pushes:want=f'0:function[0] stack underflow at offset {pushes*2} (POP needs 1, has 0)'
                elif pushes-pops!=results:want=f'0:function[0] returns with {pushes-pops} values at offset {pushes+pops} but declares {results}'
                add(f'linear-{pushes}-{pops}-{results}',code,expected=want,signatures=((0,results),))
    for results in (0,1,255):
        add(f'empty-{results}',expected='1:0',signatures=((0,results),))
        for height in range(4):
            expected=f'1:{height}' if height==results else f'0:function[0] reaches its end after offset 0 with {height} values but declares {results}'
            add(f'implicit-{results}-{height}',depth(height),expected=expected,signatures=((0,results),))
    for name in ('OWN_PACK','OWN_UNPACK_LOCAL','FILE_SERVICE','FILE_CALL_REFS','FILE_CALL_INDIRECT_REFS'):
        add('unknown-effect-'+name,ins(name)+halt,expected=f'0:function[0] {name} at offset 0 has no known stack effect')
        add('dead-unknown-'+name,halt+ins(name),expected='1:0')
    for name in ('HALT','TAIL_CALL','EFFECT_RESUME'):
        code=(push if name=='EFFECT_RESUME' else b'')+ins(name)+pop
        add('terminal-'+name,code,expected='1:1' if name=='EFFECT_RESUME' else '1:0')
    add('dead-invalid-byte',halt+b'\xff',expected='D')
    add('dead-underflow',halt+pop,expected='1:0')
    add('nonzero-span',b'\xff'+push+ret+b'\xff',start=1,length=2,signatures=((0,1),),expected='1:1')
    add('invalid-start',b'',start=1,length=0,expected='D')
    add('invalid-length',b'',length=1,expected='D')
    add('no-functions',signatures=(),expected='D')
    add('bad-current',current=1,expected='D')
    for field in ('arity','results','locals','params','tag'):
        options={'arity':dict(signatures=((65536,0),)), 'results':dict(signatures=((0,256),)),
                 'locals':dict(locals_=65536), 'params':dict(imports=((65536,0),)),
                 'tag':dict(imports=((0,256),))}
        add('invalid-'+field,expected='D',**options[field])
    for local,height in ((0,65535),(1,65534),(1,65535),(65535,0),(65535,1)):
        total=local+height
        want=f'1:{height}' if total<=65535 else f'0:function[0] frame needs {total} slots ({local} locals + operand depth {height}), more than a frame can address'
        add(f'frame-{local}-{height}',depth(height)+halt,locals_=local,expected=want)
    add('two-encoded-depths-overflow',depth(65535)+depth(65535)+halt,expected='0:function[0] frame needs 131070 slots (0 locals + operand depth 131070), more than a frame can address')
    dynamic=[('CLOSURE_NEW',lambda n:ins('CLOSURE_NEW',0,n)),
             ('STRUCT_LITERAL',lambda n:ins('STRUCT_LITERAL',0,n)),
             ('UNION_CONSTRUCT',lambda n:ins('UNION_CONSTRUCT',0,0,n)),
             ('TUPLE_NEW',lambda n:ins('TUPLE_NEW',n)),
             ('AGG_PACK',lambda n:ins('AGG_PACK',0,0,0,n)),
             ('ARR_LITERAL',lambda n:ins('ARR_LITERAL',0,n)),
             ('PERFORM',lambda n:ins('PERFORM',0,n))]
    for name,encode in dynamic:
        for size in (0,1,2,255,65535):
            for delta in (-1,0,1):
                height=max(0,size+delta)
                if height>65535:continue
                want=f'1:{max(height,height-size+1)}' if height>=size else f'0:function[0] stack underflow at offset 13 ({name} needs {size}, has {height})'
                add(f'dynamic-{name}-{size}-{delta}',depth(height)+encode(size)+halt,expected=want)
    for name in ('PICK','ROLL','CALL_INDIRECT'):
        for size in (0,1,2,255,65535):
            for height in (0,min(65535,size),min(65535,size+1)):
                encoded=ins(name,size,2) if name=='CALL_INDIRECT' else ins(name,size)
                pops=size+1;after=height-pops+2 if name=='CALL_INDIRECT' else height+(name=='PICK')
                want=f'0:function[0] stack underflow at offset 13 ({name} needs {pops}, has {height})' if height<pops else f'1:{max(height,after)}'
                add(f'depth-operand-{name}-{size}-{height}',depth(height)+encoded+halt,expected=want)
    for name in ('CALL','TAIL_CALL'):
        for arity in (0,1,255,65535):
            for results in (0,1,255):
                for have in (0,arity):
                    want=f'1:{max(have,have-arity+results)}' if have>=arity else f'0:function[0] stack underflow at offset 13 ({name} needs {arity}, has {have})'
                    add(f'direct-{name}-{arity}-{results}-{have}',depth(have)+ins(name,1)+halt,
                        signatures=((0,0),(arity,results)),expected=want)
        add(f'invalid-{name}',ins(name,1)+halt,expected='D')
    for params in (0,1,255,65535):
        for results in (0,1,255,65535):
            for have in (0,params):
                want=f'0:function[0] stack underflow at offset 13 (CALL_MODULE needs {params}, has {have})' if have<params else f'1:{max(have,results)}'
                add(f'module-call-{params}-{results}-{have}',depth(have)+ins('CALL_MODULE',0,0,params,results)+halt,expected=want)
    for params in (0,1,255,65535):
        for tag in (0,1,15,255):
            for have in (0,params):
                want=f'1:{max(have,have-params+int(tag!=0))}' if have>=params else f'0:function[0] stack underflow at offset 13 (CALL_EXTERN needs {params}, has {have})'
                add(f'extern-{params}-{tag}-{have}',depth(have)+ins('CALL_EXTERN',0)+halt,imports=((params,tag),),expected=want)
    add('invalid-extern-index',ins('CALL_EXTERN',0)+halt,expected='D')
    add('dead-invalid-extern-index',halt+ins('CALL_EXTERN',0),expected='D')
    add('extern-65535',ins('CALL_EXTERN',65535)+halt,imports=((0,0),)*65536,expected='1:0')
    add('current-65535',push+ret,current=65535,signatures=((0,1),)*65536,expected='1:1')
    add('callee-65535',ins('CALL',65535)+halt,signatures=((0,1),)*65536,expected='1:1')
    # Retain consumes no value, release consumes one; I preserve underflow
    # priority over balance refusal, and check both independent merge states.
    for code,want in [(ins('GC_RELEASE')+halt,'0:function[0] stack underflow at offset 0 (GC_RELEASE needs 1, has 0)'),
                      (push+ins('GC_RELEASE')+halt,'0:function[0] releases a reference it does not hold at offset 1'),
                      (push+ins('GC_RETAIN')+ins('GC_RELEASE')+ret,'1:1'),
                      (push+ins('GC_RETAIN')+pop+ret,'0:function[0] returns at offset 3 still holding 1 retained reference(s)'),
                      (push+ins('GC_RETAIN')+pop,'0:function[0] reaches its end after offset 2 still holding 1 retained reference(s)'),
                      (push+ins('GC_RETAIN')+halt,'1:1'),
                      (ins('EFFECT_RESUME'),'0:I require one result and balanced ownership when resuming an effect.'),
                      (push+ins('GC_RETAIN')+ins('EFFECT_RESUME'),'0:I require one result and balanced ownership when resuming an effect.')]:
        add(f'ownership-explicit-{len(cases)}',code,expected=want)
    handler_size=len(ins('HANDLER_PUSH'))
    add('handler-held-priority',push+ins('GC_RETAIN')+ins('HANDLER_PUSH',0,handler_size,0,0)+halt,
        expected='0:I cannot install a nonlocal-return handler across outstanding explicit retains.')
    # A handler resets the target edge only. At the terminal boundary the
    # original helper instead checks the outgoing height before that reset.
    for height in (0,1,2):
        add(f'handler-terminal-{height}',depth(height)+ins('HANDLER_PUSH',0,handler_size,0,0),
            expected=f'1:{height}' if height==0 else f'0:function[0] reaches its end after offset 13 with {height} values but declares 0')
        add(f'handler-same-next-{height}',depth(height)+ins('HANDLER_PUSH',0,handler_size,0,0)+halt,
            expected='1:0' if height==0 else f'0:function[0] incompatible stack heights at offset 26 (0 and {height})')
    for retain in (False,True):
        a=Assembly();a.emit(push);a.branch(OPS['JMP_TRUE']['code'],'right')
        a.emit(push+(ins('GC_RETAIN') if retain else push));a.branch(OPS['JMP']['code'],'join')
        a.label('right');a.emit(push);a.label('join');a.emit(halt)
        want='0:function[0] incompatible ownership balance at offset 14 (0 and 1)' if retain else '0:function[0] incompatible stack heights at offset 14 (1 and 2)'
        add(f'join-height-or-ownership-{retain}',a.finish(),expected=want)
    for name in ('JMP','JMP_TRUE','JMP_FALSE','MATCH_TAG','HANDLER_PUSH','FILE_RESULT_BRANCH'):
        width=len(ins(name));operand=1 if name in ('MATCH_TAG','HANDLER_PUSH','FILE_RESULT_BRANCH') else 0
        kinds=OPS[name].get('operands',[]);position=1+sum(struct.calcsize('<'+FORMATS[k]) for k in kinds[:operand])
        for target in (-1,0,1,width,width+1,width+2):
            data=bytearray(ins(name));struct.pack_into('<i',data,position,target)
            add(f'branch-boundary-{name}-{target}',bytes(data)+halt)
        data=bytearray(ins(name));struct.pack_into('<i',data,position,width)
        add(f'branch-terminal-{name}',bytes(data))
        add(f'branch-nonzero-{name}',b'\xff'+bytes(data)+halt+b'\xff',start=1,length=len(data)+1)
    # FILE_RESULT_BRANCH remains ordinary fallthrough in this shared helper;
    # execution/profile-specific return branches have their own verification.
    file_branch=ins('FILE_RESULT_BRANCH',0,len(ins('FILE_RESULT_BRANCH'))+1)
    add('file-height-fallthrough',file_branch+push+halt,expected='1:1')
    rng=random.Random(1142006)
    pool=['PUSH_VOID','POP','DUP','SWAP','GC_RETAIN','GC_RELEASE','EFFECT_RESUME','NOP','RET','HALT','PICK','ROLL','I64_ADD']
    for number in range(1024):
        a=Assembly();size=rng.randrange(1,49)
        for index in range(size):
            a.label(str(index))
            if rng.randrange(4)==0:a.branch(rng.choice([OPS[n]['code'] for n in ('JMP','JMP_TRUE','JMP_FALSE')]),str(rng.randrange(size+1)))
            else:a.emit(ins(rng.choice(pool)))
        a.label(str(size));a.emit(halt)
        add(f'random-cfg-{number}',a.finish(),signatures=((0,rng.randrange(3)),),locals_=rng.choice((0,1,65535)))
    # Backedges terminate after one visit per instruction; I test long retained
    # balances as well as a late mismatch and both join error priorities.
    for size in (1,16,257,4096):
        a=Assembly();a.label('start');a.emit(ins('NOP')*size);a.branch(OPS['JMP']['code'],'start')
        add(f'loop-{size}',a.finish(),expected='1:0')
    return cases


class StackHeights(unittest.TestCase):
    def test_complete_shared_height_walk(self):
        (ROOT/'.test_output').mkdir(exist_ok=True)
        work=Path(tempfile.mkdtemp(prefix='stack-heights-',dir=ROOT/'.test_output'))
        print(f'I retain ordinary height-walk products at {work}',flush=True)
        oracle=Path(os.environ.get('NANOLANG_STACK_HEIGHTS_REFERENCE',ROOT/'obj/nvm2c_stack_heights_reference'))
        cases=corpus();paths=[]
        record=dict(cases=[dict(name=n,expected=e,sha256=hashlib.sha256(d).hexdigest()) for n,d,e in cases],routes=[])
        (work/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
        for index,(_,data,_) in enumerate(cases):
            path=work/f'case-{index}.bin';path.write_bytes(data);paths.append(path)
        def execute(command,name):
            reports=[]
            for i in range(0,len(paths),32):reports+=checked([*command,*paths[i:i+32]],timeout=600).splitlines()
            self.assertEqual(len(reports),len(cases));(work/(name+'.reports')).write_text('\n'.join(reports)+'\n');return reports
        expected=execute([oracle],'actual-c')
        failures=[(name,want,got) for (name,_,want),got in zip(cases,expected) if want is not None and want!=got]
        self.assertEqual(failures,[])
        for name,command in build_routes(ROOT/'tests/nvm2c_stack_heights_driver.nano',work):
            actual=execute(command,name)
            for (case,_,_),want,got in zip(cases,expected,actual):
                with self.subTest(route=name,case=case):self.assertEqual(got,want)
            record['routes'].append(dict(name=name,command=list(map(str,command)),reports=len(actual)))
            (work/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
        print(f'I matched {len(cases)} complete height reports per route, including '
              f'{sum(want is not None for _,_,want in cases)} explicit expectations.',flush=True)
