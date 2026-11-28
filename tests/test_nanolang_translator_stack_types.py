"""I compare complete shallow type decisions/diagnostics with the actual C walk."""
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
from tests.native_process import run_captured
from tests.test_nanolang_translator_record_stack import Assembly

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = yaml.safe_load((ROOT/'spec/nanoisa.yaml').read_text())
OPS = {entry['name']: entry for entry in SCHEMA['legacy_opcodes']}
WIDTHS = dict(U8=1, U16=2, U32=4, I32=4, I64=8, F64=8)


def encoded(name):
    entry = OPS[name]
    return bytes([entry['code']])+bytes(sum(WIDTHS[k] for k in entry.get('operands', [])))


def carrier(code=b'', *, start=0, length=None, functions=1, current=0, maximum=8, mode=0):
    return struct.pack('<6I', start, len(code)-start if length is None else length,
                       functions, current, maximum, mode)+code


def corpus():
    cases = []
    def add(name, code=b'', expected=None, **kwargs):
        cases.append((name, carrier(code, **kwargs), expected))
    for opcode in range(256):
        expected = None
        if opcode == OPS['I64_ADD']['code']: expected = 'R:1:2:1:1:1:1:0:I64_ADD'
        if opcode == OPS['CAST_U8']['code']: expected = 'R:1:1:2:1:1:0:0:CAST_U8'
        if opcode == OPS['DUP']['code']: expected = 'R:0:0:255:0:0:0:0:DUP'
        add(f'rule-{opcode}', start=opcode, length=0, mode=1, expected=expected)
    add('rule-outside', start=0xffffffff, length=0, mode=1, expected='R:0:0:255:0:0:0:0:UNKNOWN')
    constants = [('void', 'PUSH_VOID'), ('int', 'PUSH_I64'), ('u8', 'PUSH_U8'),
                 ('float', 'PUSH_F64'), ('bool', 'PUSH_BOOL'), ('string', 'PUSH_STR'),
                 ('array', 'ARR_NEW'), ('struct', 'STRUCT_NEW'),
                 ('function', 'FUNCREF'), ('hashmap', 'HM_NEW'), ('opaque', 'OPAQUE_NULL'),
                 ('unknown', 'LOAD_LOCAL')]
    pushes = {tag: encoded(name) for tag, name in constants}
    halt = encoded('HALT')
    # My corpus does not execute these instructions. Bounds/height/host/effect
    # admission remains with the other verifiers; I observe only this helper.
    for name, entry in OPS.items():
        for tag, _ in constants:
            for depth in (0, 1, 2, 3):
                add(f'transfer-{name}-{tag}-{depth}', pushes[tag]*depth+encoded(name)+halt)
    checks = [('I64_ADD', ('int', 'int')), ('I64_NEG', ('int',)),
              ('F64_ADD', ('float', 'float')), ('F64_NEG', ('float',)),
              ('BOOL_AND', ('bool', 'bool')), ('BOOL_NOT', ('bool',)),
              ('MEM_LOAD8', ('int',)), ('MEM_STORE64', ('int', 'int')),
              ('I64_ADD_CARRY', ('int', 'int', 'int')),
              ('I64_MUL_WIDE_U', ('int', 'int')), ('F64_FROM_BITS', ('int',)),
              ('F64_TO_BITS', ('float',)), ('CAST_U8', ('int',))]
    for operation, required in checks:
        for index in range(len(required)):
            for tag, _ in constants:
                values = list(required);values[index] = tag
                prefix = b''.join(pushes[value] for value in values)
                unknown = tag == 'unknown'
                identity = operation == 'CAST_U8' and tag == 'u8'
                expected = '1'
                if tag != required[index] and not unknown and not identity:
                    want = 'int or u8' if operation == 'CAST_U8' else required[index]
                    expected = f'0:function[0] {operation} at offset {len(prefix)} expects {want} but the operand is {tag}'
                add(f'explicit-{operation}-{index}-{tag}', prefix+encoded(operation)+halt, expected=expected)
    bad = pushes['string']+pushes['int']+encoded('I64_ADD')+halt
    for depth in (0, 1, 2, 255, 256, 257, 65535, 65536):
        want = 'D' if depth > 65535 else '1' if depth in (0, 1, 257, 65535) else '0:function[0] I64_ADD at offset 14 expects int but the operand is string'
        add(f'optional-depth-{depth}', bad, maximum=depth, expected=want)
    add('generic-add-is-not-typed', pushes['string']+pushes['int']+encoded('ADD')+halt, expected='1')
    add('underflow-is-not-type-error', pushes['string']+encoded('I64_ADD')+halt, expected='1')
    add('duplicate-results-are-unknown', pushes['string']+encoded('DUP')+encoded('I64_ADD')+halt, expected='1')
    add('dynamic-call-clears-types', pushes['string']+encoded('CALL')+pushes['string']+encoded('I64_ADD')+halt, expected='1')
    for count, current in ((1,0), (4,3), (65536,65535)):
        add(f'function-number-{current}', bad, functions=count, current=current,
            expected=f'0:function[{current}] I64_ADD at offset 14 expects int but the operand is string')
    add('nonzero-offset', b'\xff'+bad+b'\xff', start=1, length=len(bad),
        expected='0:function[0] I64_ADD at offset 14 expects int but the operand is string')
    add('invalid-decoder-before-optional-skip', b'\xff', maximum=0, expected='D')
    add('invalid-late-instruction', halt+b'\xff', expected='D')
    add('invalid-call-target', bytes([OPS['CALL']['code']])+struct.pack('<I',1), expected='D')
    add('no-functions', functions=0, expected='D')
    add('bad-current', current=1, expected='D')
    add('wrong-mode', mode=2, expected='D')
    add('empty', expected='1')
    # Branches visit their target first, and each queued state occurs at most
    # once. I preserve this order even when it exposes an early contradiction.
    for variant in range(16):
        a = Assembly();a.emit(pushes['unknown']);a.branch(OPS['JMP_TRUE']['code'], 'right')
        a.emit(pushes['string'] if variant&1 else pushes['int']);a.branch(OPS['JMP']['code'], 'join')
        a.label('right');a.emit(pushes['bool'] if variant&2 else pushes['int'])
        a.label('join');a.emit(pushes['int']);a.emit(encoded('I64_ADD'))
        if variant&4:a.branch(OPS['JMP_TRUE']['code'], 'right')
        if variant&8:a.emit(encoded('POP'))
        a.emit(halt);add(f'join-requeue-{variant}', a.finish())
    for opcode in ('MATCH_TAG','HANDLER_PUSH','FILE_RESULT_BRANCH'):
        entry = OPS[opcode]
        for target in (0, 1, 13, -1, 10000):
            code = bytearray(encoded(opcode));offset = 1+WIDTHS[entry['operands'][0]]
            struct.pack_into('<i',code,offset,target)
            add(f'branch-{opcode}-{target}', pushes['string']+code+pushes['int']+encoded('I64_ADD')+halt)
    for operation in ('JMP','JMP_TRUE','JMP_FALSE','MATCH_TAG','HANDLER_PUSH','FILE_RESULT_BRANCH'):
        entry=OPS[operation];width=len(encoded(operation))
        operand=1 if operation in ('MATCH_TAG','HANDLER_PUSH','FILE_RESULT_BRANCH') else 0
        position=1+sum(WIDTHS[k] for k in entry.get('operands',[])[:operand])
        for target in (-1,0,width,width+1,width+2):
            code=bytearray(encoded(operation));struct.pack_into('<i',code,position,target)
            expected='1' if target in (0,width,width+1) else 'D'
            if operation=='FILE_RESULT_BRANCH' and target>=width+1:expected='D'
            add(f'decode-boundary-{operation}-{target}',bytes(code)+halt,expected=expected)
        final=bytearray(encoded(operation));struct.pack_into('<i',final,position,width)
        add(f'decode-last-{operation}',bytes(final),expected='D' if operation=='FILE_RESULT_BRANCH' else '1')
    for operation in ('JMP','MATCH_TAG','HANDLER_PUSH','FILE_RESULT_BRANCH'):
        code=bytearray(encoded(operation));position=1 if operation=='JMP' else 1+WIDTHS[OPS[operation]['operands'][0]]
        struct.pack_into('<i',code,position,1)
        add(f'decode-before-skip-{operation}',bytes(code)+halt,maximum=257,expected='D')
    for edge in ('end','next'):
        handler=bytearray(encoded('HANDLER_PUSH'))
        distance=len(handler) if edge=='next' else len(handler)+len(pushes['int'])+len(encoded('I64_ADD'))+len(halt)
        struct.pack_into('<i',handler,5,distance)
        add(f'handler-depth-{edge}',pushes['string']+handler+pushes['int']+encoded('I64_ADD')+halt,expected='1')
    file_branch=encoded('FILE_RESULT_BRANCH')
    add('file-branch-keeps-fallthrough-types',pushes['string']+file_branch+pushes['int']+encoded('I64_ADD')+halt,
        expected='0:function[0] I64_ADD at offset 21 expects int but the operand is string')
    for operation in ('CALL','TAIL_CALL'):
        call=bytes([OPS[operation]['code']])+struct.pack('<I',65535)
        add(f'large-direct-target-{operation}',call+halt,functions=65536,expected='1')
    rng = random.Random(1142005)
    pool = ['PUSH_I64','PUSH_STR','PUSH_BOOL','PUSH_F64','PUSH_U8','PUSH_VOID',
            'LOAD_LOCAL','STORE_LOCAL','DUP','SWAP','POP','I64_ADD','F64_ADD','BOOL_NOT','CAST_U8','NOP']
    for number in range(512):
        a = Assembly();length = rng.randrange(1,33)
        for index in range(length):
            a.label(str(index))
            if rng.randrange(5)==0:a.branch(rng.choice([OPS[n]['code'] for n in ['JMP','JMP_TRUE','JMP_FALSE']]),str(rng.randrange(length+1)))
            else:a.emit(encoded(rng.choice(pool)))
        a.label(str(length));a.emit(halt)
        add(f'random-cfg-{number}',a.finish(),maximum=rng.choice((1,2,3,8,16,256,257)))
    for size in (1,16,64,257):
        # I also retain linear state traversal separately from actual requeues.
        add(f'linear-state-{size}',encoded('NOP')*size+bad,expected=f'0:function[0] I64_ADD at offset {size+14} expects int but the operand is string')
    for size in (1,16,64,257):
        a=Assembly();a.emit(pushes['int']);a.label('loop')
        a.emit(encoded('NOP')*size)
        a.emit(pushes['string']+encoded('SWAP')+encoded('POP')+pushes['unknown'])
        a.branch(OPS['JMP_TRUE']['code'],'loop');a.emit(halt)
        add(f'queue-wrap-{size}',a.finish(),expected='1')
    for count in (1,16,64):
        code = encoded('PUSH_I64')*count+encoded('I64_ADD')*max(0,count-1)+halt
        add(f'state-capacity-{count}',code,maximum=min(256,count),expected='1')
    return cases


class StackTypes(unittest.TestCase):
    def test_complete_shared_type_walk(self):
        (ROOT/'.test_output').mkdir(exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='stack-types-',dir=ROOT/'.test_output'))
        print(f'I retain shallow type-walk products at {work}',flush=True)
        oracle = Path(os.environ.get('NANOLANG_STACK_TYPES_REFERENCE',ROOT/'obj/nvm2c_stack_types_reference'))
        cases = corpus();paths=[]
        record = dict(cases=[dict(name=n,expected=e,sha256=hashlib.sha256(d).hexdigest()) for n,d,e in cases],routes=[])
        (work/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
        for index,(_,data,_) in enumerate(cases):
            path=work/f'case-{index}.bin';path.write_bytes(data);paths.append(path)
        def execute(command,name):
            reports=[]
            for i in range(0,len(paths),32):reports+=checked([*command,*paths[i:i+32]],timeout=600).splitlines()
            self.assertEqual(len(reports),len(cases));(work/(name+'.reports')).write_text('\n'.join(reports)+'\n');return reports
        expected=execute([oracle],'actual-c')
        for (name,_,want),got in zip(cases,expected):
            if want is not None:
                with self.subTest(expectation=name):self.assertEqual(got,want)
        # I observe the unchanged C helper's existing private work counter.
        # The counted side entry must retain the ordinary decision/diagnostic.
        traces=[];observed=[]
        for i in range(0,len(paths),32):
            completed=run_captured([oracle,'--trace',*paths[i:i+32]],cwd=ROOT,timeout=600)
            self.assertEqual(completed.returncode,0,completed.stderr.decode())
            observed+=completed.stdout.decode().splitlines();traces+=completed.stderr.decode().splitlines()
        self.assertEqual(observed,expected)
        (work/'actual-c.work-trace').write_text('\n'.join(traces)+'\n')
        observations={}
        for line in traces:
            path,count,maximum,charged=line.split('\t');count=int(count);maximum=int(maximum);charged=int(charged)
            visits=0
            if 0<maximum<=256 and count:
                initial=(count+1)*(maximum+4);step=maximum*4+32
                self.assertGreaterEqual(charged,initial);self.assertEqual((charged-initial)%step,0)
                visits=(charged-initial)//step
            else:self.assertEqual(charged,0)
            observations[Path(path).name]=dict(instructions=count,visits=visits,work=charged)
        wraps=[]
        for index,(name,_,_) in enumerate(cases):
            if name.startswith('queue-wrap-'):
                observation=observations[paths[index].name]
                self.assertGreater(observation['visits'],observation['instructions']+1)
                # Widening requeues my NOP chain plus PUSH_STR and SWAP.
                # SWAP already emits unknowns, so its successor is unchanged.
                self.assertEqual(observation['visits'],observation['instructions']+int(name.rsplit('-',1)[1])+2)
                wraps.append(dict(name=name,**observation))
        self.assertEqual(len(wraps),4);record['observed_queue_wraps']=wraps
        (work/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
        for name,command in build_routes(ROOT/'tests/nvm2c_stack_types_driver.nano',work):
            actual=execute(command,name)
            for (case,_,_),want,got in zip(cases,expected,actual):
                with self.subTest(route=name,case=case):self.assertEqual(got,want)
            record['routes'].append(dict(name=name,command=list(map(str,command)),reports=len(actual)))
            (work/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
        print(f'I matched {len(cases)} complete type reports per route, including '
              f'{sum(want is not None for _,_,want in cases)} explicit expectations.',flush=True)
