"""I describe strict document cases independently of the C scanner/renderer."""
import copy
import json
import re
from pathlib import Path

OK, INVALID, LIMIT, MEMORY, UNRESOLVED = range(5)
REFUSED = (INVALID, UNRESOLVED)

def encoded(value):
    return (json.dumps(value, separators=(',', ':'), ensure_ascii=False)+'\n').encode()

def reversed_keys(value):
    if isinstance(value, dict):
        return {key: reversed_keys(item) for key, item in reversed(list(value.items()))}
    if isinstance(value, list):
        return [reversed_keys(item) for item in value]
    return value

def leaf_paths(value, prefix=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaf_paths(item, prefix+(key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaf_paths(item, prefix+(index,))
    else:
        yield prefix, value

def at(value, path):
    for part in path:
        value = value[part]
    return value

def arrays(value, prefix=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from arrays(item, prefix+(key,))
    elif isinstance(value, list):
        yield prefix, value
        for index, item in enumerate(value):
            yield from arrays(item, prefix+(index,))

def corpus(root, output):
    base=json.loads((root/'tests/fixtures/nsi_file_plan.json').read_text())
    canonical=encoded(base)
    (output/'expected.json').write_bytes(canonical)
    entries=[]
    def add(name, data, statuses=REFUSED, preallocation=False, filename=None):
        filename=filename or f'case-{len(entries):04}.json'
        (output/filename).write_bytes(data)
        entries.append(dict(name=name,filename=filename,statuses=list(statuses),preallocation=preallocation))
    add('valid',(root/'tests/fixtures/nsi_file_plan.json').read_bytes(),(OK,),filename='valid.json')
    add('canonical',canonical,(OK,))
    add('reversed-object-keys',encoded(reversed_keys(base)),(OK,))
    add('escaped-slashes',canonical.replace(b'/',b'\\/'),(OK,))
    # I escape each character of every JSON key/value string, preserving facts.
    escaped=re.sub(rb'"(?:[^"\\]|\\.)*"',
                   lambda m: ('"'+''.join(f'\\u{ord(c):04x}' for c in json.loads(m[0]))+'"').encode(),canonical)
    add('escaped-key-and-value-spellings',escaped,(OK,))
    omitted=copy.deepcopy(base)
    for method in omitted['methods']:method.pop('idempotent')
    add('legacy-default-idempotent-false',encoded(omitted),(OK,))
    add('maximum-document',canonical+b' '*(1048576-len(canonical)),(OK,))
    add('document-one-over',canonical+b' '*(1048577-len(canonical)),(LIMIT,),True)
    # Each scalar catalog fact is changed alone, including every parameter mode
    # and member/arm identity; no reduced per-method document is substituted.
    for path,value in leaf_paths(base):
        changed=copy.deepcopy(base)
        replacement=(not value) if isinstance(value,bool) else value+1 if isinstance(value,int) else value+'x'
        at(changed,path[:-1])[path[-1]]=replacement
        add('mutate-'+'.'.join(map(str,path)),encoded(changed))
        removed=copy.deepcopy(base)
        at(removed,path[:-1]).pop(path[-1])
        add('omit-'+'.'.join(map(str,path)),encoded(removed),
            (OK,) if path[-1]=='idempotent' else REFUSED)
        alternatives={
            'direction':('in','out','inout','return'),
            'ownership':('borrow','transfer','copy'),
            'lifetime':('call','caller','callee','resource'),
            'mutability':('immutable','mutable'),
            'streaming':('none','in','out','bidi'),
            'type':('nsi:core/int','nsi:core/bool','nsi:core/unit','nsi:nanolang/filesystem#FileError')}
        for alternative in alternatives.get(path[-1],()):
            if alternative==value:continue
            changed=copy.deepcopy(base);at(changed,path[:-1])[path[-1]]=alternative
            add('valid-enum-or-type-'+'.'.join(map(str,path))+'-'+alternative,encoded(changed))
    for path,value in arrays(base):
        if not value:continue
        for action in ('remove','duplicate','reverse'):
            if action=='reverse' and len(value)<2:continue
            changed=copy.deepcopy(base);target=at(changed,path)
            if action=='remove':target.pop()
            elif action=='duplicate':target.append(copy.deepcopy(target[0]))
            else:target.reverse()
            add(action+'-'+'.'.join(map(str,path)),encoded(changed))
    for key in base:
        changed=copy.deepcopy(base);changed.pop(key)
        add('missing-root-'+key,encoded(changed))
    for kind,field,value in [('array','element','nsi:core/int'),('callback','method','nsi:nanolang/filesystem#temp'),('async','result','nsi:core/int')]:
        changed=copy.deepcopy(base);changed['types'][0]['kind']=kind;changed['types'][0][field]=value
        add('allocation-shape-'+kind,encoded(changed),(INVALID,),filename='alloc-'+kind+'.json')
    add('legacy-open-path-catalog',(root/'schema/nsi/modules/filesystem.nsi.json').read_bytes())
    invalid=[('empty',b''),('trailing-document',canonical+b'{}'),('trailing-garbage',canonical+b'x'),
        ('raw-nul',canonical+b'\0'),('counted-nul-middle',canonical[:20]+b'\0'+canonical[20:]),
        ('raw-control-in-string',b'{"x":"\x01"}'),('invalid-utf8',b'{"x":"\xff"}'),
        ('encoded-utf8-surrogate',b'{"x":"\xed\xa0\x80"}'),('nul-escape-value',b'{"x":"\\u0000"}'),
        ('nul-escape-key',b'{"\\u0000":0}'),('high-surrogate',b'{"x":"\\ud800"}'),
        ('low-surrogate',b'{"x":"\\udc00"}'),('bad-surrogate-pair',b'{"x":"\\ud800\\u0041"}'),
        ('bad-escape',b'{"x":"\\q"}'),('unfinished-string',b'{"x":"oops'),('short-unicode',b'{"x":"\\u0"}'),
        ('array-trailing-comma',b'{"x":[0,]}'),('object-trailing-comma',b'{"x":0,}'),
        ('leading-zero',b'{"x":01}'),('missing-fraction',b'{"x":1.}'),('missing-exponent',b'{"x":1e+}'),
        ('non-object-root',b'[]'),('bad-close',b'{"x":[0}'),('bad-literal',b'{"x":tru}')]
    for name,data in invalid:add(name,data,(INVALID,),True)
    add('duplicate-root-key',canonical.replace(b'{',b'{"nsi_version":0,',1),(INVALID,))
    add('duplicate-decoded-root-key',canonical.replace(b'{',b'{"\\u006esi_version":0,',1),(INVALID,))
    add('duplicate-nested-param-key',canonical.replace(b'"direction":"return"',b'"direction":"return","direction":"return"',1),(INVALID,))
    add('escaped-literal-not-nul',b'{"x":"\\\\u0000"}',(UNRESOLVED,))
    add('valid-surrogate-pair',b'{"x":"\\ud800\\udc00"}',(UNRESOLVED,))
    add('valid-nonascii-unknown-fact','{"x":"é"}'.encode(),(UNRESOLVED,))
    add('valid-control-escapes',b'{"x":"\\b\\f\\n\\r\\t\\\"\\\\\\/"}',(UNRESOLVED,))
    for depth in (64,65):
        add(f'depth-{depth}',b'{"x":'+b'['*(depth-1)+b'0'+b']'*(depth-1)+b'}',
            (UNRESOLVED,) if depth==64 else (LIMIT,),depth>64)
    for members in (64,65):
        add(f'members-{members}',encoded({f'k{i}':0 for i in range(members)}),
            (UNRESOLVED,) if members==64 else (LIMIT,),members>64)
    for elements in (256,257):
        add(f'elements-{elements}',encoded({'x':[0]*elements}),
            (UNRESOLVED,) if elements==256 else (LIMIT,),elements>256)
    for objects in (256,257):
        add(f'objects-{objects}',encoded({'x':[{} for _ in range(objects-1)]}),
            (UNRESOLVED,) if objects==256 else (LIMIT,),objects>256)
    for length in (4096,4097):
        status=(UNRESOLVED,) if length==4096 else (LIMIT,)
        add(f'string-{length}',encoded({'x':'a'*length}),status,length>4096)
        add(f'number-{length}',b'{"x":'+b'1'*length+b'}',status,length>4096)
    # Independent tokenization counts punctuation plus strings/literals; nested
    # arrays stay below every other cap so only the token boundary changes.
    matrix=[[0]*255 for _ in range(15)]+[[0]*252]
    matrix[0][0]=[]
    token_re=rb'"(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[{}\[\],:]'
    tokens=encoded({'x':matrix});assert len(re.findall(token_re,tokens))==8192
    add('tokens-8192',tokens,(UNRESOLVED,))
    matrix[0][1]=[]
    tokens=encoded({'x':matrix});assert len(re.findall(token_re,tokens))==8193
    add('tokens-8193',tokens,(LIMIT,),True)
    header=['typedef struct {const char *name,*filename; unsigned statuses; int preallocation;} BindingCase;',
            'static const BindingCase binding_cases[]={']
    for row in entries:
        mask=sum(1<<s for s in row['statuses'])
        header.append('{'+json.dumps(row['name'])+','+json.dumps(row['filename'])+f',{mask},{int(row["preallocation"])}'+'},')
    header.append('};')
    (output/'binding_cases.h').write_text('\n'.join(header)+'\n')
    (output/'cases.json').write_text(json.dumps(entries,indent=2)+'\n')
    return entries
