"""I describe the same explicit requests for independent C and Nano builders.

I do not parse NSI documents here: validated catalog/completeness are input
preconditions. This corpus tests plans, not source publication or File effects.
"""
from copy import deepcopy

NAMES = ['File', 'FileError', 'ReadByte', 'OpenResult', 'WriteResult',
         'PositionResult', 'ReadResult', 'CloseResult', 'temp', 'write_byte',
         'rewind', 'read_byte', 'close']
NO_INDEX = 4294967295

def span(data, size=None):
    return {'data': data, 'size': len(data.encode()) if size is None else size}

def request(module='binding', base=1):
    return dict(module=span(module), interface=span('nsi:nanolang/filesystem'),
                catalog=None, version=1, line=7, column=9,
                bindings=[dict(id=base+i, kind=int(i>=8), ordinal=i-8 if i>=8 else i,
                               name=span(name)) for i,name in enumerate(NAMES)])

def corpus():
    cases=[]
    def add(name, status=0, mutate=None):
        c=dict(name=name, status=status, requests=[request()], aliases=[], ordinary=[])
        if mutate: mutate(c)
        cases.append(c)
        return c
    add('canonical')
    add('reordered_bindings', mutate=lambda c:c['requests'][0]['bindings'].reverse())
    add('two_modules', mutate=lambda c:c['requests'].append(request('other',101)))
    add('reordered_requests', mutate=lambda c:c.update(requests=[request('other',101),request()]))
    add('partial_utf8_prefix', mutate=lambda c:c['requests'][0].update(module=span('éx',1)))
    add('non_ascii_module', mutate=lambda c:c['requests'][0].update(module=span('mód/λ')))
    add('prefix_module', mutate=lambda c:c['requests'][0].update(module=span('binding_suffix',7)))
    add('prefix_interface', mutate=lambda c:c['requests'][0].update(interface=span('nsi:nanolang/filesystem_suffix',len('nsi:nanolang/filesystem'))))
    add('prefix_name', mutate=lambda c:c['requests'][0]['bindings'][0].update(name=span('File_suffix',4)))
    add('prefix_nul', mutate=lambda c:c['requests'][0].update(module=span('a\0b',1)))
    add('counted_nul',1,lambda c:c['requests'][0].update(module=span('a\0b',3)))
    add('empty_module',1,lambda c:c['requests'][0].update(module=span('')))
    add('too_long_module',1,lambda c:c['requests'][0].update(module=span('m'*4097)))
    add('module_boundary',mutate=lambda c:c['requests'][0].update(module=span('m'*4096)))
    add('zero_extent',1,lambda c:c['requests'][0].update(module=span('binding',0)))
    add('duplicate_module',1,lambda c:c['requests'].append(request('binding',101)))
    add('unknown_interface',4,lambda c:c['requests'][0].update(interface=span('nsi:other/filesystem')))
    add('bad_version',4,lambda c:c['requests'][0].update(version=2))
    add('bad_catalog',4,lambda c:c['requests'][0].update(catalog=span('file-source-catalog1;changed')))
    for field in ('line','column'):
        for value in (0,2147483648):
            add(f'{field}_{value}',1,lambda c,f=field,v=value:c['requests'][0].update({f:v}))
    for field,values in (('id',(0,2147483648)),('kind',(2,4294967295)),('ordinal',(8,4294967295))):
        for value in values:
            add(f'binding_{field}_{value}',1,lambda c,f=field,v=value:c['requests'][0]['bindings'][0].update({f:v}))
    add('method_ordinal5',1,lambda c:c['requests'][0]['bindings'][8].update(ordinal=5))
    add('duplicate_ordinal',1,lambda c:c['requests'][0]['bindings'].__setitem__(1,deepcopy(c['requests'][0]['bindings'][0])))
    add('duplicate_id',1,lambda c:c['requests'][0]['bindings'][1].update(id=1))
    for name in ('file','1File','Fi\0le','Fíle','x'*129):
        add('bad_name_'+str(len(cases)),1,lambda c,n=name:c['requests'][0]['bindings'][0].update(name=span(n)))
    add('binding_count12',1,lambda c:c['requests'][0]['bindings'].pop())
    add('no_requests',1,lambda c:c.update(requests=[]))
    add('request_boundary',mutate=lambda c:c.update(requests=[request('m'+str(i),1+13*i) for i in range(16)]))
    add('id_boundary',mutate=lambda c:c['requests'][0]['bindings'][0].update(id=2147483647))
    add('request_limit',2,lambda c:c.update(requests=[request('m'+str(i),1+13*i) for i in range(17)]))
    def alias(target=1,name='Handle',id=500,module='user'):
        return dict(module=span(module),name=span(name),id=id,target=target)
    add('alias_file',mutate=lambda c:c['aliases'].append(alias()))
    add('alias_close',mutate=lambda c:c['aliases'].append(alias(13,'finish')))
    add('alias_unknown',4,lambda c:c['aliases'].append(alias(999)))
    add('alias_chain',4,lambda c:c['aliases'].extend([alias(),alias(500,'Again',501)]))
    add('alias_duplicate_id',1,lambda c:c['aliases'].append(alias(id=1)))
    add('alias_namespace_collision',1,lambda c:c['aliases'].append(alias(name='File',module='binding')))
    add('alias_duplicate_name',1,lambda c:c['aliases'].extend([alias(),alias(2,id=501)]))
    add('alias_different_module',mutate=lambda c:c['aliases'].extend([alias(),alias(2,id=501,module='other')]))
    add('alias_boundary',mutate=lambda c:c.update(aliases=[alias(id=500+i,name='n'+str(i)) for i in range(64)]))
    add('alias_limit',2,lambda c:c.update(aliases=[alias(id=500+i,name='n'+str(i)) for i in range(65)]))
    def ordinary(id=700,name='value',module='binding'):
        return dict(module=span(module),name=span(name),id=id)
    add('ordinary',mutate=lambda c:c['ordinary'].append(ordinary()))
    add('ordinary_id_collision',1,lambda c:c['ordinary'].append(ordinary(id=1)))
    add('ordinary_name_collision',1,lambda c:c['ordinary'].append(ordinary(name='File')))
    add('ordinary_other_module',mutate=lambda c:c['ordinary'].append(ordinary(name='File',module='other')))
    add('ordinary_boundary',mutate=lambda c:c.update(ordinary=[ordinary(700+i,'n'+str(i)) for i in range(256)]))
    add('ordinary_limit',2,lambda c:c.update(ordinary=[ordinary(700+i,'n'+str(i)) for i in range(257)]))
    return cases

def c_string(s):
    return '"'+''.join('\\%03o'%v for v in s.encode())+'"'

def nano_string(s):
    return '"'+s.replace('\\','\\\\').replace('"','\\"').replace('\0','\\0').replace('\n','\\n').replace('\r','\\r').replace('\t','\\t')+'"'

def c_span(s):
    return '{'+c_string(s['data'])+','+str(s['size'])+'}'

def nano_span(s):
    return 'FileSourceText { data: '+nano_string(s['data'])+', size: '+str(s['size'])+' }'

def expected_rows(c):
    rows=[]
    if c['status']: return rows
    for r,q in enumerate(c['requests']):
        for b in q['bindings']:
            k,o=b['kind'],b['ordinal']
            category=5 if k else 1 if o==0 else 2 if o==3 else 3 if o<3 else 4
            rows.append([q['module'],b['name'],b['id'],b['id'],r,k,o,category,
                         [0,1,1,1,2][o] if k else NO_INDEX,
                         3+o if k else NO_INDEX,NO_INDEX,NO_INDEX,q['line'],q['column']])
    bases={r[2]:r for r in rows}
    for a in c['aliases']:
        r=deepcopy(bases[a['target']]);r[:4]=[a['module'],a['name'],a['id'],a['target']];rows.append(r)
    return rows

def expected(cases):
    out=bytearray()
    for index,c in enumerate(cases):
        rows=expected_rows(c)
        out.extend(f"CASE {index} {c['status']} {len(rows)}\n".encode())
        for r in rows:
            for s in r[:2]:
                data=s['data'].encode()[:s['size']]
                out.extend(str(len(data)).encode()+b':'+data+b'|')
            out.extend(('|'.join(map(str,r[2:]))+'\n').encode())
    return bytes(out)

def generate_c(cases):
    out=['#include <stdio.h>\n#include <stdlib.h>\n#include <stdint.h>\n#include "src/nanoisa/file_source_plan.h"\n']
    out.append('static char catalog[32768];static size_t catalog_size;\n')
    out.append('static void emit(unsigned n,NlFileSourceStatus s,NlFileSourcePlan *p){printf("CASE %u %u %zu\\n",n,(unsigned)s,nl_file_source_plan_count(p));for(size_t i=0;i<nl_file_source_plan_count(p);i++){NlFileSourceRow r;if(!nl_file_source_plan_row(p,i,&r))abort();printf("%zu:%.*s|%zu:%.*s|%u|%u|%u|%u|%u|%u|%u|%u|%u|%u|%u|%u\\n",r.module.size,(int)r.module.size,r.module.data,r.name.size,(int)r.name.size,r.name.data,r.id,r.target,r.request,r.kind,r.ordinal,r.category,r.input_mode,r.result_ordinal,r.global_layout,r.import_index,r.line,r.column);}nl_file_source_plan_free(p);}\n')
    for n,c in enumerate(cases):
        out.append(f'static void case_{n}(void){{\n')
        for j,q in enumerate(c['requests']):
            bs=','.join('{%d,%d,%d,%s}'%(b['id'],b['kind'],b['ordinal'],c_span(b['name'])) for b in q['bindings'])
            out.append(f'NlFileSourceBinding b{j}[]={{'+bs+'};\n')
        if c['requests']:
            qs=[]
            for j,q in enumerate(c['requests']):
                cat='{catalog,catalog_size-1}' if q['catalog'] is None else c_span(q['catalog'])
                qs.append('{%s,%s,%s,%d,%d,%d,b%d,%d}'%(c_span(q['module']),c_span(q['interface']),cat,q['version'],q['line'],q['column'],j,len(q['bindings'])))
            out.append('NlFileSourceRequest q[]={'+','.join(qs)+'};\n')
        for key,typ in [('aliases','NlFileSourceAlias'),('ordinary','NlFileSourceOrdinary')]:
            if c[key]:
                values=['{'+c_span(a['module'])+','+c_span(a['name'])+','+str(a['id'])+(','+str(a['target']) if key=='aliases' else '')+'}' for a in c[key]]
                out.append(typ+' '+key+'[]={'+','.join(values)+'};\n')
        args=[('q' if c['requests'] else 'NULL'),str(len(c['requests']))]
        for key in ('aliases','ordinary'):args.extend([key if c[key] else 'NULL',str(len(c[key]))])
        out.append('NlFileSourcePlan *p=(NlFileSourcePlan *)(uintptr_t)1;NlFileSourceStatus s=nl_file_source_plan_build('+','.join(args)+',&p);\n')
        out.append(f'if(s!={c["status"]}) {{ abort(); }}\n')
        out.append('if(s) {\n if(p!=(NlFileSourcePlan *)(uintptr_t)1) { abort(); }\n p=NULL;\n}\n')
        out.append(f'emit({n},s,p);\n}}\n')
    out.append('int main(void){if(!nl_file_source_catalog_view(catalog,sizeof catalog,&catalog_size))abort();printf("CAT:%s\\n",catalog);')
    out.extend(f'case_{i}();' for i in range(len(cases)));out.append('return 0;}\n')
    return ''.join(out)

def generate_nano(cases):
    out=['module "src_nano/compiler/file_source_plan.nano" as Plan\n']
    # Imported type names are the actual declarations, not copied definitions.
    out.append('fn emit(index: int, plan: FileSourcePlan) -> void {\n (println (+ "CASE " (+ (int_to_string index) (+ " " (+ (int_to_string plan.status) (+ " " (int_to_string (array_length plan.rows))))))))\n let mut i: int = 0\n while (< i (array_length plan.rows)) {\n let row: FileSourceRow = (at plan.rows i)\n let mut line: string = (+ (int_to_string (str_length row.module_id)) (+ ":" (+ row.module_id (+ "|" (+ (int_to_string (str_length row.name)) (+ ":" (+ row.name "|")))))))\n')
    fields=['id','target','request','kind','ordinal','category','input_mode','result_ordinal','global_layout','import_index','line','column']
    for j,f in enumerate(fields):
        out.append(' set line (+ line '+('(int_to_string row.'+f+')' if j==0 else '(+ "|" (int_to_string row.'+f+'))')+')\n')
    out.append(' (println line) set i (+ i 1)\n }\n}\nshadow emit { let rows: array<FileSourceRow> = [] (emit -1 FileSourcePlan { status: 1, rows: rows, logical_bytes: 0 }) assert (== (array_length rows) 0) }\n')
    for n,c in enumerate(cases):
        out.append(f'fn case_{n}() -> FileSourcePlan {{\n let catalog: string = (Plan.file_source_catalog_view)\n')
        for j,q in enumerate(c['requests']):
            out.append(f' let mut b{j}: array<FileSourceBinding> = []\n')
            for b in q['bindings']:
                out.append(f' set b{j} (array_push b{j} FileSourceBinding {{ id: {b["id"]}, kind: {b["kind"]}, ordinal: {b["ordinal"]}, name: {nano_span(b["name"])} }})\n')
        out.append(' let mut requests: array<FileSourceRequest> = []\n')
        for j,q in enumerate(c['requests']):
            cat='FileSourceText { data: catalog, size: (str_length catalog) }' if q['catalog'] is None else nano_span(q['catalog'])
            out.append(' set requests (array_push requests FileSourceRequest { module_id: '+nano_span(q['module'])+', interface_id: '+nano_span(q['interface'])+', catalog_view: '+cat+f', catalog_version: {q["version"]}, line: {q["line"]}, column: {q["column"]}, bindings: b{j}'+ ' })\n')
        for key,typ in [('aliases','FileSourceAlias'),('ordinary','FileSourceOrdinary')]:
            out.append(f' let mut {key}: array<{typ}> = []\n')
            for a in c[key]:
                out.append(f' set {key} (array_push {key} {typ} {{ module_id: '+nano_span(a['module'])+', name: '+nano_span(a['name'])+f', id: {a["id"]}'+(f', target: {a["target"]}' if key=='aliases' else '')+' })\n')
        out.append(' return (Plan.file_source_plan requests aliases ordinary)\n}\n')
        out.append(f'shadow case_{n} {{ assert (== (case_{n}).status {c["status"]}) }}\n')
    out.append(NANO_BOUNDARIES)
    out.append('fn main() -> int { (println (+ "CAT:" (Plan.file_source_catalog_view)))\n')
    out.extend(f' (emit {i} (case_{i}))\n' for i in range(len(cases)));out.append(' return 0\n}\n')
    return ''.join(out)

NANO_BOUNDARIES = r'''
fn nano_extent_case(size: int) -> int {
    let names: array<string> = ["File", "FileError", "ReadByte", "OpenResult", "WriteResult", "PositionResult", "ReadResult", "CloseResult", "temp", "write_byte", "rewind", "read_byte", "close"]
    let mut bindings: array<FileSourceBinding> = []
    let mut i: int = 0
    while (< i 13) {
        let name: string = (at names i)
        set bindings (array_push bindings FileSourceBinding { id: (+ i 1), kind: (cond ((>= i 8) 1) (else 0)), ordinal: (cond ((>= i 8) (- i 8)) (else i)), name: FileSourceText { data: name, size: (str_length name) } })
        set i (+ i 1)
    }
    let catalog: string = (Plan.file_source_catalog_view)
    let aliases: array<FileSourceAlias> = []
    let ordinary: array<FileSourceOrdinary> = []
    let q: FileSourceRequest = FileSourceRequest {
        module_id: FileSourceText { data: "a", size: size },
        interface_id: FileSourceText { data: "nsi:nanolang/filesystem", size: 23 },
        catalog_view: FileSourceText { data: catalog, size: (str_length catalog) },
        catalog_version: 1, line: 1, column: 1, bindings: bindings
    }
    let requests: array<FileSourceRequest> = [q]
    let plan: FileSourcePlan = (Plan.file_source_plan requests aliases ordinary)
    return plan.status
}
shadow nano_extent_case {
    assert (== (nano_extent_case 1) 0)
    assert (== (nano_extent_case -1) 1)
    assert (== (nano_extent_case -9223372036854775807) 1)
    assert (== (nano_extent_case 2) 1)
}
fn nano_budget_case(extra: int) -> int {
    let catalog: string = (Plan.file_source_catalog_view)
    let mut bindings: array<FileSourceBinding> = []
    let names: array<string> = ["File", "FileError", "ReadByte", "OpenResult", "WriteResult", "PositionResult", "ReadResult", "CloseResult", "temp", "write_byte", "rewind", "read_byte", "close"]
    let mut used: int = (+ 7 (+ 24 (+ (str_length catalog) 1)))
    let mut i: int = 0
    while (< i 13) {
        let name: string = (at names i)
        let kind: int = (cond ((>= i 8) 1) (else 0))
        let ordinal: int = (cond ((>= i 8) (- i 8)) (else i))
        set bindings (array_push bindings FileSourceBinding { id: (+ i 1), kind: kind, ordinal: ordinal, name: FileSourceText { data: name, size: (str_length name) } })
        set used (+ used (+ 7 (* 2 (+ (str_length name) 1))))
        set i (+ i 1)
    }
    let q: FileSourceRequest = FileSourceRequest { module_id: FileSourceText { data: "module", size: 6 }, interface_id: FileSourceText { data: "nsi:nanolang/filesystem", size: 23 }, catalog_view: FileSourceText { data: catalog, size: (str_length catalog) }, catalog_version: 1, line: 1, column: 2, bindings: bindings }
    let requests: array<FileSourceRequest> = [q]
    let aliases: array<FileSourceAlias> = []
    let mut ordinary: array<FileSourceOrdinary> = []
    let mut large: string = "m"
    set i 0
    while (< i 12) { set large (+ large large) set i (+ i 1) }
    set i 0
    while (< used 1048576) {
        let name: string = (+ "n" (int_to_string i))
        let remain: int = (- 1048576 used)
        let mut width: int = (- remain (+ (str_length name) 2))
        if (> width 4096) { set width 4096 }
        assert (> width 0)
        set used (+ used (+ width (+ (str_length name) 2)))
        if (== used 1048576) { set width (+ width extra) }
        set ordinary (array_push ordinary FileSourceOrdinary { module_id: FileSourceText { data: large, size: width }, name: FileSourceText { data: name, size: (str_length name) }, id: (+ 1000 i) })
        set i (+ i 1)
    }
    assert (<= (array_length ordinary) 256)
    let plan: FileSourcePlan = (Plan.file_source_plan requests aliases ordinary)
    return plan.status
}
shadow nano_budget_case {
    assert (== (nano_budget_case 0) 0)
    assert (== (nano_budget_case 1) 2)
}
'''
