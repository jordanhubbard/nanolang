"""I verify failed later-fixture readiness and contextual array-write diagnostics."""
from pathlib import Path
import hashlib,json,re,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,records,data,js=read_archive(base,'4144dc21c00b43c9d9da3685170120b6b3df99aa453a1bcdda81ead1f05e9f0d')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert m['accepted_failure_archive'] and not m['accepted_bounded_readiness']
assert not any(m[n] for n in ['accepted_component','whole_release','full_five_routes'])
assert m['source_commit']=='901caef8e6f995080ac7d3c48c4cf23c75c99315'
q=js('readiness/result.json');d=js('localization/result.json');extra=js('localization/additional-traces.json');small=js('put-control/result.json')
assert q['terminal'] and q['all_guards_match'] and not q['accepted_bounded_readiness'] and q['readiness_only']
assert len(q['commands'])==27 and q['ordinary_seconds']==10 and not q['accepted_component'] and not q['whole_release']
assert q['cpu_round']['share']>=.75 and q['free_bytes']>15*1024**3 and q['filesystem']!='nfs'
assert all(q['environment_before'][n] is None for n in ['NANO_CC','NANO_CFLAGS','NANO_LDFLAGS','CMD_TIMEOUT','NANO_SHADOW_TIMEOUT_SECONDS','NANOLANG_TRANSLATOR_SOURCE_COMPILER','NANOLANG_TRANSLATOR_SOURCE_MODULE'])
for group in ['source','tools','hosts']:
 assert q[group+'_before']==q[group+'_after']
 for n,h in q[group+'_before'].items():assert records['readiness/source/'+n if group=='source' else 'external/'+n.lstrip('/')]['sha256']==h
for n,h in q['modules'].items():assert records['readiness/'+n]['sha256']==h
assert len(q['modules'])==6
failed={'analysis-c-translation','analysis-genuine-translation','public-c-translation','public-genuine-translation'}
assert {c['label'] for c in q['commands'] if c['exit']}==failed
baseline=data('probe/original-actual-c.reports');assert sha(baseline)=='f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0'
expected=baseline.splitlines()[336]+b'\n';assert expected.decode()==q['expected_probe']
assert q['probe']['name']=='empty-identical' and q['probe_after']==q['probe']['sha256']==records['external/'+q['probe']['path'].lstrip('/')]['sha256']
assert len(data('external/'+q['probe']['path'].lstrip('/')))==169
probes=[]
for c in q['commands']:
 label=c['label'];assert c['timeout']==(60 if label.endswith(('-typed','-probe')) else 600) and not c.get('timeout_expired')
 assert c['exit']==(1 if label in failed else 0)
 if label in failed:assert b'I require the matching scalar shape at this array write' in data('readiness/'+label+'.stderr')
 if label.endswith('-probe'):
  assert c['probe_matches'] and data('readiness/'+label+'.stdout')==expected;probes.append(label)
assert len(probes)==9
assert d['terminal'] and d['read_only_debugger'] and d['all_guards_match'] and not d['product_acceptance']
assert extra['terminal'] and extra['read_only_debugger'] and not extra['product_acceptance'] and len(extra['traces'])==3
for state in [d,*extra['traces']]:
 assert state['exit']==0 and state['before']==state['after']
 for n,h in state['before'].items():assert records['external/'+n.lstrip('/')]['sha256']==h
for label,idx,dumpname,tracename,decoded in [('analysis-c',4,'module.stdout','trace.stdout','function-decoded.json'),('analysis-genuine',152,'analysis-genuine-module.stdout','analysis-genuine.stdout','analysis-genuine-decoded.json'),('public-c',23,'public-c-module.stdout','public-c.stdout','public-c-decoded.json'),('public-genuine',198,'public-genuine-module.stdout','public-genuine.stdout','public-genuine-decoded.json')]:
 functions=re.findall(r'^\.function (\S+)',data('localization/'+dumpname).decode(),re.M);assert functions[idx]=='put' or functions[idx].endswith('_put')
 trace=data('localization/'+tracename).decode()
 for text in ['idx = '+str(idx),'start = 42','val.kind = 3','arr.kind = 2','arr.origin = 0','facts->final = 0','b->canonical_records = false','b->track_shapes = 0','fn->code_length = 45']:assert text in trace,(label,text)
 rows=js('localization/'+decoded);at={r['offset']:r for r in rows}
 assert [(at[n]['name'],at[n]['operands']) for n in [33,36,39,42,43,44]]==[('LOAD_LOCAL',[0]),('LOAD_LOCAL',[1]),('LOAD_LOCAL',[2]),('ARR_SET',[]),('POP',[]),('RET',[])]
assert small['terminal'] and small['all_guards_match'] and small['all_nontranslation_controls_pass'] and not small['refusing_translations'] and len(small['commands'])==8
assert small['ordinary_seconds']==10 and small['before']==small['after']
for n,h in small['before'].items():assert records['external/'+n.lstrip('/')]['sha256']==h
for c in small['commands']:
 assert c['exit']==0 and c['timeout']==(60 if c['label'].endswith('-probe') else 600)
 if c['label'].endswith('-probe'):assert c['probe_matches'] and data('put-control/'+c['label']+'.stdout')==b'7\n'
source=data('readiness/source/src_nano/nvm2c/affine_analysis.nano');a=source.index(b'fn put(');b=source.index(b'\nfn transferred(',a)
assert source[a:b] in data('put-control/put.nano')
# I bind all selected source inputs to the declared pin with one exact query copy.
p=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE);changed=[]
for n,h in q['source_before'].items():
 p.stdin.write((m['source_commit']+':'+n+'\n').encode());p.stdin.flush();parts=p.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob',n
 old=p.stdout.read(int(parts[2]));assert p.stdout.read(1)==b'\n';actual=data('readiness/source/'+n)
 if old!=actual:
  changed.append(n);assert n=='src_nano/nvm2c/ownership_queries.nano'
  needle=b'    return Query { status: 0, number: out.number, fields: (array_slice selected 0 out.number), globals: out.globals,'
  replacement=b'    let mut copied_fields: array<int> = []\n    for field_index in (range 0 out.number) { set copied_fields (array_push copied_fields (at selected field_index)) }\n    return Query { status: 0, number: out.number, fields: copied_fields, globals: out.globals,'
  assert old.count(needle)==1 and old.replace(needle,replacement)==actual
 assert sha(actual)==h
p.stdin.close();assert p.wait()==0 and changed==['src_nano/nvm2c/ownership_queries.nano']
print(json.dumps(dict(terminal=True,accepted_failure_archive=True,readiness_commands=27,translation_refusals=4,single_carrier_probes=9,standalone_positive_commands=8,accepted_component=False,whole_release=False)))
