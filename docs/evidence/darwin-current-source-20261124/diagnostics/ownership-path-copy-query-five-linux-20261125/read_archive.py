"""I verify my unchanged complete query gate and one explicit pure copy expression."""
from pathlib import Path
import hashlib,json,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,records,data,js=read_archive(base,'8de5afcc2d1cd1f3428618fa22ad5d1d3da101ebcbfcb01b3057020fbe02c709')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert m['source_commit']=='67d99e5121521428740d385d10071b9679e66013' and m['working_overlay'] and m['accepted_component'] and m['full_five_routes'] and not m['whole_release']
q=js('comparison/cohort-result.json');s=js('comparison/result.json');admit=js('comparison/admission-result.json')
assert q['terminal'] and q['accepted_component'] and q['all_guards_match'] and not q['whole_release'] and q['working_overlay']
assert q['tests']==dict(tests=1,failures=0,errors=0,skips=0,passed=True)
assert q['original_methods']==1 and q['new_methods']==0 and q['original_query_cases']==3080 and len(q['commands'])==1360
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and s['ordinary_seconds']==q['ordinary_seconds']==10
for name in ['CMD_TIMEOUT','NANO_SHADOW_TIMEOUT_SECONDS','NANO_CC','NANO_CFLAGS','NANO_LDFLAGS','NANOLANG_TRANSLATOR_SOURCE_COMPILER','NANOLANG_TRANSLATOR_SOURCE_MODULE']:assert s['environment_before'][name] is None
assert admit['terminal'] and admit['accepted'] and not admit['other_jobs'] and admit['free_bytes']>15*1024**3 and admit['filesystem']!='nfs'
assert all(r['share']>=.75 for r in s['cpu_rounds'])
assert s['readiness_publication']==m['source_commit'] and s['readiness_manifest_sha256']=='059daeb734e96daac1eaec34d926b4449a0b74a36f3df57d233c03fcd43d2ac9'
assert q['producer_source_pin']==s['bootstrap_source_pin']=='7b9f08ee0941c23810d18fa3f8ebfbb759c968de'
assert q['producer_sha256']==q['producer_after']==s['producer_sha256']=='5a537570a4f6708212fd22c7ccedd8be69207e53a11db4e5bc579a443126b1c3'
root=Path(m['roots']['comparison']);source=root/'source'
def mapped(name):
 p=Path(name)
 return 'comparison/'+str(p.relative_to(root)) if p.is_relative_to(root) else 'external/'+str(p).lstrip('/')
for group in ['source','images','tools','hosts']:
 assert q[group+'_before']==q[group+'_after']==s[group+'_before']==s[group+'_after']
 for n,h in q[group+'_before'].items():assert records['comparison/source/'+n if group in ['source','images'] else mapped(n)]['sha256']==h,n
assert q['object_inputs']==q['objects_after']
for n,h in q['object_inputs'].items():assert records[mapped(n)]['sha256']==h
assert records[mapped(s['producer'])]['sha256']==s['producer_sha256']
assert q['observer_before']==q['observer_after']
for n,h in q['observer_before'].items():assert records[mapped(n)]['sha256']==h
origin=js('origin/nanoc_bootstrap.json');assert origin['phase']=='stage2' and origin['raw_stage1_stage2_equal'] and origin['artifacts']['nanoc_stage2']==q['producer_sha256']
for c in q['commands']:
 assert c['exit']==0 and not c.get('interrupted_or_timeout')
 assert c['timeout']==(60 if c['number']<=2 else 600)
 assert c['build']==(389<=c['number']<=395)
 assert records['comparison/command-%04d.stdout'%c['number']] and records['comparison/command-%04d.stderr'%c['number']]
assert sum(c['build'] for c in q['commands'])==7
san=q['sanitized_reference_argv'];assert '-fsanitize=address,undefined' in san and '-fno-sanitize-recover=all' in san
assert {Path(v).name for v in san if v.endswith('.c')}=={'nvm2c_ownership_queries_reference.c','ownership_contracts.c','retained_layouts.c','reference_places.c','nvm_v2_layouts.c','nvm_v2_cursor.c','isa.c'}
assert any(v.endswith('/nanoisa/affine_bytecode.o') for v in q['reference_argv'])
# I reconstruct the complete terminal state from the append-only journal.
state=None;history=[];committed=False
for i,line in enumerate(data('comparison/cohort-journal.jsonl').splitlines(),1):
 e=json.loads(line);assert e['sequence']==i and not committed
 if e['kind']=='metadata':state=e['value'];assert state['commands']==[]
 elif e['kind']=='command':
  number=e['number'];assert number==e['value']['number'] and number in (len(history),len(history)+1) and number>0
  if number==len(history)+1:history.append(e['value'])
  else:history[-1]=e['value']
 elif e['kind']=='commit':
  assert state['terminal'] and e['commands']==len(history);state['commands']=history
  raw=(json.dumps(state,indent=2)+'\n').encode();assert len(raw)==e['bytes'] and sha(raw)==e['sha256'] and raw==data('comparison/cohort-result.json');committed=True
 else:raise AssertionError('I require a known journal event')
assert committed and state==q
# I check every independent criterion, raw carrier, report and command output.
paths=[n for n in records if n.startswith('comparison/tmp/nano-ownership-queries-') and n.endswith('/manifest.json')];assert len(paths)==1
manifest=js(paths[0]);assert len(manifest['corpus'])==3080 and len(manifest['routes'])==7
expected_labels=['actual-c','actual-c-sanitized','c-seed-native','c-producer-nanoisa-vm','c-producer-sanitized-native','nanolang-producer-nanoisa-vm','nanolang-producer-sanitized-native']
assert [r['label'] for r in manifest['routes']]==expected_labels
report_dir=paths[0].rsplit('/',1)[0];baseline=data(report_dir+'/actual-c.reports')
assert len(baseline.splitlines())==3080 and sha(baseline)=='b5973145e3390a20d652a55102bc12b925892c221507c78b261566a7ee9071a0'
for case,line in zip(manifest['corpus'],baseline.decode().splitlines()):
 assert records[mapped(case['path'])]['sha256']==case['sha256'] and records[mapped(case['path'])]['bytes']==case['bytes']
 if case['expected'] is not None:assert line==case['expected'],case['name']
 if case['expected_status'] is not None:assert line.startswith(str(case['expected_status'])+','),case['name']
for route in manifest['routes']:
 raw=data(report_dir+'/'+route['label']+'.reports');assert raw==baseline and route['reports']==3080 and route['sha256']==sha(raw)
 prefix=route['command'];commands=[c for c in q['commands'] if c['argv'][:len(prefix)]==prefix]
 assert len(commands)==193
 assert [n for c in commands for n in c['argv'][len(prefix):]]==[r['path'] for r in manifest['corpus']]
 assert all(len(c['argv'])<=len(prefix)+16 for c in commands)
 assert b''.join(data('comparison/command-%04d.stdout'%c['number']) for c in commands)==raw
assert len(q['providers'])==2
for name,r in q['providers'].items():
 assert r['read_only'] and r['before_translation'] and r['module_sha256']==r['module_after']==records[mapped(name)]['sha256']
 assert r['libraries_before']==r['libraries_after']=={}
# I compare every product/test source byte against Git, allowing only the published copy expression.
p=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE);changed=[]
for n,h in q['source_before'].items():
 p.stdin.write((m['source_commit']+':'+n+'\n').encode());p.stdin.flush();parts=p.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob',n
 old=p.stdout.read(int(parts[2]));assert p.stdout.read(1)==b'\n';actual=data('comparison/source/'+n)
 if old!=actual:
  changed.append(n);needle=b'    return Query { status: 0, number: out.number, fields: (array_slice selected 0 out.number), globals: out.globals,'
  replacement=b'    let mut copied_fields: array<int> = []\n    for field_index in (range 0 out.number) { set copied_fields (array_push copied_fields (at selected field_index)) }\n    return Query { status: 0, number: out.number, fields: copied_fields, globals: out.globals,'
  assert old.count(needle)==1 and old.replace(needle,replacement)==actual,n
 assert sha(actual)==h
p.stdin.close();assert p.wait()==0 and changed==['src_nano/nvm2c/ownership_queries.nano']
failed=js('failed-launch/cohort-result.json');assert failed['terminal'] and failed['all_guards_match'] and not failed.get('accepted_component') and len(failed['commands'])==1 and failed['commands'][0]['exit']==1 and not failed.get('tests')
assert 'undefined reference' in data('failed-launch/command-0001.stderr').decode() and 'nvm_affine_analyze_function' in data('failed-launch/command-0001.stderr').decode()
assert data('failed-launch/candidate.diff')==data('comparison/candidate.diff')
print('PASS %d records; 1360 commands, 3080 unchanged cases x seven products = 21560 complete identical reports, original budgets and full journal/source/Git/image/tool/object/host/provider/observer guards. Prior launcher failure retained. Linux query gate only; no global/array/paired-platform/compiler repair/emission/cutover/release acceptance.'%len(records))
