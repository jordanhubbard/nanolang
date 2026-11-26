"""I verify the complete isolated fresh-function preflight gate, not product cutover."""
from pathlib import Path
import hashlib,json,subprocess,sys,shlex
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,records,data,js=read_archive(base,'MANIFEST_SHA')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
q=js('comparison/cohort-result.json');s=js('comparison/result.json')
assert m['methods']==1 and m['report_fix_commit']==q['report_fix_commit']=='c886e6dc73c2680d17a6db76b18e4ec00050e592'
assert m['accepted_component'] and m['full_five_routes'] and m['working_overlay'] and not m['whole_release']
assert m['source_commit']==q['source_pin']==s['source_base']=='67d99e5121521428740d385d10071b9679e66013'
assert q['terminal'] and q['accepted_component'] and q['all_guards_match'] and not q['whole_release']
assert q['tests']==dict(tests=1,failures=0,errors=0,skips=0,passed=True)
assert q['original_frame_cases']==1001 and q['new_cases']==dict(preflight=228)
root=Path(m['roots']['comparison']);source=root/'source'
def mapped(name):
 p=Path(name)
 return 'comparison/'+str(p.relative_to(root)) if p.is_relative_to(root) else 'external/'+str(p).lstrip('/')
for group in ['source','images','tools','hosts']:
 assert q[group+'_before']==q[group+'_after']==s[group+'_before']==s[group+'_after']
 for n,h in q[group+'_before'].items():assert records['comparison/source/'+n if group in ['source','images'] else mapped(n)]['sha256']==h,n
assert q['object_inputs']==q['objects_after']
for n,h in q['object_inputs'].items():assert records[mapped(n)]['sha256']==h
assert q['observer_before']==q['observer_after']
for n,h in q['observer_before'].items():assert records[mapped(n)]['sha256']==h
assert q['producer_after']==q['producer_sha256']==s['producer_sha256']=='5a537570a4f6708212fd22c7ccedd8be69207e53a11db4e5bc579a443126b1c3'
assert q['producer_source_pin']==s['bootstrap_source_pin']=='7b9f08ee0941c23810d18fa3f8ebfbb759c968de'
assert records[mapped(s['producer'])]['sha256']==q['producer_sha256']
origin=js('origin/nanoc_bootstrap.json');assert origin['phase']=='stage2' and origin['raw_stage1_stage2_equal'] and origin['artifacts']['nanoc_stage2']==q['producer_sha256']
fix_names={'tests/nvm2c_affine_'+leaf+'_driver.nano' for leaf in ['aggregates','effects','value_calls','analysis_cache']}
parent=js('query-dependency/cohort-result.json');assert parent['terminal'] and parent['accepted_component'] and parent['all_guards_match'] and {n:h for n,h in parent['source_before'].items() if n not in fix_names}=={n:h for n,h in q['source_before'].items() if n not in fix_names} and q['overlay_sha256']==s['overlays']
assert parent['overlay_sha256']=={n:h for n,h in q['overlay_sha256'].items() if n not in fix_names}
assert q['report_fix_manifest_sha256']==s['report_fix_manifest_sha256']=='1cce1d7d6614bef2cfb49980e9bbc474cf22d68c707acf4f7be23808250cb94f'
assert q['query_evidence']==s['query_evidence'] and s['query_evidence']['result_sha256']==sha(data('query-dependency/cohort-result.json'))
assert s['query_evidence']['commit']=='dec468a6bda61adf5d268c365832811c448d69f0' and s['query_evidence']['manifest_sha256']=='8de5afcc2d1cd1f3428618fa22ad5d1d3da101ebcbfcb01b3057020fbe02c709'
admit=js('comparison/admission-result.json');assert admit['terminal'] and admit['accepted'] and not admit['other_jobs'] and admit['free_bytes']>15*1024**3 and admit['filesystem']!='nfs'
assert all(r['share']>=.75 for r in s['cpu_rounds']) and s['ordinary_seconds']==q['ordinary_seconds']==10
for name in ['CMD_TIMEOUT','NANO_SHADOW_TIMEOUT_SECONDS','NANO_CC','NANO_CFLAGS','NANO_LDFLAGS','NANOLANG_TRANSLATOR_SOURCE_COMPILER','NANOLANG_TRANSLATOR_SOURCE_MODULE']:assert s['environment_before'][name] is None
assert all(c['exit']==0 and not c.get('interrupted_or_timeout') for c in q['commands'])
assert sum(c['build'] for c in q['commands'])==7 and len(q['providers'])==2
for label,key in [('global-array','global_array'),('foundation','foundation')]:
 parent=js(label+'-dependency/cohort-result.json');assert parent['terminal'] and parent['accepted_component'] and parent['all_guards_match']
 assert sha(data(label+'-dependency/cohort-result.json'))==s[key+'_evidence']['result_sha256']
 assert q[key+'_evidence']==s[key+'_evidence'] and parent['producer_sha256']==q['producer_sha256']
 if key=='global_array':assert {n:h for n,h in parent['source_before'].items() if n not in fix_names}=={n:h for n,h in q['source_before'].items() if n not in fix_names} and parent['overlay_sha256']=={n:h for n,h in q['overlay_sha256'].items() if n not in fix_names}
assert m['cache_fix_commit']==q['cache_fix_commit']==s['cache_fix_commit']=='901caef8e6f995080ac7d3c48c4cf23c75c99315'
assert q['cache_key_seed_manifest_sha256']==s['cache_key_seed_manifest_sha256']=='3fd80a1344ff25b6ccfa6cfe689e8bafce613b773cb879c42b4583bcf998f73e'
triple=js('corrected-three-dependency/cohort-result.json');assert triple['terminal'] and triple['accepted_component'] and triple['all_guards_match'] and triple['tests']==dict(tests=3,failures=0,errors=0,skips=0,passed=True)
assert q['corrected_three_evidence']==s['corrected_three_evidence'] and s['corrected_three_evidence']['result_sha256']==sha(data('corrected-three-dependency/cohort-result.json'))
assert s['corrected_three_evidence']['commit']=='4494fe46ae1cfa3db1c6c4abec4c1dd74b926cf5' and s['corrected_three_evidence']['manifest_sha256']=='3d54cfc4a833259baad781c211ee16f4d855a233a9da741298e4fd39efe4adb2'
cache_name='tests/nvm2c_affine_analysis_cache_driver.nano'
assert {n:h for n,h in q['source_before'].items() if n!=cache_name}=={n:h for n,h in triple['source_before'].items() if n!=cache_name}
controls=js('key-seed-controls/cohort-result.json');assert controls['terminal'] and controls['accepted_four_route_report_controls'] and controls['all_guards_match']
assert q['source_before'][cache_name]==controls['source_before'][cache_name]
cache_parent=js('cache-full-dependency/cohort-result.json');assert cache_parent['terminal'] and cache_parent['accepted_component'] and cache_parent['all_guards_match'] and cache_parent['tests']==dict(tests=1,failures=0,errors=0,skips=0,passed=True)
assert q['cache_full_evidence']==s['cache_full_evidence'] and s['cache_full_evidence']['result_sha256']==sha(data('cache-full-dependency/cohort-result.json'))
assert s['cache_full_evidence']['commit']=='25e4c2dc7cbfe86383da891e44520d3a1f4f6804' and s['cache_full_evidence']['manifest_sha256']=='0854e17db9032208aef0c6d259ac16b3b63670e4e4054452c23bfed12e677de2'
assert q['source_before']==cache_parent['source_before'] and q['overlay_sha256']==cache_parent['overlay_sha256']
for group in ['images','tools','hosts']:assert q[group+'_before']==cache_parent[group+'_before']
six=js('six-dependency/cohort-result.json');assert six['terminal'] and six['accepted_component'] and six['all_guards_match'] and six['tests']==dict(tests=6,failures=0,errors=0,skips=0,passed=True)
assert six['new_cases']==dict(owners=14542,aggregates=15222,effects=8128,value_calls=15644,call_context=834,analysis_cache=3317)
assert q['six_evidence']==s['six_evidence'] and sha(data('six-dependency/cohort-result.json'))==s['six_evidence']['result_sha256']
assert sha(data('six-proof/manifest.json'))==s['six_evidence']['manifest_sha256'] and s['six_evidence']['producer_sha256']==q['producer_sha256']==six['producer_sha256']
assert q['source_before']==six['source_before'] and q['overlay_sha256']==six['overlay_sha256']
for group in ['images','tools','hosts']:assert q[group+'_before']==six[group+'_before']==six[group+'_after']
publication=s['six_evidence']['commit'];assert len(publication)==40 and all(c in '0123456789abcdef' for c in publication)
for name in ['manifest.json','read_archive.py']:
 path='docs/evidence/darwin-current-source-20261124/diagnostics/ownership-path-copy-six-leaves-corrected-five-linux-20261126/'+name
 assert data('six-proof/'+name)==subprocess.check_output(['git','show',publication+':'+path],cwd=base)
# I replay every command/metadata event into the exact terminal JSON.
state=None;history=[];committed=False
for i,line in enumerate(data('comparison/cohort-journal.jsonl').splitlines(),1):
 e=json.loads(line);assert e['sequence']==i and not committed
 if e['kind']=='metadata':state=e['value'];assert state['commands']==[]
 elif e['kind']=='command':
  n=e['number'];assert n==e['value']['number'] and n in (len(history),len(history)+1) and n>0
  if n==len(history)+1:history.append(e['value'])
  else:history[-1]=e['value']
 elif e['kind']=='commit':
  assert state['terminal'] and e['commands']==len(history);state['commands']=history;raw=(json.dumps(state,indent=2)+'\n').encode()
  assert len(raw)==e['bytes'] and sha(raw)==e['sha256'] and raw==data('comparison/cohort-result.json');committed=True
 else:raise AssertionError('I require a known journal event')
assert committed and state==q
# I retain exact imported module/provider bindings before each translation.
for index,(name,r) in enumerate(q['providers'].items()):
 assert r['read_only'] and r['before_translation'] and r['module_sha256']==r['module_after']==records[mapped(name)]['sha256']
 assert r['libraries_before']==r['libraries_after']
 for n,h in r['libraries_before'].items():assert records[mapped(n)]['sha256']==h
 lines=data('comparison/provider-%d.stdout'%index).decode().splitlines()
 assert [shlex.split(line) for line in lines if line.startswith('.import ')]==r['bindings']
def inspect_case(group,new,case,line):
 expected=case['expected'];assert isinstance(expected,str)
 if group=='original' and not case['name'].startswith(('global-algebra-','bounded-fifo-')):assert line.startswith(expected)
 else:assert line==expected

# I reconstruct each complete report stream and check all independent criteria.
manifests=[n for n in records if n.startswith('comparison/tmp/nano-affine-') and n.endswith('/manifest.json')];assert len(manifests)==1
case_counts={'preflight':228};seen=set();total=0;report_commands=set()
report_hashes={'preflight':'bf39b8692e5b95198377eef1c35683dc78404c94667024967a339cc8884b8a2c'}
for name in manifests:
 doc=js(name);folder=name.rsplit('/',1)[0];groups={r['group'] for r in doc['cases']};new=(groups-{'original'}).pop();assert groups=={'original',new} and new in case_counts and new not in seen;seen.add(new)
 assert len(doc['routes'])==12
 for group,count in [('original',1001),(new,case_counts[new])]:
  cases=[r for r in doc['cases'] if r['group']==group];assert len(cases)==count
  baseline=data(folder+'/'+group+'-actual-c.reports');lines=baseline.decode().splitlines();assert len(lines)==count
  if group=='original':assert sha(baseline)=='f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0'
  for case,line in zip(cases,lines):
   assert records[mapped(case['path'])]['sha256']==case['sha256']
   inspect_case(group,new,case,line)
  if group!='original':assert sha(baseline)==report_hashes[new]
  physical=root/folder.removeprefix('comparison/')
  leaf={'calls':'value_calls','context':'call_context','cache':'analysis_cache'}.get(new,new)
  reference=q['reference_argv'][leaf]
  prefixes={'actual-c':[reference[reference.index('-o')+1]],'c-seed-native':[str(physical/'decoder-native')],
   'c-producer-nanoisa-vm':[str(source/'bin/nano_vm'),str(physical/'decoder.nvm'),'--'],
   'c-producer-sanitized-native':[str(physical/'decoder-aot')],
   'nanolang-producer-nanoisa-vm':[str(source/'bin/nano_vm'),str(physical/'source-decoder.nvm'),'--'],
   'nanolang-producer-sanitized-native':[str(physical/'source-decoder-aot')]}
  routes=[r for r in doc['routes'] if r['group']==group];assert {r['label'] for r in routes}==set(prefixes)
  for route in routes:
   raw=data(folder+'/'+group+'-'+route['label']+'.reports');assert raw==baseline and route['sha256']==sha(raw) and route['cases']==count
   prefix=prefixes[route['label']];paths={r['path'] for r in cases}
   commands=[c for c in q['commands'] if c['argv'][:len(prefix)]==prefix and c['argv'][len(prefix)] in paths]
   assert [n for c in commands for n in c['argv'][len(prefix):]]==[r['path'] for r in cases]
   for c in commands:
    inputs=c['argv'][len(prefix):];assert all(n in paths for n in inputs) and c['timeout']==(600 if group=='original' else 60)
    assert len(inputs)<=(8 if group=='original' else 32)
    if group!='original':assert len(inputs)==1 or sum(records[mapped(n)]['bytes'] for n in inputs)<=8192
    assert c['number'] not in report_commands;report_commands.add(c['number'])
   assert b''.join(data('comparison/command-%04d.stdout'%c['number']) for c in commands)==raw
   total+=count
assert seen==set(case_counts) and total==7374 and len(report_commands)+8==len(q['commands'])
for c in q['commands']:
 if c['number'] not in report_commands:assert c['timeout']==(600 if c['build'] else 60)
# I bind source/test bytes to Git with exactly the published copy expression.
p=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE);changed=[]
for n,h in q['source_before'].items():
 p.stdin.write((m['source_commit']+':'+n+'\n').encode());p.stdin.flush();parts=p.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob',n
 old=p.stdout.read(int(parts[2]));assert p.stdout.read(1)==b'\n';actual=data('comparison/source/'+n)
 if old!=actual:
  changed.append(n)
  if n in fix_names:
   assert actual==subprocess.check_output(['git','show',(m['cache_fix_commit'] if n.endswith('affine_analysis_cache_driver.nano') else m['report_fix_commit'])+':'+n],cwd=base),n
   continue
  needle=b'    return Query { status: 0, number: out.number, fields: (array_slice selected 0 out.number), globals: out.globals,'
  replacement=b'    let mut copied_fields: array<int> = []\n    for field_index in (range 0 out.number) { set copied_fields (array_push copied_fields (at selected field_index)) }\n    return Query { status: 0, number: out.number, fields: copied_fields, globals: out.globals,'
  assert old.count(needle)==1 and old.replace(needle,replacement)==actual,n
 assert sha(actual)==h
p.stdin.close();assert p.wait()==0 and set(changed)==fix_names|{'src_nano/nvm2c/ownership_queries.nano'}
print('PASS %d records; %d commands, 7374 complete actual-C/five-route reports, full original limits/journal/Git/source/image/tool/object/host/provider/observer guards and published full query dependency. Isolated Linux fresh-function preflight gate only; no paired-platform/compiler repair/integration/emission/cutover/release acceptance.'%(len(records),len(q['commands'])))
