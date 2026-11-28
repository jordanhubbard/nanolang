"""I verify the compiler-observer diagnostic without qualifying a product."""
from pathlib import Path
import json,gzip,hashlib,subprocess,sys,argparse,ast,collections,re,shlex
p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=Path);a=p.parse_args();base=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest();m=json.loads((base/'manifest.json').read_bytes());rows={r['path']:r for r in m['files']};assert len(rows)==len(m['files'])
supplement=json.loads((base/'supplement.json').read_bytes());assert supplement['base_manifest_sha256']==sha((base/'manifest.json').read_bytes())=='c9ddffbb5bb4d8672f4c0eef06fa7e7f260b12d5c3e404d823eab1762cdd47e1' and supplement['post_terminal_observer_validation'] and not supplement['compiler_accepted'] and not supplement['whole_release']
assert len(supplement['files'])==22 and not (set(rows)&{v['path'] for v in supplement['files']});rows.update({v['path']:v for v in supplement['files']})
def raw(name):
 r=rows[name];p=Path(r['stored']);assert not p.is_absolute() and '..' not in p.parts;packed=(base/p).read_bytes();assert len(packed)==r['stored_bytes'] and sha(packed)==r['stored_sha256'];data=gzip.decompress(packed);assert len(data)==r['bytes'] and sha(data)==r['sha256'];return data
for n in rows:raw(n)
load=lambda name:json.loads(raw(name));q=load('controller/profile-result.json');r=load('controller/request.json');audit=load('terminal-audit.json')
assert q['terminal'] and q['diagnostic_valid'] and q['diagnostic_only'] and not q['accepted'] and not q['whole_release'] and not q.get('exception') and m['diagnostic_only'] and not m['compiler_accepted'] and not m['whole_release']
assert m['source_pin']==q['source_pin']==r['source_pin']=='d907f264d16d348b4589d3760bd213803e1c6f35' and m['publication']==r['publication']=='cdbb2dfafb9f8131abfb5f332814fab82a755edf' and r['profile_kind']=='real-compiler-wrapper'
parent=a.repo/'docs/evidence/darwin-current-source-20261124/diagnostics/linux-lowering-helper-diagnostic-20261126'
assert sha((parent/'complete/manifest.json').read_bytes())==m['parent_complete_manifest_sha256']==r['parent_complete_manifest_sha256']=='b562894bd99ae58f41e3aaeb504518a45dd7d1c7d26548c262e8600b787277df'
assert sha((parent/'failed-full-name/manifest.json').read_bytes())==m['parent_failed_manifest_sha256']==r['parent_failed_manifest_sha256']=='7fa6d133f8cc867488f10ebf03a8a0cec73a033c375c7a6b9dcaf565361bdd5e'
check=subprocess.run([sys.executable,str(parent/'read_terminal.py'),'--repo',str(a.repo)],capture_output=True,text=True);assert check.returncode==0,check.stderr;parent_result=json.loads(check.stdout);assert parent_result['records']==151 and not parent_result['compiler_accepted'] and not parent_result['whole_release']
pm=json.loads((parent/'complete/manifest.json').read_bytes());pr={v['path']:v for v in pm['files']};parentraw=lambda n:gzip.decompress((parent/'complete'/pr[n]['stored']).read_bytes());pq=json.loads(parentraw('controller/result.json'));request=json.loads(parentraw('controller/request.json'))
assert q['source_before']==q['source_after']==r['baseline_sources']==request['baseline_sources'];assert q['images_before']==q['images_after']==r['copied_images']==pq['images_after'];assert q['tools_before']==q['tools_after']==r['tools']==pq['tools_after']
assert q['accepted_sources_before']==q['accepted_sources_after']==r['accepted_source_guard']==pq['accepted_sources_after']
metadata={};batch=subprocess.Popen(['git','cat-file','--batch'],cwd=a.repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in r['baseline_sources'].items():
 batch.stdin.write((q['source_pin']+':'+name+'\n').encode());batch.stdin.flush();head=batch.stdout.readline().split();assert head[1]==b'blob';size=int(head[2]);data=batch.stdout.read(size);assert len(data)==size and batch.stdout.read(1)==b'\n' and sha(data)==h,name
 for target in ['file_product','nanoisa']:
  if name=='modules/'+target+'/module.json':
   v=json.loads(data);metadata[target]={k:v.get(k,[]) for k in ['c_sources','shared_c_sources']}
batch.stdin.close();assert batch.wait()==0
assert metadata==load('controller/profile-module-inputs.json')
assert q['observer_sources_before']==q['observer_sources_after']==r['observer_sources']
for name,h in r['observer_sources'].items():assert sha(raw('controller/'+name))==h
assert q['observer_tools_before']==q['observer_tools_after'] and set(q['observer_tools_before'])=={'/usr/bin/python3'}
for name,h in q['observer_tools_after'].items():assert sha(raw('observer-tools/'+name.lstrip('/')))==h
assert q['guards_match'] and q['reservation_released'] and not q['remaining_owned_processes']
assert audit['controller_absent_or_reused'] and not audit['remaining_owned_processes'] and not audit['reservation_exists'] and all(v for k,v in audit.items() if k.endswith('_match'))
assert [(c['label'],c['timeout']) for c in q['commands']]==[('seed-prefix',720),('witness-compile',180),('witness-dump',60),('witness-run',180)] and all(c['exit']==0 and not c.get('timed_out') for c in q['commands'])
for c in q['commands']:
 for stream in ['stdout','stderr']:assert c[stream+'_sha256']==sha(raw('controller/'+c['label']+'.'+stream))
prefix=load('controller/prefix-result.json');b=prefix['manifest'];root=Path(q['root']);source=root/'source';assert prefix['prepared_prefix_only'] and prefix['stopped_before_stage1'] and not prefix['bootstrap_accepted'] and not prefix['whole_release'] and prefix['budgets']==[dict(label='seed',seconds=600),dict(label='seed-verify',seconds=60)]
assert b['compiler']==[str(root/'compiler_observer.py')] and b['cflags']==['-O1'] and b['ldflags']==[] and b['stage_timeout_seconds']==600 and b['shadow_timeout_seconds']==30 and b['module_build_cache']==str(source/'obj/module_cache')
assert [v['label'] for v in b['steps']]==['seed','seed-verify'] and all(v['exit_code']==0 for v in b['steps'])
for step in b['steps']:assert sha(raw('bootstrap/'+Path(step['log']).name))==step['log_sha256']
assert q['hosts_before']==q['hosts_after']==b['hosts'] and len(b['hosts'])==6 and q['seed_before']==q['seed_after']==b['artifacts']['nanoc_seed.nvm']==sha(raw('products/bin/nanoc_seed.nvm'))
assert q['working_sources_before']==q['working_sources_after']==b['sources'] and not q['unexpected_native_compiler']
extra=set(b['sources'])-set(q['source_after']);assert extra=={'modules/index.json'}
for name,h in b['sources'].items():assert (sha(raw('inputs/'+name)) if name in extra else q['source_after'][name])==h
assert q['ordinary_shadow_seconds']==10 and q['observation_environment']['NANO_TRACE_BUILD']=='1' and q['observation_environment']['CC']==q['observation_environment']['NANO_CC']==str(Path(b['work'])/'guard-native-compiler')
assert q['witness_module_sha256']==sha(raw('controller/artifact-profile.nvm'))
imports={shlex.split(line)[1] for line in raw('controller/witness-dump.stdout').decode().splitlines() if line.startswith('.import ') and shlex.split(line)[1]};assert imports==set(q['provider_hosts']) and imports
for name,h in {**b['hosts'],**q['provider_hosts'],**q['returned_hosts']}.items():assert sha(raw('hosts/'+str(Path(name).relative_to(source))))==h
assert q['same_prefix_hosts']=={name:name in b['hosts'] and h==b['hosts'][name] for name,h in q['returned_hosts'].items()} and all(q['same_prefix_hosts'].values())
assert q['lookup_phases_sha256']==sha(raw('controller/lookup-phases.jsonl')) and q['compiler_events_sha256']==sha(raw('controller/compiler-events.jsonl'))
analysis_source=raw('controller/analyze_profile.py').decode();tree=ast.parse(analysis_source);namespace=dict(Path=Path,json=json,hashlib=hashlib,collections=collections,re=re)
for name in ['category','analyze']:
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);exec(ast.get_source_segment(analysis_source,node),namespace)
report=namespace['analyze'](raw('controller/compiler-events.jsonl'),raw('controller/lookup-phases.jsonl'),metadata);assert report==load('controller/compiler-profile-analysis.json') and report['compiler_events']==4486 and report['compiler_invocations']==2243
assert report['phase_invocations']=={'seed-prefix':1213,'witness-compile':24,'file_product-first':331,'file_product-repeat':331,'nanoisa-first':172,'nanoisa-repeat':172}
assert [v['compiler_count'] for v in report['lookups']]==[331,331,172,172,0] and all(not v['compiler_spans_overlap'] for v in report['lookups'])
for v in report['lookups']:
 for group in v['categories'].values():assert group['exit_statuses']=={'0':group['count']}
lines=raw('controller/witness-run.stdout').decode().splitlines();expected_labels=['file_product-first','file_product-repeat','nanoisa-first','nanoisa-repeat','missing-first'];assert len(lines)==5
results={}
for label,line in zip(expected_labels,lines):
 assert line.startswith('result:'+label+':');path=line.split(':',2)[2];assert (path in q['returned_hosts']) if label!='missing-first' else path=='';results[label]=path
for target in ['file_product','nanoisa']:assert results[target+'-first']==results[target+'-repeat']
reuses=re.findall(rb'^I checked build evidence: phase=reuse expected=(\d+) observed=(\d+) accepted=([01])$',raw('controller/witness-run.stderr'),re.M);assert reuses==[(b'1',b'1',b'1')]*4
transparency=load('observer-transparency/result.json');assert transparency['terminal'] and transparency['accepted_observer'] and not transparency['compiler_accepted'] and not transparency['whole_release']
assert transparency['real_compiler_before']==transparency['real_compiler_after']==q['tools_after']['/usr/bin/cc'] and transparency['python_before']==transparency['python_after']==q['observer_tools_after']['/usr/bin/python3'] and transparency['observer_sha256']==q['observer_sources_after']['compiler_observer.py']==sha(raw('observer-transparency/compiler_observer.py'))
assert [(v['name'],v['expected_exit']) for v in transparency['cases']]==[('version',0),('refusal',1),('stdin',0),('descriptor',0)]
for case in transparency['cases']:
 assert case['matched'] and case['timeout']==30
 for variant in ['direct','observed']:
  assert case[variant]['exit']==case['expected_exit']
  for stream in ['stdout','stderr']:assert case[variant][stream+'_sha256']==sha(raw('observer-transparency/'+case['name']+'.'+variant+'.'+stream))
 for stream in ['stdout','stderr']:assert raw('observer-transparency/'+case['name']+'.direct.'+stream)==raw('observer-transparency/'+case['name']+'.observed.'+stream)
assert raw('observer-transparency/stdin.direct.stdout').strip()==b'int value = 7;' and raw('observer-transparency/descriptor.direct.stdout').strip()==b'int inherited = 19;'
transparency_events=raw('observer-transparency/compiler-events.jsonl');assert sha(transparency_events)==transparency['events_sha256'];te=[json.loads(line) for line in transparency_events.splitlines()];assert len(te)==8 and all(v['phase']=='transparency' for v in te)
print(json.dumps(dict(verified=True,records=len(rows),source_blobs_verified=len(r['baseline_sources']),parent_records=parent_result['records'],compiler_events=report['compiler_events'],compiler_invocations=report['compiler_invocations'],lookup_compiler_counts=[v['compiler_count'] for v in report['lookups']],observer_transparency_cases=4,cache_reuses=4,diagnostic_valid=True,compiler_accepted=False,whole_release=False,current_liveness_checked=False)))
