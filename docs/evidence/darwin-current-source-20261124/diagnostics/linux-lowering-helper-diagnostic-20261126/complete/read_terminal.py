"""I verify my bounded Linux observation and keep compiler acceptance false."""
from pathlib import Path
import json,hashlib,gzip,subprocess,sys,argparse,os
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--failed-name-guard',action='store_true');a=p.parse_args();b=Path(__file__).resolve().parent;sha=lambda d:hashlib.sha256(d).hexdigest();m=json.loads((b/'manifest.json').read_bytes());rows={x['path']:x for x in m['files']};assert len(rows)==len(m['files'])
def raw(n):
 r=rows[n];p=Path(r['stored']);assert not p.is_absolute() and '..' not in p.parts;packed=(b/p).read_bytes();assert len(packed)==r['stored_bytes'] and sha(packed)==r['stored_sha256'];d=gzip.decompress(packed);assert len(d)==r['bytes'] and sha(d)==r['sha256'];return d
for n in rows:raw(n)
load=lambda n:json.loads(raw(n));q=load('controller/result.json');r=load('controller/request.json');audit=load('terminal-audit.json');binding=load('controller/overlay-binding.json')
assert q['terminal'] and q['diagnostic_only'] and q['diagnostic_valid'] and not q['accepted'] and not q['whole_release'] and m['diagnostic_only'] and not m['compiler_accepted'] and not m['whole_release'] and not q.get('exception')
assert m['source_pin']==q['source_pin']==r['source_pin']=='d907f264d16d348b4589d3760bd213803e1c6f35'
assert m['publication']==r['publication']=='f4588455d2443bddc7eb65aaa6ed414a4e146735'
assert (r['seed_seconds'],r['verify_seconds'],r['compiler_shadow_seconds'],r['ordinary_seconds'],r['target_entry_seconds'],r['target_window_seconds'])==(600,60,30,10,600,120)
combined=raw('controller/observed-codegen.nano').decode();old=raw('controller/baseline-codegen.nano').decode();observed=raw('controller/parent-codegen.nano').decode();name=binding['path'];assert name=='src_nano/compiler/nanoisa_codegen.nano' and binding['target_function']=='nisa_emit_module_indexed_body' and binding['offset_units']=='unicode-codepoints'
assert sha(old.encode())==binding['baseline_sha256']==r['baseline_sources'][name] and sha(observed.encode())==binding['observed_sha256']==sha(observed.encode())
start,end=binding['start'],binding['end'];body=old[start:end]
for op in binding['operations']:
 assert body.count(op['before'])==1 and op['after'].count(op['before'])==1;body=body.replace(op['before'],op['after'],1)
assert len(binding['operations'])==12 and binding['added_shadow'].count('\nshadow ')==1 and observed==old[:start]+body+binding['added_shadow']+old[end:]
for op in reversed(binding['operations']):assert body.count(op['after'])==1;body=body.replace(op['after'],op['before'],1)
assert body==old[start:end]
helper=load('controller/helper-overlay-binding.json');assert helper['path']==name and helper['target_function']=='nisa_emit_function_effects' and helper['offset_units']=='unicode-codepoints'
assert helper['parent_sha256']==sha(observed.encode()) and helper['observed_sha256']==sha(combined.encode())
hstart,hend=helper['start'],helper['end'];hbody=observed[hstart:hend]
for op in helper['operations']:
 assert hbody.count(op['before'])==1 and op['after'].count(op['before'])==1;hbody=hbody.replace(op['before'],op['after'],1)
assert len(helper['operations'])==10 and helper['added_shadow'].startswith('\nshadow nisa_emit_function_effects {') and helper['added_shadow'].count('\nshadow ')==1
assert combined==observed[:hstart]+hbody+helper['added_shadow']+observed[hend:]
for op in reversed(helper['operations']):assert hbody.count(op['after'])==1;hbody=hbody.replace(op['after'],op['before'],1)
assert hbody==observed[hstart:hend]
prior=a.repo/'docs/evidence/darwin-current-source-20261124/diagnostics/linux-lowering-detail-diagnostic-20261126'
assert sha((prior/'manifest.json').read_bytes())==r['parent_linux_manifest_sha256']=='3b8ee2a19660c6ca457fef7e2d9eee3ec5bbb369784728ab0ed3cae86317c8ef'
prior_check=subprocess.run([sys.executable,str(prior/'read_terminal.py'),'--repo',str(a.repo)],capture_output=True,text=True);assert prior_check.returncode==0,prior_check.stderr;prior_result=json.loads(prior_check.stdout)
expected=dict(r['baseline_sources']);expected[name]=helper['observed_sha256'];assert q['source_before']==q['source_after']==expected and q['guards_match']
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=a.repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for n,h in r['baseline_sources'].items():
 proc.stdin.write((q['source_pin']+':'+n+'\n').encode());proc.stdin.flush();head=proc.stdout.readline().split();assert head[1]==b'blob';length=int(head[2]);data=proc.stdout.read(length);assert len(data)==length and proc.stdout.read(1)==b'\n' and sha(data)==h,n
proc.stdin.close();assert proc.wait()==0
assert q['images_before']==q['images_after']==r['copied_images'] and q['tools_before']==q['tools_after']==r['tools'] and q['accepted_sources_before']==q['accepted_sources_after']==r['accepted_source_guard']
# I bind the copied tools to my published original Linux bootstrap receipt.
parent=a.repo/'docs/evidence/darwin-current-source-20261124/diagnostics/typechecker-vector-prefix-cold-bootstrap-linux-20261126';assert sha((parent/'manifest.json').read_bytes())=='3c2f440e292cb2c0c54f8b3d344b239c927223bed505c874e90831b842f1c5d7'
checked=subprocess.run([sys.executable,str(parent/'read_archive.py')],capture_output=True,text=True);assert checked.returncode==0,checked.stderr
sys.path.insert(0,str(parent.parent));from read_chunks import read_archive
pm,pr,pdata,pjs=read_archive(parent,'3c2f440e292cb2c0c54f8b3d344b239c927223bed505c874e90831b842f1c5d7');pq=pjs('bootstrap/bootstrap-result.json')
assert pq['source_after']==r['accepted_source_guard'] and pq['tools_after']==r['tools']
assert {n:h for n,h in r['copied_images'].items() if n!='bin/generate_module_index'}==pq['images_after']
assert r['copied_images']['bin/generate_module_index']==pr['bootstrap/source/bin/generate_module_index']['sha256']
commands=q['commands'];assert [c['label'] for c in commands]==['seed-prefix','lowering-window'];assert commands[0]['exit']==0 and not commands[0].get('timed_out')
for c in commands:
 for stream in ['stdout','stderr']:assert sha(raw('controller/'+c['label']+'.'+stream))==c[stream+'_sha256']
prefix=load('controller/prefix-result.json');receipt=prefix['manifest'];assert prefix['prepared_prefix_only'] and prefix['stopped_before_stage1'] and not prefix['bootstrap_accepted'] and not prefix['whole_release']
assert prefix['budgets']==[{'label':'seed','seconds':600},{'label':'seed-verify','seconds':60}]
assert receipt['compiler']==['/usr/bin/cc'] and receipt['cflags']==['-O1'] and receipt['ldflags']==[] and receipt['module_build_cache']==str(Path(q['root'])/'source/obj/module_cache')
assert receipt['shadow_timeout_seconds']==30 and receipt['stage_timeout_seconds']==600 and [c['label'] for c in receipt['steps']]==['seed','seed-verify'] and all(c['exit_code']==0 for c in receipt['steps'])
for step in receipt['steps']:assert sha(raw('bootstrap/'+Path(step['log']).name))==step['log_sha256']
assert q['hosts_before']==q['hosts_after']==receipt['hosts'] and len(receipt['hosts'])==6
for n,h in receipt['hosts'].items():assert sha(raw('hosts/'+str(Path(n).relative_to(Path(q['root'])/'source'))))==h
assert q['seed_before']==q['seed_after']==sha(raw('products/bin/nanoc_seed.nvm'))==receipt['artifacts']['nanoc_seed.nvm']
assert q['working_sources_before']==q['working_sources_after']==receipt['sources'] and not q['unexpected_native_compiler']
extra=set(receipt['sources'])-set(expected);assert extra=={'modules/index.json'}
for n,h in receipt['sources'].items():assert (sha(raw('inputs/'+n)) if n in extra else expected[n])==h
assert q['phase_record_complete'];events=[json.loads(line) for line in raw('controller/phases.jsonl').splitlines()];assert events==q['phase_events'] and all(x['monotonic']<=y['monotonic'] for x,y in zip(events,events[1:]));targets=[e for e in events if e['phase']=='program-lowering-start'];assert len(targets)==1
last=commands[-1];assert last['target_started_monotonic']==targets[0]['monotonic'] and 0<=last['target_started_monotonic']-last['started_monotonic']<=600 and last['target_entry_seconds']==600 and last['target_window_seconds']==120
assert last['stop_reason'] in ['target-return-marker','target-window-deadline','vm-exited']
if last['stop_reason']=='target-window-deadline':assert last['deadline_monotonic']==last['target_started_monotonic']+120 and last['stop_monotonic']>=last['deadline_monotonic'] and last['observer_stopped_vm'] and last['exit']==-15
if last['stop_reason']=='target-return-marker':
 done=[e for e in events if e['phase']=='program-lowering-done'];assert len(done)==1 and 0<=done[0]['monotonic']-targets[0]['monotonic']<=120 and last['observer_stopped_vm'] and last['exit']==-15
entries=[e for e in events if e['phase'].startswith('lower-function-')];assert [int(e['phase'].split(':')[0].removeprefix('lower-function-')) for e in entries]==[0]+list(range(32,65))
assert q['reservation_released'] and not q['remaining_owned_processes'] and audit['controller_absent_or_reused'] and not audit['remaining_owned_processes'] and not audit['reservation_exists']
assert all(audit[k] for k in ['source_match','images_match','tools_match','accepted_sources_match','hosts_match','seed_match'])
helper_events=[e for e in events if e['phase'].startswith('helper-')]
labels=['enter','params-start','params-done','body-done','fallthrough-done','exact-done','header-done','contract-start','contract-done','parameters-done']
targets_by_queue={int(e['phase'].split(':')[0].removeprefix('lower-function-')):e['phase'].split(':',1)[1] for e in entries}
assert targets_by_queue[41].endswith('_publish_service_source') and targets_by_queue[49].endswith('_assemble_text_save')
if a.failed_name_guard:
 assert not helper_events and last['stop_reason']=='vm-exited' and last['exit']==-15
 stop=load('controller/helper-name-mismatch-stop.json');assert not stop['helper_acceptance'] and stop['pid']==last['pid']==stop['pgid'] and stop['process_start']==last['process_start'] and stop['signal']=='SIGTERM'
 for target in [targets_by_queue[41],targets_by_queue[49]]:assert ('(== func.name "'+target+'")') not in combined
else:
 for target in [targets_by_queue[41],targets_by_queue[49]]:
  actual=[e for e in helper_events if e['phase'].endswith(':'+target)];assert [e['phase'].removeprefix('helper-').split(':')[0] for e in actual]==labels
 assert len(helper_events)==20
if not a.failed_name_guard:
 expected_probe=dict(source=q['source_after'],images=q['images_after'],tools=q['tools_after'],hosts=q['hosts_after'],seed=q['seed_after'])
 def common_probe(x,prefix):
  assert x['terminal'] and not x['accepted'] and not x['whole_release'] and x['diagnostic_only'] and not x.get('exception') and x['parent_result_sha256']==sha(raw('controller/result.json'))
  assert x['before']==x['after']==expected_probe and x['guards_match'] and not x['unexpected_native_compiler']
  assert x['stdout_observer']=='raw PTY with normal libc line buffering and no preload' and x['commands'][-1]['argv'][0]==str(Path(q['root'])/'source/bin/nano_vm')
  assert x['ordinary_shadow_seconds']==10 and x['fixture_sha256']==sha(raw('controller/'+prefix+'artifact-boundary.nano')) and x['module_sha256']==sha(raw('controller/'+prefix+'artifact-boundary.nvm'))
  assert [(c['label'],c['timeout']) for c in x['commands']]==[('artifact-compile',180),('artifact-dump',60),('artifact-run',180)] and all(c['exit']==0 and not c.get('timed_out') for c in x['commands'])
  for c in x['commands']:
   for stream in ['stdout','stderr']:assert c[stream+'_sha256']==sha(raw('controller/'+prefix+c['label']+'.'+stream))
  import shlex
  imports={shlex.split(line)[1] for line in raw('controller/'+prefix+'artifact-dump.stdout').decode().splitlines() if line.startswith('.import ') and shlex.split(line)[1]};assert imports==set(x['provider_hosts']) and imports
  for n,h in {**x['provider_hosts'],**x['returned_hosts']}.items():assert sha(raw('probe-hosts/'+str(Path(n).relative_to(Path(q['root'])/'source'))))==h
  assert x['same_prefix_hosts']=={n:n in q['hosts_after'] and h==q['hosts_after'][n] for n,h in x['returned_hosts'].items()}
  pevents=x['events'];assert [e['line'] for e in pevents]==raw('controller/'+prefix+'artifact-run.stdout').decode().splitlines() and all(u['monotonic']<=v['monotonic'] for u,v in zip(pevents,pevents[1:]))
  intervals=[]
  for label,iterations in [('file_product',['first','repeat']),('nanoisa',['first','repeat']),('missing',['first'])]:
   for iteration in iterations:
    begins=[e for e in pevents if e['line']=='probe-start:'+label+':'+iteration];ends=[e for e in pevents if e['line'].startswith('probe-result:'+label+':'+iteration+':')];assert len(begins)==len(ends)==1;result=ends[0]['line'].split(':',3)[3]
    assert (result in x['returned_hosts']) if label!='missing' else result==''
    intervals.append(dict(label=label,iteration=iteration,seconds=ends[0]['monotonic']-begins[0]['monotonic'],result=result))
  assert intervals==x['intervals']
  for label in ['file_product','nanoisa']:assert [v['result'] for v in intervals if v['label']==label][0]==[v['result'] for v in intervals if v['label']==label][1]
 probe=load('controller/artifact-pty-result.json');common_probe(probe,'pty-valid-');post=load('controller/artifact-pty-post-terminal-audit.json')
 assert not probe['diagnostic_valid'] and not probe.get('reservation_released') and len(probe['remaining_owned_processes'])==1
 watcher=probe['remaining_owned_processes'][0];assert watcher['command'].startswith('toolscheme --root '+q['root']+' -e ') and 'artifact-pty-result.json' in watcher['command']
 assert post['terminal'] and post['post_terminal'] and post['raw_probe_result_sha256']==sha(raw('controller/artifact-pty-result.json')) and not post['raw_diagnostic_valid'] and post['timing_observation_valid'] and not post['compiler_accepted'] and not post['whole_release']
 assert post['controller_absent'] and post['guards_match'] and not post['remaining_owned_processes'] and post['reservation_released'] and post['reservation_owner']==dict(root=q['root'],pid=probe['pid'],process_start=probe['process_start'])
 assert len(post['observer_checks'])==1 and post['observer_checks'][0]['pid']==watcher['pid'] and post['observer_checks'][0]['absent_or_reused'] and q['root'] not in post['observer_checks'][0]['current_cmdline']
 assert post['reservation_owner_sha256']==sha(json.dumps(post['reservation_owner']).encode())
 buffered=load('controller/artifact-probe-result.json');stop=load('controller/artifact-buffering-stop.json')
 assert buffered['terminal'] and not buffered['diagnostic_valid'] and buffered['guards_match'] and buffered['reservation_released'] and not buffered['remaining_owned_processes'] and not buffered['events'] and not stop['timing_observation_valid']
 assert stop['pid']==stop['pgid']==buffered['commands'][-1]['pid'] and stop['process_start']==buffered['commands'][-1]['process_start'] and stop['signal']=='SIGTERM' and buffered['commands'][-1]['exit']==-15
 assert buffered['fixture_sha256']==sha(raw('controller/buffered-artifact-boundary.nano')) and buffered['parent_result_sha256']==sha(raw('controller/result.json'))
 for c in buffered['commands']:
  for stream in ['stdout','stderr']:assert c[stream+'_sha256']==sha(raw('controller/buffered-'+c['label']+'.'+stream))
 refused=load('controller/artifact-linebuffered-result.json');assert refused['terminal'] and not refused['diagnostic_valid'] and refused['guards_match'] and refused['reservation_released'] and not refused['remaining_owned_processes']
 assert refused['commands'][-1]['argv'][:2]==['/usr/bin/stdbuf','-oL'] and refused['commands'][-1]['exit']==1 and b'capture failure' in raw('controller/linebuffered-refusal-artifact-run.stderr')
 for c in refused['commands']:
  for stream in ['stdout','stderr']:assert c[stream+'_sha256']==sha(raw('controller/linebuffered-refusal-'+c['label']+'.'+stream))
 assert refused['observer_tools_before']==refused['observer_tools_after'] and set(refused['observer_tools_before'])=={'/usr/bin/stdbuf','/usr/libexec/coreutils/libstdbuf.so'}
 for n,h in refused['observer_tools_before'].items():assert sha(raw('observer-tools/'+n.lstrip('/')))==h
 traced=load('controller/artifact-trace-result.json');partial=load('controller/artifact-trace-partial-observation.json');assert traced['terminal'] and not traced['diagnostic_valid'] and traced['guards_match'] and traced['reservation_released'] and not traced['remaining_owned_processes'] and not traced['unexpected_native_compiler']
 assert traced['commands'][-1]['argv'][0]=='/usr/bin/strace' and traced['commands'][-1]['timeout']==180 and traced['commands'][-1]['timed_out'] and traced['commands'][-1]['exit']==-15
 for c in traced['commands']:
  for stream in ['stdout','stderr']:assert c[stream+'_sha256']==sha(raw('controller/trace-failed-'+c['label']+'.'+stream))
 assert partial['raw_result_sha256']==sha(raw('controller/artifact-trace-result.json')) and partial['trace_sha256']==sha(raw('controller/artifact-process.trace')) and not partial['timing_observation_valid'] and not partial['trace_complete'] and partial['original_timeout']==180
 for n,h in partial['returned_hosts'].items():assert sha(raw('probe-hosts/'+str(Path(n).relative_to(Path(q['root'])/'source'))))==h
 debug=load('controller/artifact-buildtrace-result.json');common_probe(debug,'');assert debug['diagnostic_valid'] and debug['reservation_released'] and not debug['remaining_owned_processes'] and debug['build_trace_enabled'] and debug['environment']['NANO_TRACE_BUILD']=='1'
 import ast,re,collections
 tree=ast.parse(raw('controller/analyze_buildtrace.py').decode());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='summarize');namespace=dict(re=re,json=json,hashlib=hashlib,collections=collections);exec(ast.get_source_segment(raw('controller/analyze_buildtrace.py').decode(),node),namespace)
 build_summary=namespace['summarize'](raw('controller/artifact-run.stderr'));assert build_summary==load('controller/artifact-buildtrace-summary.json')
 assert build_summary['phases']['reuse']==dict(count=4,accepted=3,rejected=1) and build_summary['timings_ms']['tool-query-ms']==dict(count=10,sum_observed=163,min_observed=3,max_observed=37)

print(json.dumps(dict(verified=True,records=len(rows),source_blobs_verified=len(expected),module_reversal=True,helper_reversal=True,parent_records=prior_result['records'],bootstrap_parent_records=len(pr),diagnostic_valid=True,helper_observation_valid=not a.failed_name_guard,artifact_intervals=[] if a.failed_name_guard else probe['intervals'],stop_reason=last['stop_reason'],seconds=last['seconds'],exit=last['exit'],helper_events=[dict(phase=e['phase'],seconds=e['monotonic']-last['target_started_monotonic']) for e in helper_events],compiler_accepted=False,whole_release=False,current_liveness_checked=False)))
