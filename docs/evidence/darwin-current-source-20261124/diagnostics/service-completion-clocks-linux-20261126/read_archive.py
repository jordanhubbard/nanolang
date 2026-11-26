"""I verify externally delayed real service and native-compiler completion."""
from pathlib import Path
import json,hashlib,sys,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
sha=lambda b:hashlib.sha256(b).hexdigest()
m,rows,data,js=read_archive(base,'5b242b6e877c00e1df5e8a2f8d16b279ca8d7504a8b4ed66841b6f648e1baf2e')
pm,pr,pd,pj=read_archive(base.parent/'typechecker-vector-prefix-cold-bootstrap-linux-20261126','3c2f440e292cb2c0c54f8b3d344b239c927223bed505c874e90831b842f1c5d7')
assert m['diagnostic_only'] and m['service_late_publication'] and m['native_compile_late_publication']
assert not any(m[k] for k in ['product_source_edits','ordinary_qualification','product_repair','paired','whole_release'])
assert m['source_commit']==pm['source_commit']=='04cee11a861af22dfd4540f369eb2c6bb42a9cfd'
assert pm['original_make_pass'] and pm['original_native36_pass'] and pm['original_emitter12_pass']
q=js('observed/result.json');failed=js('failed/result.json');wait=js('wait-control/result.json');prep=js('observed/preparation.json');admit=js('observed/admission.json');boot=Path(pm['roots']['bootstrap']);root=boot/'source';observed=Path(m['roots']['observed'])
assert q['terminal'] and q['accepted_diagnostic'] and q['two_late_publications_observed'] and q['all_guards_match']
assert q['diagnostic_only'] and not q['production_edits'] and not q['whole_release']
assert (q['original_shadow_seconds'],q['original_native_compile_seconds'])==(10,120)
assert q['source_before']==q['source_after']==pj('bootstrap/bootstrap-result.json')['source_after'] and len(q['source_before'])==3971
for name,h in q['source_before'].items():assert pr['bootstrap/source/'+name]['sha256']==h
assert q['tool_images_before']==q['tool_images_after'] and q['host_tools_before']==q['host_tools_after']
for path,h in q['tool_images_after'].items():
 p=Path(path)
 if p.is_relative_to(observed):row=rows['observed/'+str(p.relative_to(observed))]
 elif p.is_relative_to(root):row=pr['bootstrap/source/'+str(p.relative_to(root))]
 else:row=rows['external/'+path.lstrip('/')]
 assert row['sha256']==h,path
for path,h in q['host_tools_after'].items():assert rows['external/'+path.lstrip('/')]['sha256']==h
assert q['diagnostic_inputs_before']==q['diagnostic_inputs_after']
for name,h in q['diagnostic_inputs_after'].items():assert rows['observed/'+name]['sha256']==h
assert q['observer_sha256']==rows['observed/fork_observer.so']['sha256']
assert data('observed/fork_observer.c')==data('failed/fork_observer.c')==data('unrun-plan/fork_observer.c')
assert b'pid_t fork(void)' in data('observed/fork_observer.c') and b'clock_gettime' not in data('observed/fork_observer.c')
assert q['owned_cleanup']['terminal'] and q['owned_cleanup']['pidfd_identity'] and not q['owned_cleanup']['remaining'] and not q['owned_cleanup']['records']
assert admit['terminal'] and admit['accepted'] and admit['source_after']==q['source_after'] and admit['free_bytes']>15*1024**3 and admit['filesystem']!='nfs' and not admit['other_jobs'] and all(v['cpu']>=1 and v['share']>=.75 for v in admit['cpu_rounds'])
for name,h in admit['runtime_inputs'].items():assert pr['bootstrap/source/'+name]['sha256']==h
for label in ['observed','failed']:
 c=js(label+'/cleanup-control-v2.json');assert c['terminal'] and c['accepted'] and not c['cleanup']['remaining'] and c['helper_sha256']==rows[label+'/owned_cleanup.py']['sha256']
assert wait['terminal'] and wait['accepted'] and wait['product_commands']==0 and not wait['product_source_edits'] and wait['controlled_child_exit']==-9
assert wait['tool_before']==wait['tool_after']==rows['external/home/jkh/.local/bin/toolscheme']['sha256']
assert [c['expected_satisfied'] for c in wait['commands']]==[False,True,False,True,False]
for c in wait['commands']:
 assert c['exit']==0 and 'timeout-ms 500' in c['argv'][-1]
 out=data('wait-control/'+c['label']+'.stdout');assert sha(out)==c['stdout_sha256']
 assert sha(data('wait-control/'+c['label']+'.stderr'))==c['stderr_sha256']
 assert (b'(satisfied #t)' if c['expected_satisfied'] else b'(satisfied #f)') in out
 assert (b'(reason satisfied)' if c['expected_satisfied'] else b'(reason timeout)') in out
assert b'T (stopped)' in data('wait-control/stopped-status.txt')
assert prep['wait_helper_result_sha256']==rows['wait-control/result.json']['sha256'] and prep['controller_sha256']==rows['observed/observe.py']['sha256']
assert prep['controller_reversal_exact'] and prep['changed_regions']==3 and prep['all_original_four_cases_and_assertions_preserved']
reverse=data('observed/observe.py')
for before,after in reversed(js('observed/controller-regions.json')):
 assert reverse.count(after.encode())==1;reverse=reverse.replace(after.encode(),before.encode(),1)
assert reverse==data('observed/controller-before.py.txt')==data('failed/observe.py')
assert sha(reverse)==prep['failed_controller_sha256']
assert failed['terminal'] and not failed['accepted_diagnostic'] and failed['all_guards_match'] and 'TimeoutExpired' in failed['exception']
assert not failed['owned_cleanup']['remaining'] and not failed['owned_cleanup']['records'] and failed['source_after']==q['source_after']
assert [c['label'] for c in failed['cases']]==['baseline-module','late-service-module'] and failed['cases'][0]['exit']==0
assert data('failed/late-service-module/product')==b'I retain my prior service product.\n'
assert data('failed/original-observe.py.txt')==data('unrun-plan/observe.py')
assert sha(data('unrun-plan/observe.py'))==js('failed/preparation.json')['original_controller_sha256']
assert [c['label'] for c in q['cases']]==['baseline-module','late-service-module','baseline-native','late-native-compile']
for index,c in enumerate(q['cases']):
 label=c['label'];prefix='observed/'+label+'/';assert c['terminal'] and c['exit']==0 and not c['product']['prior_preserved']
 assert c['record_counts']=={'SELECT':5,'START':5,'DONE':5}
 assert c['stdout_sha256']==rows[prefix+'stdout']['sha256'] and c['stderr_sha256']==rows[prefix+'stderr']['sha256']
 assert c['product']['sha256']==rows[prefix+'product']['sha256'] and c['product']['bytes']==rows[prefix+'product']['bytes']
 lines=data(prefix+'stderr').decode().splitlines();selected=[x.removeprefix('SELECT ') for x in lines if x.startswith('SELECT ')];started=[x.removeprefix('START ') for x in lines if x.startswith('START ')];done=[x.removeprefix('DONE ') for x in lines if x.startswith('DONE ')];assert len(selected)==5 and selected==started==done
 assert data(prefix+'fork-events').decode()==c['fork_events'];events=[line.split() for line in c['fork_events'].splitlines()];assert len(events)==(1 if index<2 else 2)
 assert [int(e[1]) for e in events]==list(range(1,len(events)+1)) and all(e[0]=='FORK' and int(e[2])==c['pid'] for e in events)
 assert c['argv'][0]==str(observed/'tools/nanoc_c') and c['argv'][1]==str(observed/'source.nano') and c['argv'].count('--allow-temporary-files')==1
 assert ('--emit-nvm' in c['argv'])==(index<2)
 if index in [1,3]:
  deadline=10 if index==1 else 120;fork=1 if index==1 else 2;assert c['deadline_seconds']==deadline and c['selected_fork']==fork
  assert c['child_resumed_monotonic']-c['parent_stopped_monotonic']>deadline
  assert c['child_exit_observed_monotonic']>=c['child_resumed_monotonic'] and c['seconds']>deadline
  assert 'T (stopped)' in c['parent_status_stopped'] and 'T (stopped)' in c['child_status_stopped'] and 'T (stopped)' in c['parent_status_before_resume'] and 'hrtimer_nanosleep' in c['parent_wchan_before_stop']
  e=events[fork-1];assert int(e[3])==c['child_pid'] and e[4]==c['caller_offset']
  caller=data('observed/'+label+'-fork-caller.stdout').decode();assert caller==c['caller_source'] and ('service_shadows.c:' if index==1 else 'service_product.c:') in caller
  assert all(v['exit']==0 and '(satisfied #t)' in v['stdout'] and not v.get('outer_timeout') for v in c['wait_conditions'])
  if index==3:
   assert c['retained_staging']
   for stage in c['retained_staging']:assert any(n.startswith('observed/'+str(Path(stage).relative_to(observed))+'/') for n in rows)
assert q['cases'][0]['product']['sha256']==q['cases'][1]['product']['sha256']
assert data('observed/source.nano')==pd('bootstrap/source/tests/fixtures/nsi_file_binding_expected.nano.txt')+b'\nfn main()->int{return 0}\n'
assert data('observed/interface.nsi.json')==pd('bootstrap/source/tests/fixtures/nsi_file_plan.json')
assert q['fixture']['golden_sha256']==pr['bootstrap/source/tests/fixtures/nsi_file_binding_expected.nano.txt']['sha256'] and q['fixture']['combined_sha256']==rows['observed/source.nano']['sha256'] and q['fixture']['interface_sha256']==rows['observed/interface.nsi.json']['sha256']
assert rows['observed/tools/nanoc_c']['sha256']==pr['bootstrap/source/bin/nanoc_c']['sha256']
for c in q['commands']:
 assert c['exit']==0
 if c['label'].endswith('-execute'):assert c['args'][1:]==['--allow-temporary-files']
 assert 'observed/'+c['label']+'.stdout' in rows and 'observed/'+c['label']+'.stderr' in rows
assert [c['label'] for c in q['commands'] if c['label'].endswith('-execute')]==['baseline-native-execute','late-native-compile-execute']
refusal=js('collector-source/live-refusal.json');assert refusal['terminal'] and refusal['accepted'] and refusal['exit']==1 and not refusal['archive_created']
assert refusal['result_sha256']==rows['collector-source/live-result.json']['sha256'] and not js('collector-source/live-result.json')['terminal']
assert refusal['collector_sha256']==rows['collector-source/package.py']['sha256'] and b'AssertionError' in data('collector-source/live-refusal.stderr')
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();p=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in q['source_before'].items():
 p.stdin.write((m['source_commit']+':'+name+'\n').encode());p.stdin.flush();header=p.stdout.readline().split();assert len(header)==3 and header[1]==b'blob';raw=p.stdout.read(int(header[2]));assert p.stdout.read(1)==b'\n' and sha(raw)==h,name
p.stdin.close();assert p.wait()==0
print('PASS %d records: both real late publications at original10/120; qualified products/source, exact File fixtures, baseline forks/records, real stopped states/pidfd exits/callsites, native execution, retained harness failure, five wait controls, exact three-region reversal and all guards/cleanup. Diagnostic only; C repair and ordinary full qualification remain open.'%len(rows))
