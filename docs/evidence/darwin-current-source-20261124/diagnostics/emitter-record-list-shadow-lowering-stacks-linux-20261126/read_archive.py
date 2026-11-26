"""I verify bounded current Stage2 stacks without promoting them to acceptance."""
from pathlib import Path
import hashlib,json,re,subprocess,sys
base=Path(__file__).resolve().parent
sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
sha=lambda b:hashlib.sha256(b).hexdigest()
m,rows,data,js=read_archive(base,'b7683d4db66498d9fa6a741fafcbf1d65da03189cf36a1e16f7d49060a74ef8d')
pin='771a14330475bcd784d7cb19a215d6a4f15728d4'
producer='8d7c240099edbe11cde499a268705f34660c2d940da4895378fc2a2961c886a5'
s=js('observation/result.json');a=js('observation/admission-result.json');q=js('observation/stack-result.json')
assert m['source_commit']==s['source_base']==q['source_pin']==pin and m['diagnostic_only'] and m['bootstrap_stage2_image']
assert not any(m[k] for k in ['development_image','factory_gate_acceptance','new_bootstrap','whole_release','historical_cause','performance_claim'])
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and not s['working_overlay'] and not s['overlays'] and not s['stage_alias']
assert a['terminal'] and a['accepted'] and not a['other_jobs'] and a['filesystem']!='nfs' and a['free_bytes']>15*1024**3
assert all(v['cpu']>=1 and v['share']>=.75 for v in s['cpu_rounds'])
assert q['terminal'] and q['accepted_observation'] and q['all_guards_match'] and q['gdb_exit']==0 and not q['inferior_remaining']
assert q['diagnostic_only'] and q['bootstrap_stage2_image'] and not q['development_image'] and not any(q[k] for k in ['whole_release','product_acceptance','original_consumer_acceptance'])
assert q['ordinary_seconds']==10 and q['outer_seconds']==300 and q['sample_seconds']==8 and q['wait_phases']==['shadow-lowering-start']
assert q['inferior_killed_monotonic']-q['start_monotonic']<300 and not q.get('outer_deadline_fired')
root=Path(m['roots']['observation'])
assert q['inferior_argv']==q['observed_argv']==[s['producer'],str(root/'source/tests/selfhost_shadow_emitter.nano'),'-o',str(root/'products/profile.native')]
for group in ['source','images','tools','hosts']:
 assert s[group+'_before']==s[group+'_after']==q[group+'_after']
 for name,h in s[group+'_before'].items():assert rows['observation/source/'+name if group in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
assert len(s['source_before'])==3971
assert q['producer_sha256']==q['producer_after']==s['producer_sha256']==rows['observation/source/bin/nanoc_emitter_phase']['sha256']==producer
assert q['driver_sha256']==q['driver_after']==rows['observation/sample.py']['sha256']
assert q['debugger_sha256']==q['debugger_after']==rows['external/'+q['debugger'].lstrip('/')]['sha256']
assert q['wait_tool_sha256']==rows['external/'+q['wait_tool'].lstrip('/')]['sha256']
assert q['phase_sources_before']==q['phase_sources_after']
for name,h in q['phase_sources_before'].items():assert rows['observation/'+name]['sha256']==h
phases=[json.loads(line) for line in data('observation/phases.jsonl').splitlines()]
assert q['phases']==phases and phases[0]['phase']=='inputs-start'
assert all(phases[i]['monotonic']<phases[i+1]['monotonic'] for i in range(len(phases)-1))
indices=[]
for window in q['windows']:
 phase=window['phase'];assert phase in q['wait_phases'] and window['requested_seconds']==8 and 0<window['seconds']<10
 assert window['start_monotonic']<q['start_monotonic']+285
 startphases=[json.loads(line) for line in q['phases_at_sample_start'][phase].splitlines()]
 assert any(p['phase']==phase and p['monotonic']<=window['start_monotonic'] for p in startphases)
 assert all(p in phases for p in startphases)
 assert b'(satisfied #t)' in data('observation/'+phase+'-wait.stdout')
 for i in window['sample_indices']:
  indices.append(i);v=q['samples'][i];assert v['phase']==phase and window['start_monotonic']<=v['monotonic']<window['start_monotonic']+10
assert indices==list(range(len(q['samples']))) and len(indices)>=2
for v in q['samples']:
 assert v['stack'].encode() in data('observation/gdb-mi.log')
 assert v['functions']==re.findall(r'func="([^"]+)"',v['stack'])
for missed in q.get('missed_windows',[]):assert missed['phase'] in q['wait_phases'] and missed['phase'] not in [w['phase'] for w in q['windows']]
assert set(w['phase'] for w in q['windows'])|set(w['phase'] for w in q.get('missed_windows',[]))==set(q['wait_phases'])
cleanup=q['owned_cleanup'];assert q['subreaper'] and cleanup['terminal'] and cleanup['owned_only'] and cleanup['pidfd_identity'] and not cleanup['remaining']
for v in cleanup['records']:assert v['waitpid'][0]==v['pid'] and v['ppid']>0 and v['starttime'].isdigit()
controls=[n for n in rows if n.startswith('observation/mi-control-') and n.endswith('/stack-result.json')]
assert len(controls)==1
for name in controls:
 c=js(name);assert c['terminal'] and c['accepted_observation'] and c['gdb_exit']==0 and not c['inferior_remaining'] and not c['owned_cleanup']['remaining'] and len(c['samples'])>=2
 assert c['driver_sha256']==q['driver_sha256'] and c['debugger_sha256']==q['debugger_sha256']
 for v in c['samples']:assert v['stack'].encode() in data(str(Path(name).parent/'gdb-mi.log'))
control=js('observation/cleanup-control-v2.json')
assert control['terminal'] and control['accepted'] and control['helper_sha256']==rows['observation/owned_cleanup.py']['sha256'] and control['control_sha256']==rows['observation/cleanup_control_v2.py']['sha256']
assert control['child']['pgid']==control['child']['pid'] and not control['cleanup']['remaining'] and any(v['pid']==control['child']['pid'] and v['waitpid']==[v['pid'],9] for v in control['cleanup']['records'])
refusal=js('collector-source/live-refusal-result.json');assert refusal['accepted'] and refusal['collector_exit']==1 and not refusal['archive_created'] and not js('collector-source/live-stack-result.json')['terminal']
assert refusal['live_result_sha256']==rows['collector-source/live-stack-result.json']['sha256'] and b'AssertionError' in data('collector-source/live-refusal.stderr')
pm,pr,pd,pj=read_archive(base.parent/'emitter-record-list-cold-bootstrap-linux-20261126','d230b2bd730561baa99e3496e8c58eb14fd570e24176c90bba9cb255f9df9f8b')
b=pj('bootstrap/bootstrap-result.json');installed=pj('bootstrap/source/bin/nanoc_bootstrap.json')
assert b['terminal'] and b['original_make_pass'] and b['exit']==0 and installed['phase']=='stage2' and installed['raw_stage1_stage2_equal'] and len(installed['steps'])==17 and all(v['exit_code']==0 for v in installed['steps'])
assert s['bootstrap_source_pin']==b['source_pin']==pin and s['bootstrap_source_inputs']==b['source_after']==s['source_before']
assert s['bootstrap_result_sha256']==pr['bootstrap/bootstrap-result.json']['sha256'] and s['bootstrap_receipt_sha256']==pr['bootstrap/source/bin/nanoc_bootstrap.json']['sha256'] and s['native_consumer_receipt_sha256']==pr['bootstrap/native-gates-result.json']['sha256']
assert s['hosts_before']==installed['hosts'] and producer==installed['artifacts']['nanoc_stage2']==pr['bootstrap/source/bin/nanoc_stage2']['sha256'] and not s['native_consumer_accepted']
for name,h in s['images_before'].items():assert pr['bootstrap/source/'+name]['sha256']==h
helperbase=base.parent/'owned-body-lowering-stack-linux-20261125';assert sha((helperbase/'manifest.json').read_bytes())=='2cf5d0d94a865c6749e5f30645f9384a300d782938e3d1e93c1751ad6d51dd03'
helperrows={r['path']:r for r in json.loads((helperbase/'manifest.json').read_bytes())['files']}
for name in ['owned_cleanup.py','cleanup_control_v2.py']:assert helperrows['observation/'+name]['sha256']==rows['observation/'+name]['sha256']
assert not any(n.endswith('/source/bin/nanoc_stage2') or n.endswith('/source/bin/nanoc_bootstrap.json') for n in rows)
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip()
assert s['bootstrap_publication']=='08c223f945474f459364c24d5e8767e4ede1b0ab' and s['current_compiler_closure_matches']
assert subprocess.check_output(['git','diff',pin,s['current_head'],'--','src','modules','stdlib','src_nano',':(exclude)src_nano/nvm2c/**','tests/selfhost_shadow_emitter.nano'],cwd=repo)==b''
subprocess.run(['git','merge-base','--is-ancestor',s['bootstrap_publication'],s['current_head']],cwd=repo,check=True)
for name in ['manifest.json','read_archive.py']:
 origin=base.parent/'emitter-record-list-cold-bootstrap-linux-20261126'/name
 assert subprocess.check_output(['git','show',s['bootstrap_publication']+':'+str(origin.relative_to(repo))],cwd=repo)==origin.read_bytes()
p=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in s['source_before'].items():
 p.stdin.write((pin+':'+name+'\n').encode());p.stdin.flush();head=p.stdout.readline().split();assert len(head)==3 and head[1]==b'blob';raw=p.stdout.read(int(head[2]));assert p.stdout.read(1)==b'\n' and sha(raw)==h,name
p.stdin.close();assert p.wait()==0
print('PASS %d records: genuine8d7c240/exact771a origin, current closure, bounded phase windows/stacks/transcript, fresh admission, debugger and adopted-child controls, complete guards, nonterminal refusal. Sampling establishes no product, factory, paired, cause, speedup or release acceptance.'%len(rows))

assert s['phase_publication']=='fa182043809cc83131d1b81f26841549597f15f1' and s['phase_manifest_sha256']=='4bf05eea5d3dc4d046557a9be010e173beba9a0ebf919060eb33f5ba6ff27558'
phasebase=base.parent/'emitter-record-list-current-stage2-phase-linux-20261126'
read_archive(phasebase,s['phase_manifest_sha256'])
for n in ['manifest.json','read_archive.py']:assert subprocess.check_output(['git','show',s['phase_publication']+':'+str((phasebase/n).relative_to(repo))],cwd=repo)==(phasebase/n).read_bytes()
subprocess.run(['git','merge-base','--is-ancestor',s['phase_publication'],s['current_head']],cwd=repo,check=True)
