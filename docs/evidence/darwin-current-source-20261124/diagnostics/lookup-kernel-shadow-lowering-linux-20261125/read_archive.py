"""I verify the current Stage2 phase, bounded stacks and launcher failure."""
from pathlib import Path
import hashlib,json,subprocess,sys
base=Path(__file__).resolve().parent
sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'97fba060a5bf00bf50167b0f33f01bfff16302ba93ac2e019cf15f0f254ec7f3')
sha=lambda b:hashlib.sha256(b).hexdigest()
pin='95a4062efebe0e97d8cde271117eac1deab21595'
producer='78228e7dbd4485d1b243088a2d25ac5578816e08e1a48d4e0792cd47bf086739'
assert m['source_commit']==pin and m['diagnostic_only'] and m['bootstrap_stage2_image']
assert not any(m[k] for k in ['original_factory_acceptance','whole_release','historical_cause','performance_claim'])
s=js('observation/result.json');a=js('observation/admission-result.json');q=js('observation/stack-result.json')
p=js('phase/factory-result.json');bad=js('failed-output-directory/cohort-result.json')
pm,pr,pd,pj=read_archive(base.parent/'lookup-kernel-cold-bootstrap-linux-20261125','667a97fd742fd62558ed12eefa3578906acc55f8364131bd0b1f0171feb883d1')
b=pj('bootstrap/bootstrap-result.json');installed=pj('bootstrap/source/bin/nanoc_bootstrap.json')
assert b['terminal'] and b['original_make_pass'] and b['exit']==0 and installed['phase']=='stage2' and installed['raw_stage1_stage2_equal']
for label in ['observation','phase','failed-output-directory']:
 setup=js(label+'/result.json')
 assert setup['terminal'] and setup['admitted_setup'] and setup['all_guards_match'] and setup['source_base']==pin
 assert not setup['working_overlay'] and not setup['overlays'] and not setup['stage_alias']
 assert setup['bootstrap_result_sha256']==pr['bootstrap/bootstrap-result.json']['sha256']
 assert setup['bootstrap_receipt_sha256']==pr['bootstrap/source/bin/nanoc_bootstrap.json']['sha256']
 assert setup['source_before']==setup['source_after']==b['source_after']
 assert setup['producer_sha256']==rows[label+'/source/bin/nanoc_emitter_phase']['sha256']==installed['artifacts']['nanoc_stage2']==producer
 assert setup['hosts_before']==setup['hosts_after']==installed['hosts']
 for group in ['source','images','tools','hosts']:
  assert setup[group+'_before']==setup[group+'_after']
  for name,h in setup[group+'_before'].items():
   key=label+'/source/'+name if group in ['source','images'] else 'external/'+name.lstrip('/')
   assert rows[key]['sha256']==h,(label,group,name)
 admission=js(label+'/admission-result.json')
 assert admission['terminal'] and admission['accepted'] and not admission['other_jobs'] and admission['filesystem']!='nfs' and admission['free_bytes']>15*1024**3
 assert all(v['cpu']>=1 and v['share']>=.75 for v in setup['cpu_rounds'])
assert p['terminal'] and p['timed_out'] and p['exit']==-9 and p['all_guards_match'] and not p['product']
assert p['timeout']==300 and p['ordinary_seconds']==10 and not p['factory_gate_acceptance'] and not p['unchanged_full_retry']
assert p['producer_sha256']==p['producer_after']==producer and p['phase_sources']==p['phase_sources_after']
for group in ['source','images','tools','hosts']:assert p[group+'_before']==p[group+'_after']
phases=[json.loads(line) for line in data('phase/phases.jsonl').splitlines()]
expected=['inputs-start','imports-done','lex-done','parse-done','checked-emit-start','typecheck-start','typecheck-done','program-lowering-start','program-lowering-done','shadow-lowering-start','shadow-lowering-done','checked-emit-done','shadow-assembly-start']
assert phases==p['phases'] and [v['phase'] for v in phases]==expected
assert all(phases[i]['monotonic']<phases[i+1]['monotonic'] for i in range(len(phases)-1))
assert phases[-1]['monotonic']<p['deadline_observed_monotonic'] and 300<=p['seconds']<302
assert data('phase/stdout')==data('phase/stderr')==b''
assert q['terminal'] and q['accepted_observation'] and q['all_guards_match'] and q['gdb_exit']==0
assert not q['inferior_remaining'] and not q['owned_cleanup']['remaining'] and q['owned_cleanup']['owned_only'] and q['owned_cleanup']['pidfd_identity']
assert q['ordinary_seconds']==10 and q['outer_seconds']==300 and q['sample_seconds']==8 and q['wait_phase']=='shadow-lowering-start'
assert not q['product_acceptance'] and not q['original_consumer_acceptance'] and q['bootstrap_stage2_image'] and not q['development_image']
assert len(q['samples'])==8 and 8<=q['sample_elapsed']<9 and q['inferior_killed_monotonic']-q['start_monotonic']<300 and not q.get('outer_deadline_fired')
assert q['inferior_argv']==q['observed_argv']==[s['producer'],str(Path(m['roots']['observation'])/'source/tests/selfhost_shadow_emitter.nano'),'-o',str(Path(m['roots']['observation'])/'products/profile.native')]
count=lambda suffix:sum(any(name.endswith(suffix) for name in sample['functions']) for sample in q['samples'])
assert count('__memcpy_sve')==4 and count('_nisa_call_index')==5 and count('_nisa_fn_index')==1 and count('_nisa_enum_index')==2 and count('_native_phase_marker')==1
for sample in q['samples']:assert sample['stack'].encode() in data('observation/gdb-mi.log')
for group in ['source','images','tools','hosts']:assert q[group+'_after']==s[group+'_after']
assert q['producer_sha256']==q['producer_after']==producer
assert q['driver_sha256']==q['driver_after']==rows['observation/sample.py']['sha256']
assert q['debugger_sha256']==q['debugger_after']==rows['external/'+q['debugger'].lstrip('/')]['sha256']
assert q['phase_sources_before']==q['phase_sources_after']
for name,h in q['phase_sources_before'].items():assert rows['observation/'+name]['sha256']==h
assert b'280000' in data('observation/sample.py')
controls=[n for n in rows if n.startswith('observation/mi-control-') and n.endswith('/stack-result.json')]
assert len(controls)==1
for name in controls:
 c=js(name);assert c['terminal'] and c['accepted_observation'] and c['gdb_exit']==0 and not c['inferior_remaining'] and not c['owned_cleanup']['remaining'] and len(c['samples'])==3 and c['driver_sha256']==q['driver_sha256']
cleanup=js('observation/cleanup-control-v2.json')
assert cleanup['terminal'] and cleanup['accepted'] and cleanup['helper_sha256']==rows['observation/owned_cleanup.py']['sha256'] and cleanup['control_sha256']==rows['observation/cleanup_control_v2.py']['sha256']
assert cleanup['child']['pgid']==cleanup['child']['pid'] and not cleanup['cleanup']['remaining']
assert any(v['pid']==cleanup['child']['pid'] and v.get('waitpid',[0,0])[1]==9 for v in cleanup['cleanup']['records'])
assert bad['terminal'] and bad['all_guards_match'] and len(bad['commands'])==1
command=bad['commands'][0];assert command['exit']==1 and command['timeout']==600 and not command.get('timed_out')
err=data('failed-output-directory/command-1.stderr')
assert b'I require separate native and generated-C outputs:' in err and b'Could not create temporary C file:' in err
assert not any(n.startswith('failed-output-directory/products/') or n.endswith('/permanent-test.stderr') for n in rows)
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip()
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in b['source_after'].items():
 proc.stdin.write((pin+':'+name+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob'
 raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n' and sha(raw)==h,name
proc.stdin.close();assert proc.wait()==0
print('PASS %d records: actual78228/exact95a origin, original300 phase timeout, eight bounded shadow-lowering samples, native debugger/adopted-orphan cleanup controls, stable guards and failed output-directory launcher. No factory/full-route/paired/release or cause/gain acceptance.'%len(rows))
