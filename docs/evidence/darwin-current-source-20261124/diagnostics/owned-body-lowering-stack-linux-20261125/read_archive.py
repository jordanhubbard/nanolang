"""I verify eight actual lowering stacks from the exact retained native image."""
from pathlib import Path
import sys,json,hashlib,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'2cf5d0d94a865c6749e5f30645f9384a300d782938e3d1e93c1751ad6d51dd03');sha=lambda raw:hashlib.sha256(raw).hexdigest();s=js('observation/result.json');a=js('observation/admission-result.json');q=js('observation/stack-result.json')
assert m['source_commit']==m['development_derivation_source_commit']==s['source_base']==q['source_pin']=='00025ff76979b4009b84f10f8cb8f07492c62210'
assert m['diagnostic_only'] and m['development_image'] and m['observation_pass'] and not any(m[k] for k in ['bootstrap_stage2_image','first_documentation_admission_refusal','original_factory_acceptance','new_bootstrap','whole_release','historical_cause','performance_claim'])
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and not s['working_overlay'] and not s['overlays'] and not s['stage_alias'] and not s['recorded_documentation_difference']
assert a['terminal'] and a['accepted'] and not a['other_jobs'] and a['filesystem']!='nfs' and a['free_bytes']>15*1024**3 and all(c['cpu']>=1 and c['share']>=.75 for c in s['cpu_rounds'])
assert q['terminal'] and q['accepted_observation'] and q['all_guards_match'] and q['gdb_exit']==0 and not q['inferior_remaining'] and q['subreaper'] and not q['owned_cleanup']['remaining'] and q['owned_cleanup']['owned_only'] and q['owned_cleanup']['pidfd_identity']
assert q['ordinary_seconds']==10 and q['outer_seconds']==300 and q['sample_seconds']==8 and q['wait_phase']=='program-lowering-start' and not q['product_acceptance'] and not q['original_consumer_acceptance']
assert len(q['samples'])==8 and 8<=q['sample_elapsed']<9 and q['inferior_killed_monotonic']-q['start_monotonic']<300 and not q.get('outer_deadline_fired')
assert q['inferior_argv']==q['observed_argv']==[s['producer'],str(Path(m['roots']['observation'])/'source/tests/selfhost_shadow_emitter.nano'),'-o',str(Path(m['roots']['observation'])/'products/profile.native')]
count=lambda suffix:sum(any(name.endswith(suffix) for name in sample['functions']) for sample in q['samples'])
assert count('__memcpy_sve')==5 and count('_nisa_call_index')==2 and count('_nisa_expr_type')==4 and count('_nisa_union_index')==1 and count('_nisa_json_signature')==1 and count('_nanoisa_has_borrows')==1 and count('_nisa_module_declarations_valid')==1
for sample in q['samples']:assert sample['stack'].encode() in data('observation/gdb-mi.log')
for group in ['source','images','tools','hosts']:
 assert q[group+'_after']==s[group+'_before']==s[group+'_after']
 for name,h in s[group+'_before'].items():assert rows['observation/source/'+name if group in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
assert q['producer_sha256']==q['producer_after']==s['producer_sha256']==rows['observation/source/bin/nanoc_emitter_phase']['sha256']=='385db586c833e94a5a7d02eefa6289a5020bf3e46ecb75936d466f45f712fb4c'
assert q['driver_sha256']==q['driver_after']==rows['observation/sample.py']['sha256'] and q['debugger_sha256']==q['debugger_after']==rows['external/'+q['debugger'].lstrip('/')]['sha256']
assert q['phase_sources_before']==q['phase_sources_after']
for name,h in q['phase_sources_before'].items():assert rows['observation/'+name]['sha256']==h
assert str(Path(m['roots']['observation'])/'mark.py').encode() in data('observation/phase-recorder')
phases=[json.loads(line) for line in data('observation/phases.jsonl').splitlines()];assert q['phases']==phases and [v['phase'] for v in phases]==['inputs-start','imports-done','lex-done','parse-done','checked-emit-start','typecheck-start','typecheck-done','program-lowering-start']
assert all(phases[i]['monotonic']<phases[i+1]['monotonic'] for i in range(len(phases)-1)) and phases[-1]['monotonic']<=q['sample_start_monotonic']<q['samples'][0]['monotonic']
assert b'280000' in data('observation/sample.py')
controls=[n for n in rows if n.startswith('observation/mi-control-') and n.endswith('/stack-result.json')];assert len(controls)==a['debugger_controls']==2
for name in controls:
 c=js(name);assert c['terminal'] and c['accepted_observation'] and c['gdb_exit']==0 and not c['inferior_remaining'] and not c['owned_cleanup']['remaining'] and len(c['samples'])==3 and c['driver_sha256']==q['driver_sha256']
cleanup=js('observation/cleanup-control-v2.json');assert cleanup['terminal'] and cleanup['accepted'] and cleanup['helper_sha256']==rows['observation/owned_cleanup.py']['sha256'] and cleanup['control_sha256']==rows['observation/cleanup_control_v2.py']['sha256']
assert cleanup['child']['pgid']==cleanup['child']['pid'] and not cleanup['cleanup']['remaining'] and any(v['pid']==cleanup['child']['pid'] and v.get('waitpid',[0,0])[1]==9 for v in cleanup['cleanup']['records'])
assert a['cleanup_control_sha256']==rows['observation/cleanup-control-v2.json']['sha256'] and data('observation/owned_cleanup_initial.py')!=data('observation/owned_cleanup.py')
assert not any(n.endswith('/source/bin/nanoc_stage2') or n.endswith('/source/bin/nanoc_bootstrap.json') or n=='observation/products/profile.native' for n in rows)
pm,pr,pd,pj=read_archive(base.parent/'owned-body-compiler-development-linux-20261125','7b3a3ce30c132413fdd8b27af6d387cdc5e6da98d88ae91a07a9f44ce002e465');d=pj('development/development-native-result.json');n=pj('development/native36-result.json');f=pj('development/factory-result.json')
assert d['terminal'] and d['development_native_accepted'] and d['all_guards_match'] and n['accepted_component'] and n['methods']==36 and f['timed_out'] and not f['product']
assert s['development_source_inputs']==d['source_after']==s['source_before'] and len(s['source_before'])==3903
for key,name in [('development_result_sha256','development-native-result.json'),('development_native36_sha256','native36-result.json'),('development_factory_sha256','factory-result.json')]:assert s[key]==rows['origin/'+name]['sha256']==pr['development/'+name]['sha256']
assert s['hosts_before']==d['hosts_after'] and s['producer_sha256']==pr['development/products/compiler-development.native']['sha256']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in s['source_before'].items():
 proc.stdin.write((m['source_commit']+':'+name+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob';raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n' and sha(raw)==h,name
proc.stdin.close();assert proc.wait()==0
print('PASS %d records: actual385db origin/exact3903 inputs, eight lowering stacks inside300/default10/original280 phase wait, both native GDB controls and real separate-group adopted-child pidfd cleanup. No product/factory/bootstrap/paired/pure-translator/release or cause/gain acceptance.'%len(rows))
