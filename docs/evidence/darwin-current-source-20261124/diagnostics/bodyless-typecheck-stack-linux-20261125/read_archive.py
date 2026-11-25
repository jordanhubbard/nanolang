"""I verify my actual bootstrap Stage2 path observation, not consumer acceptance."""
from pathlib import Path
import sys,json,hashlib,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'e79c9be408569914bc5eae816e0603f4248dbd7af15c5cf94091be7bebd2c8c4');sha=lambda b:hashlib.sha256(b).hexdigest();s=js('observation/result.json');a=js('observation/admission-result.json');q=js('observation/stack-result.json')
assert m['source_commit']==s['source_base']==q['source_pin']=='598db452b59373eb9faa41e545ea45300e1ae535'
assert m['diagnostic_only'] and m['development_image'] and not m['bootstrap_stage2_image'] and m['observation_pass'] and m['first_documentation_admission_refusal'] and not any(m[k] for k in ['original_factory_acceptance','historical_cause','performance_claim','new_bootstrap','whole_release'])
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and not s['working_overlay'] and not s['stage_alias'] and not s['overlays']
assert a['terminal'] and a['accepted'] and not a['other_jobs'] and a['filesystem']!='nfs' and a['free_bytes']>15*1024**3 and all(v['cpu']>=1 and v['share']>=.75 for v in s['cpu_rounds'])
assert q['terminal'] and q['accepted_observation'] and q['all_guards_match'] and q['gdb_exit']==0 and not q['inferior_remaining']
assert q['ordinary_seconds']==10 and q['outer_seconds']==300 and q['sample_seconds']==8 and q['development_image'] and not q['bootstrap_stage2_image'] and not q['original_consumer_acceptance'] and q['wait_phase']=='typecheck-start'
assert len(q['samples'])==8 and 8<=q['sample_elapsed']<9 and q['inferior_killed_monotonic']-q['start_monotonic']<300 and not q.get('outer_deadline_fired')
count=lambda suffix:sum(any(name.endswith(suffix) for name in sample['functions']) for sample in q['samples'])
assert count('_env_copy_symbols')==2 and count('_env_lookup')==3 and count('__memcpy_sve')==4 and count('__memcpy_generic')==1 and count('_tc_layout_visit')==1 and count('nmap_collect')==1 and count('_merge_struct_metadata')==1
for sample in q['samples']:assert sample['stack'].encode() in data('observation/gdb-mi.log')
assert q['inferior_argv']==q['observed_argv']==[s['producer'],str(Path(m['roots']['observation'])/'source/tests/selfhost_shadow_emitter.nano'),'-o',str(Path(m['roots']['observation'])/'products/profile.native')]
for group in ['source','images','tools','hosts']:
 assert q[group+'_after']==s[group+'_before']==s[group+'_after']
 for name,h in s[group+'_before'].items():assert rows['observation/source/'+name if group in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
assert q['producer_sha256']==q['producer_after']==s['producer_sha256']==rows['observation/source/bin/nanoc_emitter_phase']['sha256']=='aa6472382c2130bb166742d7bee32f180c68341d77a2bebd9d63a0f81194844e'
assert q['driver_sha256']==q['driver_after']==rows['observation/sample.py']['sha256'] and q['debugger_sha256']==q['debugger_after']==rows['external/'+q['debugger'].lstrip('/')]['sha256']
assert q['phase_sources_before']==q['phase_sources_after']
for name,h in q['phase_sources_before'].items():assert rows['observation/'+name]['sha256']==h
assert str(Path(m['roots']['observation'])/'mark.py').encode() in data('observation/phase-recorder')
phases=[json.loads(line) for line in data('observation/phases.jsonl').splitlines()];assert phases==q['phases'] and [r['phase'] for r in phases]==['inputs-start','imports-done','lex-done','parse-done','checked-emit-start','typecheck-start']
assert all(phases[i]['monotonic']<phases[i+1]['monotonic'] for i in range(len(phases)-1)) and phases[-1]['monotonic']<=q['sample_start_monotonic']<q['samples'][0]['monotonic']
controls=[name for name in rows if name.startswith('observation/mi-control-') and name.endswith('/stack-result.json')];assert len(controls)==1;c=js(controls[0]);assert c['terminal'] and c['accepted_observation'] and c['gdb_exit']==0 and not c['inferior_remaining'] and len(c['samples'])==3 and c['driver_sha256']==q['driver_sha256']
assert not any(name.endswith('/source/bin/nanoc_stage2') or name.endswith('/source/bin/nanoc_bootstrap.json') or name=='observation/products/profile.native' for name in rows)
pm,pr,pd,pj=read_archive(base.parent/'bodyless-compiler-development-linux-20261125','e6a8aa9cd29b1bcb7623c02e7bb2514e8b05b126bb2646458018f13e605499d8');d=pj('development/development-native-result.json');n=pj('development/native36-result.json');f=pj('development/factory-result.json')
assert d['terminal'] and d['development_native_accepted'] and d['all_guards_match'] and n['accepted_component'] and n['methods']==36 and f['timed_out'] and not f['product']
assert s['development_source_pin']==m['development_derivation_source_commit']==d['source_base']=='bf9fef87f2c965db3f8c9f6c813a542a15333675' and s['development_source_inputs']==d['source_after']
for key,name in [('development_result_sha256','development-native-result.json'),('development_native36_sha256','native36-result.json'),('development_factory_sha256','factory-result.json')]:assert s[key]==rows['origin/'+name]['sha256']==pr['development/'+name]['sha256']
assert s['development_native36_accepted'] and not s['factory_gate_accepted'] and s['development_image'] and not s['bootstrap_stage2_image']
assert s['hosts_before']==d['hosts_after'] and s['producer_sha256']==pr['development/products/compiler-development.native']['sha256']
for name,h in s['images_before'].items():assert h==pr['development/source/'+name]['sha256']
assert s['source_before'].keys()==d['source_after'].keys() and len(s['source_before'])==3903
changes={name:dict(before=d['source_after'][name],after=h) for name,h in s['source_before'].items() if h!=d['source_after'][name]}
assert changes==s['recorded_documentation_difference']=={'docs/ROADMAP.md':dict(before='c40ac0dd921586691b2023f6c8443c6c30aa22a204b2d6a4724f40a1fecb1539',after='db6ed6098e7481a0abc528f6db7c053bc793f47daaa168fcc710a9ce38747083')}
failed=js('first-admission/result.json');assert failed['terminal'] and failed['exception']=='AssertionError()' and not failed.get('admitted_setup') and failed['source_before']==s['source_before']
assert 'first-admission/stack-result.json' not in rows and 'first-admission/source/bin/nanoc_emitter_phase' not in rows
for name,h in failed['source_before'].items():assert rows['first-admission/source/'+name]['sha256']==h
first_controls=[name for name in rows if name.startswith('first-admission/mi-control-') and name.endswith('/stack-result.json')];assert len(first_controls)==1;fc=js(first_controls[0]);assert fc['terminal'] and fc['accepted_observation'] and fc['gdb_exit']==0 and not fc['inferior_remaining'] and len(fc['samples'])==3 and fc['driver_sha256']==q['driver_sha256']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();p=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in s['source_before'].items():
 p.stdin.write((m['source_commit']+':'+name+'\n').encode());p.stdin.flush();header=p.stdout.readline().split();assert len(header)==3 and header[1]==b'blob';raw=p.stdout.read(int(header[2]));assert p.stdout.read(1)==b'\n' and sha(raw)==h,name
p.stdin.close();assert p.wait()==0
assert sha(subprocess.check_output(['git','show','bf9fef87f2c965db3f8c9f6c813a542a15333675:docs/ROADMAP.md'],cwd=repo))==changes['docs/ROADMAP.md']['before']
print('PASS %d records: actual aa647 updated development origin,3903 exact current inputs with one explicitly bound ROADMAP difference, eight bounded stacks/native control and clean reaping inside300/default10. First overstrict documentation preflight retained; no product/factory/bootstrap/paired/full-translator/release acceptance or cause/gain attribution.'%len(rows))
