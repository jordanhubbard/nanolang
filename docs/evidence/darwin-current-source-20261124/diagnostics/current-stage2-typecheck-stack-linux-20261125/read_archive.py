"""I verify my actual bootstrap Stage2 path observation, not consumer acceptance."""
from pathlib import Path
import sys,json,hashlib,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'1a2bedaf8f4bb6c00511092b30d4389ec0207386e66fbdb81de9bff90e728550');sha=lambda b:hashlib.sha256(b).hexdigest();s=js('stack/result.json');a=js('stack/admission-result.json');q=js('stack/stack-result.json')
assert m['source_commit']==s['source_base']==q['source_pin']=='9c12c920d45677e98b3d08665d2ae58501d96ec3'
assert m['diagnostic_only'] and not m['development_image'] and m['bootstrap_stage2_image'] and not any(m[k] for k in ['product_acceptance','historical_cause','performance_claim','new_bootstrap','own_stage2_acceptance','whole_release','original_consumer_acceptance'])
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and not s['working_overlay'] and not s['stage_alias'] and not s['overlays']
assert a['terminal'] and a['accepted'] and not a['other_jobs'] and a['filesystem']!='nfs' and a['free_bytes']>15*1024**3 and all(v['cpu']>=1 and v['share']>=.75 for v in s['cpu_rounds'])
assert q['terminal'] and q['accepted_observation'] and q['all_guards_match'] and q['gdb_exit']==0 and not q['inferior_remaining']
assert q['ordinary_seconds']==10 and q['outer_seconds']==300 and q['sample_seconds']==8 and not q['development_image'] and q['bootstrap_stage2_image'] and not q['original_consumer_acceptance'] and q['wait_phase']=='typecheck-start'
assert len(q['samples'])==8 and 8<=q['sample_elapsed']<9 and q['inferior_killed_monotonic']-q['start_monotonic']<300 and not q.get('outer_deadline_fired')
count=lambda suffix:sum(any(name.endswith(suffix) for name in sample['functions']) for sample in q['samples'])
assert count('_env_copy_symbols')==3 and count('_env_lookup')==1 and count('__memcpy_sve')==6 and count('_purity_node')==1 and count('nroot_add')==1 and count('_merge_struct_metadata')==1
for sample in q['samples']:assert sample['stack'].encode() in data('stack/gdb-mi.log')
assert q['inferior_argv']==q['observed_argv']==[s['producer'],str(Path(m['roots']['stack'])/'source/tests/selfhost_shadow_emitter.nano'),'-o',str(Path(m['roots']['stack'])/'products/profile.native')]
for group in ['source','images','tools','hosts']:
 assert q[group+'_after']==s[group+'_before']==s[group+'_after']
 for name,h in s[group+'_before'].items():assert rows['stack/source/'+name if group in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
assert q['producer_sha256']==q['producer_after']==s['producer_sha256']==rows['stack/source/bin/nanoc_emitter_phase']['sha256']=='2eb02fd33d8cc8d8ca5f2684b4399d55540b648a1f4eb2570e6ead118e03a63d'
assert q['driver_sha256']==q['driver_after']==rows['stack/sample.py']['sha256'] and q['debugger_sha256']==q['debugger_after']==rows['external/'+q['debugger'].lstrip('/')]['sha256']
assert q['phase_sources_before']==q['phase_sources_after']
for name,h in q['phase_sources_before'].items():assert rows['stack/'+name]['sha256']==h
assert str(Path(m['roots']['stack'])/'mark.py').encode() in data('stack/phase-recorder')
phases=[json.loads(line) for line in data('stack/phases.jsonl').splitlines()];assert phases==q['phases'] and [r['phase'] for r in phases]==['inputs-start','imports-done','lex-done','parse-done','checked-emit-start','typecheck-start']
assert all(phases[i]['monotonic']<phases[i+1]['monotonic'] for i in range(len(phases)-1)) and phases[-1]['monotonic']<=q['sample_start_monotonic']<q['samples'][0]['monotonic']
controls=[name for name in rows if name.startswith('stack/mi-control-') and name.endswith('/stack-result.json')];assert len(controls)==1;c=js(controls[0]);assert c['terminal'] and c['accepted_observation'] and c['gdb_exit']==0 and not c['inferior_remaining'] and len(c['samples'])==3 and c['driver_sha256']==q['driver_sha256']
assert not any(name.endswith('/source/bin/nanoc_stage2') or name.endswith('/source/bin/nanoc_bootstrap.json') or name=='stack/products/profile.native' for name in rows)
pm,pr,pd,pj=read_archive(base.parent/'symbol-copy-cold-bootstrap-linux-20261124','abaa42d83391abc7656f41b9b435c6a1f1eb4f34c6c17555692cf51eed1ef994');b=pj('bootstrap/bootstrap-result.json');installed=pj('bootstrap/source/bin/nanoc_bootstrap.json');consumer=pj('bootstrap/native-gates-result.json')
assert b['terminal'] and b['original_make_pass'] and b['exit']==0 and installed['phase']=='stage2' and installed['raw_stage1_stage2_equal'] and all(row['exit_code']==0 for row in installed['steps'])
assert s['bootstrap_source_pin']==b['source_pin']=='8b2f65d5ea8fbc8a81825808c0b101111db58553' and s['bootstrap_source_inputs']==b['source_after']==s['source_before'] and len(s['source_before'])==3903
assert s['bootstrap_result_sha256']==pr['bootstrap/bootstrap-result.json']['sha256'] and s['bootstrap_receipt_sha256']==pr['bootstrap/source/bin/nanoc_bootstrap.json']['sha256'] and s['native_consumer_receipt_sha256']==pr['bootstrap/native-gates-result.json']['sha256']
assert not s['native_consumer_accepted'] and not consumer['accepted'] and s['hosts_before']==installed['hosts'] and s['producer_sha256']==installed['artifacts']['nanoc_stage2']==pr['bootstrap/source/bin/nanoc_stage2']['sha256']
for name,h in s['images_before'].items():assert h==pr['bootstrap/source/'+name]['sha256']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();p=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in s['source_before'].items():
 p.stdin.write((m['source_commit']+':'+name+'\n').encode());p.stdin.flush();header=p.stdout.readline().split();assert len(header)==3 and header[1]==b'blob';raw=p.stdout.read(int(header[2]));assert p.stdout.read(1)==b'\n' and sha(raw)==h,name
p.stdin.close();assert p.wait()==0
print('PASS %d records: actual8b2 bootstrap Stage2,3903 exact source inputs, eight bounded samples/three env-copy/one lookup/six memcpy plus layout/purity/root/metadata paths, native control, independent300/default10 and clean reaping. Original consumer acceptance remains FALSE; no whole-run cause/gain, new repair, paired, full pure-translator or release claim.'%len(rows))
