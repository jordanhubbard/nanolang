"""I bind current phase observations to the actual exact-source bootstrap Stage2."""
from pathlib import Path
import sys,json,hashlib,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'3476f8d279e25d1333870a893a391073db1fcee1f85bcd7ec24c92722292dd88');sha=lambda b:hashlib.sha256(b).hexdigest();s=js('observation/result.json');a=js('observation/admission-result.json');q=js('observation/factory-result.json')
assert m['source_commit']==s['source_base']==q['source_pin']=='8fa1025477ca2b7db9266f1019b7192de20175d1'
assert m['diagnostic_only'] and m['bootstrap_stage2_image'] and not any(m[k] for k in ['development_image','factory_gate_acceptance','new_bootstrap','whole_release','historical_cause','performance_claim'])
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and not s['working_overlay'] and not s['stage_alias'] and not s['overlays']
assert a['terminal'] and a['accepted'] and not a['other_jobs'] and a['filesystem']!='nfs' and a['free_bytes']>15*1024**3 and all(c['cpu']>=1 and c['share']>=.75 for c in s['cpu_rounds'])
assert q['terminal'] and q['all_guards_match'] and q['diagnostic_only'] and q['bootstrap_stage2_image'] and not q['development_image'] and not q['factory_gate_acceptance'] and not q['unchanged_full_retry']
assert q['timeout']==300 and q['ordinary_seconds']==10 and q['environment']['NANO_SHADOW_TIMEOUT_SECONDS'] is None and q['environment']['NANO_CC'] is None and q['environment']['NANO_CFLAGS'] is None and q['environment']['CMD_TIMEOUT'] is None
assert q['argv']==[s['producer'],str(Path(m['roots']['observation'])/'source/tests/selfhost_shadow_emitter.nano'),'-o',str(Path(m['roots']['observation'])/'factory-products/emitter.native')]
for g in ['source','images','tools','hosts']:
 assert q[g+'_before']==q[g+'_after']==s[g+'_before']==s[g+'_after']
 for name,h in q[g+'_before'].items():assert rows['observation/source/'+name if g in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
assert q['producer_sha256']==q['producer_after']==s['producer_sha256']==rows['observation/source/bin/nanoc_emitter_phase']['sha256']=='7f431c60c2f344f877f031be669ac64d8b17bd198f218c4fe0d15754795a6695'
assert q['phase_sources']==q['phase_sources_after']
for name,h in q['phase_sources'].items():assert rows['observation/'+name]['sha256']==h
assert str(Path(m['roots']['observation'])/'mark.py').encode() in data('observation/phase-recorder')
assert q['phases']==[json.loads(line) for line in data('observation/phases.jsonl').splitlines()]
assert all(q['phases'][i]['monotonic']<q['phases'][i+1]['monotonic'] for i in range(len(q['phases'])-1))
if q.get('timed_out'):assert q['exit']==-9 and q['deadline_observed_monotonic']-q['start_monotonic']>=300
else:assert q['exit']==0
for name,v in q['product'].items():assert rows['observation/'+name]['sha256']==v['sha256'] and rows['observation/'+name]['bytes']==v['bytes']
for r in q.get('snapshot',[]):assert rows['observation/terminal-snapshot/'+str(Path(r['path']).relative_to(m['roots']['observation']))]['sha256']==r['sha256']
pm,pr,pd,pj=read_archive(base.parent/'emission-function-index-cold-bootstrap-linux-20261126','53b44bb6e4b13a351b12bd633d0c57df8211b9771cf83d2e568a9c8b0bd268b3');b=pj('bootstrap/bootstrap-result.json');installed=pj('bootstrap/source/bin/nanoc_bootstrap.json');consumer=pj('bootstrap/native-gates-result.json')
assert b['terminal'] and b['original_make_pass'] and b['exit']==0 and installed['phase']=='stage2' and installed['raw_stage1_stage2_equal'] and len(installed['steps'])==17 and all(r['exit_code']==0 for r in installed['steps'])
assert s['bootstrap_source_pin']==b['source_pin']==m['source_commit'] and s['bootstrap_source_inputs']==b['source_after']==s['source_before'] and len(s['source_before'])==3971
assert s['bootstrap_result_sha256']==pr['bootstrap/bootstrap-result.json']['sha256'] and s['bootstrap_receipt_sha256']==pr['bootstrap/source/bin/nanoc_bootstrap.json']['sha256'] and s['native_consumer_receipt_sha256']==pr['bootstrap/native-gates-result.json']['sha256']
assert not s['native_consumer_accepted'] and not consumer['accepted'] and s['hosts_before']==installed['hosts'] and s['producer_sha256']==installed['artifacts']['nanoc_stage2']==pr['bootstrap/source/bin/nanoc_stage2']['sha256']
for name,h in s['images_before'].items():assert pr['bootstrap/source/'+name]['sha256']==h
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();p=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in s['source_before'].items():
 p.stdin.write((m['source_commit']+':'+name+'\n').encode());p.stdin.flush();head=p.stdout.readline().split();assert len(head)==3 and head[1]==b'blob',name;raw=p.stdout.read(int(head[2]));assert p.stdout.read(1)==b'\n' and sha(raw)==h,name
p.stdin.close();assert p.wait()==0
assert not any(name.endswith('/source/bin/nanoc_stage2') or name.endswith('/source/bin/nanoc_bootstrap.json') for name in rows)
assert q['subreaper_enabled'] and q['observer_tools_before']==q['observer_tools_after']
for name,h in q['observer_tools_before'].items():assert rows['external/'+name.lstrip('/')]['sha256']==h
for name in ['post_run_cleanup','final_cleanup']+(['timeout_cleanup'] if q.get('timed_out') else []):
 cl=q[name];assert cl['terminal'] and cl['owned_only'] and cl['pidfd_identity'] and not cl['remaining']
 for r in cl['records']:assert r['waitpid'][0]==r['pid'] and r['ppid']>0 and r['starttime'].isdigit()
control=js('observation/cleanup-control-v2.json');assert control['terminal'] and control['accepted'] and not control['cleanup']['remaining'] and control['cleanup']['owned_only'] and control['cleanup']['pidfd_identity']
assert control['helper_sha256']==rows['observation/owned_cleanup.py']['sha256'] and control['control_sha256']==rows['observation/cleanup_control_v2.py']['sha256']
assert any(r['pid']==control['child']['pid'] and r['waitpid']==[r['pid'],9] for r in control['cleanup']['records'])
assert s['bootstrap_publication']=='0581ef3a1448b91f88a6d7334ce9ff7a343f4699' and s['bootstrap_manifest_sha256']=='53b44bb6e4b13a351b12bd633d0c57df8211b9771cf83d2e568a9c8b0bd268b3' and s['current_compiler_closure_matches']
closure=['src','modules','stdlib','src_nano',':(exclude)src_nano/nvm2c/**','tests/selfhost_shadow_emitter.nano']
assert subprocess.check_output(['git','diff',s['source_base'],s['current_head'],'--',*closure],cwd=repo)==b''
subprocess.run(['git','merge-base','--is-ancestor',s['bootstrap_publication'],s['current_head']],cwd=repo,check=True)
origin=base.parent/'emission-function-index-cold-bootstrap-linux-20261126'
for name in ['manifest.json','read_archive.py']:
 path=str((origin/name).relative_to(repo));assert subprocess.check_output(['git','show',s['bootstrap_publication']+':'+path],cwd=repo)==(origin/name).read_bytes()
refusal=js('collector-source/live-refusal-result.json');assert refusal['terminal'] and refusal['accepted'] and refusal['collector_exit']==1 and not refusal['archive_created'] and not js('collector-source/live-factory-result.json')['terminal']
assert refusal['live_result_sha256']==rows['collector-source/live-factory-result.json']['sha256'] and b'AssertionError' in data('collector-source/live-refusal.stderr')
helperbase=base.parent/'owned-body-lowering-stack-linux-20261125'
assert sha((helperbase/'manifest.json').read_bytes())=='2cf5d0d94a865c6749e5f30645f9384a300d782938e3d1e93c1751ad6d51dd03'
helperrows={r['path']:r for r in json.loads((helperbase/'manifest.json').read_bytes())['files']}
for name in ['owned_cleanup.py','cleanup_control_v2.py']:assert helperrows['observation/'+name]['sha256']==rows['observation/'+name]['sha256']
print('PASS %d records: actual8fa bootstrap Stage2 origin, exact3971 source inputs, original300/default10 diagnostic phase terminal/complete streams/snapshots/guards. Original emitter acceptance remains FALSE; no cause/gain, new repair/bootstrap, paired, full translator or release acceptance.'%len(rows))
