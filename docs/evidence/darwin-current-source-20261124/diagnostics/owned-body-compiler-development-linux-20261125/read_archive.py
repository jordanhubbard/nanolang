"""I verify the exact owned-scope compiler derivation and retain its terminal boundary."""
from pathlib import Path
import sys,json,hashlib,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'7b3a3ce30c132413fdd8b27af6d387cdc5e6da98d88ae91a07a9f44ce002e465');sha=lambda raw:hashlib.sha256(raw).hexdigest();s=js('development/result.json');a=js('development/admission-result.json');d=js('development/development-native-result.json');origin=js('development/origin-bootstrap.json');r=m['roots']['development']
assert m['source_commit']==s['source_base']==d['source_base']=='00025ff76979b4009b84f10f8cb8f07492c62210'
assert m['development_derivation'] and m['bootstrap_stage2_producer'] and not any(m[k] for k in ['factory_gate_acceptance','new_bootstrap','raw_fixed_point','stage_alias','own_stage2_acceptance','whole_release','historical_cause','performance_claim'])
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and not s['working_overlay'] and not s['overlays'] and d['terminal'] and d['all_guards_match']
assert a['terminal'] and a['accepted'] and a['filesystem']!='nfs' and not a['other_jobs'] and a['free_bytes']>15*1024**3 and all(v['cpu']>=1 and v['share']>=.75 for v in s['cpu_rounds'])
assert d['bootstrap_stage2_producer'] and d['compiler_building_seconds']==1800 and d['compiler_building_shadow_seconds']==30 and d['ordinary_seconds']==10
expected_labels=['development-emit','development-verify','development-dump','development-translate','development-native','development-help'];commands=d['commands'];assert [v['label'] for v in commands]==expected_labels[:len(commands)]
assert m['development_native_pass']==bool(d.get('development_native_accepted'))
if d.get('development_native_accepted'):assert len(commands)==6 and all(v.get('exit')==0 and not v.get('timed_out') for v in commands)
else:assert d.get('exception') or any(v.get('exit')!=0 or v.get('timed_out') for v in commands)
for i,v in enumerate(commands):
 assert v['timeout']==(1800 if i==0 else 600) and v['shadow_seconds']==('30' if i==0 else '10')
 assert 'development/'+v['label']+'.stdout' in rows and 'development/'+v['label']+'.stderr' in rows
 if v.get('timed_out'):assert v['exit']==-9 and v['seconds']>=v['timeout']
assert commands[0]['argv']==[s['producer'],'src_nano/nanoc_v06.nano','--emit-nvm','-o',r+'/products/compiler-development.nvm']
expected_commands=[[r+'/source/bin/nano_vm','--verify-only',r+'/products/compiler-development.nvm'],[r+'/source/bin/nanoisa','dump',r+'/products/compiler-development.nvm'],[r+'/source/bin/nvm2c',r+'/products/compiler-development.nvm','-o',r+'/products/compiler-development.c'],origin['compiler']+['-std=c11']+origin['cflags']+[r+'/products/compiler-development.c',r+'/source/bin/nano_aot_runtime.o','-lm','-rdynamic','-ldl']+origin['ldflags']+['-o',r+'/products/compiler-development.native'],[r+'/products/compiler-development.native','--help']]
for i,v in enumerate(commands[1:]):assert v['argv']==expected_commands[i]
assert d['producer_before']==d['producer_after']==s['producer_sha256']==origin['artifacts']['nanoc_stage2']==rows['development/source/bin/nanoc_emitter_phase']['sha256']=='a518cdc27fdc4905c28621e7f2bf5923a1997d8faae97c281bb999edf6989359'
assert d['origin_receipt_sha256']==rows['development/origin-bootstrap.json']['sha256'] and origin['phase']=='stage2' and origin['raw_stage1_stage2_equal'] and len(origin['steps'])==17 and all(v['exit_code']==0 for v in origin['steps'])
assert d['native_flags']=={k:origin[k] for k in ['compiler','cflags','ldflags']}
def guards(rec):
 for group in ['source','images','tools','hosts']:
  before=rec.get(group+'_before',{});assert before==rec.get(group+'_after',{})
  for name,h in before.items():assert rows[('development/source/'+name) if group in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
 assert rec['source_before']==s['source_after'] and rec['images_before']==s['images_after'] and rec['tools_before']==s['tools_after']
guards(d);assert d['producer_hosts_before']==d['producer_hosts_after']==s['hosts_after']
for name,h in d['producer_hosts_before'].items():assert rows['external/'+name.lstrip('/')]['sha256']==h
for name,v in d['products'].items():assert rows['development/products/'+name]['sha256']==v['sha256'] and rows['development/products/'+name]['bytes']==v['bytes']
n=js('development/native36-result.json') if 'development/native36-result.json' in rows else None;f=js('development/factory-result.json') if 'development/factory-result.json' in rows else None
assert m['native36_not_run']==(n is None) and m['factory_not_run']==(f is None) and m['native36_pass']==bool(n and n.get('accepted_component'))
if n is None:assert not d.get('development_native_accepted')
else:
 assert d.get('development_native_accepted') and n['terminal'] and n['all_guards_match'];guards(n)
 assert n['argv']==['make','-f','Makefile.gnu','test-native-shadows'] and n['development_image'] and not n['own_stage2_acceptance'] and n['ordinary_seconds']==10 and n['timeout']==600
 assert n['producer_before']==n['producer_after']==d['products']['compiler-development.native']['sha256']
 if n.get('accepted_component'):assert n['methods']==36 and n['exit']==0 and not n.get('timed_out') and b'Ran 36 tests' in data('development/native36.stderr') and data('development/native36.stderr').rstrip().endswith(b'OK') and b'skipped=' not in data('development/native36.stderr')
 else:assert n.get('exception') or n.get('exit')!=0
if f is None:assert not (n and n.get('accepted_component')) and not m['factory_native_exists']
else:
 assert n and n.get('accepted_component') and f['terminal'] and f['all_guards_match'];guards(f)
 assert f['producer_sha256']==f['producer_after']==d['products']['compiler-development.native']['sha256'] and f['timeout']==300 and f['ordinary_seconds']==10 and f['diagnostic_only'] and f['development_image'] and not f['factory_gate_acceptance'] and not f['unchanged_full_retry']
 assert f['environment']['NANO_SHADOW_TIMEOUT_SECONDS'] is None and f['argv']==[r+'/products/compiler-development.native',r+'/source/tests/selfhost_shadow_emitter.nano','-o',r+'/factory-products/emitter.native']
 phases=[json.loads(line) for line in data('development/phases.jsonl').splitlines()] if 'development/phases.jsonl' in rows else [];assert f['phases']==phases and f['phase_sources']==f['phase_sources_after']
 for name,h in f['phase_sources'].items():assert rows['development/'+name]['sha256']==h
 assert str(Path(r)/'mark.py').encode() in data('development/phase-recorder')
 for name,v in f['product'].items():assert rows['development/'+name]['sha256']==v['sha256'] and rows['development/'+name]['bytes']==v['bytes']
 for v in f.get('snapshot',[]):assert rows['development/terminal-snapshot/'+str(Path(v['path']).relative_to(r))]['sha256']==v['sha256']
 assert m['factory_native_exists']==('factory-products/emitter.native' in f['product'])
 if f.get('timed_out'):assert f['exit']==-9 and f['deadline_observed_monotonic']-f['start_monotonic']>=300
 else:assert f.get('exit') is not None
cm,cr,cd,cj=read_archive(base.parent/'owned-body-five-route-linux-20261125','a365dd9d0022f445ee19bea6ccc23e43c0384678635db87097d7aedc4b97473b');component=cj('component/cohort-result.json');assert component['terminal'] and component['accepted_component'] and component['all_guards_match'] and len(component['commands'])==132 and len(component['cases'])==24
assert s['component_dependency']['cohort_sha256']==cr['component/cohort-result.json']['sha256'] and s['component_dependency']['runtime_hosts_sha256']==cr['component/runtime-hosts-after.json']['sha256'] and s['source_before']==component['source_before']
bm,br,bd,bj=read_archive(base.parent/'lookup-projection-cold-bootstrap-linux-20261125','eae3850a91bf73dd8e1dc5fac790392f8aaaa7fc63ded80d02180bd7a56e3c2d');assert br['bootstrap/source/bin/nanoc_bootstrap.json']['sha256']==d['origin_receipt_sha256'] and br['bootstrap/source/bin/nanoc_stage2']['sha256']==d['producer_before']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
assert len(s['source_before'])==3903
for name,h in s['source_before'].items():
 proc.stdin.write((m['source_commit']+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob',name;raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n' and sha(raw)==h,name
proc.stdin.close();assert proc.wait()==0
print('PASS %d records: exact3903 source inputs, actual qualified Stage2 origin, prerequisite full132 component, explicit1800/30 derivation and terminal native36/factory300/default10 boundary with complete streams/products/guards. Product outcomes follow manifest flags; no new fixed point/own-Stage2/emitter12/paired/pure-translator/release or cause/gain acceptance.'%len(rows))
