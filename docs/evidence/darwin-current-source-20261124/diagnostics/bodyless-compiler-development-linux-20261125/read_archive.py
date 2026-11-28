"""I verify the explicit native-producer development role and original consumer limits."""
from pathlib import Path
import sys,hashlib,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'e6a8aa9cd29b1bcb7623c02e7bb2514e8b05b126bb2646458018f13e605499d8');sha=lambda raw:hashlib.sha256(raw).hexdigest()
s=js('development/result.json');a=js('development/admission-result.json');d=js('development/development-native-result.json');n=js('development/native36-result.json');f=js('development/factory-result.json');origin=js('development/origin-bootstrap.json')
assert m['source_commit']==s['source_base']==d['source_base']=='bf9fef87f2c965db3f8c9f6c813a542a15333675'
assert m['development_derivation'] and m['bootstrap_stage2_producer'] and m['development_native_pass'] and m['native36_pass']
assert not any(m[k] for k in ['factory_gate_acceptance','new_bootstrap','raw_fixed_point','stage_alias','own_stage2_acceptance','whole_release','historical_cause','performance_claim'])
assert all(v['terminal'] and v['all_guards_match'] for v in [s,d,n,f]) and s['admitted_setup'] and not s['working_overlay']
assert a['terminal'] and a['accepted'] and a['filesystem']!='nfs' and not a['other_jobs'] and a['free_bytes']>15*1024**3
assert all(v['cpu']>=1 and v['share']>=.75 for v in s['cpu_rounds'])
assert d['development_native_accepted'] and d['bootstrap_stage2_producer'] and d['compiler_building_seconds']==1800 and d['compiler_building_shadow_seconds']==30 and d['ordinary_seconds']==10
assert [v['label'] for v in d['commands']]==['development-emit','development-verify','development-dump','development-translate','development-native','development-help']
for i,v in enumerate(d['commands']):assert v['exit']==0 and not v.get('timed_out') and v['timeout']==(1800 if i==0 else 600) and v['shadow_seconds']==('30' if i==0 else '10')
assert d['commands'][0]['argv'][1:]==['src_nano/nanoc_v06.nano','--emit-nvm','-o',m['roots']['development']+'/products/compiler-development.nvm']
assert d['producer_before']==d['producer_after']==s['producer_sha256']==origin['artifacts']['nanoc_stage2']=='2eb02fd33d8cc8d8ca5f2684b4399d55540b648a1f4eb2570e6ead118e03a63d'
assert d['origin_receipt_sha256']==rows['development/origin-bootstrap.json']['sha256'] and origin['phase']=='stage2' and origin['raw_stage1_stage2_equal']
assert d['native_flags']=={k:origin[k] for k in ['compiler','cflags','ldflags']}
command=d['commands'][4]['argv'];r=m['roots']['development'];assert command==origin['compiler']+['-std=c11']+origin['cflags']+[r+'/products/compiler-development.c',r+'/source/bin/nano_aot_runtime.o','-lm','-rdynamic','-ldl']+origin['ldflags']+['-o',r+'/products/compiler-development.native']
for label,rec in [('development',d),('native36',n),('factory',f)]:
 for group in ['source','images','tools','hosts']:
  assert rec[group+'_before']==rec[group+'_after']
  for name,h in rec[group+'_before'].items():assert rows[('development/source/'+name) if group in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
 assert rec['source_before']==s['source_after'] and rec['images_before']==s['images_after'] and rec['tools_before']==s['tools_after']
assert d['producer_hosts_before']==d['producer_hosts_after']==s['hosts_after']
for name,v in d['products'].items():assert rows['development/products/'+name]['sha256']==v['sha256'] and rows['development/products/'+name]['bytes']==v['bytes']
assert n['accepted_component'] and n['methods']==36 and n['ordinary_seconds']==10 and n['timeout']==600 and n['exit']==0 and not n.get('timed_out')
assert b'skipped=' not in data('development/native36.stderr')
assert n['argv']==['make','-f','Makefile.gnu','test-native-shadows'] and n['development_image'] and not n['own_stage2_acceptance']
assert b'Ran 36 tests' in data('development/native36.stderr') and data('development/native36.stderr').rstrip().endswith(b'OK')
assert n['producer_before']==n['producer_after']==f['producer_sha256']==f['producer_after']==d['products']['compiler-development.native']['sha256']
assert f['timeout']==300 and f['ordinary_seconds']==10 and f['diagnostic_only'] and f['development_image'] and not f['factory_gate_acceptance'] and not f['unchanged_full_retry']
assert f['environment']['NANO_SHADOW_TIMEOUT_SECONDS'] is None and f['argv']==[r+'/products/compiler-development.native',r+'/source/tests/selfhost_shadow_emitter.nano','-o',r+'/factory-products/emitter.native']
assert f['phases']==[__import__('json').loads(line) for line in data('development/phases.jsonl').decode().splitlines()]
assert f['phases'][-1]['phase']=='typecheck-start' and f['timed_out'] and not f['product']
assert f['deadline_observed_monotonic']-f['start_monotonic']>=300
assert f['phase_sources']==f['phase_sources_after']
for name,h in f['phase_sources'].items():assert rows['development/'+name]['sha256']==h
for name,v in f['product'].items():assert rows['development/'+name]['sha256']==v['sha256'] and rows['development/'+name]['bytes']==v['bytes']
assert m['factory_native_exists']==('factory-products/emitter.native' in f['product'])
if f.get('timed_out'):assert f['exit']==-9
else:assert f['exit']==0 and m['factory_native_exists']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in s['source_before'].items():
 proc.stdin.write((m['source_commit']+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob',name;raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n' and sha(raw)==h,name
proc.stdin.close();assert proc.wait()==0
print('PASS %d records: exact source, actual bootstrap Stage2 producer, explicit1800/30 development derivation and unchanged native flags; original native36 and factory300/default10 terminal/complete streams/guards retained. No new raw fixed point, own-Stage2/emitter12, paired, pure translator or release acceptance; no cause or gain attribution.'%len(rows))
