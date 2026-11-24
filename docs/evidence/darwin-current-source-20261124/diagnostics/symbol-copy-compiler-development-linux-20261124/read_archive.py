"""I verify my changed development compiler without granting Stage2 acceptance."""
from pathlib import Path
import json,hashlib,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'3dccdc51bd3a92e222d69f005c52ed038418d5af63e3ef4da35a0d2edf29de68');sha=lambda b:hashlib.sha256(b).hexdigest()
s=js('development/result.json');a=js('development/admission-result.json');seed=js('development/compiler-seed-result.json');native=js('development/development-native-result.json');factory=js('development/factory-result.json');cohort=js('development/native36-result.json')
assert m['source_commit']==s['source_base']=='8da979937c9579c922ff515a8100ad9eaba7ab02'
assert m['development_image'] and m['diagnostic_only'] and not m['overlay_files']
assert not any(m[k] for k in ['product_acceptance','historical_cause','performance_claim','new_bootstrap','own_stage2_acceptance','whole_release'])
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and not s['working_overlay'] and not s['overlays'] and not s['stage_alias']
assert a['terminal'] and a['accepted'] and a['filesystem']!='nfs' and not a['other_jobs'] and a['free_bytes']>15*1024**3
assert all(v['cpu']>=1 and v['share']>=.75 for v in s['cpu_rounds'])
assert seed['terminal'] and seed['development_seed_accepted'] and seed['exit']==0 and seed['timeout']==600 and seed['ordinary_seconds']==10 and not seed['overlays']
assert seed['argv'][0].endswith('/bin/nano_virt') and seed['argv'][1:4]==['src_nano/nanoc_v06.nano','--emit-nvm','-o']
assert native['terminal'] and native['development_native_accepted'] and len(native['commands'])==5 and native['ordinary_seconds']==10
assert [c['label'] for c in native['commands']]==['development-verify','development-dump','development-translate','development-native','development-help']
assert all(c['exit']==0 and c['timeout']==600 for c in native['commands'])
assert native['native_flags']==dict(compiler=['/usr/bin/cc'],cflags=['-O1'],ldflags=['-lm','-lcrypto','-lffi','-rdynamic'])
assert seed['product']==native['input_module']==native['products']['compiler-development-seed.nvm']
oldm,oldrows,olddata,oldjs=read_archive(base.parent/'shadow-helper-cold-bootstrap-linux-20261124','c8c6cfe80993b0610a07576d13338b282a402d87fa6c636231d7680f923300e5')
qualified=oldjs('bootstrap/bootstrap-result.json');receipt=oldjs('bootstrap/source/bin/nanoc_bootstrap.json')
assert all(native['native_flags'][k]==receipt[k] for k in ['compiler','cflags','ldflags'])
assert s['tools_before']==qualified['tools_after'] and s['producer_sha256']==receipt['artifacts']['nanoc_stage2'] and s['hosts_before']==receipt['hosts']
for name,h in s['images_before'].items():assert h==oldrows['bootstrap/source/'+name]['sha256']
for name,h in qualified['source_after'].items():
 if name.endswith(('.c','.h','.inc')):assert rows['development/source/'+name]['sha256']==h
for name,v in native['products'].items():assert rows['development/products/'+name]['sha256']==v['sha256'] and rows['development/products/'+name]['bytes']==v['bytes']
assert native['products']['compiler-development.native']['sha256']=='c74f8966f810a7aaa49bd1adcdbe81d86c018a53c677f04676c024d644837fd4'
assert factory['terminal'] and factory['diagnostic_only'] and factory['development_image'] and not factory['factory_gate_acceptance'] and not factory['unchanged_full_retry'] and not factory['new_bootstrap'] and not factory['whole_release']
assert factory['timeout']==300 and factory['ordinary_seconds']==10 and factory['argv'][0]==factory['producer'] and factory['argv'][1].endswith('/tests/selfhost_shadow_emitter.nano')
assert factory['producer_sha256']==factory['producer_after']==native['products']['compiler-development.native']['sha256']
assert factory['phase_sources']==factory['phase_sources_after']
for name,h in factory['phase_sources'].items():assert rows['development/'+name]['sha256']==h
assert factory['environment']['NANO_COMPILER_PHASE_RECORDER'].endswith('/phase-recorder')
assert all(factory['environment'][k] is None for k in ['NANO_SHADOW_TIMEOUT_SECONDS','NANO_CC','NANO_CFLAGS','NANO_LDFLAGS','CMD_TIMEOUT'])
assert factory['phases']==[json.loads(line) for line in data('development/phases.jsonl').splitlines()]
assert all(p['parent_pid']>0 for p in factory['phases'])
assert [p['phase'] for p in factory['phases']]==['inputs-start','imports-done','lex-done','parse-done','checked-emit-start','typecheck-start','typecheck-done','program-lowering-start']
assert factory['phases'][-1]['monotonic']>factory['phases'][-2]['monotonic']
assert rows['development/stdout']['bytes']==0 and rows['development/stderr']['bytes']==0
for v in factory.get('snapshot',[]):assert v['stable_stat'] and rows['development/terminal-snapshot/'+v['path'].split('/nanolang-symbol-copy-development-20261124-xipm8nds/',1)[1]]['sha256']==v['sha256']
if factory.get('timed_out'):
 assert factory['exit']==-9 and factory['seconds']>=300 and not factory['product']
else:assert factory['exit']==0 and factory['product']
assert cohort['terminal'] and cohort['accepted_component'] and cohort['methods']==36 and cohort['ordinary_seconds']==10 and not cohort['own_stage2_acceptance'] and not cohort['new_bootstrap']
assert len(cohort['commands'])==1 and cohort['commands'][0]['argv']==['make','-f','Makefile.gnu','test-native-shadows'] and cohort['commands'][0]['exit']==0 and cohort['commands'][0]['budget']==600
assert cohort['producer_before']==cohort['producer_after']==factory['producer_sha256']
log=data('development/native36-command-1.stderr');assert b'Ran 36 tests' in log and log.rstrip().endswith(b'OK') and b'skipped=' not in log
for receipt in [s,seed,native,factory,cohort]:
 assert receipt['all_guards_match']
 for group in ['source','images','tools','hosts']:
  if group+'_before' not in receipt:
   assert receipt is seed and group=='hosts';continue
  assert receipt[group+'_before']==receipt[group+'_after']
  for name,h in receipt[group+'_before'].items():
   key='development/source/'+name if group in ['source','images'] else 'external/'+name.lstrip('/')
   assert rows[key]['sha256']==h,(group,name)
 assert receipt['source_before']==s['source_before'] and receipt['images_before']==s['images_before'] and receipt['tools_before']==s['tools_before']
assert not any(name.startswith('development/source/bin/nanoc_stage2') or name=='development/source/bin/nanoc_bootstrap.json' for name in rows)
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();p=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in {**s['source_before'],'Makefile.gnu':rows['development/source/Makefile.gnu']['sha256']}.items():
 p.stdin.write((m['source_commit']+':'+name+'\n').encode());p.stdin.flush();header=p.stdout.readline().split();assert len(header)==3 and header[1]==b'blob';raw=p.stdout.read(int(header[2]));assert p.stdout.read(1)==b'\n' and sha(raw)==h,name
p.stdin.close();assert p.wait()==0
print('PASS %d records: exact-source C-seed development compiler derivation, factory terminal exit%s at300/default10, original36 development native methods, retained streams/markers/products and source/tool/host guards. No own-Stage2/new bootstrap/emitter12/paired/full-pure-translator/release acceptance or cause/gain claim.'%(len(rows),factory['exit']))
