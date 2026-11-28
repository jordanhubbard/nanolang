#!/usr/bin/env python3
"""I verify my collector cost, ownership bounds and disclosed recovery limits."""
from pathlib import Path
import gzip,hashlib,json,re,subprocess
base=Path(__file__).resolve().parent
m=json.loads((base/'manifest.json').read_text());h=lambda b:hashlib.sha256(b).hexdigest()
rows={};cache={};images=set()
for r in m['files']:
 z=(base/r['artifact']).read_bytes();assert h(z)==r['compressed_sha256'],r['path']
 b=cache.get(r['sha256'])
 if b is None:
  b=gzip.decompress(z);assert len(b)==r['bytes'] and h(b)==r['sha256'];cache[r['sha256']]=b
 assert r['path'] not in rows;rows[r['path']]=b
 if b.startswith(b'\x7fELF'):
  assert int.from_bytes(b[18:20],'little')==183,r['path']
  if int.from_bytes(b[16:18],'little') in (2,3):images.add(h(b))
assert {p.resolve() for p in (base/'objects').glob('*.gz')}=={(base/r['artifact']).resolve() for r in m['files'] if r['artifact'].startswith('objects/')}
priorpath=base/m['base_manifest']['path'];assert h(priorpath.read_bytes())==m['base_manifest']['sha256'];prior=json.loads(priorpath.read_text())
def prior_record(name):
 r=next(r for r in prior['files'] if r['path']==name);z=(priorpath.parent/r['artifact']).read_bytes();assert h(z)==r['compressed_sha256'];b=gzip.decompress(z);assert h(b)==r['sha256'];return b
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE);new={}
for r in m['git_sources']:
 proc.stdin.write((m['source_pin']+':'+r['path']+'\n').encode());proc.stdin.flush();hdr=proc.stdout.readline().split();assert hdr[1]==b'blob';b=proc.stdout.read(int(hdr[2]));assert proc.stdout.read(1)==b'\n';assert b==rows['source/'+r['path']] and h(b)==r['sha256'];new[r['path']]=r['sha256']
proc.stdin.close();assert proc.wait()==0
old={r['path']:r['sha256'] for r in prior['git_sources']}
assert {p for p,v in new.items() if old.get(p)!=v}=={'src/eval.c','src/eval/carrier_runtime.c','src/eval/carrier_runtime.h','src/eval/carrier_values.c','src/eval/carrier_values.h','Makefile.gnu','docs/ROADMAP.md','docs/EVAL_CARRIER_OWNERSHIP.md','tests/test_eval_carrier_root_selection.c','tests/test_eval_carrier_field_metadata.c','tests/test_eval_carrier_field_walk.c','tests/test_eval_carrier_pressure.c'}
# I preserve every previously committed source shadow and regression assertion.
for p,v in old.items():
 if p.startswith(('src_nano/','modules/','stdlib/','tests/')):assert new[p]==v,p
final='carrier-live-pressure-20261123'
def raw(folder,name):return rows['journal/'+folder+'/'+name]
def js(folder,name):return json.loads(raw(folder,name+'.json'))
def provider(path):
 marker='/.test_output/'
 key='journal/'+path.split(marker,1)[1] if marker in path else ''
 return rows[key] if key in rows else rows['provider/'+path]
for p,v in js(final,'qualified-source').items():assert new[p]==v,p
repair=js(final,'repair');assert set(repair['modes'])=={'normal','san','production'}
for name,count in [('repair',24),('field-controls',18),('walk-controls',6),('neighbors',79),('controls',42),('extra',18),('policy',2)]:
 q=js(final,name);commands=q['commands'] if name=='repair' else q
 assert len(commands)==count and all(c['exit']==0 for c in commands),name
for p,v in repair['providers'].items():assert h(provider(p))==v,p
for mode in ['normal','san','production']:
 for subject in ['env','eval','repl','carrier_heap','carrier_runtime','carrier_values']:
  a=next(c['args'] for c in repair['commands'] if c['label']==mode+'-compile-'+subject)
  assert '-Werror' in a and ('-O3' if mode=='production' else '-O1') in a
  if mode=='san':assert '-fsanitize=address,undefined' in a
 for subject,n in [('default34',34),('private41',41),('current-private41',41)]:
  err=raw(final,mode+'-'+subject+'.stderr');assert f'Ran {n} tests'.encode() in err and b'\nOK\n' in err
  assert b'AddressSanitizer:' not in err and b'runtime error:' not in err
for mode in ['normal','san']:
 assert raw(final,mode+'-eval-run.stdout').count(b'  Testing ')==134
 assert raw(final,mode+'-checker-run.stdout').count(b'  Testing ')==88
 active=raw(final,mode+'-active-run.stdout');match=re.search(rb'after64=(\d+) after128=(\d+) peak=(\d+)',active);assert match
 a,b,peak=map(int,match.groups());assert a==b and peak<=32
 assert b'delta=0 calls=128.' in raw(final,mode+'-native-bytes-run.stdout')
controls=js(final,'final-controls');assert len(controls)==15
assert all(c['exit']==(1 if c['label']=='san-default-admission' else 0) for c in controls)
for mode in ['normal','san','production']:
 q=next(c for c in controls if c['label']==mode+'-final-make');assert len(q['products'])==8
 for p,v in q['products'].items():assert h(provider(p))==v
 assert b'AddressSanitizer:' not in raw(final,mode+'-final-make.stderr')
 assert b'Ran 7 tests' in raw(final,mode+'-default-admission.stderr')
err=raw(final,'san-default-admission.stderr');assert b'FAILED (failures=3)' in err
assert err.count(b'168 byte(s) leaked in 6 allocation(s)')==1 and err.count(b'172 byte(s) leaked in 6 allocation(s)')==2
reference=json.loads(prior_record('journal/ordinary-qualified.json'))
reference_events=[]
for line in prior_record('journal/ordinary-native/stderr').splitlines():
 try:reference_events.append(json.loads(line))
 except ValueError:pass
def selection(events,root):
 return [(e['name'],e['source'].removeprefix(root+'/'),e['line']) for e in events if e.get('event')=='shadow-start']
expected_selection=selection(reference_events,str(Path(reference['args'][1]).parent.parent))
assert len(expected_selection)==1306
for kind in ['private','ordinary']:
 q=js(final,kind+'-qualified');assert (q['exit'],q['starts'],q['completions'])==(0,1306,1306)
 assert q['timeout']==600 and q['shadow_timeout']=='default10' and not q['timed_out']
 assert q['before']==q['after'] and set(q['before'])==set(reference['before']) and len(q['before'])==1340
 for p,v in q['after'].items():assert new[p]==v,p
 for p,v in q['providers'].items():assert h(provider(p))==v,p
 assert h(provider(q['args'][0]))==q['image_sha256']
 events=[]
 for line in raw(final,kind+'-native/stderr').splitlines():
  try:events.append(json.loads(line))
  except ValueError:pass
 starts=[e for e in events if e.get('event')=='shadow-start'];ends=[e for e in events if e.get('event')=='shadow-complete']
 assert len(starts)==len(ends)==1306 and all(e['failures']==0 for e in ends)
 assert selection(events,str(Path(q['args'][1]).parent.parent))==expected_selection
 assert [(e['name'],e['source'],e['line']) for e in starts]==[(e['name'],e['source'],e['line']) for e in ends]
 assert [e for e in events if e.get('event')=='cseed-shadow-phase']==q['phases']
 status=q['phases'][-1];assert status['completed'] and status['reaped'] and not status['parent_timeout']
 assert status['signal']==status['exit_code']==0 and status['deadline_seconds']==10
 assert h(raw(final,kind+'-native/seed'))==q['product']['sha256'] and len(raw(final,kind+'-native/seed'))==q['product']['bytes']
 helpname='seed-help' if kind=='private' else 'ordinary-seed-help';assert js(final,helpname)['exit']==0 and js(final,helpname)['sha256']==q['product']['sha256']
for subject,n in [('lexical',10),('record-payload',7)]:
 q=js(final,subject+'-qualified');assert (q['tests'],q['failures'],q['errors'],q['success'])==(n,0,0,True)
assert js(final,'final-image-guards')==json.loads(rows['baseline/image-guards.json'])
closure=js(final,'product-closure');assert closure==json.loads(prior_record('journal/product-closure.json')) and len(closure)==9
for p,v in closure.items():assert h(provider(p))==v
recovery='provider-recovery-20261123';restored=js(recovery,'restored');assert restored['restored']==1957 and len(restored['blobs'])==791 and not restored['nas_action']
for origin,r in {**restored['mapping'],**js(recovery,'extras')}.items():assert h(provider(r['path']))==r['sha256'],origin
assert js(recovery,'baseline-guards')['bad']==[] and js(recovery,'baseline-guards')['checked']==324
inventory=js(recovery,'prototype-inventory');assert len(inventory['unrecovered'])==104 and inventory['deletion_cause'] is None
assert js(recovery,'retained-storage')['mount']['filesystems'][0]['fstype']=='ext4'
# My earlier 64-budget full pass remains blocked by its original peak bound.
oldtrial=js('carrier-field-index-20261123','private-qualified');assert (oldtrial['exit'],oldtrial['completions'])==(0,1306)
assert b'FAILED: peak_live<=32' in raw('carrier-field-index-20261123','normal-active-run.stdout')
assert next(c for c in js('carrier-field-index-20261123','neighbors') if c['label']=='normal-active-run')['exit']==1
for folder,n in [('carrier-walk-index-retained-20261123',168),('carrier-pressure-20261123',1076),('carrier-typed-fields-20261123',1061),('carrier-leaf-arrays-20261123',809)]:
 q=js(folder,'private-qualified');assert q['exit']==1 and q['completions']==n and q['product'] is None
for folder,n in [('carrier-cpu-sampling-20261123',9989),('carrier-index-cpu-20261123',9991),('carrier-pressure-cpu-20261123',9983),('carrier-retained-graph-cpu-20261123',9983)]:
 q=js(folder,'cpu-summary');assert q['samples']==n
 b=raw(folder,'samples/pc-'+str(q['child_pid'])+'.bin');assert len(b)==16*n
 assert h(provider(js(folder,'private-qualified')['args'][0]))==q['compiler_sha256']
assert m['acceptance']['full_private_compiler_source'] and not m['acceptance']['default_aggregate_ownership'] and not m['acceptance']['whole_release']
print('I verify',len(rows),'records,',len(new),'Git inputs and',len(images),'native ELF images; current full1306/profile/ownership bounds pass, three default leaks and104 historical provider paths remain open.')
