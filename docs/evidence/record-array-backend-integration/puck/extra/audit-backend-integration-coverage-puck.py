from pathlib import Path
import json,hashlib,sys
base=Path(sys.argv[1]);report=base/'reports'
def read(p):return json.loads(Path(p).read_text())
def good(x):return x['returncode']==0 and not x['timeout'] and x['leader_reaped'] and x['group_disappeared'] and not x['errors']
expected=['setup','configuration','discovery','generated-discovery','generated-ordinary-native','generated-clang-native','parity','ordinary-emission','ordinary-native','clang-ordinary-emission','clang-ordinary-native','wasm','old-private-vm','record_array_global_flow','record_array_execution','record_array_origins','package']
rows=read(report/'results.json');assert [(r['phase'],r['status']) for r in rows]==[(p,0 if p!='package' else 1) for p in expected]
assert read(base/'outer.json')['returncode']==1
supp=base/'package-hb-reports';suprows=read(supp/'results.json');assert [(r['phase'],r['status']) for r in suprows]==[('configuration',0),('package',0)]
assert read(base/'package-hb-outer.json')['returncode']==0
for phase in ('configuration','package'):
 assert good(read(supp/(phase+'-terminal.json')))
 for kind in ('source','tools'):assert read(supp/(phase+'-'+kind+'-before.json'))==read(supp/(phase+'-'+kind+'-after.json'))
result=[]
for phase in expected:
 terminal=read(report/(phase+'-terminal.json'))
 if phase=='package':
  assert terminal['returncode']==1 and not terminal['timeout'] and terminal['leader_reaped'] and terminal['group_disappeared'] and not terminal['errors']
 else:assert good(terminal),phase
 for kind in ('source','tools'):assert read(report/(phase+'-'+kind+'-before.json'))==read(report/(phase+'-'+kind+'-after.json'))
 if phase not in ('generated-ordinary-native','generated-clang-native','ordinary-native','clang-ordinary-native','wasm'):continue
 raw=read(report/(phase+'-artifacts.json'));art={Path(k).name:v for k,v in raw.items()}
 def artifact(name):
  row=art[name];p=Path(row['artifact']);b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==row['sha256'];return json.loads(b)
 statuses=[n for n in art if n.endswith('-status.json')]
 for n in statuses:assert good(artifact(n)),(phase,n)
 total=records=0
 c=phase.startswith('generated-');wasm=phase=='wasm'
 if not c:
  abi=artifact(('wasm' if wasm else 'native')+'-abi-comparison.json');assert abi['equal'] and len(abi['declared'])==40 and abi['declared']==abi['actual_C']
 for opt in ('O0','O2'):
  for mode in ('linked','observed'):
   for i in range(73):
    prefix=('wasm-' if wasm else '')+f'{opt}-{mode}-{i:04d}'
    assert good(artifact(prefix+('-build' if c else '-link')+'-status.json'))
    for engine in (('wasmtime','node') if wasm else ('',)):
     ep=prefix+('-'+engine if engine else '')
     if wasm:assert good(artifact(ep+'-run-status.json'))
     assert good(artifact(ep+('-baseline' if mode=='observed' else '-run')+'-status.json'))
     if mode!='observed':continue
     cov=artifact(ep+('-coverage.json' if wasm else '-fault-coverage.json'))
     assert cov['complete'];calls=cov['allocation_calls'] if c else cov['calls'];assert calls>0
     cursor=recoveries=0
     for worker in cov['workers']:
      if c:
       begin,end,modes,count=worker['begin'],worker['end'],worker['modes'],worker['recoveries'];name=worker['name']
      else:
       begin,end,modes,count=worker;name=f'{ep}-fault-{begin:06d}-{end:06d}'
      assert begin==cursor and 0<end-begin<=16 and modes==2 and count==2*(end-begin)
      assert good(artifact(name+'-status.json'));cursor=end;recoveries+=count
     assert cursor==calls and recoveries==cov['recoveries']==2*calls
     if wasm:assert cov['requested_live_peak']>8*1024*1024 and cov['linear_memory_before_pages']==16 and 16<cov['linear_memory_after_pages']<=1024
     records+=1;total+=recoveries
 assert records==(292 if wasm else 146)
 result.append({'phase':phase,'child_statuses':len(statuses),'fault_records':records,'recoveries':total})
out={'source':read(base/'source.json')['pin'],'status':'PASS','original_phases':len(rows),'continuation_phases':len(suprows),'preserved_failure':'original Apple compiler package phase; exact supported-compiler continuation separately passes','coverage':result,'scope':'Current integration conjunction: original16 passes plus retained package failure and separate two-method package pass; exact phase selection and complete 73-case product/fault accounting; original sanitizer attribution remains separate.'}
(base/'coverage-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
