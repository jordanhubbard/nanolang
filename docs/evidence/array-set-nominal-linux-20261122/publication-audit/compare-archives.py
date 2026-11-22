from pathlib import Path
import json,gzip,sys,hashlib
left,right=map(Path,sys.argv[1:3]);a=json.loads((left/'manifest.json').read_text());b=json.loads((right/'manifest.json').read_text());assert a['implementation']==b['implementation'] and a['source_hashes']==b['source_hashes'];summary=[]
def read(root,m,suffix):
 files=[e for e in m['files'] if e['path'].endswith(suffix)];assert len(files)==1,(suffix,len(files));return json.loads(gzip.decompress((root/files[0]['artifact']).read_bytes()))
for label,number in [('mutation',4),('neighbors',25)]:
 values=[]
 for root,m in [(left,a),(right,b)]:
  r=m['verification'][label];assert r['successful'] and r['ran']==r['discovered']==number and not(r['failures'] or r['errors'] or r['skipped'])
  commands=read(root,m,'/'+label+'/commands.json');hashes={i['sha256'] for c in commands for i in c['inputs'] if i['path'].endswith('.nano')};values.append((commands,hashes,r['seconds']))
 assert values[0][1]==values[1][1] and len(values[0][0])==len(values[1][0])
 summary.append(dict(label=label,methods=number,unique_nano_inputs=len(values[0][1]),commands_each=len(values[0][0]),seconds_each=[v[2] for v in values],all_captured_nano_input_hashes_identical=True))
for root,m in [(left,a),(right,b)]:
 files={e['path']:e for e in m['files']};san=m['compiler_sanitizers'];assert san['passed'] and files[san['compiler']]['sha256']==san['compiler_sha256'];assert all(m['source_hashes'][n]==v for n,v in san['source_hashes'].items())
 suffix='/compiler-asan-green/result.json' if root==left else '/compiler-sanitizers/result.json';r=read(root,m,suffix);assert r['successful'] and r['ran']==r['discovered']==4 and not(r['failures'] or r['errors'] or r['skipped'])
 for seed in m['seed_controls']:
  assert seed['status']=='passed';assert all(m['source_hashes'][n]==v for n,v in seed['sources'].items())
  assert [x['label'] for x in seed['steps']]==['seed','seed-verify']
  for name,value in seed['hosts'].items():assert files[name]['sha256']==value
  assert all(name in files for name in seed['actual_retained_host_files'])
  if seed['seed_artifact_retained']:
   matches=[e for e in m['files'] if e['path'].endswith('/bin/nanoc_seed.nvm')];assert len(matches)==1 and matches[0]['sha256']==seed['artifacts']['nanoc_seed.nvm']
 assert any(s['seed_artifact_retained'] and s['control'].endswith('/compiler-seed/manifest.json') for s in m['seed_controls'])
print(json.dumps(dict(implementation=a['implementation'],source_files=len(a['source_hashes']),source_hashes_identical=True,cohorts=summary,targeted_compiler_sanitizers_methods_each=4,all_seed_source_maps_host_images_actual_retained_inputs_and_final_artifacts_verified=True,limits='I compare actual source inputs and completed method outcomes. Seed modules/native binaries are platform/context-specific; no binary-equality, full conditional receiver admission or full bootstrap claim follows.'),indent=2))
