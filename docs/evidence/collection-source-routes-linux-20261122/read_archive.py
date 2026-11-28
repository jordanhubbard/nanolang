#!/usr/bin/env python3
"""I independently check my collection source and native opcode checkpoint."""
from pathlib import Path
import gzip,hashlib,json,subprocess
here=Path(__file__).resolve().parent;root=here.parents[2]
sha=lambda raw:hashlib.sha256(raw).hexdigest()
m=json.loads((here/'manifest.json').read_text())
assert sha((here/m['base_manifest']).read_bytes())==m['base_manifest_sha256']
rows={row['path']:row for row in m['files']};assert len(rows)==m['counts']['files']
objects={row['artifact'] for row in rows.values()};assert len(objects)==m['counts']['objects']
assert sum((here/name).stat().st_size for name in objects)==m['counts']['compressed_bytes']
def read(name):
 row=rows[name];packed=(here/row['artifact']).read_bytes()
 assert len(packed)==row['compressed_bytes'] and sha(packed)==row['compressed_sha256'],name
 raw=gzip.decompress(packed)
 assert len(raw)==row['bytes'] and sha(raw)==row['sha256'],name
 return raw
def data(name):return json.loads(read(name))
for name in rows:read(name)
assert sum(row['bytes'] for row in rows.values())==m['counts']['raw_bytes']
tree={}
for entry in subprocess.check_output(['git','ls-tree','-rz',m['source_commit']],cwd=root).split(b'\0'):
 if entry:
  meta,path=entry.split(b'\t',1);tree[path.decode()]=meta.split()[2].decode()
for name in rows:
 if name.startswith('source/'):
  path=name[7:];raw=read(name)
  if path in tree:assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==tree[path],path
  else:assert path=='modules/index.json',path
before=data('observation/product-before.json');after=data('observation/product-after.json')
assert after.pop('status')=='PASS' and before==after
for name,digest in before['sources'].items():assert sha(read('source/'+name))==digest
for name,digest in before['tools'].items():assert sha(read('tools/'+name))==digest
for name,digest in before['compilers'].items():assert sha(read('compilers/'+Path(name).name))==digest
cohorts=data('observation/product-cohorts.json')
assert [row['label'] for row in cohorts]==['product-gcc','product-clang','product-neighbors']
assert [row['expected_methods'] for row in cohorts]==[26,16,None]
for row in cohorts:
 assert row['returncode']==0;raw=read('observation/'+row['label']+'.log');assert sha(raw)==row['log_sha256']
 if row['expected_methods'] is not None:
  assert ('Ran '+str(row['expected_methods'])+' tests in ').encode() in raw and raw.rstrip().endswith(b'OK')
 assert data('observation/'+row['label']+'-after.json')==before
neighbors=read('observation/product-neighbors.log')
assert neighbors.count(b'Testing ')==88 and b'All typechecker tests passed!' in neighbors
assert b'0 reserved names' in neighbors
seed=data('observation/seed-product/manifest.json');assert seed['status']=='passed';b=seed['bootstrap_manifest']
assert b['stage_timeout_seconds']==1800 and b['shadow_timeout_seconds']==30
assert [row['label'] for row in b['steps']]==['seed','seed-verify']
for row in b['steps']:
 assert row['exit_code']==0 and sha(read('bootstrap-seed-product/'+Path(row['log']).name))==row['log_sha256']
for name,digest in b['tools'].items():assert sha(read('tools/'+name))==digest
for name,digest in b['sources'].items():assert sha(read('source/'+name))==digest
assert len(b['hosts'])==6
for name,digest in b['hosts'].items():assert sha(read('hosts/'+digest+'/'+Path(name).name))==digest
assert sha(read('tools/nanoc_seed.nvm'))==b['artifacts']['nanoc_seed.nvm']
assert data('observation/baseline.json')[0]['returncode']!=0
assert b'CAST_STRING: expected int value' in read('observation/source-initial.log')
assert b'FAILED (failures=2)' in read('observation/source-kind-refusals.log')
failed=data('observation/private-cohort-field-gcc-result.json')
assert failed['ran']==16 and len(failed['failures'])==23 and not failed['successful']
assert b'OPAQUE_NULL' in read('observation/private-cohort-field-gcc.log')
assert sha(read('observation/nvm2c-private.c'))==sha(read('source/src/nanoisa/nvm2c.c'))
assert b'self.assertEqual(len(cases),28)' in read('source/tests/test_native_nonscalar_cast.py')
assert b'self.assertEqual(len(cases), 73)' in read('source/tests/test_native_regex_host_abi.py')
# I retain actual executable products and same-image collection ABI guards.
aot_guards=[name for name in rows if name.startswith('collection-aot/') and name.endswith('/inputs-tools-before-after.json')]
assert aot_guards
qualified=0
for name in aot_guards:
 guard=data(name)
 if guard['tools'][str(root/'bin/nvm2c')]==before['tools']['nvm2c']:
  qualified+=1
  for path,digest in guard['inputs'].items():assert before['sources'].get(path,sha(read('source/'+path)))==digest
assert qualified==1
assert any(name.startswith('cases/') and name.endswith('/decoder-aot') for name in rows)
assert any(name.startswith('cases/') and name.endswith('/source-decoder-aot') for name in rows)
print(json.dumps(dict(status='PASS',source_commit=m['source_commit'],records=len(rows),gcc_methods=26,clang_methods=16,typechecker_methods=88,raw_cast_cases=28,abi_refusals=73,raw_bytes=m['counts']['raw_bytes']),indent=2))
