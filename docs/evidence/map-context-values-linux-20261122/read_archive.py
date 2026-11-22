#!/usr/bin/env python3
"""I verify source/tool bytes, retained terminals and every available result."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

here=Path(__file__).resolve().parent
root=here.parents[2]
m=json.loads((here/'manifest.json').read_text())
sha=lambda raw:hashlib.sha256(raw).hexdigest()
records={r['path']:r for r in m['files']}
assert len(records)==m['counts']['files']==5330
objects={r['artifact']:r for r in m['files']}
assert len(objects)==m['counts']['objects']
assert sum(r['bytes'] for r in m['files'])==m['counts']['raw_bytes']
assert sum((here/p).stat().st_size for p in objects)==m['counts']['compressed_bytes']
def read(name):
 r=records[name];compressed=(here/r['artifact']).read_bytes()
 assert len(compressed)==r['compressed_bytes'] and sha(compressed)==r['compressed_sha256']
 raw=gzip.decompress(compressed)
 assert len(raw)==r['bytes'] and sha(raw)==r['sha256']
 return raw
for name in records:read(name)
def data(name):return json.loads(read(name))
# I check retained source against the actual commit, without running its compiler.
tree={}
for row in subprocess.check_output(['git','ls-tree','-rz',m['source_commit']],cwd=root).split(b'\0'):
 if row:
  metadata,path=row.split(b'\t',1);tree[path.decode()]=metadata.split()[2].decode()
for name in records:
 if name.startswith('source/'):
  path=name[len('source/'):];raw=read(name)
  if path in tree:
   assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==tree[path],path
  else:assert path=='modules/index.json',path
seeds=[]
for version in ('seed','seed-emitter'):
 s=data('observation/'+version+'/manifest.json');assert s['status']=='passed'
 b=s['bootstrap_manifest'];assert [step['label'] for step in b['steps']]==['seed','seed-verify']
 assert all(step['exit_code']==0 for step in b['steps'])
 assert b['stage_timeout_seconds']==1800 and b['shadow_timeout_seconds']==30
 artifact='observation/first-seed.nvm' if version=='seed' else 'tools/nanoc_seed.nvm'
 assert sha(read(artifact))==b['artifacts']['nanoc_seed.nvm']
 for name,digest in b['tools'].items():assert sha(read('tools/'+name))==digest
 for name,digest in b['hosts'].items():assert sha(read('hosts/'+digest+'/'+Path(name).name))==digest
 assert len(b['hosts'])==6
 seeds.append(b)
assert {name for name in seeds[0]['sources'] if seeds[0]['sources'][name]!=seeds[1]['sources'][name]}=={'src_nano/compiler/nanoisa_codegen.nano'}
guards=data('observation/final-guards.json');assert guards['status']=='PASS'
for name,digest in guards['sources'].items():assert sha(read('source/'+name))==digest
for name,digest in guards['tools'].items():assert sha(read('tools/'+name))==digest
for file,tests,failures in (('focused-first.json',7,12),('focused-before-emitter.json',7,4),('focused-emitter.json',7,0),('clang18.json',3,0)):
 r=data('observation/'+file)
 assert (r['tests'],r['failures'],r['errors'],r['skips'])==(tests,failures,0,0)
 assert all(r[key]=='PASS' for key in ('source_guards','tool_guards','host_guards','seed_guard'))
 if not failures:
  for directory in r['workdirs']:
   path='cases/'+Path(directory).name+'/commands.jsonl'
   rows=[json.loads(line) for line in read(path).splitlines()]
   assert rows and all(not row['timed_out'] for row in rows)
neighbor=data('observation/neighbors/result.json')
assert neighbor['discovered']==neighbor['ran']==12 and neighbor['successful']
assert not neighbor['failures'] and not neighbor['errors'] and not neighbor['skipped']
assert b'0 reserved names' in read('observation/identifiers.log')
assert b'Ran 7 tests' in read('observation/focused-emitter.log')
assert b'Ran 3 tests' in read('observation/clang18.log')
assert b'Ran 12 tests' in read('observation/neighbors.log')
print(json.dumps(dict(status='PASS',source_commit=m['source_commit'],records=len(records),original_plus_expanded_methods=7,clang18_methods=3,nominal_receiver_methods=12,initial_fixture_refusals=12,initial_lowering_refusals=4,raw_bytes=m['counts']['raw_bytes']),indent=2))
