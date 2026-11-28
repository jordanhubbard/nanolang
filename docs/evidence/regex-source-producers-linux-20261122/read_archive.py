#!/usr/bin/env python3
"""I independently check regex source descriptors, native results and retained failures."""
from pathlib import Path
import gzip,hashlib,json,subprocess
here=Path(__file__).resolve().parent;root=here.parents[2];sha=lambda raw:hashlib.sha256(raw).hexdigest()
m=json.loads((here/'manifest.json').read_text());base=here/m['base_manifest'];assert sha(base.read_bytes())==m['base_manifest_sha256']
records={r['path']:r for r in m['files']};assert len(records)==m['counts']['files']
objects={r['artifact'] for r in m['files']};assert len(objects)==m['counts']['objects'];assert sum((here/p).stat().st_size for p in objects)==m['counts']['compressed_bytes']
def read(name):
 r=records[name];packed=(here/r['artifact']).read_bytes();assert len(packed)==r['compressed_bytes'] and sha(packed)==r['compressed_sha256'];raw=gzip.decompress(packed);assert len(raw)==r['bytes'] and sha(raw)==r['sha256'];return raw
def data(name):return json.loads(read(name))
for name in records:read(name)
assert sum(r['bytes'] for r in m['files'])==m['counts']['raw_bytes']
tree={}
for entry in subprocess.check_output(['git','ls-tree','-rz',m['source_commit']],cwd=root).split(b'\0'):
 if entry:
  meta,path=entry.split(b'\t',1);tree[path.decode()]=meta.split()[2].decode()
for name in records:
 if name.startswith('source/'):
  path=name[7:];raw=read(name)
  if path in tree:assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==tree[path],path
  else:assert path=='modules/index.json',path
before=data('observation/qualification-before.json');after=data('observation/qualification-after.json');assert after.pop('status')=='PASS' and before==after
for name,digest in before['sources'].items():assert sha(read('source/'+name))==digest
for name,digest in before['tools'].items():assert sha(read('tools/'+name))==digest
for name,digest in before['compilers'].items():assert sha(read('compilers/'+Path(name).name))==digest
for row in data('observation/prerequisite-identity.json'):
 if 'sha256' in row:assert row['matches_published'] and sha(read('source/'+row['path']))==row['sha256']
 else:assert row['diff_except_blob_index_matches_published']
for version in ['seed-qualified','seed-final','seed-opaque']:
 s=data('observation/'+version+'/manifest.json');assert s['status']=='passed';b=s['bootstrap_manifest'];assert b['stage_timeout_seconds']==1800 and b['shadow_timeout_seconds']==30
 assert [r['label'] for r in b['steps']]==['seed','seed-verify'] and all(r['exit_code']==0 for r in b['steps'])
 assert len(b['artifacts']['nanoc_seed.nvm'])==64
 for row in b['steps']:
  assert sha(read('bootstrap-'+version+'/'+Path(row['log']).name))==row['log_sha256']
 for name,digest in b['tools'].items():assert sha(read('tools/'+name))==digest
 for name,digest in b['hosts'].items():assert sha(read('hosts/'+digest+'/'+Path(name).name))==digest
 assert len(b['hosts'])==6
for version in ['seed-initial','seed-corrected']:assert data('observation/'+version+'/manifest.json')['status']=='failed'
b=data('observation/seed-opaque/manifest.json')['bootstrap_manifest'];assert sha(read('tools/nanoc_seed.nvm'))==b['artifacts']['nanoc_seed.nvm']
for name,digest in b['sources'].items():assert sha(read('source/'+name))==digest
cohorts=data('observation/cohorts.json');assert [r['label'] for r in cohorts]==['gcc-source','clang-native','neighbors','math-neighbors']
for row in cohorts:
 assert row['returncode']==0;log=read('observation/'+row['label']+'.log');assert sha(log)==row['log_sha256']
 if row['expected_methods'] is not None:assert ('Ran '+str(row['expected_methods'])+' tests in ').encode() in log and log.rstrip().endswith(b'OK')
 assert data('observation/'+row['label']+'-after.json')==before
assert cohorts[0]['expected_methods']==10 and cohorts[1]['expected_methods']==5 and cohorts[3]['expected_methods']==3
assert b'0 reserved names' in read('observation/neighbors.log')
assert b'conflicting types for' in read('observation/source-baseline.log') and b'exact builtin host ABI' in read('observation/source-baseline.log')
assert b'CALL_EXTERN 1' in read('observation/provider-calls-initial.log')
nominal=json.loads(read('observation/nominal-observation.log'));assert nominal['returncode']==0 and not nominal['prior_preserved'] and '.import "" "nl_regex_compile" opaque string' in nominal['imports']
link=data('observation/final-link-note.json');assert link['before_sha256']!=link['after_sha256'] and before['tools']['nanoc_c']==link['after_sha256']
# My original ABI refusal matrix remains exact; each test retains its real commands and prior output.
source=read('source/tests/test_native_regex_host_abi.py');assert b'self.assertEqual(len(cases), 73)' in source
assert any(name.startswith('cases/') and name.endswith('/decoder-aot') for name in records)
assert any(name.startswith('cases/') and name.endswith('/source-decoder-aot') for name in records)
for name in records:
 if name.startswith('runtime/') and name.endswith('/run.stdout'):assert read(name)==b'I passed 24 regex replacement cases.\n'
 if name.startswith(('cases/','runtime/')) and name.endswith(('.stdout','.stderr')):
  raw=read(name);assert not any(s in raw for s in [b'ERROR: AddressSanitizer',b'SUMMARY: AddressSanitizer',b'SUMMARY: UndefinedBehaviorSanitizer',b'runtime error:']),name
print(json.dumps(dict(status='PASS',source_commit=m['source_commit'],records=len(records),gcc_methods=10,clang_methods=5,math_neighbor_methods=3,abi_refusals=73,raw_bytes=m['counts']['raw_bytes']),indent=2))
