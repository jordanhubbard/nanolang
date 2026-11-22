#!/usr/bin/env python3
"""I verify actual root metadata source, tools, failures and passing products."""
from pathlib import Path
import gzip,hashlib,json,subprocess
here=Path(__file__).resolve().parent;root=here.parents[2];sha=lambda b:hashlib.sha256(b).hexdigest()
m=json.loads((here/'manifest.json').read_text());base=here/m['base_manifest'];assert sha(base.read_bytes())==m['base_manifest_sha256'];bm=json.loads(base.read_text());base_records={r['path']:r for r in bm['files']}
records={r['path']:r for r in m['files']};assert len(records)==m['counts']['files']==4065
objects={r['artifact'] for r in m['files']};assert len(objects)==m['counts']['objects'];assert sum((here/p).stat().st_size for p in objects)==m['counts']['compressed_bytes']
def read(name):
 r=records[name];c=(here/r['artifact']).read_bytes();assert len(c)==r['compressed_bytes'] and sha(c)==r['compressed_sha256'];b=gzip.decompress(c);assert len(b)==r['bytes'] and sha(b)==r['sha256'];return b
def data(name):return json.loads(read(name))
for name in records:read(name)
assert sum(r['bytes'] for r in m['files'])==m['counts']['raw_bytes']
tree={}
for entry in subprocess.check_output(['git','ls-tree','-rz',m['source_commit']],cwd=root).split(b'\0'):
 if entry:
  meta,path=entry.split(b'\t',1);tree[path.decode()]=meta.split()[2].decode()
for name in records:
 if name.startswith('source/'):
  path=name[len('source/'):];raw=read(name)
  if path in tree:assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==tree[path],path
  else:assert path=='modules/index.json',path
for version,artifact in (('seed','observation/pure-only-seed.nvm'),('seed-final','tools/nanoc_seed.nvm')):
 s=data('observation/'+version+'/manifest.json');assert s['status']=='passed';b=s['bootstrap_manifest'];assert b['shadow_timeout_seconds']==30 and b['stage_timeout_seconds']==1800
 assert [r['label'] for r in b['steps']]==['seed','seed-verify'] and all(r['exit_code']==0 for r in b['steps'])
 assert sha(read(artifact))==b['artifacts']['nanoc_seed.nvm']
 for name,digest in b['tools'].items():
  actual=records['tools/'+name]['sha256'] if version=='seed-final' else base_records['tools/'+name]['sha256']
  assert actual==digest,(version,name)
 for name,digest in b['hosts'].items():assert sha(read('hosts/'+digest+'/'+Path(name).name))==digest
 assert len(b['hosts'])==6
baseline=data('observation/c-baseline/result.json');assert baseline['ran']==11 and len(baseline['failures'])==3 and not baseline['errors']
pure=data('observation/pure-first/result.json');assert pure['ran']==11 and pure['successful']
for profile in ('c','pure','asan'):
 r=data('observation/'+profile+'-release/result.json');assert r['discovered']==r['ran']==13 and r['successful'];assert not r['failures'] and not r['errors'] and not r['skipped']
assert all(r['returncode']==0 for r in data('observation/release-controls.json'))
native=data('observation/direct-native-release/result.json');assert len(native)==14 and all(r['returncode']==0 for r in native)
scan=data('observation/sanitizer-scan.json');assert scan['status']=='PASS' and scan['streams']==376 and not scan['matches'];assert sha(read('tools/nano_virt_root_metadata_asan'))==scan['tool_sha256']
patterns=(b'ERROR: AddressSanitizer',b'SUMMARY: AddressSanitizer',b'SUMMARY: UndefinedBehaviorSanitizer',b'runtime error:')
for name in records:
 if name.startswith(('observation/asan/','observation/asan-release/')) and name.endswith(('.stdout','.stderr')):assert not any(p in read(name) for p in patterns),name
link=data('observation/asan-link.json');assert link['returncode']==0
for name,digest in link['sources'].items():assert sha(read('source/'+name))==digest
assert b'0 reserved names' in read('observation/contracts.log') and b'FAILED' not in read('observation/contracts.log')
guards=data('observation/final-guards.json');assert guards['status']=='PASS'
for name,digest in guards['sources'].items():assert sha(read('source/'+name))==digest
for name,digest in guards['tools'].items():assert sha(read('tools/'+name))==digest
print(json.dumps(dict(status='PASS',source_commit=m['source_commit'],records=len(records),methods_per_producer=13,producers=['C','pure','targeted-parser-typechecker-ASan'],direct_native_controls=7,retained_C_failures=3,raw_bytes=m['counts']['raw_bytes']),indent=2))
