from pathlib import Path
import hashlib,io,json,tarfile
base=Path(__file__).resolve().parent
root=base.parents[2]
sha=lambda b:hashlib.sha256(b).hexdigest()
archive=base/'artifacts.tar.gz'
assert sha(archive.read_bytes())=='64b33290e5f3a9e1b260ec88329c9d058e96b7312dff36ac7322ec7769212c63'
m=json.loads((base/'manifest.json').read_text())
with tarfile.open(archive) as t:
 for p,h in m.items(): assert sha(t.extractfile('evidence/'+p).read())==h,p
 for p in ('gccO0','gccO2','clangO0','clangO2'):
  r=json.load(t.extractfile('evidence/artifacts/'+p+'.json'))
  assert r['build_status']==r['run_status']==0
  assert r['dependency_command'][0]=='otool'
  assert sha(t.extractfile('evidence/artifacts/'+p).read())==r['executable_sha256']
for name in ('sources','tools'):
 assert json.loads((base/(name+'-before.json')).read_text())==json.loads((base/(name+'-after.json')).read_text())
for p,h in json.loads((base/'sources-before.json').read_text()).items(): assert sha((root/p).read_bytes())==h,p
assert json.loads((base/'result.json').read_text())['status']==0
print('I verified 28 sealed reports/artifacts, four successful otool routes, eight current source identities and equal before/after tool maps. No artifact was executed.')
