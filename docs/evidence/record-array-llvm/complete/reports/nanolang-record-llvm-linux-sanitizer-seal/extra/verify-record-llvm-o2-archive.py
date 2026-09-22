import hashlib,json,sys,tarfile
from pathlib import Path
archive,manifest=map(Path,sys.argv[1:3]);expected=json.loads(manifest.read_text());seen={};links=[]
with tarfile.open(archive,'r:gz') as tar:
 for m in tar:
  if m.isdir():continue
  assert m.name in expected,m.name
  assert m.mode==expected[m.name]['mode'],m.name
  if m.islnk():links.append((m.name,m.linkname));continue
  assert m.isfile(),m.name
  h=hashlib.sha256();f=tar.extractfile(m)
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
  assert h.hexdigest()==expected[m.name]['sha256'] and m.size==expected[m.name]['bytes'],m.name
  seen[m.name]=h.hexdigest()
for name,target in links:
 assert target in seen and seen[target]==expected[name]['sha256'];seen[name]=seen[target]
assert set(seen)==set(expected)
h=hashlib.sha256()
with archive.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
print(json.dumps({'archive_sha256':h.hexdigest(),'members':len(seen),'status':'PASS'}))
