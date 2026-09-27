from pathlib import Path
import hashlib,json,sys,tarfile,posixpath
archive,index=map(Path,sys.argv[1:3]);expected=json.loads(index.read_text());seen={};names=set();links=[]
with tarfile.open(archive,'r:gz') as t:
 for m in t:
  assert m.name in expected and m.name not in names,m.name
  names.add(m.name);row=expected[m.name];assert m.mode==row['mode'],m.name
  if m.issym():
   assert row['type']=='symlink' and m.linkname==row['linkname'],m.name
   target=posixpath.normpath(posixpath.join(posixpath.dirname(m.name),m.linkname))
   assert not target.startswith('/') and not target.startswith('../') and target in expected,m.name
   links.append((m.name,target));continue
  assert row['type']=='file',m.name
  if m.islnk():links.append((m.name,m.linkname));continue
  assert m.isfile() and m.size==row['bytes'],m.name
  h=hashlib.sha256();f=t.extractfile(m)
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
  assert h.hexdigest()==row['sha256'],m.name
  seen[m.name]=h.hexdigest()
while links:
 pending=[]
 for name,target in links:
  if target not in seen:pending.append((name,target));continue
  assert seen[target]==expected[name]['sha256'] and expected[target]['bytes']==expected[name]['bytes'],name
  seen[name]=seen[target]
 assert len(pending)<len(links),'I cannot resolve archived links'
 links=pending
assert set(seen)==set(expected)
h=hashlib.sha256()
with archive.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
print(json.dumps({'archive_sha256':h.hexdigest(),'members':len(seen),'status':'PASS'}))
