from pathlib import Path
import hashlib,json,tarfile
base=Path('/private/tmp');reports=['nanolang-file-public-integrated-puck']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
unique={};refs=0
with tarfile.open(base/'nanolang-file-public-integrated-puck-reports.tar.gz','w:gz') as t:
 for name in reports:
  root=base/name
  for p in root.iterdir():
   if p.is_file():t.add(p,arcname=name+'/'+p.name)
  for p in list(root.glob('*-artifacts.json'))+list(root.glob('*-providers-before.json')):
   for v in json.loads(p.read_text()).values():
    refs+=1;digest=v['sha256'];src=Path(v['artifact'])
    if digest not in unique:
     assert sha(src)==digest;unique[digest]=src
with tarfile.open(base/'nanolang-file-public-integrated-puck-artifacts.tar.gz','w:gz') as t:
 for digest,p in unique.items():t.add(p,arcname=digest)
archives={}
for kind in ('reports','artifacts'):
 p=base/('nanolang-file-public-integrated-puck-'+kind+'.tar.gz');archives[p.name]={'sha256':sha(p),'bytes':p.stat().st_size}
out={'reports':reports,'references':refs,'unique':len(unique),'archives':archives}
(base/'nanolang-file-public-integrated-puck-packaging.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
