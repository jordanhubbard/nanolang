from pathlib import Path
import hashlib,json,tarfile
roots=[Path('/tmp/nanolang-file-private-vm-cfb-darwin')]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for data in iter(lambda:f.read(1048576),b''):h.update(data)
 return h.hexdigest()
unique={};entries=0
for root in roots:
 for m in [*root.glob('*-artifacts.json'),*root.glob('*-inputs-before.json')]:
  for v in json.loads(m.read_text()).values():
   p=Path(v['artifact']);s=v['sha256'];entries+=1
   if s not in unique:assert digest(p)==s;unique[s]=p
with tarfile.open('/tmp/nanolang-file-private-vm-puck-artifacts.tar.gz','w:gz') as t:
 for s,p in unique.items():t.add(p,arcname=s)
with tarfile.open('/tmp/nanolang-file-private-vm-puck-reports.tar.gz','w:gz') as t:
 for root in roots:
  for p in root.iterdir():
   if p.is_file():t.add(p,arcname=root.name+'/'+p.name)
summary={'artifact_entries':entries,'unique_artifacts':len(unique),'bytes':sum(p.stat().st_size for p in unique.values()),'archives':{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in (Path('/tmp/nanolang-file-private-vm-puck-artifacts.tar.gz'),Path('/tmp/nanolang-file-private-vm-puck-reports.tar.gz'))}}
Path('/tmp/nanolang-file-private-vm-puck-packaging.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
