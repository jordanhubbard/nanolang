import hashlib,json,tarfile,shutil
from pathlib import Path
repo=Path('/home/jkh/Src/nanolang-record-array-v3');dest=repo/'docs/evidence/private-layout';dest.mkdir(parents=True,exist_ok=False)
store=Path('/tmp/nanolang-private-layout-artifacts');store.mkdir(exist_ok=False)
archive=Path('/tmp/nanolang-private-layout-puck-evidence.tar.gz');download=Path('/tmp/nanolang-private-layout-puck-retained');download.mkdir(exist_ok=False)
with tarfile.open(archive) as t:t.extractall(download,filter='data')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
artifacts={};pairs=[];refs=0;terminals=[]
roots=[('linux',Path('/tmp/nanolang-private-layout-83b-linux')),('puck',download/'nanolang-private-layout-83b-puck')]
for host,root in roots:
 out=dest/host;out.mkdir(parents=True)
 for p in sorted(root.iterdir()):
  if p.is_file():shutil.copyfile(p,out/p.name)
 for p in (root/'artifacts').iterdir():
  h=digest(p);assert p.name==h
  if h not in artifacts:shutil.copyfile(p,store/h);artifacts[h]={'path':str(store/h),'bytes':p.stat().st_size}
 for p in root.glob('*-source-before.json'):
  for domain in ('source','tools'):
   before=root/p.name.replace('-source-',f'-{domain}-');after=before.with_name(before.name.replace('-before.json','-after.json'))
   a=json.loads(before.read_text());b=json.loads(after.read_text());assert a==b
   pairs.append({'host':host,'before':before.name,'after':after.name,'count':len(a)})
 for p in root.glob('*-artifacts.json'):
  for name,row in json.loads(p.read_text()).items():assert row['sha256'] in artifacts,(host,name);refs+=1
 for p in root.glob('*-terminal.json'):
  d=json.loads(p.read_text());expected=0
  assert d['returncode']==expected and not d['timeout'] and d['group_disappeared'] and not d['errors'],p
  terminals.append({'host':host,'phase':p.name,'status':d['returncode']})
shutil.copyfile(__file__,dest/'seal.py')
dump(dest/'artifact-store.json',artifacts)
dump(dest/'qualification-summary.json',{'production':'f4200bff56069e49fe44ba41f5cb66dc8299ccc7','fixture':'83b04f11c','unique_artifacts':len(artifacts),'artifact_bytes':sum(v['bytes'] for v in artifacts.values()),'artifact_references':refs,'equal_source_tool_pairs':pairs,'terminals':terminals,'puck_archive':{'path':str(archive),'sha256':digest(archive),'bytes':archive.stat().st_size}})
manifest={str(p.relative_to(dest)):digest(p) for p in sorted(dest.rglob('*')) if p.is_file()};dump(dest/'report-sha256.json',manifest)
print(json.dumps({'reports':len(manifest),'artifacts':len(artifacts),'bytes':sum(v['bytes'] for v in artifacts.values()),'references':refs,'pairs':len(pairs),'terminals':len(terminals)}))
