import hashlib,json,tarfile,shutil
from pathlib import Path
repo=Path('/home/jkh/Src/nanolang-file-indirect-contract');dest=repo/'docs/evidence/file-indirect-targets';dest.mkdir(parents=True,exist_ok=False)
store=Path('/tmp/nanolang-file-indirect-artifacts');store.mkdir(exist_ok=False)
archive=Path('/tmp/nanolang-file-indirect-5acd-puck-reports.tar.gz');download=Path('/tmp/nanolang-file-indirect-puck-retained');download.mkdir(exist_ok=False)
with tarfile.open(archive) as t:t.extractall(download,filter='data')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
artifacts={};pairs=[];refs=0;terminals=[]
for host,root in [('linux',Path('/tmp/nanolang-file-indirect-5acd-linux')),('puck',download/'nanolang-file-indirect-5acd-puck')]:
 out=dest/host;out.mkdir()
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
  d=json.loads(p.read_text());assert d['returncode']==0 and not d['timeout'] and d['group_disappeared'] and not d['errors'],p
  terminals.append({'host':host,'phase':p.name,'status':0})
shutil.copyfile('/tmp/nanolang-file-indirect-5acd-puck-extraction-first.json',dest/'extraction-first.json')
shutil.copyfile('/tmp/seal-file-indirect-root.py',dest/'seal.py')
dump(dest/'artifact-store.json',artifacts)
dump(dest/'qualification-summary.json',{'source':'5acd1b370','unique_artifacts':len(artifacts),'artifact_bytes':sum(v['bytes'] for v in artifacts.values()),'artifact_references':refs,'equal_source_tool_pairs':pairs,'terminals':terminals,'puck_archive':{'path':str(archive),'sha256':digest(archive),'bytes':archive.stat().st_size}})
manifest={str(p.relative_to(dest)):digest(p) for p in sorted(dest.rglob('*')) if p.is_file()};dump(dest/'report-sha256.json',manifest)
print(json.dumps({'reports':len(manifest),'artifacts':len(artifacts),'references':refs,'pairs':len(pairs),'terminals':len(terminals)}))
