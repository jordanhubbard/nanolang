import json,hashlib,shutil,tarfile
from pathlib import Path
repo=Path('/home/jkh/Src/nanolang-record-array-v3');dest=repo/'docs/evidence/ordinary-array-authority';dest.mkdir();store=Path('/tmp/nanolang-ordinary-array-artifacts');store.mkdir()
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,d):p.write_text(json.dumps(d,sort_keys=True,indent=2)+'\n')
archive=Path('/tmp/nanolang-ordinary-array-puck-seal-v2.tar.gz');assert sha(archive)=='10ab440c7fb6599a10568e8b097945fd54f8ba44d39ae3aba6ff3bff7ed9ecf9'
download=Path('/tmp/nanolang-ordinary-array-puck-download');download.mkdir()
with tarfile.open(archive) as t:t.extractall(download,filter='data')
objects={};summaries={}
for host,root in [('linux',Path('/tmp/nanolang-ordinary-array-linux-seal')),('puck',download/'nanolang-ordinary-array-puck-seal-v2')]:
 out=dest/host;out.mkdir()
 for p in root.iterdir():
  if p.name=='artifacts':continue
  if p.is_dir():shutil.copytree(p,out/p.name)
  else:shutil.copyfile(p,out/p.name)
 for p in (root/'artifacts').iterdir():
  h=sha(p);assert p.name==h
  if h not in objects:shutil.copyfile(p,store/h);objects[h]={'path':str(store/h),'bytes':p.stat().st_size}
 summaries[host]=json.loads((root/'summary.json').read_text())
for name in ['nanolang-ordinary-array-baseline-check','nanolang-array-neighbor-cdd-baseline']:
 root=Path('/tmp')/name;out=dest/'baseline'/name;out.mkdir(parents=True)
 for p in root.iterdir():
  if not p.is_file():continue
  if p.suffix in ['.json','.c','.stderr','.stdout'] or p.name.endswith(('.txt','.log')):shutil.copyfile(p,out/p.name)
  else:
   h=sha(p)
   if h not in objects:shutil.copyfile(p,store/h);objects[h]={'path':str(store/h),'bytes':p.stat().st_size}
   dump(out/(p.name+'.artifact.json'),{'path':str(p),'sha256':h})
shutil.copyfile(__file__,dest/'seal.py')
dump(dest/'artifact-store.json',objects)
dump(dest/'qualification-summary.json',{'query_production':'4f5f26126','query_fixture':'ef4500e40','neighbor_fixture':'2a30d68e23618a8c97f033f5bb12b80fbc55f657','hosts':summaries,'unique_artifacts':len(objects),'artifact_bytes':sum(r['bytes'] for r in objects.values()),'puck_archive':{'path':str(archive),'sha256':sha(archive)},'historical_linux_query_actual_python_hash_captured':False})
dump(dest/'report-sha256.json',{str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()})
print(json.dumps({'reports':len(json.loads((dest/'report-sha256.json').read_text())),'objects':len(objects),'bytes':sum(r['bytes'] for r in objects.values())}))
