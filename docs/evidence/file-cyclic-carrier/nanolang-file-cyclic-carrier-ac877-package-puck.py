import hashlib,json,tarfile
from pathlib import Path
root=Path('/private/tmp/nanolang-file-cyclic-carrier-ac877');report=Path('/tmp/nanolang-file-cyclic-carrier-ac877-puck')
archive=Path('/tmp/nanolang-file-cyclic-carrier-ac877-puck-evidence.tar.gz')
supplement=Path('/tmp/nanolang-file-cyclic-carrier-history-puck')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
results=json.loads((report/'results.json').read_text());assert len(results)==11 and all(r['status']==0 for r in results[:-1]) and results[-1]['phase']=='public' and results[-1]['status']==1
assert [r['status'] for r in json.loads((supplement/'results.json').read_text())]==[0]
last='public-linked';sources=json.loads((supplement/(last+'-source-after.json')).read_text());tools=json.loads((supplement/(last+'-tools-after.json')).read_text());products=json.loads((supplement/(last+'-artifacts.json')).read_text())
for name,digest in sources.items():assert sha(root/name)==digest,name
for name,record in tools.items():assert sha(Path(record['path']))==record['sha256'],name
current={name:record for name,record in products.items() if Path(name).resolve().is_relative_to(root.resolve())}
for name,record in current.items():assert sha(Path(name))==record['sha256'],name
artifacts={}
for directory in (report,supplement):
 for pattern in ('*-inputs-before.json','*-artifacts.json'):
  for manifest in directory.glob(pattern):
   for record in json.loads(manifest.read_text()).values():
    path=Path(record['artifact']);digest=record['sha256'];assert sha(path)==digest
    artifacts[digest]=path
(supplement/'current-verification.json').write_text(json.dumps({'source_count':len(sources),'tool_count':len(tools),'provider_count':len(current),'root':str(root),'phase':'public-linked','zero_mismatches':True},indent=2)+'\n')
with tarfile.open(archive,'w:gz') as tar:
 for p in sorted(report.iterdir()):
  if p.is_file():tar.add(p,arcname='reports/darwin/'+p.name)
 for p in sorted(supplement.iterdir()):
  if p.is_file():tar.add(p,arcname='reports/darwin-public-linked/'+p.name)
 for name in ('input.json','source.c'):
  p=Path('/tmp/nanolang-file-cyclic-carrier-history-'+name);tar.add(p,arcname='history/'+p.name)
 for digest,path in sorted(artifacts.items()):tar.add(path,arcname='artifacts/'+digest)
record={'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'artifacts':len(artifacts),'reports':len([p for p in report.iterdir() if p.is_file()])}
Path('/tmp/nanolang-file-cyclic-carrier-ac877-puck-packaging.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
