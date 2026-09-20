import hashlib,json,tarfile
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
groups={pin:Path('/tmp/nanolang-file-cyclic-dispatch-'+pin+'-puck') for pin in ('3ea2',)}
records={};current=[]
for pin,report in groups.items():
 results=json.loads((report/'results.json').read_text())
 if pin=='3ea2':assert len(results)==9 and all(r['status']==0 for r in results),results
 else:assert len(results)==4 and results[-1]['status']==1,results
 tree=Path('/private/tmp/nanolang-file-cyclic-dispatch-'+pin)
 phase=results[-1]['phase']
 sources=json.loads((report/(phase+'-source-after.json')).read_text())
 tools=json.loads((report/(phase+'-tools-after.json')).read_text())
 products=json.loads((report/(phase+'-artifacts.json')).read_text())
 for n,d in sources.items():assert sha(tree/n)==d,n
 for n,r in tools.items():assert sha(Path(r['path']))==r['sha256'],n
 providers={n:r for n,r in products.items() if Path(n).resolve().is_relative_to(tree.resolve())}
 for n,r in providers.items():assert sha(Path(n))==r['sha256'],n
 current.append(dict(pin=pin,sources=len(sources),tools=len(tools),providers=len(providers),mismatches=0))
 for pattern in ('*-inputs-before.json','*-artifacts.json'):
  for p in report.glob(pattern):
   for r in json.loads(p.read_text()).values():
    f=Path(r['artifact']);assert sha(f)==r['sha256'];records[r['sha256']]=f
archive=Path('/tmp/nanolang-file-cyclic-dispatch-integration-puck-evidence.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for pin,report in groups.items():
  for p in sorted(report.iterdir()):
   if p.is_file():t.add(p,arcname='reports/'+pin+'/'+p.name)
 for digest,p in sorted(records.items()):t.add(p,arcname='artifacts/'+digest)
result=dict(current=current,artifacts=len(records),archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size)
Path('/tmp/nanolang-file-cyclic-dispatch-integration-puck-packaging.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
