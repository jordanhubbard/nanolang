import hashlib,json,shutil,tarfile
from pathlib import Path
base=Path('/tmp/nanolang-file-cyclic-public-final-puck-package');base.mkdir(exist_ok=False)
reports=base/'reports';reports.mkdir();store=base/'artifacts';store.mkdir()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
groups={'darwin-integration':'final'};index={};refs=0;current=[]
for label,suffix in groups.items():
 root=Path('/tmp/nanolang-file-cyclic-public-'+suffix).resolve();src=Path('/tmp/nanolang-file-cyclic-public-'+suffix+'-puck');dst=reports/label;dst.mkdir()
 for p in src.iterdir():
  if p.is_file():shutil.copyfile(p,dst/p.name)
 results=json.loads((src/'results.json').read_text());last=results[-1]['phase'];assert len(results)==11 and all(x['status']==0 for x in results)
 source=json.loads((src/(last+'-source-after.json')).read_text());tools=json.loads((src/(last+'-tools-after.json')).read_text());products=json.loads((src/(last+'-artifacts.json')).read_text())
 for p,h in source.items():assert sha(root/p)==h,(label,p)
 for p,v in tools.items():assert sha(v['path'])==v['sha256'],(label,p)
 for p,v in products.items():assert sha(p)==v['sha256'],(label,p)
 current.append({'group':label,'root':str(root),'last':last,'sources':len(source),'tools':len(tools),'products':len(products),'mismatches':0})
 for pattern in ('*-artifacts.json','*-inputs-before.json'):
  for p in src.glob(pattern):
   for v in json.loads(p.read_text()).values():
    refs+=1;h=v['sha256'];f=Path(v['artifact']);assert sha(f)==h
    if h not in index:shutil.copyfile(f,store/h);index[h]=(store/h).stat().st_size
(reports/'current-puck.json').write_text(json.dumps(current,indent=2)+'\n')
summary={'objects':len(index),'bytes':sum(index.values()),'references':refs,'current':current};(reports/'packaging-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for name in ('reports','artifacts'):
 dest=Path('/tmp/nanolang-file-cyclic-public-final-puck-'+name+'.tar.gz')
 with tarfile.open(dest,'w:gz') as tar:tar.add(base/name,arcname=name)
 summary[name+'_archive']={'path':str(dest),'sha256':sha(dest),'bytes':dest.stat().st_size}
Path('/tmp/nanolang-file-cyclic-public-final-puck-packaging.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
