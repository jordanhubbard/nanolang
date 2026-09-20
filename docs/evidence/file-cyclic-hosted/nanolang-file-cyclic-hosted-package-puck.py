import hashlib,json,tarfile,shutil
from pathlib import Path
roots={'darwin':Path('/tmp/nanolang-file-cyclic-hosted-46912-puck')}
base=Path('/tmp/nanolang-file-cyclic-hosted-puck-package');base.mkdir(exist_ok=False);reports=base/'reports';reports.mkdir();objects=base/'artifacts';objects.mkdir()
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();index={};references=0
for group,root in roots.items():
 results=json.loads((root/'results.json').read_text());assert len(results)==11 and all(x['status']==0 for x in results)
 dst=reports/group;dst.mkdir()
 for p in root.iterdir():
  if p.is_file():shutil.copy2(p,dst/p.name)
 for p in root.glob('*-artifacts.json'):
  for d in json.loads(p.read_text()).values():
   references+=1;sha=d['sha256'];source=Path(d['artifact']);assert h(source)==sha
   if sha not in index:shutil.copyfile(source,objects/sha);index[sha]={'bytes':source.stat().st_size}
 for p in root.glob('*-inputs-before.json'):
  for d in json.loads(p.read_text()).values():
   references+=1;sha=d['sha256'];source=Path(d['artifact']);assert h(source)==sha
   if sha not in index:shutil.copyfile(source,objects/sha);index[sha]={'bytes':source.stat().st_size}
current=[]
for group,root,tree,phase in [('darwin',roots['darwin'],Path('/tmp/nanolang-file-cyclic-hosted-46912'),'ordinary')]:
 sources=json.loads((root/(phase+'-source-before.json')).read_text());tools=json.loads((root/(phase+'-tools-before.json')).read_text())
 for n,sha in sources.items():assert h(tree/n)==sha,n
 for n,d in tools.items():assert h(Path(d['path']))==d['sha256'],n
 inputs=json.loads((root/(phase+'-inputs-before.json')).read_text())
 for n,d in inputs.items():assert h(Path(n))==d['sha256'],n
 current.append({'providers':len(inputs),'group':group,'tree':str(tree),'sources':len(sources),'tools':len(tools),'mismatches':0})
archives={}
for name,source in [('reports',reports),('artifacts',objects)]:
 archive=Path('/tmp/nanolang-file-cyclic-hosted-puck-'+name+'.tar.gz')
 with tarfile.open(archive,'w:gz') as tar:tar.add(source,arcname=name)
 archives[str(archive)]={'sha256':h(archive),'bytes':archive.stat().st_size}
summary={'references':references,'unique_artifacts':len(index),'bytes':sum(d['bytes'] for d in index.values()),'current':current,'archives':archives}
Path('/tmp/nanolang-file-cyclic-hosted-puck-packaging.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
