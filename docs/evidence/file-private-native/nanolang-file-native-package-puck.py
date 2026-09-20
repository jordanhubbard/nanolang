import hashlib,json,tarfile,shutil
from pathlib import Path
roots={'darwin':Path('/tmp/nanolang-file-native-4e8-puck-corrected'),'darwin-corrected':Path('/tmp/nanolang-file-native-6fdb-puck-ready')}
base=Path('/tmp/nanolang-file-native-puck-package');base.mkdir(exist_ok=False);reports=base/'reports';reports.mkdir();objects=base/'artifacts';objects.mkdir()
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();index={};references=0
for group,root in roots.items():
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
for group,root,tree,phase in [('darwin',roots['darwin'],Path('/tmp/nanolang-file-native-4e8-corrected'),'normal'),('darwin-corrected',roots['darwin-corrected'],Path('/tmp/nanolang-file-native-6fdb-ready'),'sanitizer')]:
 sources=json.loads((root/(phase+'-source-before.json')).read_text());tools=json.loads((root/(phase+'-tools-before.json')).read_text())
 for n,sha in sources.items():assert h(tree/n)==sha,n
 for n,d in tools.items():assert h(Path(d['path']))==d['sha256'],n
 current.append({'group':group,'tree':str(tree),'sources':len(sources),'tools':len(tools),'mismatches':0})
extras=['/tmp/nanolang-file-native-6fdb-puck-ready-provider-reuse.json']
for n in extras:shutil.copy2(n,reports/Path(n).name)
archives={}
for name,source in [('reports',reports),('artifacts',objects)]:
 archive=Path('/tmp/nanolang-file-native-puck-'+name+'.tar.gz')
 with tarfile.open(archive,'w:gz') as tar:tar.add(source,arcname=name)
 archives[str(archive)]={'sha256':h(archive),'bytes':archive.stat().st_size}
summary={'references':references,'unique_artifacts':len(index),'bytes':sum(d['bytes'] for d in index.values()),'current':current,'archives':archives}
Path('/tmp/nanolang-file-native-puck-packaging.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
