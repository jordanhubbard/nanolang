import hashlib,json,shutil,tarfile
from pathlib import Path
root=Path('/home/jkh/Src/nanolang-file-cyclic-carrier');out=root/'docs/evidence/file-cyclic-carrier';out.mkdir(parents=True,exist_ok=False)
store=Path('/tmp/nanolang-file-cyclic-carrier-artifacts');store.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
groups={'first-linux':Path('/tmp/nanolang-file-cyclic-carrier-c5d-linux'),'linux':Path('/tmp/nanolang-file-cyclic-carrier-ac877-linux'),'darwin':Path('/tmp/nanolang-file-cyclic-carrier-puck-downloaded/reports/darwin'),'darwin-public-linked':Path('/tmp/nanolang-file-cyclic-carrier-puck-downloaded/reports/darwin-public-linked')}
index={};references=0;pairs=[];statuses={};provider_pairs=[]
for group,source in groups.items():
 dst=out/group;dst.mkdir()
 for p in source.iterdir():
  if p.is_file():shutil.copyfile(p,dst/p.name)
 statuses[group]=json.loads((dst/'results.json').read_text())
 if group=='first-linux':assert [r['status'] for r in statuses[group]]==[0,0,1]
 elif group=='darwin':assert len(statuses[group])==11 and all(r['status']==0 for r in statuses[group][:-1]) and statuses[group][-1]['phase']=='public' and statuses[group][-1]['status']==1
 elif group=='darwin-public-linked':assert len(statuses[group])==1 and statuses[group][0]['status']==0
 else:assert len(statuses[group])==12 and all(r['status']==0 for r in statuses[group])
 for p in dst.glob('*-source-before.json'):
  phase=p.name.removesuffix('-source-before.json');before=json.loads(p.read_text());after=json.loads((dst/(phase+'-source-after.json')).read_text());assert before==after
  tools=json.loads((dst/(phase+'-tools-before.json')).read_text());assert tools==json.loads((dst/(phase+'-tools-after.json')).read_text())
  pairs.append({'group':group,'phase':phase,'sources':len(before),'tools':len(tools),'equal':True})
  inputs=json.loads((dst/(phase+'-inputs-before.json')).read_text());products=json.loads((dst/(phase+'-artifacts.json')).read_text())
  missing=[n for n in inputs if n not in products];assert not missing,(group,phase,missing)
  changed=[n for n in inputs if inputs[n]['sha256']!=products[n]['sha256']]
  assert not changed or phase in ('public','public-linked'),(group,phase,changed)
  provider_pairs.append({'group':group,'phase':phase,'before_count':len(inputs),'changed_existing':changed,'created':[n for n in products if n not in inputs and Path(n).suffix in ('.o','.a')],'unchanged':not changed,'public_package_rebuild':phase in ('public','public-linked')})
 for pattern in ('*-inputs-before.json','*-artifacts.json'):
  for p in dst.glob(pattern):
   for rec in json.loads(p.read_text()).values():
    references+=1;digest=rec['sha256'];src=Path(rec['artifact']) if not group.startswith('darwin') else Path('/tmp/nanolang-file-cyclic-carrier-puck-downloaded/artifacts')/digest
    assert sha(src)==digest
    if digest not in index:
     target=store/digest;shutil.copyfile(src,target);assert sha(target)==digest;index[digest]={'path':str(target),'bytes':target.stat().st_size}
current=[]
for group,tree,phase in [('first-linux',Path('/home/jkh/Src/nanolang-file-cyclic-carrier-c5d'),'ordinary'),('linux',Path('/home/jkh/Src/nanolang-file-cyclic-carrier-ac877'),'public')]:
 sources=json.loads((out/group/(phase+'-source-after.json')).read_text());tools=json.loads((out/group/(phase+'-tools-after.json')).read_text());products=json.loads((out/group/(phase+'-artifacts.json')).read_text())
 for name,digest in sources.items():assert sha(tree/name)==digest,name
 for name,rec in tools.items():assert sha(Path(rec['path']))==rec['sha256'],name
 inputs={name:rec for name,rec in products.items() if Path(name).resolve().is_relative_to(tree.resolve())}
 for name,rec in inputs.items():assert sha(Path(name))==rec['sha256'],name
 current.append({'group':group,'tree':str(tree),'sources':len(sources),'tools':len(tools),'providers':len(inputs),'mismatches':0})
for name in ('/tmp/nanolang-file-cyclic-carrier-seal.py','/tmp/nanolang-file-cyclic-carrier-ac877-package-puck.py','/tmp/nanolang-file-cyclic-carrier-ac877-driver.py','/tmp/nanolang-file-cyclic-carrier-ac877-puck-launch.py','/tmp/nanolang-file-cyclic-carrier-ac877-puck-packaging.json','/tmp/nanolang-file-cyclic-carrier-c5d-first-manifest.json','/tmp/nanolang-file-cyclic-carrier-history-driver.py','/tmp/nanolang-file-cyclic-carrier-history-puck-launch.py'):
 p=Path(name);shutil.copyfile(p,out/p.name)
shutil.copytree('/tmp/nanolang-file-cyclic-carrier-puck-downloaded/history',out/'historical-git')
archive=Path('/tmp/nanolang-file-cyclic-carrier-artifacts.tar.gz')
with tarfile.open(archive,'w:gz') as tar:tar.add(store,arcname='artifacts')
write(out/'artifact-archive.json',{'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size})
write(out/'artifact-store.json',index)
write(out/'qualification-summary.json',{'production':'d4070d6729712372f83bef6d68c80a7ca9542163','first_fixture':'c5d048fd89fdc0c856155e9da6b609e7e16b2293','qualified_fixture':'ac8772ae8befb4a344992db42abe403c30b16bc5','scope':'private manual cyclic carrier protocol; no cyclic VM/native/public dispatch','whole_program_sanitizer':False,'preparation_peak_measured':False,'phases':statuses,'source_tool_pairs':pairs,'provider_pairs':provider_pairs,'current_linux':current,'unique_artifacts':len(index),'artifact_references':references,'artifact_bytes':sum(v['bytes'] for v in index.values())})
write(out/'report-sha256.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
print(json.dumps({'reports':len(json.loads((out/'report-sha256.json').read_text())),'artifacts':len(index),'references':references,'pairs':len(pairs),'archive_sha256':sha(archive)},indent=2))
