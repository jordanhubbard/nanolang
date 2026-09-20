import hashlib,json,shutil,tarfile,subprocess
from pathlib import Path
root=Path('/home/jkh/Src/nanolang-file-cyclic-dispatch-plan')
out=root/'docs/evidence/file-cyclic-dispatch';out.mkdir(parents=True,exist_ok=False)
store=Path('/tmp/nanolang-file-cyclic-dispatch-artifacts');store.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
groups={pin+'-linux':Path('/tmp/nanolang-file-cyclic-dispatch-'+pin+'-linux') for pin in ('9284','7913','100c','6a65')}
download=Path('/tmp/nanolang-file-cyclic-dispatch-puck-downloaded')
groups.update({pin+'-darwin':download/'reports'/pin for pin in ('9284','7913','100c')})
index={};references=0;pairs=[];statuses={};provider_pairs=[];current=[]
for group,source in groups.items():
 dst=out/group;dst.mkdir()
 for p in sorted(source.iterdir()):
  if p.is_file():shutil.copyfile(p,dst/p.name)
 statuses[group]=json.loads((dst/'results.json').read_text())
 expected={'6a65-linux':15,'100c-darwin':14}.get(group)
 if expected:assert len(statuses[group])==expected and all(r['status']==0 for r in statuses[group]),group
 else:assert len(statuses[group])==4 and [r['status'] for r in statuses[group]]==[0,0,0,1],group
 for p in sorted(dst.glob('*-source-before.json')):
  phase=p.name.removesuffix('-source-before.json');before=json.loads(p.read_text());after=json.loads((dst/(phase+'-source-after.json')).read_text());assert before==after,(group,phase,'source')
  tools=json.loads((dst/(phase+'-tools-before.json')).read_text());assert tools==json.loads((dst/(phase+'-tools-after.json')).read_text()),(group,phase,'tools')
  pairs.append(dict(group=group,phase=phase,sources=len(before),tools=len(tools),equal=True))
  inputs=json.loads((dst/(phase+'-inputs-before.json')).read_text());products=json.loads((dst/(phase+'-artifacts.json')).read_text())
  assert not set(inputs)-set(products),(group,phase,'missing provider')
  changed=[n for n in inputs if inputs[n]['sha256']!=products[n]['sha256']]
  assert not changed or phase=='public',(group,phase,changed)
  provider_pairs.append(dict(group=group,phase=phase,before_count=len(inputs),changed_existing=changed,created=[n for n in products if n not in inputs and Path(n).suffix in ('.o','.a')],unchanged=not changed,public_package_rebuild=phase=='public'))
 for pattern in ('*-inputs-before.json','*-artifacts.json'):
  for p in sorted(dst.glob(pattern)):
   for rec in json.loads(p.read_text()).values():
    references+=1;digest=rec['sha256'];src=download/'artifacts'/digest if group.endswith('darwin') else Path(rec['artifact'])
    assert sha(src)==digest,(group,src)
    if digest not in index:
     target=store/digest;shutil.copyfile(src,target);assert sha(target)==digest;index[digest]=dict(path=str(target),bytes=target.stat().st_size)
 if group.endswith('linux'):
  pin=group.split('-')[0];tree=Path('/home/jkh/Src/nanolang-file-cyclic-dispatch-'+pin);phase=statuses[group][-1]['phase']
  sources=json.loads((dst/(phase+'-source-after.json')).read_text());tools=json.loads((dst/(phase+'-tools-after.json')).read_text());products=json.loads((dst/(phase+'-artifacts.json')).read_text())
  for n,d in sources.items():assert sha(tree/n)==d,(group,n)
  for n,r in tools.items():assert sha(Path(r['path']))==r['sha256'],(group,n)
  providers={n:r for n,r in products.items() if Path(n).resolve().is_relative_to(tree.resolve())}
  for n,r in providers.items():assert sha(Path(n))==r['sha256'],(group,n)
  current.append(dict(group=group,tree=str(tree),sources=len(sources),tools=len(tools),providers=len(providers),mismatches=0))
for name in ('/tmp/nanolang-file-cyclic-dispatch-seal.py','/tmp/nanolang-file-cyclic-dispatch-package-puck.py','/tmp/nanolang-file-cyclic-dispatch-puck-packaging.json','/tmp/nanolang-file-cyclic-dispatch-7913-launch-first.json','/tmp/nanolang-file-cyclic-dispatch-generated-identity.json'):
 p=Path(name);shutil.copyfile(p,out/p.name)
for pin in ('9284','7913','100c'):
 for suffix in ('puck-launch.py','puck-launch.log'):
  p=Path('/tmp/nanolang-file-cyclic-dispatch-'+pin+'-'+suffix)
  if p.exists():shutil.copyfile(p,out/p.name)
production_files=subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r','4a12db5e0'],cwd=root,text=True).splitlines()
production={p:subprocess.check_output(['git','rev-parse','4a12db5e0:'+p],cwd=root,text=True).strip() for p in production_files if p.startswith('src/')}
for pin in ('9284','7913','100c','6a65'):
 tree=Path('/home/jkh/Src/nanolang-file-cyclic-dispatch-'+pin)
 for p,blob in production.items():assert subprocess.check_output(['git','rev-parse','HEAD:'+p],cwd=tree,text=True).strip()==blob,(pin,p)
write(out/'production-identity.json',production)
archive=Path('/tmp/nanolang-file-cyclic-dispatch-artifacts.tar.gz')
with tarfile.open(archive,'w:gz') as tar:tar.add(store,arcname='artifacts')
write(out/'artifact-archive.json',dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size))
write(out/'artifact-store.json',index)
write(out/'qualification-summary.json',dict(production='4a12db5e0d32d41751eecf6affd81827c4eff929',linux_fixture='6a65b87ff',darwin_fixture='100c5a1fd',scope='private matched cyclic VM/native dispatch; public and source cyclic admission remain refused',whole_program_sanitizer=False,preparation_peak_measured=False,phases=statuses,source_tool_pairs=pairs,provider_pairs=provider_pairs,current_linux=current,unique_artifacts=len(index),artifact_references=references,artifact_bytes=sum(v['bytes'] for v in index.values())))
write(out/'report-sha256.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
print(json.dumps(dict(reports=len(json.loads((out/'report-sha256.json').read_text())),artifacts=len(index),references=references,pairs=len(pairs),archive_sha256=sha(archive)),indent=2),flush=True)
