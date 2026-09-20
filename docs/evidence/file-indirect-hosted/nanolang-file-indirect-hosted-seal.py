import hashlib,json,shutil,subprocess,tarfile
from pathlib import Path
root=Path('/home/jkh/Src/nanolang-file-indirect-hosted-plan');out=root/'docs/evidence/file-indirect-hosted';out.mkdir(parents=True,exist_ok=False)
store=Path('/tmp/nanolang-file-indirect-hosted-artifacts');store.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
groups={'linux-first':Path('/tmp/nanolang-file-indirect-hosted-225b-linux'),'linux':Path('/tmp/nanolang-file-indirect-hosted-5c26-linux'),'darwin-first':Path('/tmp/nanolang-file-indirect-hosted-puck-downloaded/reports/darwin-first'),'darwin':Path('/tmp/nanolang-file-indirect-hosted-puck-downloaded/reports/darwin')}
index={};references=0;pairs=[];statuses={};provider_pairs=[]
for group,source in groups.items():
 dst=out/group;dst.mkdir()
 for p in source.iterdir():
  if p.is_file():shutil.copyfile(p,dst/p.name)
 statuses[group]=json.loads((dst/'results.json').read_text())
 assert (len(statuses[group])==4 and [x['status'] for x in statuses[group]]==[0,0,0,1]) if group.endswith('first') else (len(statuses[group])==(16 if group=='linux' else 15) and all(x['status']==0 for x in statuses[group]))
 for p in dst.glob('*-source-before.json'):
  phase=p.name.removesuffix('-source-before.json');before=json.loads(p.read_text());after=json.loads((dst/(phase+'-source-after.json')).read_text());assert before==after
  tools=json.loads((dst/(phase+'-tools-before.json')).read_text());assert tools==json.loads((dst/(phase+'-tools-after.json')).read_text())
  pairs.append({'group':group,'phase':phase,'sources':len(before),'tools':len(tools),'equal':True})
  inputs=json.loads((dst/(phase+'-inputs-before.json')).read_text());artifacts=json.loads((dst/(phase+'-artifacts.json')).read_text())
  for name,record in inputs.items():assert name in artifacts and record['sha256']==artifacts[name]['sha256'],(group,phase,name)
  provider_pairs.append({'group':group,'phase':phase,'existing_input_count':len(inputs),'unchanged':True,'creation_during_setup':phase=='setup'})
 for pattern in ('*-artifacts.json','*-inputs-before.json'):
  for p in dst.glob(pattern):
   for record in json.loads(p.read_text()).values():
    references+=1;digest=record['sha256'];src=Path(record['artifact']) if group.startswith('linux') else Path('/tmp/nanolang-file-indirect-hosted-puck-downloaded/artifacts')/digest
    assert sha(src)==digest
    if digest not in index:
     target=store/digest;shutil.copyfile(src,target);assert sha(target)==digest;index[digest]={'path':str(target),'bytes':target.stat().st_size}
current=[]
for group,tree in [('linux-first',Path('/home/jkh/Src/nanolang-file-indirect-hosted-qualified-225b')),('linux',Path('/home/jkh/Src/nanolang-file-indirect-hosted-qualified-5c26'))]:
 src=json.loads((out/group/'ordinary-source-before.json').read_text());tools=json.loads((out/group/'ordinary-tools-before.json').read_text())
 for name,digest in src.items():assert sha(tree/name)==digest,name
 for name,info in tools.items():assert sha(Path(info['path']))==info['sha256'],name
 inputs=json.loads((out/group/'ordinary-inputs-before.json').read_text())
 for name,info in inputs.items():assert sha(Path(name))==info['sha256'],name
 current.append({'group':group,'tree':str(tree),'sources':len(src),'tools':len(tools),'providers':len(inputs),'mismatches':0})
for name in ('/tmp/nanolang-file-indirect-hosted-seal.py','/tmp/nanolang-file-indirect-hosted-package-puck.py','/tmp/nanolang-file-indirect-hosted-5c26-driver.py','/tmp/nanolang-file-indirect-hosted-5c26-puck-launch.py','/tmp/nanolang-file-indirect-hosted-puck-packaging.json'):
 p=Path(name);shutil.copyfile(p,out/p.name)
archive=Path('/tmp/nanolang-file-indirect-hosted-artifacts.tar.gz')
with tarfile.open(archive,'w:gz') as tar:tar.add(store,arcname='artifacts')
write(out/'artifact-archive.json',{'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size})
write(out/'artifact-store.json',index)
write(out/'qualification-summary.json',{'production':'95c04f751a909b122cdbc0ac207516b2496511a7','qualified_fixture':'5c26faaa0bd3892f3528c3732a5791ffc925ea4b','first_fixture':'225b67a6da942b32896ab167d664a95040701281','query_only':True,'runtime_admitted':False,'whole_program_sanitizer':False,'phases':statuses,'source_tool_pairs':pairs,'provider_pairs':provider_pairs,'current_linux':current,'unique_artifacts':len(index),'artifact_references':references,'artifact_bytes':sum(x['bytes'] for x in index.values())})
manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()};write(out/'report-sha256.json',manifest)
print(json.dumps({'reports':len(manifest),'unique_artifacts':len(index),'references':references,'pairs':len(pairs),'archive_sha256':sha(archive)},indent=2))
