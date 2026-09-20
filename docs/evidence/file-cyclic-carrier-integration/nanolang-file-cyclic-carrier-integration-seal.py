import hashlib,json,shutil,subprocess,tarfile
from pathlib import Path
root=Path('/home/jkh/Src/nanolang-file-cyclic-carrier-ready');out=root/'docs/evidence/file-cyclic-carrier-integration';out.mkdir(parents=True,exist_ok=False)
store=Path('/tmp/nanolang-file-cyclic-carrier-integration-artifacts');store.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
groups={'linux':Path('/tmp/nanolang-file-cyclic-carrier-5c71-linux'),'darwin':Path('/tmp/nanolang-file-cyclic-carrier-integration-puck-downloaded/reports/darwin')}
index={};references=0;pairs=[];statuses={};provider_pairs=[]
for group,source in groups.items():
 dst=out/group;dst.mkdir()
 for p in source.iterdir():
  if p.is_file():shutil.copyfile(p,dst/p.name)
 statuses[group]=json.loads((dst/'results.json').read_text())
 assert len(statuses[group])==6 and all(x['status']==0 for x in statuses[group])
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
    references+=1;digest=record['sha256'];src=Path(record['artifact']) if group.startswith('linux') else Path('/tmp/nanolang-file-cyclic-carrier-integration-puck-downloaded/artifacts')/digest
    assert sha(src)==digest
    if digest not in index:
     target=store/digest;shutil.copyfile(src,target);assert sha(target)==digest;index[digest]={'path':str(target),'bytes':target.stat().st_size}
current=[]
for group,tree in [('linux',Path('/home/jkh/Src/nanolang-file-cyclic-carrier-5c71'))]:
 src=json.loads((out/group/'ordinary-source-before.json').read_text());tools=json.loads((out/group/'ordinary-tools-before.json').read_text())
 for name,digest in src.items():assert sha(tree/name)==digest,name
 for name,info in tools.items():assert sha(Path(info['path']))==info['sha256'],name
 inputs=json.loads((out/group/'ordinary-inputs-before.json').read_text())
 for name,info in inputs.items():assert sha(Path(name))==info['sha256'],name
 current.append({'group':group,'tree':str(tree),'sources':len(src),'tools':len(tools),'providers':len(inputs),'mismatches':0})
for name in ('/tmp/nanolang-file-cyclic-carrier-integration-seal.py','/tmp/nanolang-file-cyclic-carrier-integration-package-puck.py','/tmp/nanolang-file-cyclic-carrier-integration-driver.py','/tmp/nanolang-file-cyclic-carrier-5c71-puck-launch.py','/tmp/nanolang-file-cyclic-carrier-integration-puck-packaging.json'):
 p=Path(name);shutil.copyfile(p,out/p.name)
archive=Path('/tmp/nanolang-file-cyclic-carrier-integration-artifacts.tar.gz')
with tarfile.open(archive,'w:gz') as tar:tar.add(store,arcname='artifacts')
write(out/'artifact-archive.json',{'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size})
write(out/'artifact-store.json',index)
write(out/'qualification-summary.json',{'production':'d4070d6729712372f83bef6d68c80a7ca9542163','integration_pin':'5c71b7b38292ac368cbb425be802dc172495d76a','qualified_fixture':'ac8772ae8befb4a344992db42abe403c30b16bc5','cyclic_dispatch':False,'manual_private_carrier':True,'acyclic_scalar_control':True,'integrated_sanitizers':False,'runtime_admitted':False,'whole_program_sanitizer':False,'phases':statuses,'source_tool_pairs':pairs,'provider_pairs':provider_pairs,'current_linux':current,'unique_artifacts':len(index),'artifact_references':references,'artifact_bytes':sum(x['bytes'] for x in index.values())})
manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()};write(out/'report-sha256.json',manifest)
print(json.dumps({'reports':len(manifest),'unique_artifacts':len(index),'references':references,'pairs':len(pairs),'archive_sha256':sha(archive)},indent=2))
