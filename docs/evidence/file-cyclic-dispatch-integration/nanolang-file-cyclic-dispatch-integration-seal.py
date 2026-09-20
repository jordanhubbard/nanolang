import hashlib,json,shutil,subprocess,tarfile
from pathlib import Path
root=Path('/home/jkh/Src/nanolang-file-cyclic-dispatch-ready');out=root/'docs/evidence/file-cyclic-dispatch-integration';out.mkdir(parents=True,exist_ok=False)
store=Path('/tmp/nanolang-file-cyclic-dispatch-integration-artifacts');store.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
groups={'linux':Path('/tmp/nanolang-file-cyclic-dispatch-3ea2-linux'),'darwin':Path('/tmp/nanolang-file-cyclic-dispatch-integration-puck-downloaded/reports/3ea2')}
index={};references=0;pairs=[];statuses={};provider_pairs=[]
for group,source in groups.items():
 dst=out/group;dst.mkdir()
 for p in source.iterdir():
  if p.is_file():shutil.copyfile(p,dst/p.name)
 statuses[group]=json.loads((dst/'results.json').read_text())
 assert len(statuses[group])==9 and all(x['status']==0 for x in statuses[group])
 for p in dst.glob('*-source-before.json'):
  phase=p.name.removesuffix('-source-before.json');before=json.loads(p.read_text());after=json.loads((dst/(phase+'-source-after.json')).read_text());assert before==after
  tools=json.loads((dst/(phase+'-tools-before.json')).read_text());assert tools==json.loads((dst/(phase+'-tools-after.json')).read_text())
  pairs.append({'group':group,'phase':phase,'sources':len(before),'tools':len(tools),'equal':True})
  inputs=json.loads((dst/(phase+'-inputs-before.json')).read_text());artifacts=json.loads((dst/(phase+'-artifacts.json')).read_text())
  changed=[]
  for name,record in inputs.items():
   assert name in artifacts,(group,phase,name)
   if record['sha256']!=artifacts[name]['sha256']:changed.append(name)
  assert not changed or phase=='public',(group,phase,changed)
  provider_pairs.append({'group':group,'phase':phase,'existing_input_count':len(inputs),'unchanged':not changed,'changed_existing':changed,'creation_during_setup':phase=='setup'})
 for pattern in ('*-artifacts.json','*-inputs-before.json'):
  for p in dst.glob(pattern):
   for record in json.loads(p.read_text()).values():
    references+=1;digest=record['sha256'];src=Path(record['artifact']) if group.startswith('linux') else Path('/tmp/nanolang-file-cyclic-dispatch-integration-puck-downloaded/artifacts')/digest
    assert sha(src)==digest
    if digest not in index:
     target=store/digest;shutil.copyfile(src,target);assert sha(target)==digest;index[digest]={'path':str(target),'bytes':target.stat().st_size}
current=[]
for group,tree in [('linux',Path('/home/jkh/Src/nanolang-file-cyclic-dispatch-3ea2'))]:
 src=json.loads((out/group/'archive-boundary-source-after.json').read_text());tools=json.loads((out/group/'archive-boundary-tools-after.json').read_text())
 for name,digest in src.items():assert sha(tree/name)==digest,name
 for name,info in tools.items():assert sha(Path(info['path']))==info['sha256'],name
 inputs=json.loads((out/group/'archive-boundary-inputs-before.json').read_text())
 for name,info in inputs.items():assert sha(Path(name))==info['sha256'],name
 current.append({'group':group,'tree':str(tree),'sources':len(src),'tools':len(tools),'providers':len(inputs),'mismatches':0})
for name in ('/tmp/nanolang-file-cyclic-dispatch-integration-seal.py','/tmp/nanolang-file-cyclic-dispatch-integration-package-puck.py','/tmp/nanolang-file-cyclic-dispatch-integration-driver.py','/tmp/nanolang-file-cyclic-dispatch-3ea2-puck-launch.py','/tmp/nanolang-file-cyclic-dispatch-integration-puck-packaging.json','/tmp/nanolang-file-cyclic-dispatch-incoming-identity.json','/tmp/nanolang-file-cyclic-dispatch-python-current.json'):
 p=Path(name);shutil.copyfile(p,out/p.name)
archive=Path('/tmp/nanolang-file-cyclic-dispatch-integration-artifacts.tar.gz')
with tarfile.open(archive,'w:gz') as tar:tar.add(store,arcname='artifacts')
write(out/'artifact-archive.json',{'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size})
write(out/'artifact-store.json',index)
write(out/'qualification-summary.json',{'production':'4a12db5e0d32d41751eecf6affd81827c4eff929','integration_pin':'3ea2dfa983fe67fb97e625781e90e2d42e8c547f','qualified_fixture':'6a65b87ff','private_cyclic_dispatch':True,'public_cyclic_admission':False,'indirect_runtime_admission':False,'integrated_sanitizers':False,'whole_program_sanitizer':False,'historical_linux_python_hash_captured':False,'top_level_driver_copy':'prospectively corrected after gates; per-host driver.py preserves actual executed runner','phases':statuses,'source_tool_pairs':pairs,'provider_pairs':provider_pairs,'current_linux':current,'unique_artifacts':len(index),'artifact_references':references,'artifact_bytes':sum(x['bytes'] for x in index.values())})
manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()};write(out/'report-sha256.json',manifest)
print(json.dumps({'reports':len(manifest),'unique_artifacts':len(index),'references':references,'pairs':len(pairs),'archive_sha256':sha(archive)},indent=2))
