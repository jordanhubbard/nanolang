import hashlib,json,shutil,subprocess,tarfile
from pathlib import Path
root=Path('/home/jkh/Src/nanolang-file-private-native');out=root/'docs/evidence/file-private-native';out.mkdir(parents=True,exist_ok=False);store=Path('/tmp/nanolang-file-native-artifacts');store.mkdir(exist_ok=False)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();index={};refs=0;pairs=[];statuses={}
groups={'linux-first':Path('/tmp/nanolang-file-native-acff-linux'),'linux':Path('/tmp/nanolang-file-native-4e8-linux'),'linux-corrected':Path('/tmp/nanolang-file-native-6fdb-linux'),'darwin':Path('/tmp/nanolang-file-native-puck-downloaded/reports/darwin'),'darwin-corrected':Path('/tmp/nanolang-file-native-puck-downloaded/reports/darwin-corrected')}
for group,source in groups.items():
 target=out/group;target.mkdir()
 for f in source.iterdir():
  if f.is_file():shutil.copy2(f,target/f.name)
 statuses[group]=json.loads((target/'results.json').read_text())
 for f in target.glob('*-source-before.json'):
  phase=f.name.removesuffix('-source-before.json');before=json.loads(f.read_text());after=json.loads((target/(phase+'-source-after.json')).read_text());assert before==after
  tools=json.loads((target/(phase+'-tools-before.json')).read_text());assert tools==json.loads((target/(phase+'-tools-after.json')).read_text())
  pairs.append({'group':group,'phase':phase,'sources':len(before),'tools':len(tools),'equal':True})
 for pattern in ['*-artifacts.json','*-inputs-before.json']:
  for f in target.glob(pattern):
   for record in json.loads(f.read_text()).values():
    refs+=1;sha=record['sha256'];src=Path(record['artifact']) if group.startswith('linux') else Path('/tmp/nanolang-file-native-puck-downloaded/artifacts')/sha
    assert h(src)==sha
    if sha not in index:
     dst=store/sha;shutil.copyfile(src,dst);assert h(dst)==sha;index[sha]={'path':str(dst),'bytes':dst.stat().st_size}
extra=['/tmp/nanolang-file-native-darwin-preparation-failure.json','/tmp/nanolang-file-native-4e8-provider-reuse.json','/tmp/nanolang-file-native-6fdb-provider-reuse.json','/tmp/nanolang-file-native-generated-byte-equivalence.json','/tmp/nanolang-file-native-puck-packaging.json','/tmp/nanolang-file-native-package-puck.py','/tmp/nanolang-file-native-acff-driver.py','/tmp/nanolang-file-native-4e8-puck-checked-launch.py','/tmp/nanolang-file-native-6fdb-puck-launch.py','/tmp/nanolang-file-native-6fdb-puck-ready-launch.py','/tmp/nanolang-file-native-puck-downloaded/reports/nanolang-file-native-6fdb-puck-ready-provider-reuse.json']
for f in extra:shutil.copy2(f,out/Path(f).name)
for path in ['/tmp/nanolang-file-native-4e8-puck-launch','/tmp/nanolang-file-native-6fdb-puck-launch','/tmp/nanolang-file-native-6fdb-puck-ready-launch']:
 shutil.copytree(path,out/Path(path).name)
current=[]
for group,tree,phase in [('linux-first','/home/jkh/Src/nanolang-file-native-acff','normal'),('linux','/home/jkh/Src/nanolang-file-native-4e8','normal'),('linux-corrected','/home/jkh/Src/nanolang-file-native-6fdb','clang')]:
 source=json.loads((out/group/(phase+'-source-before.json')).read_text());tools=json.loads((out/group/(phase+'-tools-before.json')).read_text())
 for n,sha in source.items():assert h(Path(tree)/n)==sha,n
 for n,d in tools.items():assert h(Path(d['path']))==d['sha256'],n
 current.append({'group':group,'tree':tree,'sources':len(source),'tools':len(tools),'mismatches':0})
(out/'artifact-store.json').write_text(json.dumps(index,sort_keys=True,indent=2)+'\n')
summary={'production':'e501b6efc','whitespace_correction':'6fdb7a60e','passed_earlier_fixture':'4e8ec0d67','initial_fixture':'acff55b42','source_tool_pairs':pairs,'phases':statuses,'unique_artifacts':len(index),'artifact_references':refs,'bytes':sum(d['bytes'] for d in index.values()),'current_linux':current,'private_native':True,'public_admission':False,'whole_program_sanitizer':False}
(out/'qualification-summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
manifest={str(f.relative_to(out)):h(f) for f in sorted(out.rglob('*')) if f.is_file()};(out/'report-sha256.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n');print(json.dumps({'reports':len(manifest),'artifacts':len(index),'references':refs,'pairs':len(pairs)},indent=2))
