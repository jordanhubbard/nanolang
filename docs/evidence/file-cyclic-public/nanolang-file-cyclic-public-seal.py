import hashlib,json,shutil,subprocess,tarfile
from pathlib import Path
root=Path('/home/jkh/Src/nanolang-file-cyclic-public-integrated');out=root/'docs/evidence/file-cyclic-public';out.mkdir(parents=True,exist_ok=False)
store=Path('/tmp/nanolang-file-cyclic-public-artifacts');store.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
groups={}
for label,suffix in [('linux-first','aff3'),('linux-header-corrected','414c'),('linux-config-corrected','414c-config'),('linux-native-counter','a11c')]:groups[label]=Path('/tmp/nanolang-file-cyclic-public-'+suffix+'-linux')
for label in ('darwin-first','darwin-schema-first','darwin-header-corrected','darwin-native-corrected'):groups[label]=Path('/tmp/nanolang-file-cyclic-public-puck-downloaded/reports')/label
expected={'linux-first':(4,1),'linux-header-corrected':(7,1),'linux-config-corrected':(14,0),'linux-native-counter':(3,0),'darwin-first':(4,1),'darwin-schema-first':(1,1),'darwin-header-corrected':(7,1),'darwin-native-corrected':(13,0)}
index={};seen=set();refs=0;pairs=[];providers=[];statuses={};current=[]
for label,src in groups.items():
 dst=out/label;dst.mkdir()
 for p in src.iterdir():
  if p.is_file():shutil.copyfile(p,dst/p.name)
 results=load(dst/'results.json');count,last=expected[label];assert len(results)==count and results[-1]['status']==last and all(p['status']==0 for p in results[:-1]),label;statuses[label]=results
 for result in results:
  phase=result['phase'];before=load(dst/(phase+'-source-before.json'));after=load(dst/(phase+'-source-after.json'));assert before==after
  tools=load(dst/(phase+'-tools-before.json'));assert tools==load(dst/(phase+'-tools-after.json'))
  terminal=load(dst/(phase+'-terminal.json'));assert terminal['returncode']==result['status'] and not terminal['timeout'] and terminal['leader_reaped'] and terminal['group_disappeared'] and not terminal['errors']
  pairs.append({'group':label,'phase':phase,'sources':len(before),'tools':len(tools),'equal':True})
  ins=load(dst/(phase+'-inputs-before.json'));products=load(dst/(phase+'-artifacts.json'))
  changed=[p for p,v in ins.items() if p not in products or v['sha256']!=products[p]['sha256']];added=[p for p in products if p not in ins]
  providers.append({'group':label,'phase':phase,'existing':len(ins),'changed_existing':changed,'added':len(added),'note':'Full install rebuilds providers with selected per-phase flags; no global provider immutability claim.'})
 for pattern in ('*-artifacts.json','*-inputs-before.json'):
  for p in dst.glob(pattern):
   for v in load(p).values():
    refs+=1;h=v['sha256'];f=Path(v['artifact']) if label.startswith('linux') else Path('/tmp/nanolang-file-cyclic-public-puck-downloaded/artifacts')/h
    if str(f) not in seen:assert sha(f)==h;seen.add(str(f))
    if h not in index:target=store/h;shutil.copyfile(f,target);assert sha(target)==h;index[h]={'path':str(target),'bytes':target.stat().st_size}
 if label.startswith('linux'):
  suffix={'linux-first':'aff3','linux-header-corrected':'414c','linux-config-corrected':'414c-config','linux-native-counter':'a11c'}[label];tree=Path('/home/jkh/Src/nanolang-file-cyclic-public-qualified-'+suffix);lastphase=results[-1]['phase'];source=load(dst/(lastphase+'-source-after.json'));tools=load(dst/(lastphase+'-tools-after.json'));products=load(dst/(lastphase+'-artifacts.json'))
  for p,h in source.items():assert sha(tree/p)==h,(label,p)
  for p,v in tools.items():assert sha(v['path'])==v['sha256'],(label,p)
  for p,v in products.items():assert sha(p)==v['sha256'],(label,p)
  current.append({'group':label,'root':str(tree),'last':lastphase,'sources':len(source),'tools':len(tools),'products':len(products),'mismatches':0})
for f in ('/tmp/nanolang-file-cyclic-public-seal.py','/tmp/nanolang-file-cyclic-public-package-puck.py','/tmp/nanolang-file-cyclic-public-puck-packaging.json','/tmp/nanolang-file-cyclic-public-restack-identities.json'):
 shutil.copyfile(f,out/Path(f).name)
for pattern in ('/tmp/nanolang-file-cyclic-public-*-driver.py','/tmp/nanolang-file-cyclic-public-*-puck-launch.py','/tmp/nanolang-file-cyclic-public-*-puck-launch.log'):
 for p in Path('/tmp').glob(Path(pattern).name):shutil.copyfile(p,out/p.name)
shutil.copyfile('/tmp/nanolang-file-cyclic-public-puck-downloaded/reports/current-puck.json',out/'current-puck.json')
write(out/'artifact-store.json',index);write(out/'qualification-summary.json',{'source':'414c1042fefeefc4c51392c3e4276f03ee6a8750','native_assertion':'a11cdad02b43c9d4eba24903475e223600b3a9f7','phases':statuses,'source_tool_pairs':pairs,'provider_deltas':providers,'current_linux':current,'unique_artifacts':len(index),'artifact_references':refs,'artifact_bytes':sum(v['bytes'] for v in index.values()),'limits':['No full SDK/OS identity claim.','Allocation hooks cover listed rebuilt fixture providers and generated C; install rebuilds separately change common provider bytes.','The capture recipe ignores caller CFLAGS; compiler-command selection preserves this gate, separate full5.1 follow-up stays open.','No source/indirect/richer-borrow admission or full5.1 closure.']})
archive=Path('/tmp/nanolang-file-cyclic-public-artifacts.tar.gz')
with tarfile.open(archive,'w:gz') as t:t.add(store,arcname='artifacts')
write(out/'artifact-archive.json',{'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size})
manifest={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()};write(out/'report-sha256.json',manifest)
print(json.dumps({'reports':len(manifest),'artifacts':len(index),'references':refs,'pairs':len(pairs),'archive':sha(archive)},indent=2))
