import hashlib,json,shutil,subprocess,tarfile
from pathlib import Path
root=Path('/home/jkh/Src/nanolang-ownership-declaration-qualified');out=root/'docs/evidence/ownership-declaration-projection';out.mkdir(parents=True,exist_ok=False)
store=Path('/tmp/nanolang-mixed-declaration-artifacts');store.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
groups={'linux-first':Path('/tmp/nanolang-mixed-declaration-dd5a-linux'),'linux-corrected':Path('/tmp/nanolang-mixed-declaration-5a367-linux'),'darwin-corrected':Path('/tmp/nanolang-mixed-declaration-puck-downloaded/reports/darwin-corrected')}
expected={'linux-first':(5,1),'linux-corrected':(15,0),'darwin-corrected':(17,0)}
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
  providers.append({'group':label,'phase':phase,'existing':len(ins),'changed_existing':changed,'added':len(added),'note':'Query translation units rebuild with selected fixture flags; neighbor targets add separately retained products.'})
 for pattern in ('*-artifacts.json','*-inputs-before.json'):
  for p in dst.glob(pattern):
   for v in load(p).values():
    refs+=1;h=v['sha256'];f=Path(v['artifact']) if label.startswith('linux') else Path('/tmp/nanolang-mixed-declaration-puck-downloaded/artifacts')/h
    if str(f) not in seen:assert sha(f)==h;seen.add(str(f))
    if h not in index:target=store/h;shutil.copyfile(f,target);assert sha(target)==h;index[h]={'path':str(target),'bytes':target.stat().st_size}
 if label.startswith('linux'):
  tree=Path('/home/jkh/Src/nanolang-mixed-declaration-'+('dd5a' if label=='linux-first' else '5a367'));lastphase=results[-1]['phase'];source=load(dst/(lastphase+'-source-after.json'));tools=load(dst/(lastphase+'-tools-after.json'));products=load(dst/(lastphase+'-artifacts.json'))
  for p,h in source.items():assert sha(tree/p)==h,(label,p)
  for p,v in tools.items():assert sha(v['path'])==v['sha256'],(label,p)
  for p,v in products.items():assert sha(p)==v['sha256'],(label,p)
  current.append({'group':label,'root':str(tree),'last':lastphase,'sources':len(source),'tools':len(tools),'products':len(products),'mismatches':0})
for f in ('/tmp/nanolang-mixed-declaration-seal.py','/tmp/nanolang-mixed-declaration-package-puck.py','/tmp/nanolang-mixed-declaration-puck-packaging.json','/tmp/nanolang-mixed-declaration-identities.json','/tmp/nanolang-mixed-declaration-driver.py','/tmp/nanolang-mixed-declaration-5a367-puck-launch.py','/tmp/nanolang-mixed-declaration-reuse-linux.py','/tmp/nanolang-mixed-declaration-linux-reuse.json'):
 shutil.copyfile(f,out/Path(f).name)
shutil.copyfile('/tmp/nanolang-mixed-declaration-puck-downloaded/reports/current-puck.json',out/'current-puck.json')
write(out/'artifact-store.json',index);write(out/'qualification-summary.json',{'source':'dd5a3ab637959c5cae1c3fd0a6212b516bb64265','corrected_fixture':'5a367e16e61587a424f7572cfd9bb68b1b4ee68f','production':'31c68af10c3c970cf4fd9289fb0b03613bba1e6d','phases':statuses,'source_tool_pairs':pairs,'provider_deltas':providers,'current_linux':current,'unique_artifacts':len(index),'artifact_references':refs,'artifact_bytes':sum(v['bytes'] for v in index.values()),'limits':['Allocation hooks and query sanitizers cover ownership_contracts.c and nvm_v2_layouts.c; ordinary common providers and reference helper allocation remain outside that scope.','Selected tools and SDK/header identities are recorded, not a complete SDK/OS/toolchain closure.','Unchanged Python legacy neighbors delete their own temporary native fixtures; retained logs establish their outcomes but no deleted-binary archive is claimed.','Non-admitting declaration query only; no source/runtime/public mixed admission or full5.1 closure.']})
archive=Path('/tmp/nanolang-mixed-declaration-artifacts.tar.gz')
with tarfile.open(archive,'w:gz') as t:t.add(store,arcname='artifacts')
write(out/'artifact-archive.json',{'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size})
manifest={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()};write(out/'report-sha256.json',manifest)
print(json.dumps({'reports':len(manifest),'artifacts':len(index),'references':refs,'pairs':len(pairs),'archive':sha(archive)},indent=2))
