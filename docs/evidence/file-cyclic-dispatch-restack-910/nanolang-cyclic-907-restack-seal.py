import hashlib,json,pathlib,shutil,subprocess,shlex
root=pathlib.Path('/home/jkh/Src/nanolang-file-cyclic-dispatch-ready');out=root/'docs/evidence/file-cyclic-dispatch-restack-910';out.mkdir(exist_ok=False);cas=pathlib.Path('/tmp/nanolang-cyclic-907-restack-artifacts');cas.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,j):p.write_text(json.dumps(j,indent=2)+'\n')
store={};pairs=[];refs=0;current=[];trace_counts={};remote={}
for platform,source,tree in [('linux',pathlib.Path('/tmp/nanolang-cyclic-907-restack-linux'),pathlib.Path('/home/jkh/Src/nanolang-cyclic-907-restack-3df')),('puck',pathlib.Path('/tmp/nanolang-cyclic-907-restack-download/nanolang-cyclic-907-restack-puck'),pathlib.Path('/tmp/nanolang-cyclic-907-restack-3df-py314'))]:
 dest=out/platform;dest.mkdir()
 for p in source.iterdir():
  if p.is_file():shutil.copy2(p,dest/p.name)
 results=json.loads((source/'results.json').read_text());assert [x['phase'] for x in results]==['setup','configuration','discovery','ordinary'];assert all(x['status']==0 for x in results)
 for phase in ['setup','configuration','discovery','ordinary']:
  s=json.loads((source/(phase+'-source-before.json')).read_text());t=json.loads((source/(phase+'-tools-before.json')).read_text());assert s==json.loads((source/(phase+'-source-after.json')).read_text());assert t==json.loads((source/(phase+'-tools-after.json')).read_text())
  ins=json.loads((source/(phase+'-inputs-before.json')).read_text());arts=json.loads((source/(phase+'-artifacts.json')).read_text());assert all(k in arts and v['sha256']==arts[k]['sha256'] for k,v in ins.items());pairs.append({'host':platform,'phase':phase,'sources':len(s),'tools':len(t),'existing_providers':len(ins),'equal':True})
  for record in list(ins.values())+list(arts.values()):
   refs+=1;h=record['sha256'];p=source/'artifacts'/h;assert sha(p)==h
   q=cas/h
   if not q.exists():shutil.copy2(p,q)
   store[h]={'path':str(q),'bytes':q.stat().st_size}
  terminal=json.loads((source/(phase+'-terminal.json')).read_text());assert terminal['returncode']==0 and not terminal['timeout'] and not terminal['errors'] and terminal['group_disappeared']
 actual={str(tree/k):v for k,v in s.items()};actual.update({v['path']:v['sha256'] for v in t.values()});actual.update({k:v['sha256'] for k,v in ins.items()})
 if platform=='linux':assert all(sha(k)==v for k,v in actual.items())
 else:remote=actual
 current.append({'host':platform,'tree':str(tree),'sources':len(s),'tools':len(t),'providers':len(ins),'match':True})
 capture=next(k for k in arts if k.endswith('linked-capture-run-stdout.log'));cap=pathlib.Path(store[arts[capture]['sha256']]['path']).read_text();traces=[x for x in cap.splitlines() if x.startswith('TRACE ')];assert 'PASS cyclic VM capture: 23 exact modules' in cap
 trace_counts[platform]={'vm_traces':len(traces),'exact_modules':23}
 for opt in ['O0','O2']:
  name=next(k for k in arts if k.endswith('linked-'+opt+'-run-stdout.log'));text=pathlib.Path(store[arts[name]['sha256']]['path']).read_text();native=[x for x in text.splitlines() if x.startswith('TRACE ')];assert native==traces;trace_counts[platform][opt]=len(native)
code="import json,hashlib,sys,pathlib\nm=json.load(sys.stdin)\nfor p,w in m.items():\n h=hashlib.sha256()\n with open(p,'rb') as f:\n  for b in iter(lambda:f.read(1048576),b''):h.update(b)\n assert h.hexdigest()==w,p\nprint(len(m))\n"
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','puck.local','python3 -c '+shlex.quote(code)],input=json.dumps(remote).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180);assert r.returncode==0,r.stderr
for name in ['nanolang-cyclic-907-puck-first.log','nanolang-cyclic-907-puck-launch.py','nanolang-cyclic-907-puck-launch-py314.py','nanolang-cyclic-907-restack-seal.py']:
 shutil.copy2(pathlib.Path('/tmp')/name,out/name)
archive=pathlib.Path('/tmp/nanolang-cyclic-907-restack-puck.tar.gz');h=sha(archive);assert h==pathlib.Path('/tmp/nanolang-cyclic-907-puck-archive.sha256').read_text().split()[0]
write(out/'artifact-store.json',store);write(out/'qualification-summary.json',{'pin':'3df2998922ff6cf1168038ce4b05e0ad248ed021','pairs':pairs,'references':refs,'unique_artifacts':len(store),'artifact_bytes':sum(v['bytes'] for v in store.values()),'current':current,'traces':trace_counts,'archive':{'path':str(archive),'bytes':archive.stat().st_size,'sha256':h},'scope':'Fresh ordinary linked cyclic VM/native O0/O2 only; unchanged original sanitizer acceptance remains at old pins.'})
manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()};write(out/'report-sha256.json',manifest);print(json.dumps({'reports':len(manifest),'artifacts':len(store),'bytes':sum(v['bytes'] for v in store.values()),'references':refs,'pairs':len(pairs),'current':current,'traces':trace_counts},indent=2))
