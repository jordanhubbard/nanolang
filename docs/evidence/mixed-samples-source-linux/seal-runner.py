import pathlib as p,json,hashlib,shutil,subprocess
root=p.Path.cwd(); out=root/'docs/evidence/mixed-samples-source-linux';out.mkdir(exist_ok=False)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
paths={'bootstrap':'/tmp/nanolang-mixed-source-bootstrap-70c511dad','source':'/tmp/nanolang-mixed-source-qualified-70c511dad','runtime':'/tmp/nanolang-mixed-source-runtime-70c511dad','service-corrected':'/tmp/nanolang-mixed-source-service-corrected-70c511dad'}
reports={}
for label,s in paths.items():
 base=p.Path(s)
 for f in sorted(base.rglob('*')):
  if f.is_file() and f.suffix in ('.json','.log','.py','.mk') and 'native' not in f.relative_to(base).parts:
   rel=p.Path(label)/f.relative_to(base); dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest);assert sha(f)==sha(dest);reports[str(rel)]=sha(dest)
for label,s in {'source-fixtures':'/tmp/nano-source-borrows-xaj6ccfr','runtime-native':paths['runtime']+'/native','service-fixtures':'/tmp/nano-service-module-b775aweb'}.items():
 base=p.Path(s);assert base.is_dir(); data={str(f):{'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(base.rglob('*')) if f.is_file()}; dest=out/(label+'-retained.json');dest.write_text(json.dumps(data,indent=2)+'\n');reports[dest.name]=sha(dest)
rows=json.load(open('/tmp/nanolang-mixed-source-ledger-all-current.json')); prefixes=['78c9ce','24c910','0b47e5','9c4df6','8f91de','6f81cf']; selected=[{'id':r['id'],'state':r['state']} for r in rows if any(v in r['id'] for v in prefixes)];assert len(selected)==6 and all(r['state']=='completed' for r in selected)
f=out/'prior-children-current.json';f.write_text(json.dumps(selected,indent=2)+'\n');reports[f.name]=sha(f)
assert json.load(open(paths['source']+'/status.json'))['status']==0
assert json.load(open(paths['service-corrected']+'/status.json'))['steps'][0]['status']==0
for label in paths:
 s=json.load(open(paths[label]+'/status.json'));assert s['sources_unchanged'] and s['clean']
f=out/'seal-runner.py';shutil.copyfile(__file__,f);reports[f.name]=sha(f)
(out/'manifest.json').write_text(json.dumps({'qualification_pin':'70c511dad439f77eba15fe0bdc189e90ff9c9486','reports':reports,'scope':'Linux only; setup cache changes retained separately; no Darwin or full release claim'},indent=2)+'\n')
for rel,h in reports.items():assert sha(out/rel)==h
print(len(reports),'sealed reports')
