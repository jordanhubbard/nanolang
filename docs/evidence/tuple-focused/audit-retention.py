from pathlib import Path
import hashlib,json
base=Path('/home/jkh/nanolang-qualification');store=base/'tuple-focused-artifacts'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
objects={p.name:p.stat().st_size for p in store.iterdir() if p.is_file()}
for h in objects:assert len(h)==64 and sha(store/h)==h,h
refs=[];pairs=[];terminals=[];reports={}
def walk(x,p):
 if isinstance(x,dict):
  if 'archive' in x and 'sha256' in x:
   h=x['sha256'];assert h in objects,(p,h)
   assert x.get('bytes',objects[h])==objects[h],(p,h)
   refs.append((str(p),h))
  for v in x.values():walk(v,p)
 elif isinstance(x,list):
  for v in x:walk(v,p)
for label in ('tuple-e456-prepare','tuple-e456-focused','tuple-73b-prepare','tuple-73b-focused'):
 root=base/(label+'-puck-copy')
 for p in root.rglob('*'):
  if not p.is_file() or p.suffix not in ('.json','.jsonl','.log','.stdout','.stderr','.txt','.mk'):continue
  reports[str(p)]=sha(p)
  if p.suffix=='.json':
   x=json.loads(p.read_text());walk(x,p)
   if p.name.endswith('-status.json') or p.name=='terminal.json':
    assert 'status' in x or 'returncode' in x,p
    terminals.append({'path':str(p),'status':x.get('status',x.get('returncode')),'timeout':x.get('timeout')})
  if p.name.endswith('-before.json') and any(s in p.name for s in ('-inputs-','-sources-','-tools-')):
   after=p.with_name(p.name.replace('-before.json','-after.json'));assert after.exists(),after
   assert json.loads(p.read_text())==json.loads(after.read_text()),p
   pairs.append([str(p),str(after)])
failures=[x for x in terminals if x['status'] not in (0,'PASS')]
assert len(failures)==2 and all('tuple-e456-focused' in x['path'] and x['status']==1 and not x['timeout'] for x in failures),failures
out={'objects':len(objects),'bytes':sum(objects.values()),'artifact_references':len(refs),'equal_pairs':len(pairs),'report_files':len(reports),'terminals':terminals,'failures':failures,'pairs':pairs,'reports':reports,'scope':'local retained report/CAS integrity only; no new product execution; nested terminal uses returncode, outer uses status'}
(base/'tuple-focused-retention-audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('terminals','pairs','reports')},indent=2))
