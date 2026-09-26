from pathlib import Path
import hashlib,json,gzip,tarfile,io,shutil
base=Path('/home/jkh/nanolang-qualification');out=Path('/home/jkh/Src/nanolang-evaluator-collection-owners/docs/evidence/collection-dfbf');out.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
scope=json.loads((base/'collection-dfbf-drivers/scope.json').read_text());expected={p:v['sha256'] for p,v in scope['files'].items()}
seal={'source':scope['pin'],'lanes':{},'objects':{}}
for label,r,status in [('linux-ordinary',base/'collection-dfbf-ordinary',0),('darwin-ordinary',base/'collection-dfbf-puck-reports/collection-dfbf-ordinary',0),('linux-sanitizer',base/'collection-dfbf-sanitizer',0),('darwin-sanitizer',base/'collection-dfbf-puck-reports/collection-dfbf-sanitizer',0)]:
 objects={}
 for p in (r/'artifacts').iterdir():
  if p.is_file():
   assert sha(p)==p.name;objects[p.name]={'bytes':p.stat().st_size,'path':str(p)};seal['objects'].setdefault(p.name,objects[p.name])
 pairs=0
 for kind in ['source','tools']:
  for p in r.glob('*-'+kind+'-before.json'):
   a=json.loads(p.read_text());assert a==json.loads((r/p.name.replace('-before','-after')).read_text());pairs+=1
   if kind=='source':assert a==expected
 refs=0
 for p in r.glob('*-artifacts.json'):
  for v in json.loads(p.read_text()).values():assert v['sha256'] in objects;refs+=1
 terminals={p.name:json.loads(p.read_text()) for p in r.glob('*-terminal.json')}
 assert len(terminals)==1
 for t in terminals.values():assert t['returncode']==status and t['leader_reaped'] and t['group_disappeared'] and not t['errors'] and not t['timeout']
 results=json.loads((r/'results.json').read_text());assert len(results)==1 and results[0]['status']==status
 log=(r/'test-evaluator-collection-ownership.log').read_text();assert 'I passed evaluator collection ownership controls.' in log
 for kind,count in [(0,12),(1,11),(2,13),(5,9),(6,9)]:assert f'I checked collection kind {kind} across {count} allocation positions.' in log
 if status:assert '8568 byte(s) leaked in 441 allocation(s)' in log
 else:assert json.loads((r/'final.json').read_text())['status']=='PASS'
 for p in r.rglob('retention.json'):
  rec=json.loads(p.read_text());exe=p.parent/Path(rec['source']).name;assert sha(exe)==rec['sha256'];assert exe.stat().st_mode&511==rec['mode']
 if status:
  rec=json.loads((r/'failed-executable.json').read_text());assert rec['sha256'] in objects
 bundle=out/(label+'.tar.gz');reports=[]
 with bundle.open('wb') as f,gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z,tarfile.open(fileobj=z,mode='w|') as t:
  for p in sorted(r.rglob('*')):
   if not p.is_file() or 'artifacts' in p.relative_to(r).parts:continue
   if p.name=='test_evaluator_collection_ownership':continue
   b=p.read_bytes();name=str(p.relative_to(r));e=tarfile.TarInfo(name);e.size=len(b);e.mode=p.stat().st_mode&511;t.addfile(e,io.BytesIO(b));reports.append({'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
 with tarfile.open(bundle) as t:
  for v in reports:assert hashlib.sha256(t.extractfile(v['path']).read()).hexdigest()==v['sha256']
 seal['lanes'][label]={'bundle':bundle.name,'sha256':sha(bundle),'reports':reports,'cas_objects':len(objects),'refs':refs,'equal_pairs':pairs,'terminals':terminals,'results':results}

seal['projection_cases']=12;seal['totals']={'reports':sum(len(x['reports']) for x in seal['lanes'].values()),'objects':len(seal['objects']),'bytes':sum(x['bytes'] for x in seal['objects'].values())}
(out/'seal.json').write_text(json.dumps(seal,indent=2)+'\n');shutil.copy2(__file__,out/'seal.py');print(seal['totals'])
