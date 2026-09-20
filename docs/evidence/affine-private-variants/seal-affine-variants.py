import pathlib,json,hashlib,os,shutil,sys
out=pathlib.Path(sys.argv[1]);out.mkdir(exist_ok=False);reports=out/'reports';reports.mkdir();cas=out/'objects';cas.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
roots=[pathlib.Path(p) for p in sys.argv[2:]];store={};pairs=[];terminals=[]
for root in roots:
 label=root.name
 for p in root.rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(root)
  if p.parent.name=='objects' and len(p.name)==64:
   if p.name not in store:
    assert sha(p)==p.name,p
    q=cas/p.name
    try:os.link(p,q)
    except OSError:shutil.copyfile(p,q)
    store[p.name]={'bytes':p.stat().st_size,'path':str(q),'original':str(p)}
   continue
  if 'objects' in rel.parts or 'products' in rel.parts:continue
  if p.suffix not in ('.json','.log','.txt','.py','.mk'):continue
  q=reports/label/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
 for p in root.glob('*-before.json'):
  if p.name.endswith('products-before.json'):continue
  q=root/p.name.replace('-before.json','-after.json')
  if q.exists():
   a=json.loads(p.read_text());b=json.loads(q.read_text());pairs.append({'before':str(p),'after':str(q),'equal':a==b,'changed':[k for k in a.keys()&b.keys() if a[k]!=b[k]],'before_count':len(a),'after_count':len(b)})
 for p in root.glob('*-status.json'):
  v=json.loads(p.read_text());terminals.append({'path':str(p),'returncode':v.get('returncode'),'timeout':v.get('timeout'),'seconds':v.get('seconds')})
 for p in root.glob('*/inputs-before.json'):
  q=p.parent/'inputs-after.json'
  if q.exists():
   a=json.loads(p.read_text());b=json.loads(q.read_text());pairs.append({'before':str(p),'after':str(q),'equal':a==b,'before_count':len(a),'after_count':len(b)})
# All archived references in retained reports must identify a byte-checked object.
refs=0
for p in reports.rglob('*.json'):
 def walk(x):
  global refs
  if isinstance(x,dict):
   if 'sha256' in x and 'archive' in x:
    assert x['sha256'] in store,(p,x['archive']);assert store[x['sha256']]['bytes']==x['bytes'];refs+=1
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(json.loads(p.read_text()))
summary={'roots':list(map(str,roots)),'artifacts':len(store),'artifact_bytes':sum(v['bytes'] for v in store.values()),'references':refs,'pairs':pairs,'terminals':terminals,'scope':'Private/public constructor fault sweeps, scoped compiler matrices, integrated actual host/ownership queries/runtime and unchanged emitter; retained51d crash and660 link terminals remain failures'}
(reports/'artifact-store.json').write_text(json.dumps(store,indent=2)+'\n');(reports/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
manifest={str(p.relative_to(reports)):sha(p) for p in sorted(reports.rglob('*')) if p.is_file()};(reports/'report-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'reports':len(manifest),'artifacts':len(store),'bytes':summary['artifact_bytes'],'refs':refs,'pairs':len(pairs),'unequal':sum(not p['equal'] for p in pairs)}))
