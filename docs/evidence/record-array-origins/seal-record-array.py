import hashlib,json,shutil,sys
from pathlib import Path
host=sys.argv[1];out=Path(sys.argv[2]);out.mkdir(exist_ok=False);reports=out/'reports';reports.mkdir();objects=out/'objects';objects.mkdir();index={};store={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def add(p):
 p=Path(p);h=sha(p);target=objects/h
 if not target.exists():shutil.copyfile(p,target)
 assert sha(target)==h;store[h]={'bytes':p.stat().st_size,'path':str(target)};return h
def save(p,name):
 p=Path(p);dst=reports/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst);index[name]={'sha256':sha(dst),'bytes':dst.stat().st_size}
def refs(v):
 if isinstance(v,dict):
  if 'artifact' in v and 'sha256' in v:yield v
  for x in v.values():yield from refs(x)
 elif isinstance(v,list):
  for x in v:yield from refs(x)
count=0;pairs=[];terms=[];current={}
for short in ['55045','2e9fa']:
 root=Path('/tmp/nanolang-record-array-'+short+'-'+host)
 for p in sorted(root.rglob('*')):
  if not p.is_file() or any(x in p.relative_to(root).parts for x in ('artifacts','controls')):continue
  name=str(Path(root.name)/p.relative_to(root));save(p,name)
  if p.suffix=='.json':
   value=json.loads(p.read_text())
   for v in refs(value):assert add(v['artifact'])==v['sha256'];count+=1
   if p.name.endswith(('-source-before.json','-tools-before.json')):
    other=p.with_name(p.name.replace('-before','-after'));assert value==json.loads(other.read_text());pairs.append(name)
   if p.name.endswith('-terminal.json'):terms.append({'path':name,**value})
 for name in ['driver.py']:
  assert (root/name).is_file()
 for kind in ['source','tools']:
  phase='ordinary' if short=='55045' else 'complete-declarations';p=root/(phase+'-'+kind+'-after.json');rows=json.loads(p.read_text());n=0
  for key,v in rows.items():
   path=Path('/tmp/nanolang-record-array-qualified-'+short)/key if kind=='source' else Path(v['path']);digest=v if kind=='source' else v['sha256'];assert sha(path)==digest,(kind,path);n+=1
  current[short+'/'+kind]={'entries':n,'map':str(p),'sha256':sha(p)}
 p=root/(('ordinary' if short=='55045' else 'complete-declarations')+'-artifacts.json');rows=json.loads(p.read_text())
 for name,v in rows.items():assert sha(name)==v['sha256'],name
 current[short+'/providers-and-last-fixture-artifacts']={'entries':len(rows),'map':str(p),'sha256':sha(p)}
 for suffix in ['source.json','launch.py']:
  p=Path('/tmp/nanolang-record-array-'+short+'-'+suffix);save(p,'extra/'+p.name)
 p=Path('/tmp/nanolang-record-array-'+short+'-'+host+'-launch.log');save(p,'extra/'+p.name)
 p=Path('/tmp/nanolang-record-array-'+short+'-source.tar.gz');h=add(p)
 manifest=json.loads(Path('/tmp/nanolang-record-array-'+short+'-source.json').read_text());assert h==manifest['archive_sha256']
 current[short+'/source-archive']={'sha256':h,'bytes':p.stat().st_size,'artifact':str(objects/h)}
extra=reports/'current-inputs.json';extra.write_text(json.dumps(current,indent=2)+'\n');index['current-inputs.json']={'sha256':sha(extra),'bytes':extra.stat().st_size};count+=2
summary={'host':host,'reports':len(index),'objects':len(store),'bytes':sum(v['bytes'] for v in store.values()),'references':count,'equal_pairs':len(pairs),'pairs':pairs,'terminals':terms,'scope':'private non-admitting query; seven total selected compiler configurations; first55045 fixture encoding refusal retained'}
for name,value in [('report-sha256.json',index),('artifact-store.json',store),('summary.json',summary)]: (out/name).write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('pairs','terminals')},indent=2))
