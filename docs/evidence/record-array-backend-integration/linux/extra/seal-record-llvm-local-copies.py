import argparse,hashlib,json,os,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--history',action='append',default=[]);p.add_argument('--extra',action='append',default=[]);args=p.parse_args()
out=Path(args.output);out.mkdir(exist_ok=False);reports=out/'reports';reports.mkdir();objects=out/'objects';objects.mkdir();index={};store={};references=0;pairs=[];terminals=[]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def add(path,expected=None):
 path=Path(path)
 if not path.is_file() and expected is not None:
  candidates=[Path(h)/'artifacts'/expected for h in args.history]
  found=[p for p in candidates if p.is_file()]
  assert found,('missing retained object',path,expected)
  path=found[0]
 digest=sha(path)
 if expected is not None:assert digest==expected,(path,digest,expected)
 dst=objects/digest
 if not dst.exists():
  try:os.link(path,dst)
  except OSError:shutil.copyfile(path,dst)
 assert sha(dst)==digest
 store[digest]={'bytes':path.stat().st_size,'path':str(dst)}
 return digest
def refs(v):
 if isinstance(v,dict):
  if 'artifact' in v and 'sha256' in v:yield v
  for x in v.values():yield from refs(x)
 elif isinstance(v,list):
  for x in v:yield from refs(x)
def save(path,name):
 global references
 path=Path(path);dst=reports/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dst)
 index[name]={'sha256':sha(dst),'bytes':dst.stat().st_size}
 if path.suffix=='.json':
  value=json.loads(path.read_text())
  for row in refs(value):add(row['artifact'],row['sha256']);references+=1
  if path.name.endswith(('-source-before.json','-tools-before.json')):
   after=path.with_name(path.name.replace('-before','-after'));assert value==json.loads(after.read_text());pairs.append(name)
  if path.name.endswith('-terminal.json'):terminals.append({'path':name,**value})
for directory in args.history:
 root=Path(directory);assert (root/'results.json').is_file()
 # Every selected phase must have its terminal and fully captured postphase maps.
 for row in json.loads((root/'results.json').read_text()):
  for suffix in ('terminal.json','source-after.json','tools-after.json','artifacts.json'):
   assert (root/(row['phase']+'-'+suffix)).is_file(),(root,row,suffix)
 for path in sorted(root.rglob('*')):
  if path.is_file() and not any(x in ('artifacts','controls','package-artifacts') for x in path.relative_to(root).parts):save(path,str(Path(root.name)/path.relative_to(root)))
for filename in args.extra:
 path=Path(filename)
 if path.suffix in ('.gz','.xz'):add(path)
 else:save(path,'extra/'+path.name)
summary={'histories':args.history,'reports':len(index),'objects':len(store),'object_bytes':sum(v['bytes'] for v in store.values()),'references':references,'equal_pairs':len(pairs),'pairs':pairs,'terminals':terminals,'scope':'Only the explicitly selected completed histories. No current host-input rehash or full matrix acceptance is inferred; independent final inventory required.'}
for name,value in [('report-sha256.json',index),('artifact-store.json',store),('summary.json',summary)]: (out/name).write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('pairs','terminals')},indent=2))
