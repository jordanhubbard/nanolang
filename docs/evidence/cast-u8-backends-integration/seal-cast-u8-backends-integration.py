import hashlib,json,pathlib,shutil,sys
host=sys.argv[1];out=pathlib.Path(sys.argv[2]);out.mkdir(exist_ok=False);objects=out/'objects';objects.mkdir();reports=out/'reports';reports.mkdir();index={};store={};refs=0;pairs=0;terminals={}
def digest(p):
 with p.open('rb') as f:
  h=hashlib.sha256()
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
  return h.hexdigest()
def walk(v):
 global refs
 if isinstance(v,dict):
  if 'sha256' in v and 'archive' in v:
   p=pathlib.Path(v['archive']);h=v['sha256'];assert digest(p)==h;d=objects/h
   if not d.exists():shutil.copyfile(p,d)
   assert d.stat().st_size==v['bytes'];store[h]={'bytes':v['bytes'],'path':str(d)};refs+=1
  for x in v.values():walk(x)
 elif isinstance(v,list):
  for x in v:walk(x)
roots=sorted(pathlib.Path('/tmp').glob('nanolang-cast-u8-backends-'+host+'-integration*'))
for root in roots:
 if not root.is_dir() or not (root/'manifest.json').exists():continue
 for p in sorted(root.iterdir()):
  if not p.is_file():continue
  name=root.name+'/'+p.name;d=reports/name;d.parent.mkdir(exist_ok=True);shutil.copyfile(p,d);index[name]={'sha256':digest(d),'bytes':d.stat().st_size}
  if p.suffix=='.json':
   value=json.loads(p.read_text());walk(value)
   if p.name.endswith(('-sources-before.json','-tools-before.json')):
    assert value==json.loads(p.with_name(p.name.replace('-before','-after')).read_text());pairs+=1
   if p.name.endswith('-status.json'):terminals[name]=value
for p in sorted(pathlib.Path('/tmp').glob('nanolang-cast-u8-backends-*.py')):
 d=reports/p.name;shutil.copyfile(p,d);index[p.name]={'sha256':digest(d),'bytes':d.stat().st_size}
summary={'host':host,'reports':len(index),'objects':len(store),'bytes':sum(v['bytes'] for v in store.values()),'references':refs,'equal_pairs':pairs,'terminals':terminals}
for name,v in [('report-sha256.json',index),('artifact-store.json',store),('summary.json',summary)]: (out/name).write_text(json.dumps(v,indent=2)+'\n')
print({k:v for k,v in summary.items() if k!='terminals'})
