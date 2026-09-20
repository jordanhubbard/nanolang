import hashlib,json,pathlib,shutil,sys
out=pathlib.Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
reports=out/'reports';reports.mkdir();objects=out/'objects';objects.mkdir()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
store={};refs=0;reportmap={};map_pairs=[];current=[]
for name in sys.argv[2:]:
 root=pathlib.Path(name)
 for p in sorted(root.rglob('*')):
  if not p.is_file():continue
  relative=p.relative_to(root)
  archive='objects' in relative.parts or 'products' in relative.parts or p.name.endswith('-binary')
  h=digest(p)
  if archive:
   refs+=1
   if h not in store:
    target=objects/h;shutil.copyfile(p,target);store[h]={'path':str(target),'bytes':p.stat().st_size}
   if p.parent.name=='objects' and p.name!=h:raise RuntimeError('Bad content-address '+str(p))
  else:
   dest=reports/root.name/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
   reportmap[str(dest.relative_to(reports))]=h
 for before in root.rglob('*before.json'):
  after=before.with_name(before.name.replace('before.json','after.json'))
  if after.exists():
   a=json.loads(before.read_text());b=json.loads(after.read_text())
   if a!=b:raise RuntimeError('Pair mismatch '+str(before))
   map_pairs.append({'before':str(before),'after':str(after),'count':len(a),'equal':True})
 # Current source/provider/tool bytes: every endpoint's named input rows.
 for p in root.rglob('*after.json'):
  data=json.loads(p.read_text())
  if not isinstance(data,dict):continue
  count=0
  for path,row in data.items():
   if isinstance(row,dict) and 'sha256' in row and pathlib.Path(path).is_absolute():
    if digest(pathlib.Path(path))!=row['sha256']:raise RuntimeError('Current drift '+path)
    count+=1
  if count:current.append({'map':str(p),'current_equal':True,'count':count})
dump(out/'artifact-store.json',store);dump(out/'report-sha256.json',reportmap)
dump(out/'summary.json',{'reports':len(reportmap),'unique_artifacts':len(store),'artifact_references':refs,'artifact_bytes':sum(r['bytes'] for r in store.values()),'pairs':map_pairs,'current':current})
print(json.dumps({'reports':len(reportmap),'artifacts':len(store),'pairs':len(map_pairs),'current':len(current)}),flush=True)
