import pathlib,json,hashlib,shutil,sys,tarfile
out=pathlib.Path(sys.argv[1]);out.mkdir(exist_ok=False);store=out/'objects';store.mkdir();reports=out/'reports';reports.mkdir();index={};pairs=[];current={};terminals=[];refs=0
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for arg in sys.argv[2:]:
 root=pathlib.Path(arg);assert root.is_dir();dest=reports/root.name;dest.mkdir()
 for p in root.rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(root)
  if 'products' in rel.parts:continue
  if 'objects' in rel.parts:
   digest=p.name;assert sha(p)==digest
   if digest not in index:shutil.copyfile(p,store/digest);index[digest]={'bytes':p.stat().st_size}
   continue
  target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
 for before in root.rglob('*before.json'):
  if 'products' in before.name:continue
  after=before.with_name(before.name.replace('before.json','after.json'))
  if after.exists():
   a=json.loads(before.read_text());b=json.loads(after.read_text());assert a==b,(before,after);pairs.append({'before':str(before),'after':str(after),'count':len(a)})
 for p in root.rglob('inputs-after.json'):
  for name,row in json.loads(p.read_text()).items():
   expected=row['sha256'];assert name not in current or current[name]==expected;current[name]=expected
 for p in root.glob('*-source-after.json'):
  phase=p.name.removesuffix('-source-after.json');cmd=json.loads((root/(phase+'-command.json')).read_text());tree=pathlib.Path(cmd['cwd'])
  for name,h in json.loads(p.read_text()).items():
   path=str(tree/name);assert path not in current or current[path]==h;current[path]=h
 for p in root.glob('*-tools-after.json'):
  for row in json.loads(p.read_text()).values():
   name=row['path'];expected=row['sha256'];assert name not in current or current[name]==expected;current[name]=expected
 for p in root.rglob('*-terminal.json'):
  terminals.append({'path':str(p),'status':json.loads(p.read_text())})
for name,h in current.items():assert sha(name)==h,name
for p in reports.rglob('*.json'):
 def visit(v):
  global refs
  if isinstance(v,dict):
   if 'sha256' in v and 'archive' in v:assert v['sha256'] in index,(p,v);refs+=1
   for z in v.values():visit(z)
  elif isinstance(v,list):
   for z in v:visit(z)
 visit(json.loads(p.read_text()))
summary={'report_files':sum(p.is_file() for p in reports.rglob('*')),'unique_objects':len(index),'object_bytes':sum(v['bytes'] for v in index.values()),'references':refs,'pairs':pairs,'current_files_verified':len(current),'terminals':terminals}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'artifact-store.json').write_text(json.dumps(index,indent=2)+'\n');(out/'current.json').write_text(json.dumps(current,indent=2)+'\n');shutil.copyfile(__file__,out/'collector.py')
manifest={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and 'objects' not in p.relative_to(out).parts};(out/'report-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['pairs','terminals']}),flush=True)
