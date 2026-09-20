import sys,json,hashlib,shutil,re,tarfile
from pathlib import Path
host=sys.argv[1];dest=Path(sys.argv[2]);dest.mkdir();cas=dest/'artifacts';cas.mkdir();refs=0;objects={};pairs=[];terminals=[]
roots=[Path(p) for p in sys.argv[3:]]
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def keep(p):
 h=sha(p)
 if h not in objects:shutil.copyfile(p,cas/h);objects[h]={'bytes':p.stat().st_size}
 return h
current={}
for root in roots:
 out=dest/root.name;out.mkdir()
 for p in root.iterdir():
  if p.is_file():shutil.copyfile(p,out/p.name)
 if (root/'artifacts').is_dir():
  for p in (root/'artifacts').iterdir():assert keep(p)==p.name
 for p in root.glob('*-artifacts.json'):
  for name,row in json.loads(p.read_text()).items():
   assert row['sha256'] in objects,(p,name);refs+=1
 for p in root.glob('*-source-before.json'):
  for domain in ['source','tools']:
   a=p.with_name(p.name.replace('-source-',f'-{domain}-'));b=a.with_name(a.name.replace('-before.json','-after.json'))
   assert json.loads(a.read_text())==json.loads(b.read_text()),p
   pairs.append({'root':root.name,'before':a.name,'after':b.name})
 for p in root.glob('*-terminal.json'):
  r=json.loads(p.read_text());assert not r['timeout'] and not r['errors'] and r['group_disappeared'];terminals.append({'root':root.name,'phase':p.name,**r})
 for log in root.glob('*.log'):
  for name in re.findall(r'I retain ordinary array authority artifacts at (\S+)',log.read_text(errors='replace')):
   sub=Path(name);target=out/'nested'/sub.name;target.mkdir(parents=True)
   for p in sub.iterdir():
    if p.suffix in ['.txt','.log','.json']:shutil.copyfile(p,target/p.name)
    else:keep(p)
 # I verify qualified source paths using setup argv rather than assuming checkout identity.
 setup=root/'setup-command.json'
 if setup.exists():
  driver=(root/'driver.py').read_text()
  env=json.loads((root/'environment.json').read_text())
  command=json.loads(setup.read_text())['argv']
  # source root is recoverable from a provider inventory absolute path.
  last=json.loads(sorted(root.glob('*-artifacts.json'))[-1].read_text())
  candidates=[Path(k.split('/obj/')[0]) for k in last if '/obj/' in k]
  if candidates:
   tree=candidates[0];source=json.loads(next(root.glob('*-source-before.json')).read_text())
   for rel,h in source.items():
    p=tree/rel;assert sha(p)==h,p;current[str(p)]=h
  t=json.loads(next(root.glob('*-tools-before.json')).read_text())
  for row in t.values():p=Path(row['path']);assert sha(p)==row['sha256'];current[str(p)]=row['sha256']
(dest/'summary.json').write_text(json.dumps({'host':host,'objects':len(objects),'bytes':sum(r['bytes'] for r in objects.values()),'references':refs,'pairs':pairs,'terminals':terminals},indent=2)+'\n')
(dest/'current.json').write_text(json.dumps(current,sort_keys=True,indent=2)+'\n')
shutil.copyfile(__file__,dest/'collector.py')
print(json.dumps({'host':host,'objects':len(objects),'refs':refs,'pairs':len(pairs),'current':len(current)}),flush=True)
with tarfile.open(str(dest)+'.tar.gz','w:gz') as t:t.add(dest,arcname=dest.name)
print(sha(Path(str(dest)+'.tar.gz')),flush=True)
