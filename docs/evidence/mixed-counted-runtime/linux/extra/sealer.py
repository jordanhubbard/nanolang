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
count=0;pairs=[];deltas=[];terms=[];current={}
names=['nanolang-mixed-counted-5f4f-'+host]
if host=='linux':names=['nanolang-mixed-counted-3c0b-linux',*names,'nanolang-mixed-counted-5f4f-linux-neighbors']
for name in names:
 root=Path('/tmp')/name
 for p in sorted(root.iterdir()):
  if not p.is_file():continue
  relative=str(Path(root.name)/p.name);save(p,relative)
  if p.suffix=='.json':
   value=json.loads(p.read_text())
   for v in refs(value):assert add(v['artifact'])==v['sha256'];count+=1
   if p.name.endswith(('-source-before.json','-tools-before.json')):
    other=p.with_name(p.name.replace('-before','-after'));after=json.loads(other.read_text())
    if value==after:pairs.append(relative)
    else:
     changed=[k for k in set(value)|set(after) if value.get(k)!=after.get(k)]
     assert name=='nanolang-mixed-counted-3c0b-linux' and p.name=='setup-source-before.json' and changed==['src/nanoisa/managed_native_source.h'],(name,p.name,changed)
     deltas.append({'before':relative,'after':str(Path(root.name)/other.name),'changed':changed,'scope':'preserved first setup canonical generation of approved testing hooks'})
   if p.name.endswith('-terminal.json'):terms.append({'path':relative,**value})
final=Path('/tmp')/names[-1];frozen=Path('/tmp/nanolang-mixed-counted-qualified-5f4f')
for kind in ['source','tools']:
 p=final/('origin-query-'+kind+'-after.json');rows=json.loads(p.read_text())
 for key,v in rows.items():
  path=frozen/key if kind=='source' else Path(v['path']);digest=v if kind=='source' else v['sha256'];assert sha(path)==digest,(kind,path)
 current[kind]={'entries':len(rows),'map':str(p),'sha256':sha(p)}
p=final/'origin-query-artifacts.json';rows=json.loads(p.read_text())
for name,v in rows.items():assert sha(name)==v['sha256'],name
current['providers-and-last-fixture-artifacts']={'entries':len(rows),'map':str(p),'sha256':sha(p)}
# Archive closure is a current-only supplement; original phase inventories name obj/bin.
for p in (frozen/'lib').glob('*.a'):
 h=add(p);current[str(p)]={'sha256':h,'bytes':p.stat().st_size,'artifact':str(objects/h),'scope':'current only; not historical phase map'};count+=1
for short in (['3c0b','5f4f'] if host=='linux' else ['5f4f']):
 for suffix in ['source.json','launch.py']:
  p=Path('/tmp/nanolang-mixed-counted-'+short+'-'+suffix);save(p,'extra/'+p.name)
 p=Path('/tmp/nanolang-mixed-counted-'+short+'-'+host+'-launch.log');save(p,'extra/'+p.name)
 p=Path('/tmp/nanolang-mixed-counted-'+short+'-source.tar.gz');h=add(p)
 manifest=json.loads(Path('/tmp/nanolang-mixed-counted-'+short+'-source.json').read_text());assert h==manifest['archive_sha256']
 current[short+'/source-archive']={'sha256':h,'bytes':p.stat().st_size,'artifact':str(objects/h)};count+=1
if host=='linux':
 for name in ['nanolang-mixed-counted-neighbor-launch.py','nanolang-mixed-counted-5f4f-linux-neighbors-launch.log','nanolang-mixed-counted-wasmtime-selection.json']:
  save(Path('/tmp')/name,'extra/'+name)
else:
 p=Path('/private/tmp/nanolang-mixed-counted-tools-v43/selection.json');save(p,'extra/wasmtime-selection.json');v=json.loads(p.read_text());assert add(v['binary_path'])==v['binary_sha256'];count+=1
save(__file__,'extra/sealer.py')
p=reports/'current-inputs.json';p.write_text(json.dumps(current,indent=2)+'\n');index['current-inputs.json']={'sha256':sha(p),'bytes':p.stat().st_size}
summary={'host':host,'reports':len(index),'objects':len(store),'bytes':sum(v['bytes'] for v in store.values()),'references':count,'equal_pairs':len(pairs),'pairs':pairs,'documented_source_deltas':deltas,'terminals':terms,'scope':'counted manual storage/adapters; seven total native compiler configurations, ordinary LLVM/Wasm/package; no public/source consumer admission'}
for name,value in [('report-sha256.json',index),('artifact-store.json',store),('summary.json',summary)]: (out/name).write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('pairs','terminals')},indent=2))
