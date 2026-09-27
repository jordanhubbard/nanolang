from pathlib import Path
import hashlib,json,gzip,os,shutil
root=Path('/home/jkh/Src/nanolang-generic-list-mutations')
dest=root/'docs/evidence/evaluator-lifetime-diagnostic';dest.mkdir(parents=True,exist_ok=False)
cas=Path('/tmp/nanolang-evaluator-lifetime-diagnostic-artifacts');cas.mkdir(exist_ok=False)
def digest(b):return hashlib.sha256(b).hexdigest()
def fact(p):
 b=p.read_bytes();return dict(sha256=digest(b),bytes=len(b))
manifest={'scope':'Private diagnostic only; original full graph retains ten-second timeout. First Make logs/statuses only, with full preparation maps retained at original paths. No production or fixture acceptance.','reports':{},'artifacts':{},'first_make_terminals':[],'command_pairs':0,'commands':[]}
def report(p,label):
 b=p.read_bytes();f=fact(p);relative=Path('reports')/label/p.name
 if len(b)>65536:relative=Path(str(relative)+'.gz');b=gzip.compress(b,mtime=0)
 target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
 f.update(stored_sha256=digest(b),stored_bytes=len(b),original_path=str(p));manifest['reports'][str(relative)]=f
for name in ('c1df','4abf','192e'):
 p=Path(f'/tmp/nanolang-evaluator-lifetime-{name}-linux');store=p/'objects'
 products={}
 for folder in ('compiler-root','module-cache'):
  for path in (p/folder).rglob('*'):
   if not path.is_file() or path.is_symlink():continue
   b=path.read_bytes();h=digest(b);saved=store/h
   if not saved.exists():saved.write_bytes(b)
   products[str(path)]=dict(sha256=h,bytes=len(b),archive=str(saved))
 (p/'retained-layout-products.json').write_text(json.dumps(products,indent=2)+'\n')
 for path in sorted(p.iterdir()):
  if path.is_file() and path.suffix in ('.json','.log','.mk'):report(path,name)
 for path in store.iterdir():
  f=fact(path);assert path.name==f['sha256']
  target=cas/path.name
  if not target.exists():os.link(path,target)
  manifest['artifacts'][path.name]=dict(path=str(target),bytes=f['bytes'])
 for status in p.glob('*-status.json'):
  x=json.loads(status.read_text());before=p/(status.name.replace('-status','-before'));after=p/(status.name.replace('-status','-after'))
  assert json.loads(before.read_text())==json.loads(after.read_text())
  manifest['command_pairs']+=1;manifest['commands'].append(dict(phase=name,name=status.name,status=x['returncode'],seconds=x['seconds'],reaped=x['reaped'],group_gone=x['group_gone'],timeout=x['timeout']))
for pin in ('b915','cc606'):
 for host in ('linux','puck'):
  if host=='linux':p=Path(f'/tmp/nanolang-record-lists-{pin}-linux-prepare');status=p/'build-status.json';log=p/'build.log'
  else:p=Path('/tmp/nanolang-evaluator-timing-first-terminals');status=p/f'{pin}-puck-status.json';log=p/f'{pin}-puck.log'
  x=json.loads(status.read_text());assert x['log']['sha256']==fact(log)['sha256'];assert x['sources_equal'] and x['tools_equal']
  report(status,f'first-{pin}-{host}');report(log,f'first-{pin}-{host}')
  manifest['first_make_terminals'].append(dict(pin=pin,host=host,status=x['status'],seconds=x['seconds'],full_maps='Retained at original preparation root; not part of this diagnostic report/CAS seal.'))
for path in ('/tmp/run-evaluator-lifetime-diagnostic.py','/tmp/run-evaluator-lifetime-diagnostic-corrected.py','/tmp/run-evaluator-lifetime-diagnostic-companion.py',__file__):report(Path(path),'runners')
manifest['report_count']=len(manifest['reports']);manifest['unique_artifacts']=len(manifest['artifacts']);manifest['artifact_bytes']=sum(x['bytes'] for x in manifest['artifacts'].values())
(dest/'seal.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ('report_count','unique_artifacts','artifact_bytes','command_pairs','commands')},indent=2))
