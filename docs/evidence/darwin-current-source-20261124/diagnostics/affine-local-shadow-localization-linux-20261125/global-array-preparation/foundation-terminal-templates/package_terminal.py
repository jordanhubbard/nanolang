from pathlib import Path
import os,json,hashlib,gzip,stat
control=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest()
j=Path('/var/tmp/nanolang-affine-typed-five-20261125-udz2o6c_');q=json.loads((j/'cohort-result.json').read_bytes());assert q['terminal'] and q['accepted_component'] and q['all_guards_match'] and q['tests']==dict(tests=5,failures=0,errors=0,skips=0,passed=True) and q['observer_before']==q['observer_after']
parent=Path('/var/tmp/nanolang-darwin-publication-qkk55gi3/docs/evidence/darwin-current-source-20261124');out=parent/'diagnostics/affine-typed-five-linux-20261125';out.mkdir();(out/'objects').mkdir();known={};bases=[]
for path in sorted((parent/'diagnostics').glob('*/manifest.json')):
 if path.parent==out:continue
 old=json.loads(path.read_bytes())
 if old.get('format')!=1:continue
 bases.append(dict(path=os.path.relpath(path,out),sha256=sha(path.read_bytes())))
 for h,i in old['chunks'].items():known[h]=((path.parent/i.get('artifact','objects/'+h+'.gz')).resolve(),i)
m=dict(format=1,source_commit=q['source_pin'],component_only=True,interrupted=False,accepted_component=True,full_five_routes=True,new_bootstrap=False,whole_release=False,roots={},files=[],symlinks=[],chunks={},base_manifests=bases)
if True:
 def record(p,name):
  digest=hashlib.sha256();size=0;hs=[]
  with p.open('rb') as f:
   while True:
    raw=f.read(8*1024*1024)
    if not raw:break
    digest.update(raw);size+=len(raw);h=sha(raw);hs.append(h)
    if h in m['chunks']:continue
    if h in known:
     target,info=known[h];info=dict(info);info['artifact']=os.path.relpath(target,out);assert info['bytes']==len(raw)
    else:
     z=gzip.compress(raw,compresslevel=6,mtime=0);(out/'objects'/(h+'.gz')).write_bytes(z);info=dict(bytes=len(raw),compressed_bytes=len(z),compressed_sha256=sha(z),artifact='objects/'+h+'.gz')
    m['chunks'][h]=info
  m['files'].append(dict(path=name,bytes=size,sha256=digest.hexdigest(),chunks=hs,mode=stat.S_IMODE(p.stat().st_mode)))
 def tree(label,root):
  m['roots'][label]=str(root)
  for d,ds,fs in os.walk(root,followlinks=False):
   ds[:]=sorted(n for n in ds if n not in ['.git','__pycache__','phase-artifacts'])
   for n in ds:
    p=Path(d)/n
    if p.is_symlink():m['symlinks'].append(dict(path=label+'/'+str(p.relative_to(root)),target=os.readlink(p)))
   for n in sorted(fs):
    if n=='.git':continue
    p=Path(d)/n;name=label+'/'+str(p.relative_to(root))
    if p.is_symlink():m['symlinks'].append(dict(path=name,target=os.readlink(p)))
    elif p.is_file():record(p,name)
 tree('comparison',j)
 external={**q['tools_before'],**q['hosts_before'],**q['object_inputs']}
 for provider in q['providers'].values():external.update(provider['libraries_before'])
 for path,h in external.items():
  assert sha(Path(path).read_bytes())==h;record(Path(path),'external/'+path.lstrip('/'))
 origin=Path('/var/tmp/nanolang-hoisted-lists-cold-bootstrap-20261125-ia5_k8qp/bootstrap-result.json');record(origin,'origin/bootstrap-result.json')
 record(control/'package_terminal.py','collector-source/package_terminal.py')
 (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 result=dict(terminal=True,archive=str(out),source_commit=m['source_commit'],records=len(m['files']),new_bytes=sum(v['compressed_bytes'] for v in m['chunks'].values() if v['artifact'].startswith('objects/')),sha256=sha((out/'manifest.json').read_bytes()))
 (control/'collector.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
