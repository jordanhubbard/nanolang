from pathlib import Path
import os,json,hashlib,gzip,stat
control=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest()
j=Path('/var/tmp/nanolang-preflight-after-six-five-20261126-szx5u9d2')
terminal_file=j/'cohort-result.json'
assert terminal_file.is_file(), 'I require accepted terminal preflight evidence before archiving'
q=json.loads(terminal_file.read_bytes());assert q['terminal'] and q.get('accepted_component') and q.get('all_guards_match'), 'I require accepted terminal preflight evidence before archiving'
assert q['tests']==dict(tests=1,failures=0,errors=0,skips=0,passed=True)
parent=Path('/var/tmp/nanolang-darwin-publication-qkk55gi3/docs/evidence/darwin-current-source-20261124');out=parent/'diagnostics/affine-preflight-five-linux-20261126';out.mkdir();(out/'objects').mkdir();known={};bases=[]
for path in sorted((parent/'diagnostics').glob('*/manifest.json')):
 if path.parent==out:continue
 old=json.loads(path.read_bytes())
 if old.get('format')!=1:continue
 bases.append(dict(path=os.path.relpath(path,out),sha256=sha(path.read_bytes())))
 for h,i in old['chunks'].items():known[h]=((path.parent/i.get('artifact','objects/'+h+'.gz')).resolve(),i)
m=dict(format=1,source_commit='67d99e5121521428740d385d10071b9679e66013',working_overlay=True,report_fix_commit='c886e6dc73c2680d17a6db76b18e4ec00050e592',methods=1,cache_fix_commit='901caef8e6f995080ac7d3c48c4cf23c75c99315',component_only=True,interrupted=False,accepted_component=True,accepted_public_c_vm=False,full_five_routes=True,new_bootstrap=False,whole_release=False,roots={},files=[],symlinks=[],chunks={},base_manifests=bases)
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
   ds[:]=sorted(n for n in ds if n not in ['.git','phase-artifacts'])
   for n in ds:
    p=Path(d)/n
    if p.is_symlink():m['symlinks'].append(dict(path=label+'/'+str(p.relative_to(root)),target=os.readlink(p)))
   for n in sorted(fs):
    if n=='.git':continue
    p=Path(d)/n;name=label+'/'+str(p.relative_to(root))
    if p.is_symlink():m['symlinks'].append(dict(path=name,target=os.readlink(p)))
    elif p.is_file():record(p,name)
 tree('comparison',j)
 record(Path('/var/tmp/nanolang-path-copy-query-five-20261125-nb81f3jc/cohort-result.json'),'query-dependency/cohort-result.json')
 record(Path('/var/tmp/nanolang-path-copy-global-array-five-20261125-jsj5x9iu/cohort-result.json'),'global-array-dependency/cohort-result.json')
 record(Path('/var/tmp/nanolang-affine-typed-five-20261125-udz2o6c_/cohort-result.json'),'foundation-dependency/cohort-result.json')
 record(Path('/var/tmp/nanolang-report1261-three-five-20261125-ul75rs9t/cohort-result.json'),'corrected-three-dependency/cohort-result.json')
 record(Path('/var/tmp/nanolang-cache1262-key-seed-review-20261125-3n4g07gw/controls/cohort-result.json'),'key-seed-controls/cohort-result.json')
 record(Path('/var/tmp/nanolang-cache-key-seed-five-20261126-of3upyax/cohort-result.json'),'cache-full-dependency/cohort-result.json')
 record(Path('/var/tmp/nanolang-path-copy-six-corrected-five-20261126-ghighxh3/cohort-result.json'),'six-dependency/cohort-result.json')
 six_proof=parent/'diagnostics/ownership-path-copy-six-leaves-corrected-five-linux-20261126'
 for name in ['manifest.json','read_archive.py']:record(six_proof/name,'six-proof/'+name)
 tree('control',control)
 external={**q['tools_before'],**q['hosts_before'],**q['object_inputs']}
 for n,h in external.items():
  assert sha(Path(n).read_bytes())==h;record(Path(n),'external/'+n.lstrip('/'))
 record(Path('/var/tmp/nanolang-hoisted-lists-cold-bootstrap-20261125-ia5_k8qp/source/bin/nanoc_bootstrap.json'),'origin/nanoc_bootstrap.json')
 (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 result=dict(terminal=True,archive=str(out),source_commit=m['source_commit'],records=len(m['files']),new_bytes=sum(v['compressed_bytes'] for v in m['chunks'].values() if v['artifact'].startswith('objects/')),sha256=sha((out/'manifest.json').read_bytes()))
 (control/'collector.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
