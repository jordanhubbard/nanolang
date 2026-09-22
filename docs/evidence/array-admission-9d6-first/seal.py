from pathlib import Path
import hashlib,json,tarfile,gzip,io,shutil
base=Path('/home/jkh/nanolang-qualification');r=base/'array-admission-9d6-first-copy/array-admission-9d6-puck';out=Path('/home/jkh/Src/nanolang-array-arithmetic-result-views/docs/evidence/array-admission-9d6-first');out.mkdir(parents=True,exist_ok=True);cas=base/'array-admission-9d6-cas';cas.mkdir(exist_ok=True)
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
seal={'pin':'9d6118d87b876a969468d31f58a690e6685ad5c7','objects':{}}
phases=json.loads((r/'phases.json').read_text());assert [p['returncode'] for p in phases]==[0,0,0,0,0,2]
for p in phases:assert not p['timeout'] and not p['remaining_groups']
assert json.loads((r/'terminal.json').read_text())['status']==1
assert json.loads((r/'identity.json').read_text())['tools']==json.loads((r/'tools-after-failure.json').read_text())
executables={}
for config in ('ordinary','sanitized'):
 o=r/config;sources=json.loads((o/'source-before.json').read_text());after='source-after.json' if config=='ordinary' else 'source-after-failure.json';assert sources==json.loads((o/after).read_text())
 providers=json.loads((o/'providers-before.json').read_text())
 for p in o.glob('*-providers-after.json'):assert providers==json.loads(p.read_text())
 for folder in ('bin','obj'):
  for p in (o/'source'/folder).rglob('*'):
   if not p.is_file():continue
   h=sha(p);dest=cas/h
   if not dest.exists():shutil.copy2(p,dest)
   assert sha(dest)==h;seal['objects'][h]={'path':str(dest),'bytes':p.stat().st_size}
 for p in o.glob('*-retained'):
  v=json.loads(p.with_suffix('.json').read_text());h=sha(p);assert h==v['sha256'] and p.stat().st_mode&511==v['mode'];shutil.copy2(p,cas/h);seal['objects'][h]={'path':str(cas/h),'bytes':p.stat().st_size};executables[str(p.relative_to(r))]=v
log=(r/'sanitized/sanitized-test-typechecker.log').read_text();assert 'All typechecker tests passed!' in log and '2138 byte(s) leaked in 22 allocation(s)' in log and 'test_typechecker.c:1123' in log
bundle=out/'darwin.tar.gz';reports=[]
with bundle.open('wb') as f,gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z,tarfile.open(fileobj=z,mode='w|') as t:
 for p in sorted(r.rglob('*')):
  if not p.is_file():continue
  rel=p.relative_to(r)
  if 'source' in rel.parts or 'tmp' in rel.parts or p.name.endswith('-retained'):continue
  b=p.read_bytes();e=tarfile.TarInfo(str(rel));e.size=len(b);e.mode=p.stat().st_mode&511;t.addfile(e,io.BytesIO(b));reports.append({'path':str(rel),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)})
with tarfile.open(bundle) as t:
 for v in reports:assert hashlib.sha256(t.extractfile(v['path']).read()).hexdigest()==v['sha256']
seal.update(bundle={'path':bundle.name,'sha256':sha(bundle)},reports=reports,phases=phases,executables=executables,totals={'reports':len(reports),'objects':len(seal['objects']),'bytes':sum(v['bytes'] for v in seal['objects'].values())})
(out/'seal.json').write_text(json.dumps(seal,indent=2)+'\n');shutil.copy2(__file__,out/'seal.py');print(seal['totals'])
