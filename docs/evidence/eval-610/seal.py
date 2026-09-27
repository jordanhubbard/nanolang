from pathlib import Path
import hashlib,json,tarfile,gzip,io,shutil
base=Path('/home/jkh/nanolang-qualification');out=Path('/home/jkh/Src/nanolang-array-arithmetic-result-views/docs/evidence/eval-610');out.mkdir(parents=True,exist_ok=True)
cas=base/'eval-610-cas';cas.mkdir(exist_ok=True)
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
seal={'pin':'6108067d4161746bc9b20508b9f2fce8a82e433e','lanes':{},'objects':{}}
for label,r in [('linux',base/'candidate-eval-610-linux'),('darwin',base/'candidate-eval-610-puck-copy/candidate-eval-610-puck')]:
 o=r/'ordinary';ph=json.loads((r/'phases.json').read_text());assert len(ph)==2 and [v['returncode'] for v in ph]==[0,2]
 for v in ph:assert not v['timeout'] and not v['remaining_groups']
 assert json.loads((r/'terminal.json').read_text())['status']==1
 before=json.loads((o/'source-before.json').read_text());assert before==json.loads((o/'source-after-failure.json').read_text())
 providers=json.loads((o/'providers-before.json').read_text());assert providers==json.loads((o/'providers-after.json').read_text())
 assert json.loads((r/'identity.json').read_text())['tools']==json.loads((r/'tools-after-failure.json').read_text())
 for n,v in providers.items():
  p=o/'source'/n;h=sha(p);assert h==v['sha256'];dest=cas/h
  if not dest.exists():shutil.copy2(p,dest)
  assert sha(dest)==h;seal['objects'][h]={'path':str(dest),'bytes':p.stat().st_size}
 p=o/'test_eval-retained';v=json.loads((o/'test_eval-retained.json').read_text());h=sha(p);assert h==v['sha256'];shutil.copy2(p,cas/h);seal['objects'][h]={'path':str(cas/h),'bytes':p.stat().st_size}
 log=(o/'ordinary-eval.log').read_text();assert 'Testing eval_unary_minus_int_array...' in log and 'FAILED: ok at line 2421' in log
 bundle=out/(label+'.tar.gz');reports=[]
 with bundle.open('wb') as f,gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z,tarfile.open(fileobj=z,mode='w|') as t:
  for p in sorted(r.rglob('*')):
   if not p.is_file():continue
   rel=p.relative_to(r)
   if 'source' in rel.parts or 'tmp' in rel.parts or p.name=='test_eval-retained':continue
   b=p.read_bytes();e=tarfile.TarInfo(str(rel));e.size=len(b);e.mode=p.stat().st_mode&511;t.addfile(e,io.BytesIO(b));reports.append({'path':str(rel),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)})
 with tarfile.open(bundle) as t:
  for v in reports:assert hashlib.sha256(t.extractfile(v['path']).read()).hexdigest()==v['sha256']
 seal['lanes'][label]={'bundle':bundle.name,'sha256':sha(bundle),'reports':reports,'phases':ph,'provider_count':len(providers),'retained_executable':h}
seal['totals']={'reports':sum(len(v['reports']) for v in seal['lanes'].values()),'objects':len(seal['objects']),'bytes':sum(v['bytes'] for v in seal['objects'].values())}
(out/'seal.json').write_text(json.dumps(seal,indent=2)+'\n');shutil.copy2(__file__,out/'seal.py');shutil.copy2(base/'candidate-eval-610-drivers/run.py',out/'run.py');print(seal['totals'])
