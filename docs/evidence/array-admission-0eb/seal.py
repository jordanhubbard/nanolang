from pathlib import Path
import hashlib,json,tarfile,gzip,io,shutil,subprocess
base=Path('/home/jkh/nanolang-qualification');out=Path('/home/jkh/Src/nanolang-array-arithmetic-result-views/docs/evidence/array-admission-0eb');out.mkdir(parents=True,exist_ok=True)
cas=base/'array-admission-0eb-cas';cas.mkdir(exist_ok=True)
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
seal={'pin':'0eb14852e6718881f88f26f2afc94aa496ddefee','lanes':{},'objects':{}}
for host,r in [('linux',base/'array-admission-0eb-linux'),('darwin',base/'array-admission-0eb-puck-copy/array-admission-0eb-puck')]:
 phases=json.loads((r/'phases.json').read_text());assert len(phases)==8
 for v in phases:assert v['returncode']==0 and not v['timeout'] and not v['remaining_groups']
 assert json.loads((r/'terminal.json').read_text())['status']==0
 assert json.loads((r/'identity.json').read_text())['tools']==json.loads((r/'tools-after.json').read_text())
 assert json.loads((r/'fragment-before.json').read_text())==json.loads((r/'fragment-after.json').read_text())
 for config in ('ordinary','sanitized'):
  o=r/config
  before=json.loads((o/'source-before.json').read_text());assert before==json.loads((o/'source-after.json').read_text())
  providers=json.loads((o/'providers-before.json').read_text())
  for target in ('test-typechecker','test-eval','qualification-array-refusals'):assert providers==json.loads((o/(target+'-providers-after.json')).read_text())
  for n,v in json.loads((o/'products.json').read_text()).items():
   p=o/'source'/n;h=sha(p);assert h==v;dest=cas/h
   if not dest.exists():shutil.copy2(p,dest)
   assert sha(dest)==h;seal['objects'][h]={'path':str(dest),'bytes':p.stat().st_size}
  executables={}
  for binary,target,marker in [('test_typechecker','test-typechecker','All typechecker tests passed!'),('test_eval','test-eval','All eval tests passed!'),('test_codegen','qualification-array-refusals','I checked array scalar-opcode refusal: 1 passed, 0 failed.')]:
   p=o/(binary+'-retained');v=json.loads((o/(binary+'-retained.json')).read_text());h=sha(p);assert h==v['sha256'] and p.stat().st_mode&511==v['mode'];shutil.copy2(p,cas/h);seal['objects'][h]={'path':str(cas/h),'bytes':p.stat().st_size};executables[binary]=v
   text=(o/(config+'-'+target+'.log')).read_text();assert marker in text
   for bad in ('ERROR: AddressSanitizer','ERROR: LeakSanitizer','runtime error:'):assert bad not in text
   if host=='linux' and config=='sanitized':
    symbols=subprocess.check_output(['nm','-u',str(p)],text=True);assert '__asan_' in symbols and '__ubsan_' in symbols;(o/(binary+'-sanitizer-symbols.txt')).write_text(symbols)
  if host=='darwin' and config=='sanitized':
   for binary in ('test_eval','test_typechecker','test_codegen'):
    symbols=(o/(binary+'-sanitizer-symbols.txt')).read_text();assert '__asan_' in symbols and '__ubsan_' in symbols
  label=host+'-'+config;bundle=out/(label+'.tar.gz');reports=[]
  selected=[p for p in r.iterdir() if p.is_file()]+[p for p in o.rglob('*') if p.is_file() and 'source' not in p.relative_to(o).parts and 'tmp' not in p.relative_to(o).parts and not p.name.endswith('-retained')]
  with bundle.open('wb') as f,gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z,tarfile.open(fileobj=z,mode='w|') as t:
   for p in sorted(selected):
    rel=p.relative_to(r);b=p.read_bytes();e=tarfile.TarInfo(str(rel));e.size=len(b);e.mode=p.stat().st_mode&511;t.addfile(e,io.BytesIO(b));reports.append({'path':str(rel),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)})
  with tarfile.open(bundle) as t:
   for v in reports:assert hashlib.sha256(t.extractfile(v['path']).read()).hexdigest()==v['sha256']
  seal['lanes'][label]={'bundle':bundle.name,'sha256':sha(bundle),'reports':reports,'phases':[p for p in phases if p['label'].startswith(config)],'provider_count':len(providers),'executables':executables}
seal['totals']={'reports':sum(len(v['reports']) for v in seal['lanes'].values()),'objects':len(seal['objects']),'bytes':sum(v['bytes'] for v in seal['objects'].values())}
(out/'seal.json').write_text(json.dumps(seal,indent=2)+'\n');shutil.copy2(__file__,out/'seal.py');shutil.copy2(base/'array-admission-0eb-drivers/run.py',out/'run.py');print(seal['totals'])
