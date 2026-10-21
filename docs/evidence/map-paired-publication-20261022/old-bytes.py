import os,sys,json,hashlib
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root))
from tests.native_process import run_captured
out=Path(__file__).parent;agg=Path('/Users/jkh/Src/nanolang-51-harness');rows=[]
pairs=[('seed',agg/'bin/nano_virt',out/'nano_virt'),('nano',Path('/private/tmp/nanolang-generic-record-field-20261021/canonical'),out/'canonical')]
for name,typ,value in [('scalar','int','7'),('flat','array<int>','[7]'),('depth2','array<array<int>>','[[7]]'),('depth3','array<array<array<bool>>>','[[[true]]]')]:
 source=out/'old-byte-source.nano';source.write_text(f'struct Box {{ value: {typ} }} fn main()->int {{ let box: Box = Box {{ value: {value} }} return 0 }}')
 for family,before,after in pairs:
  paths=[]
  for version,tool in [('old',before),('new',after)]:
   module=out/f'{name}-{family}-{version}.nvm';args=[str(tool),str(source),'--emit-nvm','-o',str(module)]
   p=run_captured(args,cwd=root,env=os.environ,timeout=180)
   (out/f'{name}-{family}-{version}.stdout').write_bytes(p.stdout);(out/f'{name}-{family}-{version}.stderr').write_bytes(p.stderr)
   assert p.returncode==0,(args,p.stdout,p.stderr);paths.append(module)
  assert paths[0].read_bytes()==paths[1].read_bytes(),(name,family)
  rows.append(dict(case=name,producer=family,sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),bytes=paths[0].stat().st_size,old_provider=str(before),old_provider_sha256=hashlib.sha256(before.read_bytes()).hexdigest(),new_provider_sha256=hashlib.sha256(after.read_bytes()).hexdigest()))
  (out/'old-bytes.json').write_text(json.dumps(rows,indent=2)+'\n')
print('PASS eight old scalar/flat/depth2/depth3 source module byte goldens')
