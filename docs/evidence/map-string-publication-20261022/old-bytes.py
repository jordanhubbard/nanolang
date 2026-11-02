import os,sys,json,hashlib
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root))
from tests.native_process import run_captured
out=Path(__file__).parent;agg=Path('/Users/jkh/Src/nanolang-51-harness');rows=[]
pairs=[('seed',Path('/private/tmp/nanolang-map-delete-publication-20261022/nano_virt'),out/'nano_virt'),('nano',Path('/private/tmp/nanolang-map-delete-publication-20261022/canonical'),out/'canonical')]
for name,typ,value in [('scalar','int','7'),('flat','array<int>','[7]'),('depth2','array<array<int>>','[[7]]'),('depth3','array<array<array<bool>>>','[[[true]]]')]:
 source=out/'old-byte-source.nano';source.write_text(f'struct Box {{ value: {typ} }} fn main()->int {{ let box: Box = Box {{ value: {value} }} return 0 }}')
 for family,before,after in pairs:
  paths=[]
  for version,tool in [('old',before),('new',after)]:
   module=out/f'{name}-{family}-{version}.nvm';args=[str(tool),str(source),'--emit-nvm','-o',str(module)]
   p=run_captured(args,cwd=root,env=dict(os.environ,NANO_VM="/private/tmp/nanolang-map-string-consumer-20261022/nano_vm"),timeout=180)
   (out/f'{name}-{family}-{version}.stdout').write_bytes(p.stdout);(out/f'{name}-{family}-{version}.stderr').write_bytes(p.stderr)
   assert p.returncode==0,(args,p.stdout,p.stderr);paths.append(module)
  assert paths[0].read_bytes()==paths[1].read_bytes(),(name,family)
  rows.append(dict(case=name,producer=family,sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),bytes=paths[0].stat().st_size,old_provider=str(before),old_provider_sha256=hashlib.sha256(before.read_bytes()).hexdigest(),new_provider_sha256=hashlib.sha256(after.read_bytes()).hexdigest()))
  (out/'old-bytes.json').write_text(json.dumps(rows,indent=2)+'\n')
source.write_bytes((root/'tests/nanoisa/fixtures/map_record_fields.nano').read_bytes())
old=Path('/private/tmp/nanolang-map-delete-publication-20261022')
for family,mode in [('seed','ordinary'),('nano','ordinary'),('c-shadow','shadow'),('raw','raw'),('raw','program')]:
 before=old/({'seed':'nano_virt','nano':'canonical'}.get(family,family));after=out/before.name
 data=[]
 for version,tool in [('old',before),('new',after)]:
  module=out/f'map-{family}-{mode}-{version}.nvm'
  args=([tool,source,'--emit-nvm','-o',module] if mode=='ordinary' else [tool,source] if family=='c-shadow' else [tool,source,0,mode])
  p=run_captured(list(map(str,args)),cwd=root,env=dict(os.environ,NANO_VM='/private/tmp/nanolang-map-string-consumer-20261022/nano_vm'),timeout=180)
  (out/f'map-{family}-{mode}-{version}.stderr').write_bytes(p.stderr)
  assert p.returncode==0,(args,p.stdout,p.stderr)
  contents=module.read_bytes() if mode=='ordinary' else p.stdout
  (out/f'map-{family}-{mode}-{version}.bytes').write_bytes(contents);data.append(contents)
 assert data[0]==data[1],(family,mode)
 rows.append(dict(case='map',producer=family,mode=mode,sha256=hashlib.sha256(data[0]).hexdigest(),bytes=len(data[0]),old_provider_sha256=hashlib.sha256(before.read_bytes()).hexdigest(),new_provider_sha256=hashlib.sha256(after.read_bytes()).hexdigest()))
 (out/'old-bytes.json').write_text(json.dumps(rows,indent=2)+'\n')
print('PASS eight old array/scalar modules plus five SI ordinary/shadow/raw byte goldens at identical source path')
