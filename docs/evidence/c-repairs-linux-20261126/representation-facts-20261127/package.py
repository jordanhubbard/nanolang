"""I seal my accepted classifier fact-join inputs and products."""
from pathlib import Path
import gzip,hashlib,json
base=Path('/home/jkh/.local/share/fleet-migration/puck-volume/nanolang')
first=base/'tmp/representation-facts-20261127';passed=base/'tmp/representation-facts-qualification-20261127'
out=base/'tmp/representation-facts-evidence-20261127';out.mkdir();(out/'objects').mkdir()
sha=lambda b:hashlib.sha256(b).hexdigest()
records=[];chunks={}
roots={'prototype':base/'tmp/representation-facts-20261127','passed':passed,'passed-products':Path(json.loads((passed/'result.json').read_text())['temporary_roots'][0])}
selected={}
for prefix,root in roots.items():
 for p in sorted(root.rglob('*')):
  if p.is_file():
   assert not p.is_symlink();selected[prefix+'/'+str(p.relative_to(root))]=p
work=base/'worktrees/representation-facts-20261127'
for n in ['Makefile.gnu',*json.loads((passed/'result.json').read_text())['new_sources']]:selected['source/'+n]=work/n
for n in ['nanoc_c','nano_virt','nano_vm','nvm2c','nano_aot_runtime.o']:selected['tools/'+n]=work/'bin'/n
for name,p in selected.items():
 keys=[];h=hashlib.sha256();size=0
 with p.open('rb') as f:
  for data in iter(lambda:f.read(2*1024*1024),b''):
   key=sha(data);keys.append(key);h.update(data);size+=len(data)
   if key not in chunks:
    packed=gzip.compress(data,compresslevel=6,mtime=0);(out/'objects'/(key+'.gz')).write_bytes(packed);chunks[key]={'bytes':len(data),'gzip_sha256':sha(packed)}
 records.append({'path':name,'sha256':h.hexdigest(),'size':size,'chunks':keys})
m={'version':1,'whole_release':False,'source_base':'1bd996562d8fa2b55ead6930839d233f4decf6a9','product_c_source':'6e09c07a9f1651294e4669282cdc17ee14126775','prerequisite_manifest_sha256':'4c3fef1f342e7a6b2d0fe912100f9b1d299455d4db3db2d89783ab2b6c00dd8f','files':records,'chunks':chunks}
raw=(json.dumps(m,indent=2)+'\n').encode();(out/'manifest.json').write_bytes(raw)
print(json.dumps({'records':len(records),'chunks':len(chunks),'raw_bytes':sum(r['size'] for r in records),'manifest_sha256':sha(raw)}))
