"""I retain my generated compiler index separately from Git source inputs."""
from pathlib import Path
import gzip,hashlib,json,os
root=Path.cwd();here=root/'docs/evidence/ownership-queries-linux-20261122'
manifest=json.loads((here/'manifest.json').read_text());raw=(root/'modules/index.json').read_bytes();digest=hashlib.sha256(raw).hexdigest()
name='generated/modules/index.json';assert not any(r['path']==name for r in manifest['files'])
base=root/'docs/evidence/affine-cfg-foundation-linux-20261122';old=json.loads((base/'manifest.json').read_text())
row=next((dict(r) for r in old['files'] if r['sha256']==digest),None)
if row is None:
 packed=gzip.compress(raw,mtime=0);p=here/'objects'/(digest+'.gz');p.write_bytes(packed)
 row=dict(bytes=len(raw),sha256=digest,artifact='objects/'+p.name,compressed_bytes=len(packed),compressed_sha256=hashlib.sha256(packed).hexdigest())
else:
 row['artifact']=os.path.relpath((base/row['artifact']).resolve(),here)
row['path']=name;manifest['files'].append(row)
manifest['counts']['records']+=1;manifest['counts']['logical_bytes']+=len(raw)
manifest['counts']['unique_objects']=len({r['sha256'] for r in manifest['files']})
(here/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(generated_index_sha256=digest,**manifest['counts']),indent=2))
