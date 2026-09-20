from pathlib import Path
import hashlib,json,shutil
root=Path('/home/jkh/Src/nanolang-cast-u8-backends');base=root/'docs/evidence/cast-u8-backends';base.mkdir(parents=True,exist_ok=False)
objects={};summary={}
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for host in ['linux','puck']:
 src=Path('/tmp/nanolang-byte-backend-'+host+'-seal');dst=base/host
 shutil.copytree(src/'reports',dst)
 for name in ['summary.json','report-sha256.json','artifact-store.json']:shutil.copyfile(src/name,dst/name)
 for h,v in json.loads((src/'artifact-store.json').read_text()).items():
  p=src/'objects'/h;assert sha(p)==h and p.stat().st_size==v['bytes']
  row=objects.setdefault(h,{'bytes':v['bytes'],'copies':[]});assert row['bytes']==v['bytes'];row['copies'].append(str(p))
 summary[host]=json.loads((src/'summary.json').read_text())
for name,v in [('objects.json',objects),('summary.json',summary)]: (base/name).write_text(json.dumps(v,indent=2)+'\n')
for name in ['seal-cast-u8-backends.py','publish-cast-u8-backends.py','collect-cast-u8-backends-inner.py']:shutil.copyfile('/tmp/'+name,base/name)
archive=Path('/tmp/nanolang-byte-backend-puck-evidence.tar.gz');(base/'transport.json').write_text(json.dumps({'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size},indent=2)+'\n')
manifest={str(p.relative_to(root)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(base.rglob('*')) if p.is_file()}
(base/'seal-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print({'reports':len(manifest),'objects':len(objects),'bytes':sum(x['bytes'] for x in objects.values()),'references':sum(x['references'] for x in summary.values()),'pairs':sum(x['equal_pairs'] for x in summary.values())})
