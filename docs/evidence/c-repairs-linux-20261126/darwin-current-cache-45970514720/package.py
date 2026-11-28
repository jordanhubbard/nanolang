"""I seal the exact downloaded original hosted Darwin artifact archives."""
from pathlib import Path
import gzip,hashlib,json,stat
root=Path(__file__).resolve().parent;out=root.parent/'darwin-current-cache-45970514720-archive';out.mkdir();(out/'objects').mkdir()
sha=lambda b:hashlib.sha256(b).hexdigest()
files=[];chunks={}
for name in ('cache.zip','api-artifacts.json','api-run.json','api-jobs.json','check_cache.py','cache-check.stdout','cache-check.stderr','job-terminal.json','job-terminal.log'):
 p=root/name;assert p.is_file() and not p.is_symlink();keys=[];h=hashlib.sha256();size=0
 with p.open('rb') as f:
  for block in iter(lambda:f.read(2*1024*1024),b''):
   key=sha(block);keys.append(key);size+=len(block);h.update(block)
   if key not in chunks:
    packed=gzip.compress(block,compresslevel=6,mtime=0);(out/'objects'/(key+'.gz')).write_bytes(packed);chunks[key]={'bytes':len(block),'gzip_sha256':sha(packed)}
 files.append({'path':name,'sha256':h.hexdigest(),'size':size,'chunks':keys})
manifest={'version':1,'run':45970514720,'revision':'6e09c07a9f1651294e4669282cdc17ee14126775','artifact_ids':{'cache':15191184169},'artifact_roots':{'cache':'nanolang/nanolang'},'files':files,'chunks':chunks,'scope':'I retain the exact current Darwin cache success:284 methods/26 existing skips,3 worker methods and6 array/pointer methods, matching source/tool/image guards. The original Darwin emitter fails separately; all69 release requirements remain.','whole_release':False}
raw=(json.dumps(manifest,indent=2)+'\n').encode();(out/'manifest.json').write_bytes(raw)
print(json.dumps({'records':len(files),'chunks':len(chunks),'raw_bytes':sum(v['size'] for v in files),'manifest_sha256':sha(raw)}))
