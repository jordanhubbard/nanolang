"""I seal my current-producer component outputs and retain their setup failure."""
from pathlib import Path
import gzip,hashlib,json,os,stat
base=Path(__file__).resolve().parent
inventory=json.loads((base/'current-components-inventory.json').read_text())
out=base/'current-components-20261127-archive';out.mkdir();(out/'objects').mkdir()
sha=lambda b:hashlib.sha256(b).hexdigest();records=[];chunks={}
for relative,item in sorted(inventory.items()):
 path=Path(item['path']);mode=stat.S_IMODE(path.lstat().st_mode)
 if item['kind']=='symlink':
  records.append({'path':relative,'kind':'symlink','target':os.readlink(path),'mode':mode});continue
 assert path.is_file(),path
 keys=[];h=hashlib.sha256();size=0
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(2*1024*1024),b''):
   key=sha(block);keys.append(key);h.update(block);size+=len(block)
   if key not in chunks:
    data=gzip.compress(block,compresslevel=6,mtime=0);(out/'objects'/(key+'.gz')).write_bytes(data);chunks[key]={'bytes':len(block),'gzip_sha256':sha(data)}
 assert size==item['size'],path
 records.append({'path':relative,'kind':'file','mode':mode,'sha256':h.hexdigest(),'size':size,'chunks':keys})
manifest={'version':1,'revision':'6e09c07a9f1651294e4669282cdc17ee14126775','tree':'/home/jkh/.local/share/fleet-migration/puck-volume/nanolang/worktrees/c-repair-components-current-20261127','core_raw':str(base/'current-components-20261127'),'companion_raw':str(base/'current-components-text-companion-20261127'),'prerequisite_manifest':'docs/evidence/c-repairs-linux-20261126/native-current-linux-20261127/archive/manifest.json','prerequisite_manifest_sha256':'4c3fef1f342e7a6b2d0fe912100f9b1d299455d4db3db2d89783ab2b6c00dd8f','files':records,'chunks':chunks,'scope':'I retain current-qualified-Stage2 eleven-core and three-companion component passes with identical source/tool/existing-image guards. The original fourteen-method observer remains failed29 missing-assembler errors; only that prerequisite is built before the unchanged companion module. I preserve the original limits, failures, source and qualified producer closure. This is not full structured-C, current Darwin/emitter, installed pure-translator or69-entry release acceptance.','whole_release':False}
raw=(json.dumps(manifest,indent=2)+'\n').encode();(out/'manifest.json').write_bytes(raw)
print(json.dumps({'records':len(records),'chunks':len(chunks),'raw_bytes':sum(v.get('size',0) for v in records),'manifest_sha256':sha(raw)}))
