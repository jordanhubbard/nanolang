import gzip,hashlib,json,stat
from pathlib import Path
base=Path('/home/jkh/.local/share/fleet-migration/puck-volume/nanolang/tmp/c-repairs-followups-20261126')
source=base/'hosted-linux-bootstrap-second-artifact';out=base/'hosted-bootstrap-second-archive';out.mkdir();(out/'objects').mkdir()
sha=lambda b:hashlib.sha256(b).hexdigest()
records=[];chunks={}
for f in sorted(source.rglob('*')):
 if not f.is_file():continue
 assert not f.is_symlink(),f
 h=hashlib.sha256();parts=[];size=0
 with f.open('rb') as stream:
  for block in iter(lambda:stream.read(2*1024*1024),b''):
   key=sha(block);parts.append(key);h.update(block);size+=len(block)
   if key not in chunks:
    data=gzip.compress(block,compresslevel=6,mtime=0);(out/'objects'/(key+'.gz')).write_bytes(data);chunks[key]={'bytes':len(block),'gzip_sha256':sha(data)}
 records.append({'path':str(f.relative_to(source)),'sha256':h.hexdigest(),'size':size,'mode':stat.S_IMODE(f.stat().st_mode),'chunks':parts})
manifest={'version':1,'run':45952317501,'revision':'ee5d38a4b401249ecb7dcfaa0b7b04001fdc8dc1','artifact':15178888555,'files':records,'chunks':chunks,'root':'nanolang/nanolang','scope':'I retain the original hosted Linux bootstrap and native36 pass, the emitter setup refusal, and the missing generated module-index evidence gap. This is not full native or release acceptance.'}
raw=(json.dumps(manifest,indent=2)+'\n').encode();(out/'manifest.json').write_bytes(raw)
print(json.dumps({'files':len(records),'chunks':len(chunks),'raw_bytes':sum(r['size'] for r in records),'manifest_sha256':sha(raw)}))
