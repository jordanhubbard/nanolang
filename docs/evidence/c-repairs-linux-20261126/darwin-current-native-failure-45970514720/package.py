"""I seal the exact current Darwin bootstrap/native pass and emitter refusal."""
from pathlib import Path
import gzip,hashlib,json,stat
root=Path(__file__).resolve().parent;out=root.parent/'darwin-current-native-failure-45970514720-archive';out.mkdir();(out/'objects').mkdir()
sha=lambda b:hashlib.sha256(b).hexdigest()
files=[];chunks={}
for name in ('bootstrap.zip','api-artifacts.json','job-diagnostic-live.json','job-terminal.json','job-terminal.log','job-terminal-initial-cli.stderr','check_bootstrap.py','analyze_trace.py','bootstrap-check.stdout','bootstrap-check.stderr','trace-check.stdout','trace-check.stderr'):
 p=root/name;assert p.is_file() and not p.is_symlink();keys=[];h=hashlib.sha256();size=0
 with p.open('rb') as f:
  for block in iter(lambda:f.read(2*1024*1024),b''):
   key=sha(block);keys.append(key);size+=len(block);h.update(block)
   if key not in chunks:
    packed=gzip.compress(block,compresslevel=6,mtime=0);(out/'objects'/(key+'.gz')).write_bytes(packed);chunks[key]={'bytes':len(block),'gzip_sha256':sha(packed)}
 files.append({'path':name,'sha256':h.hexdigest(),'size':size,'chunks':keys})
manifest={'version':1,'run':45970514720,'revision':'6e09c07a9f1651294e4669282cdc17ee14126775','artifact':15188575446,'artifact_root':'nanolang/nanolang','job':140409516288,'files':files,'chunks':chunks,'scope':'I retain original current Darwin bootstrap17/native36 success and the nanoc_c emitter setup shadow timeout, zero emitter methods. The bounded diagnostic completes all1139 shadows without assertion failure but the parent rejects its clean-exit observation after10 seconds. Actual exit time and exclusive tail cause remain unknown. I accept no complete native/emitter/paired/release gate.','whole_release':False}
raw=(json.dumps(manifest,indent=2)+'\n').encode();(out/'manifest.json').write_bytes(raw)
print(json.dumps({'records':len(files),'chunks':len(chunks),'raw_bytes':sum(v['size'] for v in files),'manifest_sha256':sha(raw)}))
