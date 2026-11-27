import gzip,hashlib,json,os,stat
from pathlib import Path
BASE=Path('/home/jkh/.local/share/fleet-migration/puck-volume/nanolang/tmp/c-repairs-followups-20261126')
OUT=BASE.parent/'native-followups-20261127-archive';OUT.mkdir();(OUT/'objects').mkdir()
trees=['madmax-native-original','madmax-emitter-profile','record-pointer-overlay-original','native-gate-preflight-20261127','darwin-math-link-control','capture-assembler-second']
files=['run_madmax_native.py','run_madmax_emitter_profile.py','run_pointer_overlay_tests.py','run_pointer_module_parity.py','run_pointer_module_parity_cache.py','profile-context.json','darwin-math-link-control.json','linux-cache-third.log','linux-bootstrap-third.log','darwin-bootstrap-third.log','madmax-cache-57.stdout','madmax-cache-57.stderr','madmax-controlled-boundary.stdout','madmax-controlled-boundary.stderr','madmax-corrected-boundary.stdout','madmax-corrected-boundary.stderr']
paths=[]
for name in trees:
 root=BASE/name;assert root.is_dir(),root
 for directory,dirs,names in os.walk(root,followlinks=False):
  d=Path(directory)
  for n in dirs[:]:
   p=d/n
   if p.is_symlink():paths.append(p);dirs.remove(n)
  paths.extend(d/n for n in names)
paths.extend(BASE/n for n in files)
chunks={};records=[];seen=set()
sha=lambda b:hashlib.sha256(b).hexdigest()
for path in sorted(paths):
 rel=str(path.relative_to(BASE));assert rel not in seen,rel;seen.add(rel);mode=stat.S_IMODE(path.lstat().st_mode)
 if path.is_symlink():
  records.append({'path':rel,'kind':'symlink','target':os.readlink(path),'mode':mode});continue
 assert path.is_file(),path
 parts=[];h=hashlib.sha256();size=0
 with path.open('rb') as stream:
  for raw in iter(lambda:stream.read(2*1024*1024),b''):
   key=sha(raw);parts.append(key);h.update(raw);size+=len(raw)
   if key not in chunks:
    packed=gzip.compress(raw,compresslevel=6,mtime=0);(OUT/'objects'/(key+'.gz')).write_bytes(packed);chunks[key]={'bytes':len(raw),'gzip_sha256':sha(packed)}
 records.append({'path':rel,'kind':'file','mode':mode,'size':size,'sha256':h.hexdigest(),'chunks':parts})
manifest={'version':1,'scope':'I retain original native gate failures, their bounded diagnostics and reversible layout experiments, and the preflight for code6e09c07a9f. I preserve the two old CAST_STRING assertion failures and all69 release requirements. These observations do not qualify the new native candidate or release.','code_revision':'6e09c07a9f1651294e4669282cdc17ee14126775','files':records,'chunks':chunks,'symlinks_are_metadata_only':True,'whole_release':False}
raw=(json.dumps(manifest,indent=2)+'\n').encode();(OUT/'manifest.json').write_bytes(raw)
print(json.dumps({'records':len(records),'files':sum(r['kind']=='file' for r in records),'links':sum(r['kind']=='symlink' for r in records),'chunks':len(chunks),'raw_bytes':sum(r.get('size',0) for r in records),'manifest_sha256':sha(raw)}))
