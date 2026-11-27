from pathlib import Path
import json,hashlib,gzip,re,subprocess,collections
b=Path('/home/jkh/.local/share/fleet-migration/puck-volume/nanolang');dev=b/'worktrees/string-binding-hint-20261127';diag=b/'worktrees/alias-scan-diagnostic-20261127';asan=b/'worktrees/string-binding-hint-asan-20261127';out=b/'tmp/string-binding-hint-evidence-20261127';out.mkdir();(out/'objects').mkdir();sha=lambda x:hashlib.sha256(x).hexdigest()
selected={}
for label,p in [('counter',diag/'tmp/alias-scan-diagnostic'),('checks',dev/'tmp/hint-checks'),('emitter-refusal',dev/'tmp/hint-emitter'),('hardware',dev/'tmp/hint-cpu'),('sanitizer',asan/'tmp/sanitizer')]:
 for f in sorted(p.rglob('*')):
  if f.is_file() and f.name!='perf.data' and '__pycache__' not in f.parts:selected[label+'/'+str(f.relative_to(p))]=f
for label,p in [('counter/nanoc_c',diag/'bin/nanoc_c'),('product/nanoc_c',dev/'bin/nanoc_c'),('product/normal-regression',dev/'obj/test_env_string_binding_hint'),('product/asan-initial-regression',asan/'obj/test_env_string_binding_hint'),('product/asan-index',asan/'obj/test_env_symbol_index')]:selected[label]=p
sources=['src/env.c','src/eval.c','src/module.c','src/nanolang.h','src/runtime/gc.h','tests/test_env_string_binding_hint.c','Makefile.gnu','.github/workflows/c-repair-original-gates.yml']
for name in sources:selected['source/'+name]=dev/name
# I seal the actual final sanitizer source tree; its initial observer did not record pre-build hashes.
q={n:sha((asan/n).read_bytes()) for n in subprocess.check_output(['git','ls-files','-z'],cwd=asan).decode().split('\0') if n and not n.startswith('docs/') and (asan/n).is_file()}
meta={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=asan,text=True).strip(),'source_after':q,'initial_prebuild_source_map_recorded':False};p=asan/'tmp/sanitizer-source-after.json';p.write_text(json.dumps(meta,indent=2)+'\n');selected['sanitizer-source-after.json']=p
rows=[];chunks={}
for name,p in selected.items():
 keys=[];h=hashlib.sha256();size=0
 with p.open('rb') as f:
  for data in iter(lambda:f.read(2*1024*1024),b''):
   k=sha(data);keys.append(k);h.update(data);size+=len(data)
   if k not in chunks:
    packed=gzip.compress(data,compresslevel=6,mtime=0);(out/'objects'/(k+'.gz')).write_bytes(packed);chunks[k]={'bytes':len(data),'gzip_sha256':sha(packed)}
 rows.append({'path':name,'sha256':h.hexdigest(),'size':size,'chunks':keys})
m={'version':1,'product_commit':'2006c758df3c67c27f623ff9f7e1e1baaf56c136','candidate':'d18fed86140b6a4cf983898ac32a079002f6b784','base':'fa2c7ff82346abdcbfd0ca46d503ac77984c2722','original_c_source':'6e09c07a9f1651294e4669282cdc17ee14126775','prerequisite_profile_manifest':'b4c28416c3d93b24b19e689183a8b45fe36568d802273ee0337ea8b9b6433165','qualification':False,'whole_release':False,'files':rows,'chunks':chunks};raw=(json.dumps(m,indent=2)+'\n').encode();(out/'manifest.json').write_bytes(raw);print(json.dumps({'sha256':sha(raw),'files':len(rows),'chunks':len(chunks),'raw_bytes':sum(x['size'] for x in rows)}))
