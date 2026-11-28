import gzip,hashlib,json,os,stat
from pathlib import Path
ROOT=Path('/home/jkh/.local/share/fleet-migration/puck-volume/nanolang/worktrees/c-repair-native-gates-20261127')
BASE=Path('/home/jkh/.local/share/fleet-migration/puck-volume/nanolang/tmp/c-repairs-followups-20261126')
OUT=BASE.parent/'native-current-linux-20261127-archive';OUT.mkdir();(OUT/'objects').mkdir()
q=json.loads((ROOT/'.test_output/frozen-native-qualification/result.json').read_text());assert q['terminal'] and q['component_native_gate_passed'] and q['guards_match'] and q['generated_index_matches_receipt'] and q['host_closure_matches_receipt']
b=json.loads((ROOT/'bin/nanoc_bootstrap.json').read_text());paths=[]
for directory in [ROOT/'.test_output/frozen-native-qualification',ROOT/'obj/bootstrap-nanoisa',ROOT/'obj/module_cache']:
 for current,dirs,files in os.walk(directory,followlinks=False):
  p=Path(current)
  for name in dirs[:]:
   f=p/name
   if f.is_symlink():paths.append((str(f.relative_to(ROOT)),f));dirs.remove(name)
  paths.extend((str((p/name).relative_to(ROOT)),p/name) for name in files)
for name in ['nanoc_c','nano_virt','nano_vm','nvm2c','nanoisa','nano_aot_runtime.o','nano_as_capture.so','nanoc_seed.nvm','nanoc_stage1.nvm','nanoc_stage2.nvm','nanoc_stage1','nanoc_stage2','nanoc','nanoc_bootstrap.json']:paths.append(('bin/'+name,ROOT/'bin'/name))
paths.append(('modules/index.json',ROOT/'modules/index.json'))
paths.append(('observer.py',BASE/'run_sparky_native_candidate.py'))
# I retain the actual selected external tools without depending on their replay-host paths.
tool_paths={}
for i,name in enumerate(sorted(q['tools_after'])):
 if not Path(name).is_relative_to(ROOT):
  stored='selected-tools/'+str(i)+'-'+Path(name).name;paths.append((stored,Path(name)));tool_paths[name]=stored
records=[];chunks={};seen=set();sha=lambda b:hashlib.sha256(b).hexdigest()
for rel,f in sorted(paths):
 assert rel not in seen,rel;seen.add(rel);mode=stat.S_IMODE(f.lstat().st_mode)
 if f.is_symlink():records.append({'path':rel,'kind':'symlink','target':os.readlink(f),'mode':mode});continue
 assert f.is_file(),f
 keys=[];h=hashlib.sha256();size=0
 with f.open('rb') as stream:
  for raw in iter(lambda:stream.read(2*1024*1024),b''):
   key=sha(raw);keys.append(key);h.update(raw);size+=len(raw)
   if key not in chunks:
    data=gzip.compress(raw,compresslevel=6,mtime=0);(OUT/'objects'/(key+'.gz')).write_bytes(data);chunks[key]={'bytes':len(raw),'gzip_sha256':sha(data)}
 records.append({'path':rel,'kind':'file','mode':mode,'size':size,'sha256':h.hexdigest(),'chunks':keys})
manifest={'version':1,'revision':q['head_after'],'original_root':str(ROOT),'files':records,'chunks':chunks,'external_tools':tool_paths,'symlinks_are_metadata_only':True,'scope':'I retain current-source Linux bootstrap17/native36/emitter12 success with matching source/tool/head/index/host guards. The broad CAST_STRING decision, paired current Darwin and all69 release requirements remain open.','whole_release':False}
raw=(json.dumps(manifest,indent=2)+'\n').encode();(OUT/'manifest.json').write_bytes(raw)
print(json.dumps({'records':len(records),'chunks':len(chunks),'raw_bytes':sum(r.get('size',0) for r in records),'manifest_sha256':sha(raw)}))
