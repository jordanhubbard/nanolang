from pathlib import Path
import hashlib,json,os,subprocess
root=Path.cwd();tmp=Path('/private/tmp/nanolang-mixed-record-candidate');cache=tmp/'cache';cache.mkdir(exist_ok=True)
for base in ['modules','std','stdlib']:
 for directory,dirs,files in os.walk(root/base):
  if '.build' in dirs:
   module=Path(directory);key='v2-'+hashlib.sha256(str(module.resolve()).encode()).hexdigest();dest=cache/key
   if module!=root/'modules/nanoisa' and not dest.exists():dest.symlink_to(module/'.build',target_is_directory=True)
   dirs.remove('.build')
module=root/'modules/nanoisa';key='v2-'+hashlib.sha256(str(module.resolve()).encode()).hexdigest();dest=cache/key;dest.mkdir(exist_ok=True)
meta=json.loads((module/'module.json').read_text());sources=[]
for name in meta['c_sources']:
 p=(module/name).resolve()
 if p.name in ['affine_state.c','affine_bytecode.c']:p=tmp/'c'/p.name
 sources.append(str(p))
command=['cc','-dynamiclib','-std=c99','-O2','-fPIC','-D_GNU_SOURCE','-I'+str(root/'src'),'-I'+str(root/'src/nanoisa'),'-iquote',str(root/'src/nanoisa'),*sources,'-o',str(dest/'libnanoisa.dylib')]
print(' '.join(command),flush=True);subprocess.run(command,check=True)
