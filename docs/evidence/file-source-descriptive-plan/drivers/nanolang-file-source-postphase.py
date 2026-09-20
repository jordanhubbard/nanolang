import hashlib,json,pathlib,subprocess,sys,os
root=pathlib.Path(sys.argv[1]).resolve();dest=pathlib.Path(sys.argv[2]);dest.mkdir(exist_ok=False)
def info(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for data in iter(lambda:f.read(1048576),b''):h.update(data)
 return {'sha256':h.hexdigest(),'bytes':p.stat().st_size}
def dump(name,data):(dest/name).write_text(json.dumps(data,indent=2)+'\n')
mods={str(p):info(p) for p in sorted((root/'modules/file_source_catalog/.build').rglob('*')) if p.is_file()}
dump('retained-module-generations-postphase.json',{'scope':'postphase-only inventory of retained actual module generation files; not a historical before map','files':mods})
if sys.platform=='darwin':
 queries=[('owning-clang',['/usr/bin/xcrun','--find','clang']),('sdk',['/usr/bin/xcrun','--show-sdk-path']),('sdk-version',['/usr/bin/xcrun','--show-sdk-version'])]
 values={}
 for name,args in queries:
  result=subprocess.run(args,capture_output=True,timeout=30)
  (dest/(name+'.stdout')).write_bytes(result.stdout);(dest/(name+'.stderr')).write_bytes(result.stderr)
  dump(name+'-command.json',{'argv':args,'returncode':result.returncode,'scope':'postphase-only tool identity query'})
  if result.returncode:raise RuntimeError(name)
  values[name]=result.stdout.decode().strip()
 compiler=pathlib.Path(values['owning-clang']).resolve()
 result=subprocess.run([str(compiler),'--version'],capture_output=True,timeout=30)
 (dest/'owning-clang-version.stdout').write_bytes(result.stdout);(dest/'owning-clang-version.stderr').write_bytes(result.stderr)
 dump('owning-clang-postphase.json',{'scope':'resolved owning Apple Clang hash measured only after passing gates; original maps retain selected /usr/bin/clang shim','path':str(compiler),'identity':info(compiler),'version_returncode':result.returncode,'sdk':values})
 if result.returncode:raise RuntimeError('version')
print('Retained module files:',len(mods))
