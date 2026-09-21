import hashlib,json,os,subprocess,sys,tarfile,shutil
from pathlib import Path
darwin=sys.platform=='darwin';host='puck' if darwin else 'linux'
root=Path('/tmp/nanolang-record-vm-qualified-46c4d-r2');archive=Path('/tmp/nanolang-record-vm-46c4d-source.tar.gz');manifest=json.loads(Path('/tmp/nanolang-record-vm-46c4d-source.json').read_text())
assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['archive_sha256']
assert shutil.disk_usage('/tmp').free >= 2*1024**3,'I need 2 GiB free before fresh setup'
root.mkdir(exist_ok=False)
with tarfile.open(archive) as tar:tar.extractall(root,filter='data')
for name,digest in manifest['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
(root/'.qualification-tracked').write_text('\n'.join(manifest['files'])+'\n');(root/'.qualification-pin').write_text(manifest['pin']+'\n')
prior=Path('/tmp/nanolang-record-vm-qualified-15d1a').resolve()
prior_report=Path('/tmp/nanolang-record-vm-15d1a-'+host)
last=json.loads((prior_report/'results.json').read_text())[-1]['phase']
assert last=='test-vm-ordinary-admission'
old=json.loads(Path('/tmp/nanolang-record-vm-15d1a-source.json').read_text())
changed={name for name in set(old['files'])|set(manifest['files']) if old['files'].get(name)!=manifest['files'].get(name)}
assert changed=={'docs/ROADMAP.md','tests/nanovm/test_ordinary_admission.c','tests/nanoisa/test_owned_string_proof.c'},changed
for name,digest in old['files'].items():assert hashlib.sha256((prior/name).read_bytes()).hexdigest()==digest,name
rows=json.loads((prior_report/(last+'-artifacts.json')).read_text());copied={};current_metadata={}
for directory in ('obj','bin','lib'):
 if not (prior/directory).exists():continue
 shutil.copytree(prior/directory,root/directory,symlinks=True)
 for p in (prior/directory).rglob('*'):
  if not p.is_file():continue
  relative=p.relative_to(prior);digest=hashlib.sha256(p.read_bytes()).hexdigest()
  assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest,relative
  if str(p) in rows:assert rows[str(p)]['sha256']==digest,relative
  else:
   assert p.suffix in ('.d','.stamp'),relative
   current_metadata[str(relative)]=digest
  copied[str(relative)]=digest
Path('/tmp/nanolang-record-vm-46c4d-r2-'+host+'-reuse.json').write_text(json.dumps({'original_pin':old['pin'],'pin':manifest['pin'],'changed_files':sorted(changed),'prior_terminal':last,'copied_products':copied,'current_only_build_metadata':current_metadata,'qualified_endpoint':str(prior_report/(last+'-artifacts.json'))},indent=2)+'\n')
env=dict(os.environ,LSAN_OPTIONS='')
env['CARRIER_PHASES']='configuration,owned-string-proof-build,owned-string-proof-run,test-vm-ordinary-admission,test-owned-array-overwrite,test-file-private-vm,test-file-cyclic-public' 
if darwin:
 env.update(CC='/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang',CARRIER_SAN_CC='/opt/homebrew/opt/llvm/bin/clang',PATH='/opt/homebrew/opt/llvm/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin',LIBRARY_PATH='/opt/homebrew/opt/openssl@3/lib',NMS_RUNTIME_CLANG='/opt/homebrew/opt/llvm/bin/clang')
 env['SDKROOT']=subprocess.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True,env=env).strip()
 ffi=Path('/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk/usr/include/ffi/ffi.h');assert ffi.is_file()
 env['NANO_FILE_RUNTIME_CFLAGS']='-I'+str(ffi.parent);env['CARRIER_EXTRA_TOOLS']=json.dumps({'libffi_header':str(ffi),'xcrun':'/usr/bin/xcrun'})
else:
 env.update(CC='/bin/gcc-13',CARRIER_SAN_CC='/bin/gcc-13',NMS_RUNTIME_CLANG='/usr/local/bin/clang',NMS_RUNTIME_OPT='/usr/local/bin/opt',NMS_NATIVE_CLANG_FLAGS='--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13')
print(json.dumps({'pin':manifest['pin'],'source_files':len(manifest['files']),'root':str(root.resolve()),'sdk':env.get('SDKROOT')}),flush=True)
subprocess.run([sys.executable,'/tmp/nanolang-record-vm-46c4d-driver.py',str(root),'/tmp/nanolang-record-vm-46c4d-r2-'+host],check=True,env=env)
