import hashlib,json,os,subprocess,sys,tarfile,shutil
from pathlib import Path
root=Path('/tmp/nanolang-file-cyclic-carrier-ac877'); archive=Path('/tmp/nanolang-file-cyclic-carrier-ac877-source.tar.gz')
manifest=json.loads(Path('/tmp/nanolang-file-cyclic-carrier-ac877-source.json').read_text())
assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['archive_sha256']
assert root.is_dir()
for name,digest in manifest['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
env=dict(os.environ,CC='/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang',CARRIER_SAN_CC='/opt/homebrew/opt/llvm/bin/clang',PATH='/opt/homebrew/opt/llvm/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin',LIBRARY_PATH='/opt/homebrew/opt/openssl@3/lib',LSAN_OPTIONS='',NMS_RUNTIME_CLANG='/opt/homebrew/opt/llvm/bin/clang')
env['SDKROOT']=subprocess.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True,env=env).strip()
ffi=Path('/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk/usr/include/ffi/ffi.h');assert ffi.is_file()
env['NANO_FILE_RUNTIME_CFLAGS']='-I'+str(ffi.parent)
env['CARRIER_EXTRA_TOOLS']=json.dumps({'libffi_header':str(ffi),'xcrun':'/usr/bin/xcrun'})
env['GIT_DIR']='/Users/jkh/Src/nanolang/.git'
env['GIT_WORK_TREE']=str(root.resolve())
env['CARRIER_PHASES']='public-linked'
env['CARRIER_CONFIGURATION']='/tmp/nanolang-file-cyclic-carrier-ac877-puck/configuration.log'
blob=subprocess.check_output(['git','show','f1606e2c84e67491e9652a5bf71944d235216d95:src/nanoisa/nvm2c_file_private.c'],env=env)
identity=subprocess.check_output(['git','rev-parse','f1606e2c84e67491e9652a5bf71944d235216d95:src/nanoisa/nvm2c_file_private.c'],env=env,text=True).strip()
assert identity=='732f38decf290ab7e3288d7f77421848aa235c24'
history=Path('/tmp/nanolang-file-cyclic-carrier-history-source.c');assert not history.exists();history.write_bytes(blob)
env['CARRIER_HISTORY_BLOB']=str(history)
extra=json.loads(env['CARRIER_EXTRA_TOOLS']);extra.update(git=shutil.which('git',path=env['PATH']),historical_source=str(history));env['CARRIER_EXTRA_TOOLS']=json.dumps(extra)
Path('/tmp/nanolang-file-cyclic-carrier-history-input.json').write_text(json.dumps({'git_dir':env['GIT_DIR'],'git_work_tree':env['GIT_WORK_TREE'],'commit':'f1606e2c84e67491e9652a5bf71944d235216d95','blob':identity,'sha256':hashlib.sha256(blob).hexdigest()},indent=2)+'\n')
print(json.dumps({'pin':manifest['pin'],'source_files':len(manifest['files']),'root':str(root.resolve()),'sdk':env['SDKROOT']}),flush=True)
subprocess.run(['/opt/homebrew/bin/python3','/tmp/nanolang-file-cyclic-carrier-history-driver.py',str(root),'/tmp/nanolang-file-cyclic-carrier-history-puck'],check=True,env=env)
