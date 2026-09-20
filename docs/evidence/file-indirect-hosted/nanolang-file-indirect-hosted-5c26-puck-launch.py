import hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path
root=Path('/tmp/nanolang-file-indirect-hosted-5c26'); archive=Path('/tmp/nanolang-file-indirect-hosted-5c26-source.tar.gz')
manifest=json.loads(Path('/tmp/nanolang-file-indirect-hosted-5c26-source.json').read_text())
assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['archive_sha256']
root.mkdir(exist_ok=False)
with tarfile.open(archive) as tar:tar.extractall(root,filter='data')
for name,digest in manifest['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
(root/'.qualification-tracked').write_text('\n'.join(manifest['files'])+'\n')
(root/'.qualification-pin').write_text(manifest['pin']+'\n')
env=dict(os.environ,CC='/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang',CARRIER_SAN_CC='/opt/homebrew/opt/llvm/bin/clang',PATH='/opt/homebrew/opt/llvm/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin',LIBRARY_PATH='/opt/homebrew/opt/openssl@3/lib',LSAN_OPTIONS='',NMS_RUNTIME_CLANG='/opt/homebrew/opt/llvm/bin/clang')
env['SDKROOT']=subprocess.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True,env=env).strip()
ffi=Path('/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk/usr/include/ffi/ffi.h');assert ffi.is_file()
env['NANO_FILE_RUNTIME_CFLAGS']='-I'+str(ffi.parent)
env['CARRIER_EXTRA_TOOLS']=json.dumps({'libffi_header':str(ffi),'xcrun':'/usr/bin/xcrun'})
print(json.dumps({'pin':manifest['pin'],'source_files':len(manifest['files']),'root':str(root.resolve()),'sdk':env['SDKROOT']}),flush=True)
subprocess.run(['/opt/homebrew/bin/python3','/tmp/nanolang-file-indirect-hosted-5c26-driver.py',str(root),'/tmp/nanolang-file-indirect-hosted-5c26-puck'],check=True,env=env)
