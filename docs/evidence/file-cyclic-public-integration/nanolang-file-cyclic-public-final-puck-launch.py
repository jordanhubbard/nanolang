import hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path
root=Path('/tmp/nanolang-file-cyclic-public-final'); archive=Path('/tmp/nanolang-file-cyclic-public-final-source.tar.gz')
manifest=json.loads(Path('/tmp/nanolang-file-cyclic-public-final-source.json').read_text())
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
env['CARRIER_PHASES']='schema-consistency,schema-tests,setup,configuration,discovery,ordinary,private-cyclic-dispatch,cyclic-hosted,indirect-hosted,public,archive-boundary'
env['CARRIER_SCHEMA_PYTHON']='/opt/homebrew/opt/python@3.13/bin/python3.13'
yaml_root=Path(subprocess.check_output([env['CARRIER_SCHEMA_PYTHON'],'-c','import yaml;print(yaml.__path__[0])'],text=True).strip())
extra=json.loads(env['CARRIER_EXTRA_TOOLS'])
extra.update({'schema_yaml_'+str(p.relative_to(yaml_root)):str(p) for p in yaml_root.rglob('*') if p.is_file() and p.suffix in ('.py','.so')})
env['CARRIER_EXTRA_TOOLS']=json.dumps(extra)
print(json.dumps({'pin':manifest['pin'],'source_files':len(manifest['files']),'root':str(root.resolve()),'sdk':env['SDKROOT']}),flush=True)
subprocess.run(['/opt/homebrew/bin/python3','/tmp/nanolang-file-cyclic-public-final-driver.py',str(root),'/tmp/nanolang-file-cyclic-public-final-puck'],check=True,env=env)
